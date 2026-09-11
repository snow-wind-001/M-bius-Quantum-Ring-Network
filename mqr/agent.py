"""Unified temporal-utility MQR agent and Go learning objectives.

The runtime keeps task action and memory-write action separate.  A temporal
MQR produces a persistent state, four Go heads expose placement/legality/pass/
value predictions, and a causal utility critic decides whether slow rings
receive the current observation.  Every public transaction predicts under
``theta_t`` before optional feedback mutates ``theta_(t+1)``.

This module intentionally does not depend on MiniCPM.  External encoders can
consume ``grad_features`` returned by :meth:`apply_feedback` and consolidate a
LoRA adapter on a slower schedule.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .online import OrthogonalGradientMemory
from .safety import SidecarSafetyLimits, categorical_policy_kl, tensor_linf_drift
from .temporal import MultiTimescaleMQR, TemporalMQRState
from .utility import CAUSAL_UTILITY_FEATURES, FutureUtilityGate


@dataclass(frozen=True)
class GoMultiHeadOutput:
    """Outputs for a small-board Go position.

    Shapes are ``placement_logits=[B, points]``,
    ``legality_logits=[B, points]``, ``pass_logit=[B]``, ``value=[B]``, and
    ``latent=[B, latent_dim]``.  Pass is deliberately separate from placement
    so terminal examples cannot dominate the point distribution.
    """

    placement_logits: torch.Tensor
    legality_logits: torch.Tensor
    pass_logit: torch.Tensor
    value: torch.Tensor
    latent: torch.Tensor
    legality_policy_scale: float = 0.0

    @property
    def policy_logits(self) -> torch.Tensor:
        """Return normalized joint action log-probabilities.

        ``placement_logits`` parameterize ``P(point | not-pass)`` while
        ``pass_logit`` parameterizes the binary event ``P(pass)``.  Directly
        concatenating those two kinds of logits is not probabilistically
        coherent and makes pass calibration depend on board area.  The
        factorization below is normalized over ``points + pass`` by construction.
        """

        placement = self.placement_logits
        if self.legality_policy_scale != 0.0:
            # Learned soft prior: no oracle mask is applied, so illegal points
            # remain selectable and raw legality remains a meaningful metric.
            placement = placement + float(self.legality_policy_scale) * F.logsigmoid(
                self.legality_logits
            )
        conditional_point_log_probs = F.log_softmax(placement, dim=1)
        point_log_probs = (
            F.logsigmoid(-self.pass_logit).unsqueeze(1)
            + conditional_point_log_probs
        )
        pass_log_prob = F.logsigmoid(self.pass_logit).unsqueeze(1)
        return torch.cat([point_log_probs, pass_log_prob], dim=1)

    def detached(self) -> "GoMultiHeadOutput":
        return GoMultiHeadOutput(
            placement_logits=self.placement_logits.detach().clone(),
            legality_logits=self.legality_logits.detach().clone(),
            pass_logit=self.pass_logit.detach().clone(),
            value=self.value.detach().clone(),
            latent=self.latent.detach().clone(),
            legality_policy_scale=self.legality_policy_scale,
        )


@dataclass(frozen=True)
class GoLossWeights:
    """Weights for supervised, offline-AWR, and short-horizon actor-critic losses."""

    placement: float = 1.0
    legality: float = 1.0
    pass_decision: float = 1.0
    value: float = 0.5
    awr: float = 0.0
    ppo: float = 0.0
    entropy: float = 0.0
    reference_kl: float = 0.0
    illegal_mass: float = 0.0
    policy_distillation: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.placement,
            self.legality,
            self.pass_decision,
            self.value,
            self.awr,
            self.ppo,
            self.entropy,
            self.reference_kl,
            self.illegal_mass,
            self.policy_distillation,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError("all Go loss weights must be finite and non-negative")


class GoSpatialSkipHeads(nn.Module):
    """Shared rule-neutral local observation path for the four Go heads.

    The lossless Go vector begins with own-stone, opponent-stone, and previous
    move planes.  A translation-equivariant local path preserves those
    coordinates instead of forcing every board point through a tiny global
    latent.  Output layers are zero initialized, and no legality mask or
    simulator history is supplied.
    """

    def __init__(
        self, input_dim: int, board_size: int, channels: int, *,
        geometry: bool = False, depth: int = 1,
    ) -> None:
        super().__init__()
        if input_dim < 3 * board_size * board_size:
            raise ValueError("spatial skip requires three flattened board planes")
        if channels <= 0:
            raise ValueError("spatial skip channels must be positive")
        if depth not in (1, 2):
            raise ValueError("spatial depth must be one or two")
        self.input_dim = int(input_dim)
        self.board_size = int(board_size)
        self.points = self.board_size * self.board_size
        self.channels = int(channels)
        self.extra_dim = self.input_dim - 3 * self.points
        self.geometry = bool(geometry)
        coordinates = torch.linspace(-1.0, 1.0, board_size)
        row, column = torch.meshgrid(coordinates, coordinates, indexing="ij")
        # Validity distinguishes padding from an empty intersection; signed
        # coordinates distinguish points with identical stone neighborhoods.
        self.register_buffer("geometry_planes", torch.stack((
            torch.ones_like(row), row, column, row.square() + column.square(),
        )).unsqueeze(0), persistent=False)
        self.local = nn.Conv2d(7 if geometry else 3, self.channels, kernel_size=3, padding=1)
        self.local_second = (
            nn.Conv2d(channels, channels, kernel_size=3, padding=1) if depth == 2 else None
        )
        self.placement = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.legality = nn.Conv2d(self.channels, 1, kernel_size=1)
        global_dim = self.channels + self.extra_dim
        self.pass_decision = nn.Linear(global_dim, 1)
        self.value = nn.Linear(global_dim, 1)
        for module in (
            self.placement,
            self.legality,
            self.pass_decision,
            self.value,
        ):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return [batch, channels, size, size] rule-neutral board features."""
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"x must have shape [batch, {self.input_dim}]")
        planes = x[:, : 3 * self.points].reshape(
            x.size(0), 3, self.board_size, self.board_size
        )
        if self.geometry:
            planes = torch.cat((planes, self.geometry_planes.to(x).expand(x.size(0), -1, -1, -1)), dim=1)
        hidden = torch.tanh(self.local(planes))
        if self.local_second is not None:
            hidden = torch.tanh(self.local_second(hidden))
        return hidden

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.spatial_features(x)
        placement = self.placement(hidden).flatten(1)
        legality = self.legality(hidden).flatten(1)
        pooled = hidden.mean(dim=(2, 3))
        if self.extra_dim:
            pooled = torch.cat((pooled, x[:, 3 * self.points :]), dim=1)
        return (
            placement,
            legality,
            self.pass_decision(pooled).squeeze(1),
            self.value(pooled).squeeze(1),
        )


@dataclass(frozen=True)
class SlowLoRAConsolidationSchedule:
    """Auditable schedule for slow external-adapter consolidation.

    The schedule makes no optimizer call itself.  A MiniCPM bridge should call
    :meth:`should_consolidate` before applying a returned feature gradient.
    """

    warmup_updates: int = 64
    interval: int = 32
    minimum_abs_advantage: float = 0.0

    def __post_init__(self) -> None:
        if self.warmup_updates < 0 or self.interval <= 0:
            raise ValueError("warmup_updates must be non-negative and interval positive")
        if not math.isfinite(float(self.minimum_abs_advantage)) or self.minimum_abs_advantage < 0:
            raise ValueError("minimum_abs_advantage must be finite and non-negative")

    def should_consolidate(
        self,
        update_index: int,
        *,
        write_advantage: Optional[float] = None,
    ) -> bool:
        index = int(update_index)
        if index < self.warmup_updates:
            return False
        if (index - self.warmup_updates) % self.interval != 0:
            return False
        if write_advantage is None:
            return self.minimum_abs_advantage == 0.0
        return abs(float(write_advantage)) >= self.minimum_abs_advantage


class SlowLoRAConsolidator(nn.Module):
    """Apply external feature gradients to a LoRA encoder on a slow schedule."""

    def __init__(
        self,
        schedule: SlowLoRAConsolidationSchedule,
        *,
        lr: float,
        ogd_max_rank: int = 0,
        max_grad_norm: Optional[float] = 1.0,
        max_update_norm: Optional[float] = None,
        safety_limits: Optional[SidecarSafetyLimits] = None,
    ) -> None:
        super().__init__()
        if not isinstance(schedule, SlowLoRAConsolidationSchedule):
            raise TypeError("schedule must be a SlowLoRAConsolidationSchedule")
        if not math.isfinite(float(lr)) or lr <= 0.0:
            raise ValueError("lr must be finite and positive")
        if max_grad_norm is not None and (
            not math.isfinite(float(max_grad_norm)) or max_grad_norm <= 0.0
        ):
            raise ValueError("max_grad_norm must be finite and positive or None")
        if max_update_norm is not None and (
            not math.isfinite(float(max_update_norm)) or max_update_norm <= 0.0
        ):
            raise ValueError("max_update_norm must be finite and positive or None")
        if safety_limits is not None and not isinstance(
            safety_limits, SidecarSafetyLimits
        ):
            raise TypeError("safety_limits must be a SidecarSafetyLimits or None")
        self.schedule = schedule
        self.lr = float(lr)
        self.max_grad_norm = max_grad_norm
        self.max_update_norm = max_update_norm
        self.safety_limits = safety_limits or SidecarSafetyLimits()
        self.gradient_memory = OrthogonalGradientMemory(ogd_max_rank)
        self.register_buffer("opportunities", torch.zeros((), dtype=torch.long))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def maybe_step(
        self,
        encoder: Any,
        features: torch.Tensor,
        grad_features: torch.Tensor,
        *,
        update_index: int,
        write_advantage: Optional[float] = None,
        remember_gradient: bool = False,
        project_with_memory: bool = True,
        constancy_closure: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Conditionally call ``encoder.step_from_external_gradient``.

        ``features`` must still own the original encoder graph.  Skipped calls
        never inspect or mutate encoder gradients, which makes the slow/fast
        update ratio directly auditable.
        """

        self.opportunities.add_(1)
        scheduled = self.schedule.should_consolidate(
            int(update_index),
            write_advantage=write_advantage,
        )
        if not scheduled:
            return {
                "scheduled": False,
                "did_update": False,
                "update_index": int(update_index),
                "updates": int(self.updates.item()),
            }
        if not hasattr(encoder, "step_from_external_gradient"):
            raise TypeError("encoder must expose step_from_external_gradient")
        if features.shape != grad_features.shape:
            raise ValueError("features and grad_features must have the same shape")
        if remember_gradient and self.gradient_memory.max_rank == 0:
            raise ValueError("remember_gradient requires LoRA OGD capacity")
        info = encoder.step_from_external_gradient(
            features,
            grad_features,
            lr=self.lr,
            orthogonal_memory=(
                self.gradient_memory if self.gradient_memory.max_rank > 0 else None
            ),
            remember_gradient=remember_gradient,
            project_with_memory=project_with_memory,
            max_grad_norm=self.max_grad_norm,
            max_update_norm=self.max_update_norm,
            constancy_closure=constancy_closure,
            safety_limits=self.safety_limits,
        )
        did_update = bool(
            info.get("did_update", float(info.get("update_norm", 0.0)) > 0.0)
        )
        if did_update:
            self.updates.add_(1)
        return {
            "scheduled": True,
            "did_update": did_update,
            "update_index": int(update_index),
            "updates": int(self.updates.item()),
            **info,
        }


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    bootstrap_value: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return GAE advantages and lambda returns for one or more trajectories.

    Inputs have shape ``[time, batch]``.  ``dones[t]`` indicates that reward
    ``t`` terminates the episode.  The computation is detached by design;
    policy/value gradients are created only when the returned targets are used.
    """

    if rewards.dim() != 2 or values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("rewards, values, and dones must share shape [time, batch]")
    if not rewards.is_floating_point() or not values.is_floating_point():
        raise TypeError("rewards and values must be floating point")
    if not (0.0 <= float(gamma) <= 1.0) or not (0.0 <= float(gae_lambda) <= 1.0):
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    if not bool(torch.isfinite(rewards).all()) or not bool(torch.isfinite(values).all()):
        raise ValueError("rewards and values must be finite")
    done_values = dones.to(device=rewards.device, dtype=rewards.dtype)
    if not bool(((done_values == 0.0) | (done_values == 1.0)).all()):
        raise ValueError("dones must contain only zero or one")
    if bootstrap_value is None:
        next_value = torch.zeros(rewards.size(1), device=rewards.device, dtype=rewards.dtype)
    else:
        if bootstrap_value.shape != (rewards.size(1),):
            raise ValueError("bootstrap_value must have shape [batch]")
        next_value = bootstrap_value.detach().to(device=rewards.device, dtype=rewards.dtype)

    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros_like(next_value)
    with torch.no_grad():
        for index in range(rewards.size(0) - 1, -1, -1):
            continuation = 1.0 - done_values[index]
            delta = rewards[index] + float(gamma) * continuation * next_value - values[index]
            accumulator = (
                delta
                + float(gamma) * float(gae_lambda) * continuation * accumulator
            )
            advantages[index] = accumulator
            next_value = values[index]
    return advantages, advantages + values.detach()


class TemporalUtilityMQRAgent(nn.Module):
    """Unified causal agent with temporal MQR memory and four Go heads.

    The reference runtime accepts one observation per stream transaction.  Fast
    ring zero always writes; one learned utility action jointly opens or closes
    every slow ring.  A candidate ticket preserves no-write/write shadow states
    for simulator calibration.  Single-ticket feedback differentiates one
    committed transition, while bounded trajectory feedback replays consecutive
    committed writes so future losses can train earlier memory content.  Both
    paths preserve predict-before-update ordering and support optional OGD.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        board_size: int = 5,
        ring_dim: int = 24,
        latent_dim: int = 32,
        leak_rates: Sequence[float] = (1.0, 0.10, 0.02),
        injection_rank: int = 8,
        transition_mode: str = "unistochastic",
        learn_transitions: bool = True,
        transition_structure: str = "dense_cayley",
        cyclic_givens_layers: int = 2,
        cayley_coordinate_mode: str = "projected",
        base_unitary_init: str = "identity",
        base_unitary_scale: float = 0.25,
        base_unitary_seed: Optional[int] = None,
        utility_rank: int = 8,
        utility_content_dim: int = 0,
        utility_content_seed: int = 0,
        utility_content_projection_mode: str = "random",
        initial_write_advantage: float = -0.05,
        write_warmup_observations: int = 0,
        write_exploration_probability: float = 0.0,
        write_exploration_decay_observations: int = 0,
        memory_cost: float = 0.0,
        advantage_scale: float = 1.0,
        task_lr: float = 1e-2,
        utility_lr: float = 5e-2,
        ogd_max_rank: int = 0,
        utility_ogd_max_rank: int = 0,
        max_update_norm: Optional[float] = 0.25,
        utility_max_update_norm: Optional[float] = 0.05,
        transition_update_interval: int = 1,
        safety_limits: Optional[SidecarSafetyLimits] = None,
        max_trace_horizon: int = 32,
        max_streams: int = 64,
        max_pending_tickets: int = 128,
        loss_weights: Optional[GoLossWeights] = None,
        awr_temperature: float = 1.0,
        awr_max_weight: float = 20.0,
        ppo_clip: float = 0.2,
        legality_policy_scale: float = 0.0,
        initial_pass_probability: Optional[float] = None,
        spatial_skip_channels: int = 0,
        core: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or ring_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim, ring_dim, and latent_dim must be positive")
        if board_size < 2:
            raise ValueError("board_size must be at least two")
        if not math.isfinite(float(memory_cost)) or memory_cost < 0.0:
            raise ValueError("memory_cost must be finite and non-negative")
        if not math.isfinite(float(advantage_scale)) or advantage_scale <= 0.0:
            raise ValueError("advantage_scale must be finite and positive")
        if task_lr <= 0.0 or utility_lr <= 0.0:
            raise ValueError("task_lr and utility_lr must be positive")
        if max_streams <= 0 or max_pending_tickets <= 0:
            raise ValueError("runtime capacities must be positive")
        if (
            isinstance(max_trace_horizon, bool)
            or int(max_trace_horizon) != max_trace_horizon
            or int(max_trace_horizon) <= 0
        ):
            raise ValueError("max_trace_horizon must be a positive integer")
        if (
            isinstance(transition_update_interval, bool)
            or int(transition_update_interval) != transition_update_interval
            or int(transition_update_interval) <= 0
        ):
            raise ValueError("transition_update_interval must be a positive integer")
        if safety_limits is not None and not isinstance(
            safety_limits, SidecarSafetyLimits
        ):
            raise TypeError("safety_limits must be a SidecarSafetyLimits or None")
        if awr_temperature <= 0.0 or awr_max_weight <= 0.0:
            raise ValueError("AWR temperature and max weight must be positive")
        if not (0.0 < ppo_clip < 1.0):
            raise ValueError("ppo_clip must lie in (0, 1)")
        if not math.isfinite(float(legality_policy_scale)) or legality_policy_scale < 0.0:
            raise ValueError("legality_policy_scale must be finite and non-negative")
        if isinstance(spatial_skip_channels, bool) or int(spatial_skip_channels) < 0:
            raise ValueError("spatial_skip_channels must be a non-negative integer")
        if isinstance(utility_content_dim, bool) or int(utility_content_dim) < 0:
            raise ValueError("utility_content_dim must be a non-negative integer")
        if isinstance(utility_content_seed, bool) or int(utility_content_seed) != utility_content_seed:
            raise ValueError("utility_content_seed must be an integer")
        if utility_content_projection_mode not in ("random", "identity"):
            raise ValueError(
                'utility_content_projection_mode must be "random" or "identity"'
            )
        if (
            utility_content_projection_mode == "identity"
            and int(utility_content_dim) not in (0, int(input_dim))
        ):
            raise ValueError(
                "identity utility projection requires utility_content_dim=input_dim"
            )
        for name, value in (
            ("write_warmup_observations", write_warmup_observations),
            ("write_exploration_decay_observations", write_exploration_decay_observations),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not math.isfinite(float(write_exploration_probability)) or not (
            0.0 <= float(write_exploration_probability) <= 1.0
        ):
            raise ValueError("write_exploration_probability must lie in [0, 1]")

        self.input_dim = int(input_dim)
        self.board_size = int(board_size)
        self.points = self.board_size * self.board_size
        self.action_size = self.points + 1
        if initial_pass_probability is None:
            initial_pass_probability = 1.0 / float(self.action_size)
        if not (
            math.isfinite(float(initial_pass_probability))
            and 0.0 < float(initial_pass_probability) < 1.0
        ):
            raise ValueError("initial_pass_probability must lie strictly in (0, 1)")
        self.initial_pass_probability = float(initial_pass_probability)
        self.memory_cost = float(memory_cost)
        self.advantage_scale = float(advantage_scale)
        self.task_lr = float(task_lr)
        self.utility_lr = float(utility_lr)
        self.max_update_norm = max_update_norm
        self.utility_max_update_norm = utility_max_update_norm
        self.transition_update_interval = int(transition_update_interval)
        self.safety_limits = safety_limits or SidecarSafetyLimits()
        self.max_trace_horizon = int(max_trace_horizon)
        self.max_streams = int(max_streams)
        self.max_pending_tickets = int(max_pending_tickets)
        self.loss_weights = loss_weights or GoLossWeights()
        self.awr_temperature = float(awr_temperature)
        self.awr_max_weight = float(awr_max_weight)
        self.ppo_clip = float(ppo_clip)
        self.legality_policy_scale = float(legality_policy_scale)
        self.spatial_skip_channels = int(spatial_skip_channels)
        self.utility_content_dim = int(utility_content_dim)
        self.utility_content_seed = int(utility_content_seed)
        self.utility_content_projection_mode = str(utility_content_projection_mode)
        self.write_warmup_observations = int(write_warmup_observations)
        self.write_exploration_probability = float(write_exploration_probability)
        self.write_exploration_decay_observations = int(
            write_exploration_decay_observations
        )

        if core is None:
            resolved_core: nn.Module = MultiTimescaleMQR(
                self.input_dim,
                int(ring_dim),
                int(latent_dim),
                leak_rates=leak_rates,
                write_scales=tuple(1.0 for _ in leak_rates),
                injection_rank=injection_rank,
                injection_activation="tanh",
                state_activation="tanh",
                transition_mode=transition_mode,
                learn_transitions=learn_transitions,
                transition_structure=transition_structure,
                cyclic_givens_layers=cyclic_givens_layers,
                cayley_coordinate_mode=cayley_coordinate_mode,
                base_unitary_init=base_unitary_init,
                base_unitary_scale=base_unitary_scale,
                base_unitary_seed=base_unitary_seed,
                readout_bias=True,
            )
        else:
            resolved_core = core
            required = (
                "input_dim",
                "ring_dim",
                "output_dim",
                "num_timescales",
                "zero_state",
                "forward_step",
                "readout_state",
                "max_unitary_error",
                "max_stochastic_error",
            )
            missing = [name for name in required if not hasattr(resolved_core, name)]
            if missing:
                raise TypeError(f"custom core is missing required attributes: {missing}")
            if int(resolved_core.input_dim) != self.input_dim:
                raise ValueError("custom core input_dim does not match the agent")
            if int(resolved_core.output_dim) != int(latent_dim):
                raise ValueError("custom core output_dim must match latent_dim")
            if int(resolved_core.num_timescales) < 2:
                raise ValueError("custom core must expose one fast and at least one slow state")
        self.core = resolved_core
        self.ring_dim = int(self.core.ring_dim)
        self.latent_dim = int(self.core.output_dim)
        self.placement_head = nn.Linear(self.latent_dim, self.points)
        self.legality_head = nn.Linear(self.latent_dim, self.points)
        self.pass_head = nn.Linear(self.latent_dim, 1)
        # A factorized policy needs an area-aware prior.  Zero weights and
        # P(pass)=1/(points+1) make the initial joint distribution comparable
        # to a uniform action prior while preserving a full BCE learning path.
        nn.init.zeros_(self.pass_head.weight)
        nn.init.constant_(
            self.pass_head.bias,
            math.log(
                self.initial_pass_probability
                / (1.0 - self.initial_pass_probability)
            ),
        )
        self.value_head = nn.Linear(self.latent_dim, 1)
        self.spatial_skip_heads: Optional[GoSpatialSkipHeads] = (
            GoSpatialSkipHeads(
                self.input_dim,
                self.board_size,
                self.spatial_skip_channels,
            )
            if self.spatial_skip_channels > 0
            else None
        )
        self.utility_feature_names = CAUSAL_UTILITY_FEATURES + tuple(
            f"content_projection_{index}"
            for index in range(self.utility_content_dim)
        )
        if self.utility_content_dim > 0:
            if self.utility_content_projection_mode == "identity":
                content_projection = torch.eye(self.input_dim)
            else:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.utility_content_seed)
                content_projection = torch.randn(
                    self.utility_content_dim,
                    self.input_dim,
                    generator=generator,
                )
                content_projection = F.normalize(content_projection, dim=1)
        else:
            content_projection = torch.empty(0, self.input_dim)
        # The fixed projection is marker-free and label-free.  It is persistent
        # only when enabled so legacy state dictionaries remain strict-loadable.
        self.register_buffer(
            "utility_content_projection",
            content_projection,
            persistent=self.utility_content_dim > 0,
        )
        self.utility_gate = FutureUtilityGate(
            len(self.utility_feature_names),
            rank=utility_rank,
            initial_advantage=float(initial_write_advantage) / self.advantage_scale,
        )
        self.task_gradient_memory = OrthogonalGradientMemory(ogd_max_rank)
        self.utility_gradient_memory = OrthogonalGradientMemory(utility_ogd_max_rank)

        self._stream_states: OrderedDict[Optional[Hashable], TemporalMQRState] = OrderedDict()
        self._feature_history: Dict[Optional[Hashable], Dict[str, Any]] = {}
        self._pending_tickets: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self._closed_tickets: OrderedDict[int, str] = OrderedDict()
        self._next_ticket_id = 1
        self.register_buffer("online_observations", torch.zeros((), dtype=torch.long))
        self.register_buffer("online_task_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("online_utility_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("online_parameter_version", torch.zeros((), dtype=torch.long))

    @staticmethod
    def _stream_key(stream_id: Any) -> Optional[Hashable]:
        if isinstance(stream_id, torch.Tensor):
            if stream_id.numel() != 1:
                raise ValueError("stream_id tensor must be scalar")
            stream_id = stream_id.item()
        if stream_id is None:
            return None
        try:
            hash(stream_id)
        except TypeError as exc:
            raise TypeError("stream_id must be hashable") from exc
        return stream_id

    @staticmethod
    def _clone_state(state: TemporalMQRState) -> TemporalMQRState:
        return TemporalMQRState(tuple(value.detach().clone() for value in state.rings))

    @staticmethod
    def _serialize_output(output: GoMultiHeadOutput) -> Dict[str, Any]:
        return {
            "placement_logits": output.placement_logits.detach().clone(),
            "legality_logits": output.legality_logits.detach().clone(),
            "pass_logit": output.pass_logit.detach().clone(),
            "value": output.value.detach().clone(),
            "latent": output.latent.detach().clone(),
            "legality_policy_scale": float(output.legality_policy_scale),
        }

    def _restore_output(self, record: Mapping[str, Any]) -> GoMultiHeadOutput:
        reference = next(self.parameters())

        def tensor(name: str, shape: Tuple[int, ...]) -> torch.Tensor:
            value = record.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise RuntimeError(
                    f"checkpoint output {name} must have shape {shape}"
                )
            if torch.is_complex(value) or not value.is_floating_point():
                raise RuntimeError(f"checkpoint output {name} must be real floating point")
            value = value.detach().to(device=reference.device, dtype=reference.dtype)
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"checkpoint output {name} contains NaN or Inf")
            return value

        return GoMultiHeadOutput(
            placement_logits=tensor("placement_logits", (1, self.points)),
            legality_logits=tensor("legality_logits", (1, self.points)),
            pass_logit=tensor("pass_logit", (1,)),
            value=tensor("value", (1,)),
            latent=tensor("latent", (1, self.latent_dim)),
            legality_policy_scale=float(record.get("legality_policy_scale", 0.0)),
        )

    def _restore_state(self, values: Sequence[torch.Tensor]) -> TemporalMQRState:
        if len(values) != self.core.num_timescales:
            raise RuntimeError("checkpoint state has the wrong number of rings")
        reference = next(self.parameters())
        rings = []
        for index, value in enumerate(values):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (
                1,
                self.ring_dim,
            ):
                raise RuntimeError(
                    f"checkpoint ring {index} must have shape [1, {self.ring_dim}]"
                )
            if torch.is_complex(value) or not value.is_floating_point():
                raise RuntimeError("checkpoint ring state must be real floating point")
            prepared = value.detach().to(
                device=reference.device,
                dtype=reference.dtype,
            )
            if not bool(torch.isfinite(prepared).all()):
                raise RuntimeError("checkpoint ring state contains NaN or Inf")
            rings.append(prepared)
        return TemporalMQRState(tuple(rings))

    def get_extra_state(self) -> Dict[str, Any]:
        """Serialize stream and pending-ticket state used by trace feedback."""

        tickets = []
        for ticket_id, record in self._pending_tickets.items():
            tickets.append(
                (
                    int(ticket_id),
                    {
                        "x": record["x"].detach().clone(),
                        "previous_state": [
                            value.detach().clone()
                            for value in record["previous_state"].rings
                        ],
                        "no_write_state": [
                            value.detach().clone()
                            for value in record["no_write_state"].rings
                        ],
                        "write_state": [
                            value.detach().clone()
                            for value in record["write_state"].rings
                        ],
                        "no_write_output": self._serialize_output(
                            record["no_write_output"]
                        ),
                        "write_output": self._serialize_output(
                            record["write_output"]
                        ),
                        "actual_output": self._serialize_output(
                            record["actual_output"]
                        ),
                        "causal_features": record["causal_features"].detach().clone(),
                        "effective_write": bool(record["effective_write"]),
                        "issued_parameter_version": int(
                            record["issued_parameter_version"]
                        ),
                        "stream_id": record["stream_id"],
                        "stream_step": int(record.get("stream_step", 0)),
                        "task_resolved": bool(record["task_resolved"]),
                        "utility_resolved": bool(record["utility_resolved"]),
                    },
                )
            )
        return {
            "version": 1,
            "streams": [
                (key, [value.detach().clone() for value in state.rings])
                for key, state in self._stream_states.items()
            ],
            "feature_history": [
                (
                    key,
                    {
                        "running_mean": (
                            None
                            if record["running_mean"] is None
                            else record["running_mean"].detach().clone()
                        ),
                        "previous_input": (
                            None
                            if record["previous_input"] is None
                            else record["previous_input"].detach().clone()
                        ),
                        "past_surprise": float(record["past_surprise"]),
                        "count": int(record["count"]),
                    },
                )
                for key, record in self._feature_history.items()
            ],
            "pending_tickets": tickets,
            "closed_tickets": list(self._closed_tickets.items()),
            "next_ticket_id": int(self._next_ticket_id),
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        self._stream_states.clear()
        self._feature_history.clear()
        self._pending_tickets.clear()
        self._closed_tickets.clear()
        self._next_ticket_id = 1
        if not state:
            return
        if not isinstance(state, dict) or int(state.get("version", 0)) != 1:
            raise RuntimeError("unsupported TemporalUtilityMQRAgent checkpoint version")
        for key, rings in state.get("streams", []):
            stream_key = self._stream_key(key)
            if stream_key in self._stream_states:
                raise RuntimeError("checkpoint contains duplicate stream ids")
            self._stream_states[stream_key] = self._restore_state(rings)
        if len(self._stream_states) > self.max_streams:
            raise RuntimeError("checkpoint exceeds max_streams")

        reference = next(self.parameters())
        for key, record in state.get("feature_history", []):
            stream_key = self._stream_key(key)
            restored: Dict[str, Any] = {
                "past_surprise": float(record["past_surprise"]),
                "count": int(record["count"]),
            }
            for name in ("running_mean", "previous_input"):
                value = record.get(name)
                if value is not None:
                    if not isinstance(value, torch.Tensor) or value.shape != (
                        1,
                        self.input_dim,
                    ):
                        raise RuntimeError(
                            f"checkpoint feature {name} has an invalid shape"
                        )
                    value = value.detach().to(
                        device=reference.device,
                        dtype=reference.dtype,
                    )
                restored[name] = value
            self._feature_history[stream_key] = restored

        for raw_ticket_id, record in state.get("pending_tickets", []):
            ticket_id = int(raw_ticket_id)
            if ticket_id <= 0 or ticket_id in self._pending_tickets:
                raise RuntimeError("checkpoint contains an invalid ticket id")
            x_value = record["x"].detach().to(
                device=reference.device,
                dtype=reference.dtype,
            )
            self._validate_observation(x_value)
            causal = record["causal_features"].detach().to(
                device=reference.device,
                dtype=reference.dtype,
            )
            if causal.shape != (1, len(self.utility_feature_names)):
                raise RuntimeError("checkpoint causal feature layout is incompatible")
            self._pending_tickets[ticket_id] = {
                "x": x_value,
                "previous_state": self._restore_state(record["previous_state"]),
                "no_write_state": self._restore_state(record["no_write_state"]),
                "write_state": self._restore_state(record["write_state"]),
                "no_write_output": self._restore_output(record["no_write_output"]),
                "write_output": self._restore_output(record["write_output"]),
                "actual_output": self._restore_output(record["actual_output"]),
                "causal_features": causal,
                "effective_write": bool(record["effective_write"]),
                "issued_parameter_version": int(record["issued_parameter_version"]),
                "stream_id": self._stream_key(record["stream_id"]),
                "stream_step": int(record.get("stream_step", 0)),
                "task_resolved": bool(record["task_resolved"]),
                "utility_resolved": bool(record["utility_resolved"]),
            }
        if len(self._pending_tickets) > self.max_pending_tickets:
            raise RuntimeError("checkpoint exceeds max_pending_tickets")
        for raw_ticket_id, status in state.get("closed_tickets", []):
            self._closed_tickets[int(raw_ticket_id)] = str(status)
        self._next_ticket_id = int(state.get("next_ticket_id", 1))
        maximum_ticket = max(
            [0, *self._pending_tickets.keys(), *self._closed_tickets.keys()]
        )
        if self._next_ticket_id <= maximum_ticket:
            raise RuntimeError("checkpoint next_ticket_id is not monotone")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Defining extra state would otherwise make pre-trace checkpoints fail
        # strict loading.  An absent payload means an empty runtime, which is the
        # only recoverable interpretation of the historical format.
        key = prefix + "_extra_state"
        if key not in state_dict:
            state_dict[key] = {}
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _zero_state(self, x: torch.Tensor) -> TemporalMQRState:
        return self.core.zero_state(1, device=x.device, dtype=x.dtype)

    def _state_for(self, key: Optional[Hashable], x: torch.Tensor) -> TemporalMQRState:
        state = self._stream_states.get(key)
        if state is None or any(value.shape != (1, self.ring_dim) for value in state.rings):
            return self._zero_state(x)
        return state.detached()

    def _head_output(
        self,
        latent: torch.Tensor,
        x: Optional[torch.Tensor] = None,
    ) -> GoMultiHeadOutput:
        latent = torch.tanh(latent)
        placement = self.placement_head(latent)
        legality = self.legality_head(latent)
        pass_logit = self.pass_head(latent).squeeze(1)
        value_logit = self.value_head(latent).squeeze(1)
        if self.spatial_skip_heads is not None and x is not None:
            direct_placement, direct_legality, direct_pass, direct_value = (
                self.spatial_skip_heads(x)
            )
            placement = placement + direct_placement
            legality = legality + direct_legality
            pass_logit = pass_logit + direct_pass
            value_logit = value_logit + direct_value
        return GoMultiHeadOutput(
            placement_logits=placement,
            legality_logits=legality,
            pass_logit=pass_logit,
            value=torch.tanh(value_logit),
            latent=latent,
            legality_policy_scale=self.legality_policy_scale,
        )

    def readout_state(self, state: TemporalMQRState) -> GoMultiHeadOutput:
        return self._head_output(self.core.readout_state(state))

    @staticmethod
    def _cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        denominator = torch.linalg.vector_norm(left, dim=1) * torch.linalg.vector_norm(
            right, dim=1
        )
        similarity = torch.zeros_like(denominator)
        valid = denominator > torch.finfo(left.dtype).eps
        if bool(valid.any()):
            similarity[valid] = (
                (left[valid] * right[valid]).sum(dim=1) / denominator[valid]
            )
        return 0.5 * (1.0 - similarity.clamp(-1.0, 1.0))

    @staticmethod
    def _empty_history() -> Dict[str, Any]:
        return {
            "running_mean": None,
            "previous_input": None,
            "past_surprise": 0.0,
            "count": 0,
        }

    def _causal_state_output(
        self, state: TemporalMQRState, previous_input: Optional[torch.Tensor]
    ) -> GoMultiHeadOutput:
        """State-only default; spatially conditioned agents can use past input."""
        return self.readout_state(state)

    def _causal_features(
        self,
        x: torch.Tensor,
        state: TemporalMQRState,
        key: Optional[Hashable],
    ) -> torch.Tensor:
        history = self._feature_history.get(key, self._empty_history())
        if history["running_mean"] is None:
            # With no reference state, the observation is maximally novel.  A
            # zero here made stream starts indistinguishable from exact repeats
            # and created a no-write bootstrapping deadlock.
            novelty = x.new_ones((1,))
        else:
            novelty = self._cosine_distance(
                x,
                history["running_mean"].to(device=x.device, dtype=x.dtype),
            )
        if history["previous_input"] is None:
            context_change = x.new_ones((1,))
        else:
            context_change = self._cosine_distance(
                x,
                history["previous_input"].to(device=x.device, dtype=x.dtype),
            )
        previous_output = self._causal_state_output(state, history["previous_input"])
        policy = F.softmax(previous_output.policy_logits, dim=1)
        uncertainty = -(
            policy * policy.clamp_min(1e-12).log()
        ).sum(dim=1) / math.log(self.action_size)
        slow = torch.cat(state.rings[1:], dim=1)
        saturation = torch.tanh(slow.abs().mean(dim=1))
        surprise = x.new_full((1,), float(history["past_surprise"]))
        scalar_features = torch.stack(
            [novelty, uncertainty, surprise, saturation, context_change], dim=1
        )
        if self.utility_content_dim == 0:
            return scalar_features.detach()
        projection = self.utility_content_projection.to(device=x.device, dtype=x.dtype)
        normalized_x = F.normalize(x, dim=1)
        content = torch.tanh(normalized_x @ projection.transpose(0, 1))
        return torch.cat([scalar_features, content], dim=1).detach()

    def _write_decision(
        self,
        x: torch.Tensor,
        features: torch.Tensor,
        external_write: Optional[bool],
        *,
        allow_exploration: bool = False,
    ) -> Dict[str, Any]:
        predicted = self.advantage_scale * self.utility_gate(features)
        learned = bool(float(predicted.item()) > 0.0)
        observations = int(self.online_observations.item())
        exploration_probability = 0.0
        if observations >= self.write_warmup_observations:
            exploration_probability = self.write_exploration_probability
            if self.write_exploration_decay_observations > 0:
                elapsed = observations - self.write_warmup_observations
                exploration_probability *= max(
                    0.0,
                    1.0 - elapsed / float(self.write_exploration_decay_observations),
                )
        explored = False
        if external_write is not None:
            effective = bool(external_write)
            source = "external"
        elif observations < self.write_warmup_observations:
            effective = True
            source = "warmup"
        else:
            explored = bool(
                allow_exploration
                and self.training
                and not learned
                and exploration_probability > 0.0
                and float(torch.rand((), device=x.device).item())
                < exploration_probability
            )
            effective = bool(learned or explored)
            source = "exploration" if explored else "learned"
        gates = torch.zeros(
            1,
            self.core.num_timescales,
            device=x.device,
            dtype=x.dtype,
        )
        gates[:, 0] = 1.0
        if effective:
            gates[:, 1:] = 1.0
        return {
            "predicted_advantage": predicted,
            "learned_write": learned,
            "effective_write": effective,
            "write_gate": gates,
            "write_source": source,
            "exploration_probability": exploration_probability,
            "explored_write": explored,
        }

    def _transition(
        self,
        x: torch.Tensor,
        state: TemporalMQRState,
        *,
        slow_write: bool,
    ) -> Tuple[GoMultiHeadOutput, TemporalMQRState]:
        gates = torch.zeros(
            1,
            self.core.num_timescales,
            device=x.device,
            dtype=x.dtype,
        )
        gates[:, 0] = 1.0
        if slow_write:
            gates[:, 1:] = 1.0
        latent, next_state = self.core.forward_step(x, state=state, write_gate=gates)
        return self._head_output(latent, x), next_state

    @torch.no_grad()
    def preview_step(
        self,
        x: torch.Tensor,
        *,
        stream_id: Any = None,
        external_write: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Return the old-policy prediction without mutating state or tickets."""

        self._validate_observation(x)
        key = self._stream_key(stream_id)
        previous = self._state_for(key, x)
        features = self._causal_features(x, previous, key)
        decision = self._write_decision(
            x, features, external_write, allow_exploration=False
        )
        output, next_state = self._transition(
            x, previous, slow_write=bool(decision["effective_write"]),
        )
        output = output.detached()
        return {
            "output": output,
            "policy_logits": output.policy_logits,
            "state": next_state.detached(),
            "features": features,
            "predicted_write_advantage": float(decision["predicted_advantage"].item()),
            "learned_write": decision["learned_write"],
            "effective_write": decision["effective_write"],
            "write_source": decision["write_source"],
            "exploration_probability": decision["exploration_probability"],
            "explored_write": decision["explored_write"],
            "prediction_before_update": True,
            "preview_only": True,
            "stream_id": key,
        }

    @torch.no_grad()
    def commit_step(
        self,
        x: torch.Tensor,
        *,
        stream_id: Any = None,
        external_write: Optional[bool] = None,
        issue_ticket: bool = True,
    ) -> Dict[str, Any]:
        """Commit one old-policy state transition and optionally issue a ticket."""

        self._validate_observation(x)
        key = self._stream_key(stream_id)
        if key not in self._stream_states and len(self._stream_states) >= self.max_streams:
            raise RuntimeError("state bank is full; reset a stream or increase max_streams")
        if issue_ticket and len(self._pending_tickets) >= self.max_pending_tickets:
            raise RuntimeError("candidate ticket bank is full")
        previous = self._state_for(key, x)
        features = self._causal_features(x, previous, key)
        decision = self._write_decision(
            x, features, external_write, allow_exploration=True
        )
        output, next_state = self._transition(
            x,
            previous,
            slow_write=decision["effective_write"],
        )
        no_output, no_state = self._transition(x, previous, slow_write=False)
        write_output, write_state = self._transition(x, previous, slow_write=True)
        ticket_id: Optional[int] = None
        if issue_ticket:
            ticket_id = self._next_ticket_id
            self._next_ticket_id += 1
            self._pending_tickets[ticket_id] = {
                "x": x.detach().clone(),
                "previous_state": self._clone_state(previous),
                "no_write_state": self._clone_state(no_state),
                "write_state": self._clone_state(write_state),
                "no_write_output": no_output.detached(),
                "write_output": write_output.detached(),
                "causal_features": features.detach().clone(),
                "actual_output": output.detached(),
                "effective_write": bool(decision["effective_write"]),
                "issued_parameter_version": int(self.online_parameter_version.item()),
                "stream_id": key,
                "stream_step": int(
                    self._feature_history.get(key, self._empty_history())["count"]
                ),
                "task_resolved": False,
                "utility_resolved": False,
            }
        self._stream_states[key] = next_state.detached()
        self._stream_states.move_to_end(key)
        self._update_feature_history(key, x)
        self.online_observations.add_(1)
        detached_output = output.detached()
        return {
            "output": detached_output,
            "policy_logits": detached_output.policy_logits,
            "state": next_state.detached(),
            "ticket_id": ticket_id,
            "features": features,
            "predicted_write_advantage": float(decision["predicted_advantage"].item()),
            "learned_write": decision["learned_write"],
            "effective_write": decision["effective_write"],
            "write_source": decision["write_source"],
            "exploration_probability": decision["exploration_probability"],
            "explored_write": decision["explored_write"],
            "prediction_before_update": True,
            "preview_only": False,
            "stream_id": key,
            "parameter_version": int(self.online_parameter_version.item()),
        }

    @torch.no_grad()
    def branch_step(
        self,
        x: torch.Tensor,
        state: TemporalMQRState,
        *,
        slow_write: bool,
    ) -> Tuple[GoMultiHeadOutput, TemporalMQRState]:
        """Advance an explicit shadow state without mutating runtime state."""

        self._validate_observation(x)
        output, next_state = self._transition(x, state, slow_write=bool(slow_write))
        return output.detached(), next_state.detached()

    def shadow_states(self, ticket_id: int) -> Tuple[TemporalMQRState, TemporalMQRState]:
        """Return cloned no-write/write states for simulator twin rollouts."""

        record = self._ticket_record(ticket_id)
        return (
            self._clone_state(record["no_write_state"]),
            self._clone_state(record["write_state"]),
        )

    def compute_go_loss(
        self,
        output: GoMultiHeadOutput,
        action: torch.Tensor,
        *,
        legality_target: Optional[torch.Tensor] = None,
        value_target: Optional[torch.Tensor] = None,
        awr_advantage: Optional[torch.Tensor] = None,
        ppo_advantage: Optional[torch.Tensor] = None,
        old_log_prob: Optional[torch.Tensor] = None,
        reference_log_probs: Optional[torch.Tensor] = None,
        policy_target: Optional[torch.Tensor] = None,
        weights: Optional[GoLossWeights] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compose task losses; policy_target is a detached joint [B,A] distribution."""

        selected = self.loss_weights if weights is None else weights
        if action.dim() != 1 or action.shape != (output.policy_logits.size(0),):
            raise ValueError("action must have shape [batch]")
        action = action.to(device=output.policy_logits.device, dtype=torch.long)
        if not bool(((action >= 0) & (action < self.action_size)).all()):
            raise ValueError("action contains an out-of-range index")
        zero = output.policy_logits.sum() * 0.0
        placement_mask = action < self.points
        placement = zero
        if bool(placement_mask.any()):
            placement = F.cross_entropy(
                output.placement_logits[placement_mask],
                action[placement_mask],
            )
        pass_target = (action == self.points).to(dtype=output.pass_logit.dtype)
        pass_decision = F.binary_cross_entropy_with_logits(
            output.pass_logit,
            pass_target,
        )
        legality = zero
        illegal_mass = zero
        if legality_target is not None:
            if legality_target.shape != output.legality_logits.shape:
                raise ValueError(
                    f"legality_target must have shape {tuple(output.legality_logits.shape)}"
                )
            legality_work = legality_target.detach().to(
                device=output.legality_logits.device,
                dtype=output.legality_logits.dtype,
            )
            if not bool(((legality_work == 0.0) | (legality_work == 1.0)).all()):
                raise ValueError("legality_target must contain only zero or one")
            legality = F.binary_cross_entropy_with_logits(
                output.legality_logits,
                legality_work,
            )
            point_probabilities = F.softmax(output.policy_logits, dim=1)[:, : self.points]
            illegal_mass = (
                point_probabilities * (1.0 - legality_work)
            ).sum(dim=1).mean()
        value = zero
        if value_target is not None:
            if value_target.shape != output.value.shape:
                raise ValueError(f"value_target must have shape {tuple(output.value.shape)}")
            value = F.smooth_l1_loss(
                output.value,
                value_target.detach().to(device=output.value.device, dtype=output.value.dtype),
            )

        log_probs = F.log_softmax(output.policy_logits, dim=1)
        policy_distillation = zero
        if policy_target is not None:
            target = policy_target.detach().to(log_probs)
            if (target.shape != log_probs.shape or not bool(torch.isfinite(target).all())
                    or bool((target < 0).any())
                    or not torch.allclose(target.sum(dim=1), torch.ones_like(target[:, 0]),
                                          rtol=1e-5, atol=1e-6)):
                raise ValueError("policy_target must be a finite normalized nonnegative [batch, actions] distribution")
            policy_distillation = -(target * log_probs).sum(dim=1).mean()
        chosen_log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        awr = zero
        if awr_advantage is not None:
            if awr_advantage.shape != chosen_log_prob.shape:
                raise ValueError("awr_advantage must have shape [batch]")
            awr_weight = torch.exp(
                awr_advantage.detach().to(chosen_log_prob) / self.awr_temperature
            ).clamp(max=self.awr_max_weight)
            awr = -(awr_weight * chosen_log_prob).mean()

        ppo = zero
        if ppo_advantage is not None or old_log_prob is not None:
            if ppo_advantage is None or old_log_prob is None:
                raise ValueError("ppo_advantage and old_log_prob must be supplied together")
            if ppo_advantage.shape != chosen_log_prob.shape or old_log_prob.shape != chosen_log_prob.shape:
                raise ValueError("PPO tensors must have shape [batch]")
            advantage = ppo_advantage.detach().to(chosen_log_prob)
            ratio = torch.exp(chosen_log_prob - old_log_prob.detach().to(chosen_log_prob))
            unclipped = ratio * advantage
            clipped = ratio.clamp(1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantage
            ppo = -torch.minimum(unclipped, clipped).mean()

        entropy = -(F.softmax(output.policy_logits, dim=1) * log_probs).sum(dim=1).mean()
        reference_kl = zero
        if reference_log_probs is not None:
            if reference_log_probs.shape != log_probs.shape:
                raise ValueError("reference_log_probs must match policy logits")
            reference = reference_log_probs.detach().to(log_probs)
            probabilities = log_probs.exp()
            reference_kl = (probabilities * (log_probs - reference)).sum(dim=1).mean()

        total = (
            selected.placement * placement
            + selected.legality * legality
            + selected.pass_decision * pass_decision
            + selected.value * value
            + selected.awr * awr
            + selected.ppo * ppo
            - selected.entropy * entropy
            + selected.reference_kl * reference_kl
            + selected.illegal_mass * illegal_mass
            + selected.policy_distillation * policy_distillation
        )
        return {
            "total": total,
            "placement": placement,
            "legality": legality,
            "pass": pass_decision,
            "value": value,
            "awr": awr,
            "ppo": ppo,
            "entropy": entropy,
            "reference_kl": reference_kl,
            "illegal_mass": illegal_mass,
            "policy_distillation": policy_distillation,
            "chosen_log_prob": chosen_log_prob.mean(),
        }

    def apply_feedback(
        self,
        ticket_id: int,
        action: torch.Tensor,
        *,
        legality_target: Optional[torch.Tensor] = None,
        value_target: Optional[torch.Tensor] = None,
        awr_advantage: Optional[torch.Tensor] = None,
        ppo_advantage: Optional[torch.Tensor] = None,
        old_log_prob: Optional[torch.Tensor] = None,
        reference_log_probs: Optional[torch.Tensor] = None,
        policy_target: Optional[torch.Tensor] = None,
        weights: Optional[GoLossWeights] = None,
        learn: bool = True,
        remember_gradient: bool = False,
        project_with_memory: bool = True,
        return_grad_features: bool = False,
        allow_stale: bool = False,
    ) -> Dict[str, Any]:
        """Apply delayed task feedback without changing the committed state."""

        record = self._ticket_record(ticket_id)
        if bool(record["task_resolved"]):
            raise RuntimeError(f"ticket {ticket_id} task feedback was already consumed")
        issued_version = int(record["issued_parameter_version"])
        current_version = int(self.online_parameter_version.item())
        if not allow_stale and issued_version != current_version:
            raise RuntimeError("task ticket is stale; pass allow_stale=True to audit that update")
        if remember_gradient and (not learn or self.task_gradient_memory.max_rank == 0):
            raise ValueError("remember_gradient requires learning and task OGD capacity")

        x_work = record["x"].detach().requires_grad_(bool(return_grad_features))
        previous = record["previous_state"].detached()
        output, reference_next_state = self._transition(
            x_work,
            previous,
            slow_write=bool(record["effective_write"]),
        )
        losses = self.compute_go_loss(
            output,
            action,
            legality_target=legality_target,
            value_target=value_target,
            awr_advantage=awr_advantage,
            ppo_advantage=ppo_advantage,
            old_log_prob=old_log_prob,
            reference_log_probs=reference_log_probs,
            policy_target=policy_target,
            weights=weights,
        )
        active = [
            (name, parameter, self.task_lr)
            for name, parameter in self._named_task_parameters()
            if parameter.requires_grad
        ] if learn else []
        transition_update_due = bool(
            active
            and (current_version + 1) % self.transition_update_interval == 0
        )
        allowed_names = [
            name
            for name, _parameter, _step in active
            if transition_update_due or not self._is_cayley_parameter(name)
        ]
        differentiation_targets: List[torch.Tensor] = [
            parameter for _name, parameter, _step in active
        ]
        if return_grad_features:
            differentiation_targets.append(x_work)
        gradients: Tuple[Optional[torch.Tensor], ...] = ()
        if differentiation_targets:
            gradients = torch.autograd.grad(
                losses["total"],
                differentiation_targets,
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )
        parameter_gradients: List[torch.Tensor] = []
        for (_name, parameter, _step), gradient in zip(active, gradients[: len(active)]):
            parameter_gradients.append(
                torch.zeros_like(parameter) if gradient is None else gradient.detach()
            )
        grad_features = None
        if return_grad_features:
            raw = gradients[-1]
            grad_features = torch.zeros_like(x_work) if raw is None else raw.detach()
        entries = [
            (name, gradient, step)
            for (name, _parameter, step), gradient in zip(active, parameter_gradients)
        ]
        parameter_snapshot = {
            name: parameter.detach().clone() for name, parameter, _step in active
        }
        memory_snapshot = self.task_gradient_memory.snapshot()
        projected, stats, memory_added, projection_applied = self._project_entries(
            entries,
            self.task_gradient_memory,
            remember_gradient=remember_gradient,
            project_with_memory=project_with_memory,
            allowed_names=allowed_names,
        )
        candidate_update_norm, clip_scale = self._update_geometry(
            entries,
            projected,
            self.max_update_norm,
        )
        local_guard_enabled = any(
            value is not None
            for value in (
                self.safety_limits.max_policy_kl,
                self.safety_limits.max_output_linf_drift,
                self.safety_limits.max_state_linf_drift,
            )
        )
        transition_guard_enabled = (
            self.safety_limits.max_transition_fro_drift is not None
        )
        transition_references = []
        if transition_guard_enabled and hasattr(self.core, "transition_references"):
            transition_references = [
                (index, None, reference)
                for index, reference in enumerate(self.core.transition_references())
            ]
        elif transition_guard_enabled and hasattr(self.core, "unitary_params"):
            transition_references = [
                (index, parameter, parameter.skew_hermitian_A().detach().clone())
                for index, parameter in enumerate(self.core.unitary_params)
                if hasattr(parameter, "drift_diagnostics")
            ]

        safety_violations: List[str] = []
        policy_kl: Optional[float] = None
        output_linf_drift: Optional[float] = None
        state_linf_drift: Optional[float] = None
        transition_fro_drift: Optional[float] = None
        transition_fro_bound: Optional[float] = None
        transition_bounds_certified: Optional[bool] = None
        rolled_back = False
        with torch.no_grad():
            for name, parameter, step in active:
                parameter.add_(projected[name], alpha=-step * clip_scale)
            try:
                if entries and local_guard_enabled:
                    candidate_output, candidate_next_state = self._transition(
                        x_work.detach(),
                        previous,
                        slow_write=bool(record["effective_write"]),
                    )
                    policy_kl = categorical_policy_kl(
                        output.policy_logits.detach(),
                        candidate_output.policy_logits.detach(),
                    )
                    output_linf_drift = self._output_linf_drift(
                        output.detached(), candidate_output.detached()
                    )
                    state_linf_drift = self._state_linf_drift(
                        reference_next_state.detached(), candidate_next_state.detached()
                    )
                    if not all(
                        math.isfinite(value)
                        for value in (policy_kl, output_linf_drift, state_linf_drift)
                    ):
                        safety_violations.append("nonfinite_local_constancy")
                    if (
                        self.safety_limits.max_policy_kl is not None
                        and policy_kl > self.safety_limits.max_policy_kl
                    ):
                        safety_violations.append("policy_kl")
                    if (
                        self.safety_limits.max_output_linf_drift is not None
                        and output_linf_drift
                        > self.safety_limits.max_output_linf_drift
                    ):
                        safety_violations.append("output_linf_drift")
                    if (
                        self.safety_limits.max_state_linf_drift is not None
                        and state_linf_drift > self.safety_limits.max_state_linf_drift
                    ):
                        safety_violations.append("state_linf_drift")

                if entries and transition_guard_enabled:
                    transition_audits = [
                        (
                            self.core.transition_drift_diagnostics(index, reference)
                            if hasattr(self.core, "transition_drift_diagnostics")
                            else parameter.drift_diagnostics(reference)
                        )
                        for index, parameter, reference in transition_references
                    ]
                    transition_fro_drift = max(
                        (
                            float(item["transition_fro_drift"])
                            for item in transition_audits
                        ),
                        default=0.0,
                    )
                    transition_fro_bound = max(
                        (
                            float(item["transition_fro_bound"])
                            for item in transition_audits
                        ),
                        default=0.0,
                    )
                    transition_bounds_certified = all(
                        bool(item["bounds_certified"])
                        for item in transition_audits
                    )
                    if not transition_bounds_certified:
                        safety_violations.append("transition_bound_uncertified")
                    if (
                        transition_fro_drift
                        > float(self.safety_limits.max_transition_fro_drift)
                    ):
                        safety_violations.append("transition_fro_drift")
            except Exception:
                for name, parameter, _step in active:
                    parameter.copy_(parameter_snapshot[name])
                self.task_gradient_memory.restore(memory_snapshot)
                raise

            rolled_back = bool(safety_violations)
            if rolled_back:
                for name, parameter, _step in active:
                    parameter.copy_(parameter_snapshot[name])
                self.task_gradient_memory.restore(memory_snapshot)
                memory_added = False
            did_update = bool(entries and not rolled_back and candidate_update_norm > 0.0)
            if did_update:
                self.online_task_updates.add_(1)
                self.online_parameter_version.add_(1)
            history = self._feature_history.setdefault(record["stream_id"], self._empty_history())
            normalizer = max(math.log(self.action_size), 1e-12)
            history["past_surprise"] = min(
                1.0,
                float(losses["total"].detach().abs().item()) / (2.0 * normalizer),
            )
            record["task_resolved"] = True
            self._maybe_close_ticket(ticket_id)
        retained = float(stats["retained_norm"])
        ogd_capacity_warning = bool(
            projection_applied
            and float(stats.get("raw_norm", 0.0)) > 0.0
            and retained < self.safety_limits.ogd_retained_warning_threshold
        )
        if rolled_back:
            grad_features = None
        update_norm = 0.0 if rolled_back else candidate_update_norm
        return {
            "ticket_id": ticket_id,
            "output": record["actual_output"],
            "loss": float(losses["total"].detach().item()),
            "losses": {
                name: float(value.detach().item())
                for name, value in losses.items()
                if name != "chosen_log_prob"
            },
            "chosen_log_prob": float(losses["chosen_log_prob"].detach().item()),
            "prediction_before_update": True,
            "state_mutated": False,
            "did_update": did_update,
            "grad_features": grad_features,
            "external_gradient_authorized": bool(
                return_grad_features and not rolled_back
            ),
            "issued_parameter_version": issued_version,
            "parameter_staleness": current_version - issued_version,
            "parameter_version": int(self.online_parameter_version.item()),
            "ogd_rank": self.task_gradient_memory.rank,
            "ogd_retained_norm": retained,
            "ogd_capacity_warning": ogd_capacity_warning,
            "ogd_max_abs_overlap": float(stats["max_abs_overlap"]),
            "ogd_memory_added": bool(memory_added),
            "ogd_projection_applied": projection_applied,
            "update_norm": update_norm,
            "candidate_update_norm": candidate_update_norm,
            "update_clip_scale": clip_scale,
            "transition_update_due": transition_update_due,
            "transition_refresh_required": bool(
                did_update
                and not rolled_back
                and transition_update_due
                and any(
                    self._is_cayley_parameter(name)
                    for name, _parameter, _step in active
                )
            ),
            "transition_parameters_allowed": bool(
                transition_update_due
                or not any(self._is_cayley_parameter(name) for name, _p, _s in active)
            ),
            "local_constancy_checked": bool(entries and local_guard_enabled),
            "policy_kl": policy_kl,
            "output_linf_drift": output_linf_drift,
            "state_linf_drift": state_linf_drift,
            "transition_drift_checked": bool(entries and transition_guard_enabled),
            "transition_fro_drift": transition_fro_drift,
            "transition_fro_bound": transition_fro_bound,
            "transition_bounds_certified": transition_bounds_certified,
            "safety_passed": not rolled_back,
            "safety_violations": list(safety_violations),
            "rolled_back": rolled_back,
            "max_unitary_error": self.core.max_unitary_error(),
            "max_stochastic_error": self.core.max_stochastic_error(),
        }

    def apply_trajectory_feedback(
        self,
        ticket_ids: Sequence[int],
        feedback_steps: Sequence[Mapping[str, Any]],
        *,
        loss_scales: Optional[Sequence[float]] = None,
        weights: Optional[GoLossWeights] = None,
        learn: bool = True,
        remember_gradient: bool = False,
        project_with_memory: bool = True,
        return_grad_features: bool = False,
        allow_stale: bool = False,
        credit_horizon: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Apply one bounded, causal trajectory-end update.

        Every ticket must already have been produced by :meth:`commit_step`
        under predict-before-update semantics.  This method starts from the
        detached state preceding the first ticket, replays the committed write
        actions differentiably, and only then updates parameters once.  Future
        losses can therefore train the content written by earlier injections
        without changing any already returned prediction.

        ``feedback_steps`` contains one mapping per ticket.  The only required
        key is ``action``; optional keys match :meth:`compute_go_loss` except
        that ``weights`` is shared by the trajectory.  No feedback field is ever
        passed to the write gate or its causal feature extractor.

        ``credit_horizon`` optionally detaches state every K replay steps while
        retaining all committed predictions, losses and the single update.
        It never extends credit beyond the supplied ticket window.
        """

        ids = tuple(ticket_ids)
        steps = tuple(feedback_steps)
        if len(ids) < 2:
            raise ValueError("trajectory feedback requires at least two tickets")
        if len(ids) > self.max_trace_horizon:
            raise ValueError(
                f"trajectory length {len(ids)} exceeds max_trace_horizon "
                f"{self.max_trace_horizon}"
            )
        if len(ids) != len(steps):
            raise ValueError("ticket_ids and feedback_steps must have equal length")
        if credit_horizon is not None and (
            isinstance(credit_horizon, bool) or not isinstance(credit_horizon, int)
            or not 1 <= credit_horizon <= self.max_trace_horizon
        ):
            raise ValueError("credit_horizon must be a positive integer within the trace horizon")
        if len(set(ids)) != len(ids):
            raise ValueError("ticket_ids must be unique")
        records = [self._ticket_record(ticket_id) for ticket_id in ids]
        if any(bool(record["task_resolved"]) for record in records):
            raise RuntimeError("trajectory contains task feedback already consumed")
        stream_id = records[0]["stream_id"]
        if any(record["stream_id"] != stream_id for record in records):
            raise ValueError("all trajectory tickets must belong to one stream")
        stream_steps = [int(record["stream_step"]) for record in records]
        if any(
            right != left + 1
            for left, right in zip(stream_steps[:-1], stream_steps[1:])
        ):
            raise ValueError("trajectory tickets must be consecutive within the stream")
        current_version = int(self.online_parameter_version.item())
        issued_versions = [int(record["issued_parameter_version"]) for record in records]
        if not allow_stale and any(version != current_version for version in issued_versions):
            raise RuntimeError(
                "trajectory ticket is stale; pass allow_stale=True to audit that update"
            )
        if remember_gradient and (not learn or self.task_gradient_memory.max_rank == 0):
            raise ValueError("remember_gradient requires learning and task OGD capacity")

        if loss_scales is None:
            scales = tuple(1.0 for _ in ids)
        else:
            scales = tuple(float(value) for value in loss_scales)
            if len(scales) != len(ids):
                raise ValueError("loss_scales must have one value per ticket")
            if any(not math.isfinite(value) or value < 0.0 for value in scales):
                raise ValueError("loss_scales must be finite and non-negative")
            if sum(scales) <= 0.0:
                raise ValueError("at least one loss scale must be positive")

        allowed_feedback = {
            "action",
            "legality_target",
            "value_target",
            "awr_advantage",
            "ppo_advantage",
            "old_log_prob",
            "reference_log_probs",
            "policy_target",
        }
        normalized_steps: List[Dict[str, Any]] = []
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise TypeError(f"feedback_steps[{index}] must be a mapping")
            unknown = set(step).difference(allowed_feedback)
            if unknown:
                raise ValueError(
                    f"feedback_steps[{index}] has unknown keys: {sorted(unknown)}"
                )
            if "action" not in step:
                raise ValueError(f"feedback_steps[{index}] is missing action")
            normalized_steps.append(dict(step))

        initial_state = records[0]["previous_state"].detached()
        state = initial_state
        x_work: List[torch.Tensor] = []
        outputs: List[GoMultiHeadOutput] = []
        states: List[TemporalMQRState] = []
        losses_per_step: List[Dict[str, torch.Tensor]] = []
        for index, (record, step) in enumerate(zip(records, normalized_steps)):
            # Keep the same observations, losses, and update count while
            # changing only credit across deterministic block boundaries.
            if credit_horizon is not None and index and index % credit_horizon == 0:
                state = state.detached()
            x_value = record["x"].detach().requires_grad_(True)
            output, state = self._transition(
                x_value,
                state,
                slow_write=bool(record["effective_write"]),
            )
            losses = self.compute_go_loss(
                output,
                step["action"],
                legality_target=step.get("legality_target"),
                value_target=step.get("value_target"),
                awr_advantage=step.get("awr_advantage"),
                ppo_advantage=step.get("ppo_advantage"),
                old_log_prob=step.get("old_log_prob"),
                reference_log_probs=step.get("reference_log_probs"),
                policy_target=step.get("policy_target"),
                weights=weights,
            )
            x_work.append(x_value)
            outputs.append(output)
            states.append(state)
            losses_per_step.append(losses)
        scale_total = sum(scales)
        total_loss = sum(
            scale * losses["total"]
            for scale, losses in zip(scales, losses_per_step)
        ) / scale_total

        active = [
            (name, parameter, self.task_lr)
            for name, parameter in self._named_task_parameters()
            if parameter.requires_grad
        ] if learn else []
        transition_update_due = bool(
            active
            and (current_version + 1) % self.transition_update_interval == 0
        )
        allowed_names = [
            name
            for name, _parameter, _step in active
            if transition_update_due or not self._is_cayley_parameter(name)
        ]

        future_weight = sum(scales[1:])
        future_loss = None
        if future_weight > 0.0:
            future_loss = sum(
                scale * losses["total"]
                for scale, losses in zip(scales[1:], losses_per_step[1:])
            ) / future_weight
        future_targets = [parameter for _name, parameter, _step in active]
        future_parameter_gradients: Tuple[Optional[torch.Tensor], ...] = ()
        earliest_future_gradient: Optional[torch.Tensor] = None
        if future_loss is not None:
            future_gradient_targets: List[torch.Tensor] = future_targets + [x_work[0]]
            future_values = torch.autograd.grad(
                future_loss,
                future_gradient_targets,
                allow_unused=True,
                retain_graph=True,
                create_graph=False,
            )
            future_parameter_gradients = tuple(future_values[: len(future_targets)])
            earliest_future_gradient = future_values[-1]

        differentiation_targets: List[torch.Tensor] = future_targets + x_work
        raw_gradients: Tuple[Optional[torch.Tensor], ...] = ()
        if differentiation_targets:
            raw_gradients = torch.autograd.grad(
                total_loss,
                differentiation_targets,
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )
        parameter_gradients = [
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for (_name, parameter, _step), gradient in zip(
                active, raw_gradients[: len(active)]
            )
        ]
        feature_gradients = [
            (
                torch.zeros_like(value)
                if gradient is None
                else gradient.detach()
            )
            for value, gradient in zip(x_work, raw_gradients[len(active) :])
        ]

        def block_norms(
            gradients: Sequence[Optional[torch.Tensor]],
        ) -> Dict[str, float]:
            squared = {
                "injection": 0.0,
                "transition": 0.0,
                "core_readout": 0.0,
                "core_other": 0.0,
                "heads": 0.0,
            }
            for (name, _parameter, _step), gradient in zip(active, gradients):
                if gradient is None:
                    continue
                if self._is_cayley_parameter(name):
                    block = "transition"
                elif name.startswith("core.input_down") or name.startswith(
                    "core.input_up"
                ):
                    block = "injection"
                elif name.startswith("core.readout"):
                    block = "core_readout"
                elif name.startswith("core."):
                    block = "core_other"
                else:
                    block = "heads"
                squared[block] += float(gradient.detach().square().sum().item())
            return {name: math.sqrt(value) for name, value in squared.items()}

        raw_block_gradient_norms = block_norms(parameter_gradients)
        future_block_gradient_norms = block_norms(future_parameter_gradients)
        earliest_future_gradient_norm = (
            0.0
            if earliest_future_gradient is None
            else float(torch.linalg.vector_norm(earliest_future_gradient.detach()).item())
        )

        entries = [
            (name, gradient, step)
            for (name, _parameter, step), gradient in zip(active, parameter_gradients)
        ]
        parameter_snapshot = {
            name: parameter.detach().clone() for name, parameter, _step in active
        }
        memory_snapshot = self.task_gradient_memory.snapshot()
        projected, stats, memory_added, projection_applied = self._project_entries(
            entries,
            self.task_gradient_memory,
            remember_gradient=remember_gradient,
            project_with_memory=project_with_memory,
            allowed_names=allowed_names,
        )
        candidate_update_norm, clip_scale = self._update_geometry(
            entries,
            projected,
            self.max_update_norm,
        )

        reference_outputs = [output.detached() for output in outputs]
        reference_states = [value.detached() for value in states]
        local_guard_enabled = any(
            value is not None
            for value in (
                self.safety_limits.max_policy_kl,
                self.safety_limits.max_output_linf_drift,
                self.safety_limits.max_state_linf_drift,
            )
        )
        transition_guard_enabled = (
            self.safety_limits.max_transition_fro_drift is not None
        )
        transition_references = []
        if transition_guard_enabled and hasattr(self.core, "transition_references"):
            transition_references = [
                (index, None, reference)
                for index, reference in enumerate(self.core.transition_references())
            ]
        elif transition_guard_enabled and hasattr(self.core, "unitary_params"):
            transition_references = [
                (index, parameter, parameter.skew_hermitian_A().detach().clone())
                for index, parameter in enumerate(self.core.unitary_params)
                if hasattr(parameter, "drift_diagnostics")
            ]

        safety_violations: List[str] = []
        policy_kl: Optional[float] = None
        output_linf_drift: Optional[float] = None
        state_linf_drift: Optional[float] = None
        transition_fro_drift: Optional[float] = None
        transition_fro_bound: Optional[float] = None
        transition_bounds_certified: Optional[bool] = None
        rolled_back = False
        with torch.no_grad():
            for name, parameter, step_size in active:
                parameter.add_(projected[name], alpha=-step_size * clip_scale)
            try:
                if entries and local_guard_enabled:
                    candidate_state = initial_state
                    candidate_outputs: List[GoMultiHeadOutput] = []
                    candidate_states: List[TemporalMQRState] = []
                    for record in records:
                        candidate_output, candidate_state = self._transition(
                            record["x"].detach(),
                            candidate_state,
                            slow_write=bool(record["effective_write"]),
                        )
                        candidate_outputs.append(candidate_output)
                        candidate_states.append(candidate_state)
                    policy_kl = max(
                        categorical_policy_kl(
                            reference.policy_logits,
                            candidate.policy_logits,
                        )
                        for reference, candidate in zip(
                            reference_outputs, candidate_outputs
                        )
                    )
                    output_linf_drift = max(
                        self._output_linf_drift(reference, candidate)
                        for reference, candidate in zip(
                            reference_outputs, candidate_outputs
                        )
                    )
                    state_linf_drift = max(
                        self._state_linf_drift(reference, candidate)
                        for reference, candidate in zip(
                            reference_states, candidate_states
                        )
                    )
                    if not all(
                        math.isfinite(value)
                        for value in (policy_kl, output_linf_drift, state_linf_drift)
                    ):
                        safety_violations.append("nonfinite_local_constancy")
                    if (
                        self.safety_limits.max_policy_kl is not None
                        and policy_kl > self.safety_limits.max_policy_kl
                    ):
                        safety_violations.append("policy_kl")
                    if (
                        self.safety_limits.max_output_linf_drift is not None
                        and output_linf_drift
                        > self.safety_limits.max_output_linf_drift
                    ):
                        safety_violations.append("output_linf_drift")
                    if (
                        self.safety_limits.max_state_linf_drift is not None
                        and state_linf_drift > self.safety_limits.max_state_linf_drift
                    ):
                        safety_violations.append("state_linf_drift")

                if entries and transition_guard_enabled:
                    transition_audits = [
                        (
                            self.core.transition_drift_diagnostics(index, reference)
                            if hasattr(self.core, "transition_drift_diagnostics")
                            else parameter.drift_diagnostics(reference)
                        )
                        for index, parameter, reference in transition_references
                    ]
                    transition_fro_drift = max(
                        (
                            float(item["transition_fro_drift"])
                            for item in transition_audits
                        ),
                        default=0.0,
                    )
                    transition_fro_bound = max(
                        (
                            float(item["transition_fro_bound"])
                            for item in transition_audits
                        ),
                        default=0.0,
                    )
                    transition_bounds_certified = all(
                        bool(item["bounds_certified"])
                        for item in transition_audits
                    )
                    if not transition_bounds_certified:
                        safety_violations.append("transition_bound_uncertified")
                    if (
                        transition_fro_drift
                        > float(self.safety_limits.max_transition_fro_drift)
                    ):
                        safety_violations.append("transition_fro_drift")
            except Exception:
                for name, parameter, _step in active:
                    parameter.copy_(parameter_snapshot[name])
                self.task_gradient_memory.restore(memory_snapshot)
                raise

            rolled_back = bool(safety_violations)
            if rolled_back:
                for name, parameter, _step in active:
                    parameter.copy_(parameter_snapshot[name])
                self.task_gradient_memory.restore(memory_snapshot)
                memory_added = False
            did_update = bool(
                entries and not rolled_back and candidate_update_norm > 0.0
            )
            if did_update:
                self.online_task_updates.add_(1)
                self.online_parameter_version.add_(1)
            history = self._feature_history.setdefault(stream_id, self._empty_history())
            normalizer = max(math.log(self.action_size), 1e-12)
            history["past_surprise"] = min(
                1.0,
                float(total_loss.detach().abs().item()) / (2.0 * normalizer),
            )
            for ticket_id, record in zip(ids, records):
                record["task_resolved"] = True
                self._maybe_close_ticket(ticket_id)

        retained = float(stats["retained_norm"])
        ogd_capacity_warning = bool(
            projection_applied
            and float(stats.get("raw_norm", 0.0)) > 0.0
            and retained < self.safety_limits.ogd_retained_warning_threshold
        )
        return {
            "ticket_ids": list(ids),
            "stream_id": stream_id,
            "trace_horizon": len(ids),
            "loss": float(total_loss.detach().item()),
            "step_losses": [
                float(losses["total"].detach().item())
                for losses in losses_per_step
            ],
            "loss_scales": list(scales),
            "credit_horizon": len(ids) if credit_horizon is None else credit_horizon,
            "future_loss": (
                None if future_loss is None else float(future_loss.detach().item())
            ),
            "prediction_before_update": True,
            "all_predictions_committed_before_update": True,
            "state_mutated": False,
            "did_update": did_update,
            "issued_parameter_versions": issued_versions,
            "parameter_staleness": [
                current_version - version for version in issued_versions
            ],
            "parameter_version": int(self.online_parameter_version.item()),
            "raw_block_gradient_norms": raw_block_gradient_norms,
            "future_block_gradient_norms": future_block_gradient_norms,
            "earliest_input_future_gradient_norm": earliest_future_gradient_norm,
            "grad_features": (
                feature_gradients if return_grad_features and not rolled_back else None
            ),
            "external_gradient_authorized": bool(
                return_grad_features and not rolled_back
            ),
            "ogd_rank": self.task_gradient_memory.rank,
            "ogd_retained_norm": retained,
            "ogd_capacity_warning": ogd_capacity_warning,
            "ogd_max_abs_overlap": float(stats["max_abs_overlap"]),
            "ogd_memory_added": bool(memory_added),
            "ogd_projection_applied": projection_applied,
            "candidate_update_norm": candidate_update_norm,
            "update_norm": 0.0 if rolled_back else candidate_update_norm,
            "update_clip_scale": clip_scale,
            "transition_update_due": transition_update_due,
            "transition_refresh_required": bool(
                did_update
                and not rolled_back
                and transition_update_due
                and any(
                    self._is_cayley_parameter(name)
                    for name, _parameter, _step in active
                )
            ),
            "local_constancy_checked": bool(entries and local_guard_enabled),
            "policy_kl": policy_kl,
            "output_linf_drift": output_linf_drift,
            "state_linf_drift": state_linf_drift,
            "transition_drift_checked": bool(entries and transition_guard_enabled),
            "transition_fro_drift": transition_fro_drift,
            "transition_fro_bound": transition_fro_bound,
            "transition_bounds_certified": transition_bounds_certified,
            "safety_passed": not rolled_back,
            "safety_violations": list(safety_violations),
            "rolled_back": rolled_back,
            "future_labels_used_by_gate": False,
            "max_unitary_error": self.core.max_unitary_error(),
            "max_stochastic_error": self.core.max_stochastic_error(),
        }

    def calibrate_write_critic(
        self,
        ticket_id: int,
        *,
        no_write_return: float,
        write_return: float,
        learn: bool = True,
        remember_gradient: bool = False,
        project_with_memory: bool = True,
    ) -> Dict[str, Any]:
        """Fit the causal write critic from simulator twin-trajectory returns."""

        record = self._ticket_record(ticket_id)
        if bool(record["utility_resolved"]):
            raise RuntimeError(f"ticket {ticket_id} utility feedback was already consumed")
        for name, value in (("no_write_return", no_write_return), ("write_return", write_return)):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if remember_gradient and (not learn or self.utility_gradient_memory.max_rank == 0):
            raise ValueError("remember_gradient requires learning and utility OGD capacity")
        advantage = float(write_return) - float(no_write_return) - self.memory_cost
        # Squared regression must retain the conditional mean: clipping each
        # sample can reverse sign(E[A | c]) and therefore reverse the optimal
        # write action.  The atomic parameter trust region controls outliers.
        target = advantage / self.advantage_scale
        active = [
            (f"utility_gate.{name}", parameter, self.utility_lr)
            for name, parameter in self.utility_gate.named_parameters()
            if parameter.requires_grad
        ] if learn else []
        prediction = self.utility_gate(record["causal_features"])
        target_tensor = prediction.new_full(prediction.shape, target)
        loss = F.mse_loss(prediction, target_tensor)
        raw_gradients = torch.autograd.grad(
            loss,
            [parameter for _name, parameter, _step in active],
            allow_unused=False,
        ) if active else ()
        entries = [
            (name, gradient.detach(), step)
            for (name, _parameter, step), gradient in zip(active, raw_gradients)
        ]
        projected, stats, memory_added, projection_applied = self._project_entries(
            entries,
            self.utility_gradient_memory,
            remember_gradient=remember_gradient,
            project_with_memory=project_with_memory,
        )
        update_norm, clip_scale = self._update_geometry(
            entries,
            projected,
            self.utility_max_update_norm,
        )
        with torch.no_grad():
            for name, parameter, step in active:
                parameter.add_(projected[name], alpha=-step * clip_scale)
            if entries:
                self.online_utility_updates.add_(1)
            record["utility_resolved"] = True
            self._maybe_close_ticket(ticket_id)
        return {
            "ticket_id": ticket_id,
            "no_write_return": float(no_write_return),
            "write_return": float(write_return),
            "write_advantage": advantage,
            "predicted_advantage_before_update": float(
                self.advantage_scale * prediction.detach().item()
            ),
            "utility_loss": float(loss.detach().item()),
            "did_update": bool(entries),
            "ogd_rank": self.utility_gradient_memory.rank,
            "ogd_retained_norm": float(stats["retained_norm"]),
            "ogd_max_abs_overlap": float(stats["max_abs_overlap"]),
            "ogd_memory_added": bool(memory_added),
            "ogd_projection_applied": projection_applied,
            "update_norm": update_norm,
            "update_clip_scale": clip_scale,
            "feedback_after_action": True,
        }

    def fit_write_critic_batch(
        self,
        causal_features: torch.Tensor,
        write_advantages: torch.Tensor,
        *,
        learn: bool = True,
        project_with_memory: bool = False,
    ) -> Dict[str, Any]:
        """Fit the utility gate from a bounded replay of past twin returns.

        The batch contains only features that were available before their write
        actions and scalar advantages observed afterwards.  It cannot update or
        replay task parameters, recurrent states, boards, or policy actions.
        """

        expected_features = len(self.utility_feature_names)
        if causal_features.dim() != 2 or causal_features.size(1) != expected_features:
            raise ValueError(
                f"causal_features must have shape [batch, {expected_features}]"
            )
        if write_advantages.shape != (causal_features.size(0),):
            raise ValueError("write_advantages must have shape [batch]")
        if not causal_features.is_floating_point() or not write_advantages.is_floating_point():
            raise TypeError("critic batch tensors must be floating point")
        if not bool(torch.isfinite(causal_features).all()) or not bool(
            torch.isfinite(write_advantages).all()
        ):
            raise ValueError("critic batch tensors must be finite")
        reference = next(self.utility_gate.parameters())
        features = causal_features.detach().to(reference)
        advantages = write_advantages.detach().to(reference)
        targets = (advantages / self.advantage_scale).unsqueeze(1)
        prediction = self.utility_gate(features)
        loss = F.mse_loss(prediction, targets)
        active = [
            (f"utility_gate.{name}", parameter, self.utility_lr)
            for name, parameter in self.utility_gate.named_parameters()
            if parameter.requires_grad
        ] if learn else []
        raw_gradients = torch.autograd.grad(
            loss,
            [parameter for _name, parameter, _step in active],
            allow_unused=False,
        ) if active else ()
        entries = [
            (name, gradient.detach(), step)
            for (name, _parameter, step), gradient in zip(active, raw_gradients)
        ]
        projected, stats, _memory_added, projection_applied = self._project_entries(
            entries,
            self.utility_gradient_memory,
            remember_gradient=False,
            project_with_memory=project_with_memory,
        )
        update_norm, clip_scale = self._update_geometry(
            entries,
            projected,
            self.utility_max_update_norm,
        )
        with torch.no_grad():
            for name, parameter, step in active:
                parameter.add_(projected[name], alpha=-step * clip_scale)
            if entries:
                self.online_utility_updates.add_(1)
        return {
            "batch_size": int(features.size(0)),
            "loss": float(loss.detach().item()),
            "did_update": bool(entries),
            "target_positive_rate": float((advantages > 0.0).float().mean().item()),
            "predicted_positive_rate_before_update": float(
                (prediction.detach() > 0.0).float().mean().item()
            ),
            "target_mean": float(targets.detach().mean().item()),
            "target_max_abs": float(targets.detach().abs().max().item()),
            "update_norm": update_norm,
            "update_clip_scale": clip_scale,
            "ogd_projection_applied": projection_applied,
            "feedback_after_action": True,
        }

    @torch.no_grad()
    def cancel_ticket_branch(self, ticket_id: int, branch: str) -> None:
        record = self._ticket_record(ticket_id)
        if branch == "task":
            record["task_resolved"] = True
        elif branch == "utility":
            record["utility_resolved"] = True
        else:
            raise ValueError('branch must be "task" or "utility"')
        self._maybe_close_ticket(ticket_id)

    @torch.no_grad()
    def reset_state(self, stream_id: Any = None) -> None:
        key = self._stream_key(stream_id)
        self._stream_states.pop(key, None)
        self._feature_history.pop(key, None)

    @property
    def pending_ticket_count(self) -> int:
        return len(self._pending_tickets)

    @property
    def task_parameter_count(self) -> int:
        """Number of task parameters that can actually receive online updates.

        A competitive sidecar may wrap every adaptive core in a shared frozen
        base projection.  Counting that projection as task memory would make
        the reported OGD/replay capacity larger than the differentiable task
        layout used by :meth:`apply_feedback`.
        """

        return sum(
            parameter.numel()
            for _name, parameter in self._named_task_parameters()
            if parameter.requires_grad
        )

    @property
    def utility_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.utility_gate.parameters())

    @property
    def online_state_bytes(self) -> int:
        states = sum(
            value.numel() * value.element_size()
            for state in self._stream_states.values()
            for value in state.rings
        )
        shadows = sum(
            value.numel() * value.element_size()
            for record in self._pending_tickets.values()
            for name in ("previous_state", "no_write_state", "write_state")
            for value in record[name].rings
        )
        return int(
            states
            + shadows
            + self.task_gradient_memory.storage_bytes
            + self.utility_gradient_memory.storage_bytes
        )

    def _named_task_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        result = [(f"core.{name}", parameter) for name, parameter in self.core.named_parameters()]
        for prefix, module in (
            ("placement_head", self.placement_head),
            ("legality_head", self.legality_head),
            ("pass_head", self.pass_head),
            ("value_head", self.value_head),
        ):
            result.extend((f"{prefix}.{name}", parameter) for name, parameter in module.named_parameters())
        if self.spatial_skip_heads is not None:
            result.extend(
                (f"spatial_skip_heads.{name}", parameter)
                for name, parameter in self.spatial_skip_heads.named_parameters()
            )
        return result

    @staticmethod
    def _is_cayley_parameter(name: str) -> bool:
        """Return whether a task parameter belongs to a temporal Cayley map."""

        value = str(name)
        return (value.startswith("core.unitary_params.") or ".unitary_params." in value
                or value.startswith("core.angle_controllers."))

    @staticmethod
    def _output_linf_drift(
        reference: GoMultiHeadOutput,
        candidate: GoMultiHeadOutput,
    ) -> float:
        values = (
            tensor_linf_drift(reference.policy_logits, candidate.policy_logits),
            tensor_linf_drift(reference.legality_logits, candidate.legality_logits),
            tensor_linf_drift(reference.value, candidate.value),
            tensor_linf_drift(reference.latent, candidate.latent),
        )
        return max(values)

    @staticmethod
    def _state_linf_drift(
        reference: TemporalMQRState,
        candidate: TemporalMQRState,
    ) -> float:
        if len(reference.rings) != len(candidate.rings):
            raise ValueError("state layouts changed during a candidate update")
        return max(
            (
                tensor_linf_drift(left, right)
                for left, right in zip(reference.rings, candidate.rings)
            ),
            default=0.0,
        )

    def _validate_observation(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a tensor")
        if x.shape != (1, self.input_dim):
            raise ValueError(f"x must have shape [1, {self.input_dim}], got {tuple(x.shape)}")
        if torch.is_complex(x) or not x.is_floating_point():
            raise TypeError("x must be a real floating-point tensor")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("x must contain only finite values")

    def _update_feature_history(self, key: Optional[Hashable], x: torch.Tensor) -> None:
        history = self._feature_history.setdefault(key, self._empty_history())
        detached = x.detach().clone()
        if history["running_mean"] is None:
            history["running_mean"] = detached
        else:
            history["running_mean"] = (
                0.9 * history["running_mean"].to(detached) + 0.1 * detached
            ).detach()
        history["previous_input"] = detached
        history["count"] = int(history["count"]) + 1

    def _ticket_record(self, ticket_id: int) -> Dict[str, Any]:
        if not isinstance(ticket_id, int) or isinstance(ticket_id, bool) or ticket_id <= 0:
            raise TypeError("ticket_id must be a positive integer")
        record = self._pending_tickets.get(ticket_id)
        if record is not None:
            return record
        status = self._closed_tickets.get(ticket_id)
        if status is not None:
            raise RuntimeError(f"ticket {ticket_id} is closed: {status}")
        raise KeyError(f"unknown ticket: {ticket_id}")

    def _maybe_close_ticket(self, ticket_id: int) -> None:
        record = self._pending_tickets.get(ticket_id)
        if record is None:
            return
        if bool(record["task_resolved"]) and bool(record["utility_resolved"]):
            self._pending_tickets.pop(ticket_id)
            self._closed_tickets[ticket_id] = "consumed"
            while len(self._closed_tickets) > 2 * self.max_pending_tickets:
                self._closed_tickets.popitem(last=False)

    @staticmethod
    def _project_entries(
        entries: Sequence[Tuple[str, torch.Tensor, float]],
        memory: OrthogonalGradientMemory,
        *,
        remember_gradient: bool,
        project_with_memory: bool,
        allowed_names: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Mapping[str, float], bool, bool]:
        if not entries:
            return {}, {
                "raw_norm": 0.0,
                "projected_norm": 0.0,
                "retained_norm": 0.0,
                "max_abs_overlap": 0.0,
            }, False, False
        projection_applied = bool(project_with_memory and memory.max_rank > 0)
        if projection_applied:
            projected, stats = memory.project_preconditioned(
                entries,
                allowed_names=allowed_names,
            )
        else:
            allowed = (
                {name for name, _gradient, _step in entries}
                if allowed_names is None
                else {str(name) for name in allowed_names}
            )
            unknown = allowed.difference(
                name for name, _gradient, _step in entries
            )
            if unknown:
                raise ValueError(f"allowed_names contains unknown gradients: {sorted(unknown)}")
            projected = {
                name: gradient if name in allowed else torch.zeros_like(gradient)
                for name, gradient, _step in entries
            }
            norm = math.sqrt(
                sum(
                    step * float(gradient.square().sum().item())
                    for name, gradient, step in entries
                    if name in allowed
                )
            )
            stats = {
                "raw_norm": norm,
                "projected_norm": norm,
                "retained_norm": 1.0,
                "max_abs_overlap": 0.0,
            }
        memory_added = bool(memory.observe(entries)) if remember_gradient else False
        return projected, stats, memory_added, projection_applied

    @staticmethod
    def _update_geometry(
        entries: Sequence[Tuple[str, torch.Tensor, float]],
        projected: Mapping[str, torch.Tensor],
        maximum: Optional[float],
    ) -> Tuple[float, float]:
        if not entries:
            return 0.0, 1.0
        norm = math.sqrt(
            sum(
                step * step * float(projected[name].square().sum().item())
                for name, _gradient, step in entries
            )
        )
        scale = 1.0
        if maximum is not None and norm > float(maximum):
            scale = float(maximum) / (norm + 1e-12)
        return norm * scale, scale
