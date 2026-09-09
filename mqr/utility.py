from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .online import OrthogonalGradientMemory
from .temporal import MultiTimescaleMQR, TemporalMQRState


CAUSAL_UTILITY_FEATURES = (
    "novelty",
    "uncertainty",
    "past_surprise",
    "slow_saturation",
    "context_change",
)

CONTEXTUAL_UTILITY_FEATURES = CAUSAL_UTILITY_FEATURES + (
    "context_volatility",
)


class FutureUtilityGate(nn.Module):
    """Low-rank estimator of normalized counterfactual write advantage.

    The controller does not classify event types.  Given causal features
    available before the current write, it predicts the future loss reduction
    attributable to writing the candidate.  The surrounding runtime converts
    positive predicted advantage into a hard write action.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        rank: int = 8,
        activation: str = "tanh",
        initial_advantage: float = -0.05,
    ):
        super().__init__()
        if input_dim <= 0 or rank <= 0:
            raise ValueError("input_dim and rank must be positive")
        if activation not in ("none", "tanh", "relu", "gelu"):
            raise ValueError(
                'activation must be "none", "tanh", "relu", or "gelu"'
            )
        if not math.isfinite(float(initial_advantage)):
            raise ValueError("initial_advantage must be finite")
        self.input_dim = int(input_dim)
        self.rank = int(rank)
        self.activation = str(activation)
        self.initial_advantage = float(initial_advantage)
        self.input_down = nn.Linear(self.input_dim, self.rank, bias=False)
        self.output = nn.Linear(self.rank, 1, bias=True)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.fill_(self.initial_advantage)

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation == "none":
            return value
        if self.activation == "tanh":
            return torch.tanh(value)
        if self.activation == "relu":
            return F.relu(value)
        if self.activation == "gelu":
            return F.gelu(value)
        raise RuntimeError(f"unknown activation: {self.activation}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return normalized predicted advantage with shape ``[batch, 1]``."""

        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a tensor")
        if features.dim() != 2 or features.size(1) != self.input_dim:
            raise ValueError(
                f"features must be [batch, {self.input_dim}], got "
                f"{tuple(features.shape)}"
            )
        if torch.is_complex(features) or not features.is_floating_point():
            raise TypeError("features must be a real floating-point tensor")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features must contain only finite values")
        reference = self.input_down.weight
        prepared = features.to(device=reference.device, dtype=reference.dtype)
        return self.output(self._activate(self.input_down(prepared)))


class ContextualFutureUtilityGate(nn.Module):
    """Hard-routed utility experts under a fixed total low-rank budget.

    Routing consumes only a named causal feature already present at decision
    time.  It never sees the future target or an external task identifier.
    Hard routing gives exact parameter isolation between experts; every expert
    is still trained solely by its delayed counterfactual write advantage.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        total_rank: int,
        experts: int,
        router_index: int,
        router_threshold: float,
        initial_advantage: float,
    ) -> None:
        super().__init__()
        if experts != 2:
            raise ValueError("the reference contextual gate currently requires two experts")
        if total_rank < experts:
            raise ValueError("total_rank must be at least the number of experts")
        if not (0 <= int(router_index) < int(input_dim)):
            raise ValueError("router_index is out of range")
        if not math.isfinite(float(router_threshold)):
            raise ValueError("router_threshold must be finite")
        ranks = [total_rank // experts for _ in range(experts)]
        for index in range(total_rank % experts):
            ranks[index] += 1
        self.input_dim = int(input_dim)
        self.total_rank = int(total_rank)
        self.router_index = int(router_index)
        self.router_threshold = float(router_threshold)
        self.experts = nn.ModuleList(
            FutureUtilityGate(
                self.input_dim,
                rank=rank,
                initial_advantage=initial_advantage,
            )
            for rank in ranks
        )

    def route_indices(self, features: torch.Tensor) -> torch.Tensor:
        return (
            features[:, self.router_index] > self.router_threshold
        ).to(dtype=torch.long)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs = torch.cat(
            [expert(features) for expert in self.experts], dim=1
        )
        routes = self.route_indices(features)
        return outputs.gather(1, routes.unsqueeze(1))


class UtilityDrivenMQR(nn.Module):
    """Online MQR with delayed counterfactual credit for slow-ring writes.

    Ring zero is an always-written fast candidate ring.  Remaining rings are
    slow memory.  Every issued candidate stores two bounded shadow trajectories
    that differ only in the candidate's slow write.  Once future task feedback
    arrives, their loss difference supervises :class:`FutureUtilityGate` using
    cached causal features.  A useful missed candidate is promoted as external
    forcing on the next observation; a harmful write cannot be retroactively
    removed and is reported explicitly.

    This reference runtime intentionally accepts one item per stream step.  It
    supports multiple keyed streams, but not batched heterogeneous streams.
    """

    def __init__(
        self,
        input_dim: int,
        ring_dim: int,
        output_dim: int,
        *,
        core_kwargs: Optional[Dict[str, Any]] = None,
        gate_rank: int = 8,
        utility_lr: float = 5e-2,
        readout_lr: float = 5e-2,
        utility_ogd_max_rank: int = 0,
        readout_ogd_max_rank: int = 0,
        utility_max_update_norm: Optional[float] = 0.05,
        readout_max_update_norm: Optional[float] = 0.25,
        memory_cost: float = 0.0,
        advantage_scale: float = 1.0,
        advantage_target_clip: float = 5.0,
        decision_temperature: float = 0.25,
        initial_advantage: float = -0.05,
        feature_ema_decay: float = 0.9,
        context_volatility_decay: Optional[float] = None,
        utility_experts: int = 1,
        utility_router_feature: Optional[str] = None,
        utility_router_threshold: float = 0.10,
        max_pending_candidates: int = 128,
        candidate_overflow_policy: str = "error",
        max_streams: int = 64,
        stream_overflow_policy: str = "error",
        promotion_max_norm: Optional[float] = 4.0,
    ):
        super().__init__()
        if not math.isfinite(float(utility_lr)) or utility_lr <= 0.0:
            raise ValueError("utility_lr must be finite and positive")
        if not math.isfinite(float(readout_lr)) or readout_lr <= 0.0:
            raise ValueError("readout_lr must be finite and positive")
        for name, value in (
            ("utility_max_update_norm", utility_max_update_norm),
            ("readout_max_update_norm", readout_max_update_norm),
            ("promotion_max_norm", promotion_max_norm),
        ):
            if value is not None and (
                not math.isfinite(float(value)) or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive or None")
        if not math.isfinite(float(memory_cost)) or memory_cost < 0.0:
            raise ValueError("memory_cost must be finite and non-negative")
        if not math.isfinite(float(advantage_scale)) or advantage_scale <= 0.0:
            raise ValueError("advantage_scale must be finite and positive")
        if (
            not math.isfinite(float(advantage_target_clip))
            or advantage_target_clip <= 0.0
        ):
            raise ValueError("advantage_target_clip must be finite and positive")
        if (
            not math.isfinite(float(decision_temperature))
            or decision_temperature <= 0.0
        ):
            raise ValueError("decision_temperature must be finite and positive")
        if not (0.0 <= float(feature_ema_decay) < 1.0):
            raise ValueError("feature_ema_decay must be in [0, 1)")
        if context_volatility_decay is not None and not (
            0.0 <= float(context_volatility_decay) < 1.0
        ):
            raise ValueError("context_volatility_decay must be in [0, 1) or None")
        if utility_experts <= 0:
            raise ValueError("utility_experts must be positive")
        if max_pending_candidates <= 0 or max_streams <= 0:
            raise ValueError("candidate and stream capacities must be positive")
        if candidate_overflow_policy not in ("error", "oldest"):
            raise ValueError(
                'candidate_overflow_policy must be "error" or "oldest"'
            )
        if stream_overflow_policy not in ("error", "lru"):
            raise ValueError('stream_overflow_policy must be "error" or "lru"')

        kwargs = dict(core_kwargs or {})
        self.core = MultiTimescaleMQR(
            input_dim,
            ring_dim,
            output_dim,
            **kwargs,
        )
        if self.core.num_timescales < 2:
            raise ValueError(
                "UtilityDrivenMQR requires one fast ring and at least one slow ring"
            )
        self.utility_feature_names = (
            CAUSAL_UTILITY_FEATURES
            if context_volatility_decay is None
            else CONTEXTUAL_UTILITY_FEATURES
        )
        normalized_initial = float(initial_advantage) / float(advantage_scale)
        if utility_experts == 1:
            if utility_router_feature is not None:
                raise ValueError(
                    "utility_router_feature requires utility_experts > 1"
                )
            self.utility_gate = FutureUtilityGate(
                len(self.utility_feature_names),
                rank=gate_rank,
                initial_advantage=normalized_initial,
            )
        else:
            if utility_router_feature not in self.utility_feature_names:
                raise ValueError(
                    "utility_router_feature must name an enabled causal feature"
                )
            self.utility_gate = ContextualFutureUtilityGate(
                len(self.utility_feature_names),
                total_rank=gate_rank,
                experts=int(utility_experts),
                router_index=self.utility_feature_names.index(
                    str(utility_router_feature)
                ),
                router_threshold=float(utility_router_threshold),
                initial_advantage=normalized_initial,
            )
        self.utility_experts = int(utility_experts)
        self.utility_router_feature = utility_router_feature
        self.utility_router_threshold = float(utility_router_threshold)
        self.utility_lr = float(utility_lr)
        self.readout_lr = float(readout_lr)
        self.utility_max_update_norm = (
            None
            if utility_max_update_norm is None
            else float(utility_max_update_norm)
        )
        self.readout_max_update_norm = (
            None
            if readout_max_update_norm is None
            else float(readout_max_update_norm)
        )
        self.memory_cost = float(memory_cost)
        self.advantage_scale = float(advantage_scale)
        self.advantage_target_clip = float(advantage_target_clip)
        self.decision_temperature = float(decision_temperature)
        self.feature_ema_decay = float(feature_ema_decay)
        self.context_volatility_decay = (
            None
            if context_volatility_decay is None
            else float(context_volatility_decay)
        )
        self.max_pending_candidates = int(max_pending_candidates)
        self.candidate_overflow_policy = str(candidate_overflow_policy)
        self.max_streams = int(max_streams)
        self.stream_overflow_policy = str(stream_overflow_policy)
        self.promotion_max_norm = (
            None if promotion_max_norm is None else float(promotion_max_norm)
        )
        self.utility_gradient_memory = OrthogonalGradientMemory(
            utility_ogd_max_rank
        )
        self.readout_gradient_memory = OrthogonalGradientMemory(
            readout_ogd_max_rank
        )

        self._stream_states: OrderedDict[
            Optional[Hashable], TemporalMQRState
        ] = OrderedDict()
        self._feature_states: Dict[Optional[Hashable], Dict[str, Any]] = {}
        self._pending_candidates: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self._closed_candidates: OrderedDict[int, str] = OrderedDict()
        self._pending_promotions: Dict[Optional[Hashable], torch.Tensor] = {}
        self._last_outputs: Dict[Optional[Hashable], Dict[str, Any]] = {}
        self._next_candidate_id = 1
        self._history_limit = max(16, 2 * self.max_pending_candidates)

        self.register_buffer("online_observations", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "online_parameter_version", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "online_utility_version", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "online_readout_version", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "online_utility_feedback", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "online_utility_updates", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "online_readout_updates", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "online_promotions_queued", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "online_promotions_applied", torch.zeros((), dtype=torch.long)
        )

    @staticmethod
    def _stream_key(stream_id: Any) -> Optional[Hashable]:
        if isinstance(stream_id, torch.Tensor):
            if stream_id.numel() != 1:
                raise ValueError("stream_id tensor must be scalar")
            stream_id = stream_id.item()
        try:
            hash(stream_id)
        except TypeError as exc:
            raise TypeError("stream_id must be hashable") from exc
        return stream_id

    @staticmethod
    def _clone_state(state: TemporalMQRState) -> TemporalMQRState:
        return TemporalMQRState(tuple(value.detach().clone() for value in state.rings))

    def _restore_state(self, values: Sequence[torch.Tensor]) -> TemporalMQRState:
        if len(values) != self.core.num_timescales:
            raise RuntimeError("checkpoint state has the wrong number of rings")
        rings = tuple(value.detach() for value in values)
        state = TemporalMQRState(rings)
        self.core._validate_state(state, 1)
        return state

    def get_extra_state(self) -> Dict[str, Any]:
        tickets = []
        for ticket_id, record in self._pending_candidates.items():
            tickets.append(
                (
                    int(ticket_id),
                    {
                        **{
                            name: value
                            for name, value in record.items()
                            if name not in ("no_write_state", "write_state")
                        },
                        "features": record["features"].detach().clone(),
                        "payload": record["payload"].detach().clone(),
                        "no_write_state": [
                            value.detach().clone()
                            for value in record["no_write_state"].rings
                        ],
                        "write_state": [
                            value.detach().clone()
                            for value in record["write_state"].rings
                        ],
                    },
                )
            )
        return {
            "version": 1,
            "streams": [
                (key, [value.detach().clone() for value in state.rings])
                for key, state in self._stream_states.items()
            ],
            "feature_states": [
                (
                    key,
                    {
                        "count": int(record["count"]),
                        "past_surprise": float(record["past_surprise"]),
                        "context_volatility": float(
                            record.get("context_volatility", 0.0)
                        ),
                        "previous_input": (
                            None
                            if record["previous_input"] is None
                            else record["previous_input"].detach().clone()
                        ),
                        "running_mean": (
                            None
                            if record["running_mean"] is None
                            else record["running_mean"].detach().clone()
                        ),
                    },
                )
                for key, record in self._feature_states.items()
            ],
            "pending_candidates": tickets,
            "closed_candidates": list(self._closed_candidates.items()),
            "pending_promotions": [
                (key, value.detach().clone())
                for key, value in self._pending_promotions.items()
            ],
            "last_outputs": [
                (
                    key,
                    {
                        **record,
                        "logits": record["logits"].detach().clone(),
                        "state_vector": record["state_vector"].detach().clone(),
                    },
                )
                for key, record in self._last_outputs.items()
            ],
            "next_candidate_id": int(self._next_candidate_id),
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        self._stream_states.clear()
        self._feature_states.clear()
        self._pending_candidates.clear()
        self._closed_candidates.clear()
        self._pending_promotions.clear()
        self._last_outputs.clear()
        self._next_candidate_id = 1
        if not state:
            return
        if int(state.get("version", 0)) != 1:
            raise RuntimeError("unsupported UtilityDrivenMQR checkpoint version")
        for key, rings in state.get("streams", []):
            stream_key = self._stream_key(key)
            self._stream_states[stream_key] = self._restore_state(rings)
        if len(self._stream_states) > self.max_streams:
            raise RuntimeError("checkpoint exceeds max_streams")
        for key, record in state.get("feature_states", []):
            stream_key = self._stream_key(key)
            self._feature_states[stream_key] = {
                "count": int(record["count"]),
                "past_surprise": float(record["past_surprise"]),
                "context_volatility": float(
                    record.get("context_volatility", 0.0)
                ),
                "previous_input": record["previous_input"],
                "running_mean": record["running_mean"],
            }
        for ticket_id, record in state.get("pending_candidates", []):
            restored = dict(record)
            restored["no_write_state"] = self._restore_state(
                record["no_write_state"]
            )
            restored["write_state"] = self._restore_state(record["write_state"])
            self._pending_candidates[int(ticket_id)] = restored
        if len(self._pending_candidates) > self.max_pending_candidates:
            raise RuntimeError("checkpoint exceeds max_pending_candidates")
        for ticket_id, status in state.get("closed_candidates", []):
            self._closed_candidates[int(ticket_id)] = str(status)
        for key, value in state.get("pending_promotions", []):
            self._pending_promotions[self._stream_key(key)] = value.detach()
        for key, record in state.get("last_outputs", []):
            self._last_outputs[self._stream_key(key)] = dict(record)
        self._next_candidate_id = int(state.get("next_candidate_id", 1))

    def _empty_feature_state(self) -> Dict[str, Any]:
        return {
            "count": 0,
            "past_surprise": 0.0,
            "context_volatility": 0.0,
            "previous_input": None,
            "running_mean": None,
        }

    @staticmethod
    def _cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_norm = torch.linalg.vector_norm(left, dim=1)
        right_norm = torch.linalg.vector_norm(right, dim=1)
        denominator = left_norm * right_norm
        similarity = torch.zeros_like(denominator)
        nonzero = denominator > torch.finfo(left.dtype).eps
        if bool(nonzero.any()):
            similarity[nonzero] = (
                (left[nonzero] * right[nonzero]).sum(dim=1)
                / denominator[nonzero]
            )
        return (0.5 * (1.0 - similarity.clamp(-1.0, 1.0))).clamp(0.0, 1.0)

    def _causal_features(
        self,
        x: torch.Tensor,
        state: TemporalMQRState,
        key: Optional[Hashable],
    ) -> torch.Tensor:
        history = self._feature_states.get(key, self._empty_feature_state())
        if history["running_mean"] is None:
            novelty = x.new_zeros((1,))
        else:
            novelty = self._cosine_distance(
                x,
                history["running_mean"].to(device=x.device, dtype=x.dtype),
            )
        if history["previous_input"] is None:
            context_change = x.new_zeros((1,))
        else:
            context_change = self._cosine_distance(
                x,
                history["previous_input"].to(device=x.device, dtype=x.dtype),
            )
        logits = self.core.readout_state(state)
        probabilities = F.softmax(logits, dim=1)
        if self.core.output_dim == 1:
            uncertainty = x.new_zeros((1,))
        else:
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)
            uncertainty = entropy / math.log(self.core.output_dim)
        slow = torch.cat(state.rings[1:], dim=1)
        saturation = torch.tanh(slow.abs().mean(dim=1))
        past_surprise = x.new_full((1,), float(history["past_surprise"]))
        values = [
            novelty,
            uncertainty.to(device=x.device, dtype=x.dtype),
            past_surprise,
            saturation.to(device=x.device, dtype=x.dtype),
            context_change,
        ]
        if self.context_volatility_decay is not None:
            values.append(
                x.new_full((1,), float(history["context_volatility"]))
            )
        return torch.stack(values, dim=1).detach()

    def _update_feature_history(
        self,
        key: Optional[Hashable],
        x: torch.Tensor,
    ) -> None:
        history = self._feature_states.setdefault(key, self._empty_feature_state())
        detached = x.detach().clone()
        previous_input = history["previous_input"]
        if self.context_volatility_decay is not None:
            if previous_input is None:
                current_change = 0.0
            else:
                current_change = float(
                    self._cosine_distance(
                        detached,
                        previous_input.to(device=x.device, dtype=x.dtype),
                    ).item()
                )
            if int(history["count"]) <= 1:
                volatility = current_change
            else:
                decay = self.context_volatility_decay
                volatility = (
                    decay * float(history["context_volatility"])
                    + (1.0 - decay) * current_change
                )
            history["context_volatility"] = max(0.0, min(1.0, volatility))
        if history["running_mean"] is None:
            running_mean = detached
        else:
            running_mean = (
                self.feature_ema_decay * history["running_mean"].to(
                    device=x.device,
                    dtype=x.dtype,
                )
                + (1.0 - self.feature_ema_decay) * detached
            )
        history["running_mean"] = running_mean.detach()
        history["previous_input"] = detached
        history["count"] = int(history["count"]) + 1

    def _gate_values(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized = self.utility_gate(features)
        advantage = self.advantage_scale * normalized
        probability = torch.sigmoid(advantage / self.decision_temperature)
        return advantage, probability

    def _utility_expert_index(self, features: torch.Tensor) -> int:
        if isinstance(self.utility_gate, ContextualFutureUtilityGate):
            return int(self.utility_gate.route_indices(features).item())
        return 0

    def _gate_tensor(
        self,
        x: torch.Tensor,
        slow_action: bool,
    ) -> torch.Tensor:
        gates = torch.zeros(
            1,
            self.core.num_timescales,
            device=x.device,
            dtype=x.dtype,
        )
        gates[:, 0] = 1.0
        if slow_action:
            gates[:, 1:] = 1.0
        return gates

    def _candidate_overflow(self) -> Optional[int]:
        if len(self._pending_candidates) < self.max_pending_candidates:
            return None
        if self.candidate_overflow_policy == "error":
            raise RuntimeError(
                f"candidate queue is full ({self.max_pending_candidates}); "
                "resolve/cancel feedback or choose candidate_overflow_policy='oldest'"
            )
        return next(iter(self._pending_candidates))

    def _record_closed(self, ticket_id: int, status: str) -> None:
        self._closed_candidates.pop(int(ticket_id), None)
        self._closed_candidates[int(ticket_id)] = str(status)
        while len(self._closed_candidates) > self._history_limit:
            self._closed_candidates.popitem(last=False)

    def _cancel_stream_runtime(self, key: Optional[Hashable], status: str) -> None:
        for ticket_id, record in list(self._pending_candidates.items()):
            if record["stream_id"] == key:
                self._pending_candidates.pop(ticket_id)
                self._record_closed(ticket_id, status)
        self._feature_states.pop(key, None)
        self._pending_promotions.pop(key, None)
        self._last_outputs.pop(key, None)

    def _ensure_stream_capacity(
        self,
        key: Optional[Hashable],
    ) -> Optional[Hashable]:
        if key in self._stream_states or len(self._stream_states) < self.max_streams:
            return None
        if self.stream_overflow_policy == "error":
            raise RuntimeError(
                f"state bank is full ({self.max_streams}); reset a stream or choose "
                "stream_overflow_policy='lru'"
            )
        return next(iter(self._stream_states))

    def _promotion_for(
        self,
        key: Optional[Hashable],
        x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        value = self._pending_promotions.get(key)
        if value is None:
            return None
        return value.to(device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def observe(
        self,
        x: torch.Tensor,
        *,
        stream_id: Any = None,
        issue_candidate: bool = True,
        external_write: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Advance one stream and optionally issue a counterfactual ticket.

        The write action and prediction are committed before any utility or task
        label can be supplied.  ``external_write`` is an auditable oracle or
        ablation override; the learned estimate is still returned and can later
        be trained from the resulting ticket.
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a tensor")
        if x.shape != (1, self.core.input_dim):
            raise ValueError(
                f"x must have shape [1, {self.core.input_dim}], got {tuple(x.shape)}"
            )
        if torch.is_complex(x) or not x.is_floating_point():
            raise TypeError("x must be a real floating-point tensor")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("x must contain only finite values")
        if external_write is not None and not isinstance(external_write, bool):
            raise TypeError("external_write must be bool or None")

        key = self._stream_key(stream_id)
        evicted_key = self._ensure_stream_capacity(key)
        overflow_ticket = self._candidate_overflow() if issue_candidate else None
        previous = self._stream_states.get(key)
        state_was_present = previous is not None
        if previous is None:
            previous = self.core.zero_state(1, device=x.device, dtype=x.dtype)
        features = self._causal_features(x, previous, key)
        predicted_advantage, write_probability = self._gate_values(features)
        utility_expert = self._utility_expert_index(features)
        learned_write = bool(float(predicted_advantage.item()) > 0.0)
        action = learned_write if external_write is None else external_write
        gate = self._gate_tensor(x, action)
        promotion = self._promotion_for(key, x)

        advanced_tickets: Dict[int, Tuple[TemporalMQRState, TemporalMQRState]] = {}
        for ticket_id, record in self._pending_candidates.items():
            if record["stream_id"] != key:
                continue
            _no_logits, no_state = self.core.forward_step(
                x,
                state=record["no_write_state"],
                write_gate=gate,
                promotion=promotion,
            )
            _yes_logits, yes_state = self.core.forward_step(
                x,
                state=record["write_state"],
                write_gate=gate,
                promotion=promotion,
            )
            advanced_tickets[ticket_id] = (
                no_state.detached(),
                yes_state.detached(),
            )

        logits, next_state = self.core.forward_step(
            x,
            state=previous,
            write_gate=gate,
            promotion=promotion,
        )
        next_state = next_state.detached()
        new_record: Optional[Dict[str, Any]] = None
        if issue_candidate:
            no_gate = self._gate_tensor(x, False)
            yes_gate = self._gate_tensor(x, True)
            _no_logits, no_state = self.core.forward_step(
                x,
                state=previous,
                write_gate=no_gate,
                promotion=promotion,
            )
            _yes_logits, yes_state = self.core.forward_step(
                x,
                state=previous,
                write_gate=yes_gate,
                promotion=promotion,
            )
            payload = self.core.input_forcing(x).detach()
            payload[:, 0, :] = 0.0
            new_record = {
                "features": features.detach().clone(),
                "payload": payload,
                "no_write_state": no_state.detached(),
                "write_state": yes_state.detached(),
                "stream_id": key,
                "issued_observation": int(self.online_observations.item()),
                "issued_utility_version": int(self.online_utility_version.item()),
                "issued_parameter_version": int(self.online_parameter_version.item()),
                "action_taken": bool(action),
                "learned_action": bool(learned_write),
                "external_override": external_write is not None,
                "predicted_advantage_at_issue": float(predicted_advantage.item()),
                "write_probability_at_issue": float(write_probability.item()),
                "utility_expert_at_issue": utility_expert,
            }

        if evicted_key is not None:
            self._stream_states.pop(evicted_key)
            self._cancel_stream_runtime(evicted_key, "stream_evicted")
        for ticket_id, (no_state, yes_state) in advanced_tickets.items():
            if ticket_id in self._pending_candidates:
                self._pending_candidates[ticket_id]["no_write_state"] = no_state
                self._pending_candidates[ticket_id]["write_state"] = yes_state
        if overflow_ticket is not None:
            self._pending_candidates.pop(overflow_ticket)
            self._record_closed(overflow_ticket, "evicted")
        ticket_id: Optional[int] = None
        if new_record is not None:
            ticket_id = self._next_candidate_id
            self._next_candidate_id += 1
            self._pending_candidates[ticket_id] = new_record
        self._stream_states[key] = next_state
        self._stream_states.move_to_end(key)
        self._update_feature_history(key, x)
        promotion_applied = promotion is not None
        if promotion_applied:
            self._pending_promotions.pop(key, None)
            self.online_promotions_applied.add_(1)
        self.online_observations.add_(1)
        self._last_outputs[key] = {
            "logits": logits.detach().clone(),
            "state_vector": torch.cat(next_state.rings, dim=1).detach().clone(),
            "issued_observation": int(self.online_observations.item()),
            "issued_readout_version": int(self.online_readout_version.item()),
            "feedback_consumed": False,
        }
        return {
            "logits": logits.detach().clone(),
            "state": next_state,
            "features": features.detach().clone(),
            "feature_names": self.utility_feature_names,
            "predicted_advantage": float(predicted_advantage.item()),
            "write_probability": float(write_probability.item()),
            "utility_expert": utility_expert,
            "learned_write": learned_write,
            "effective_write": bool(action),
            "write_source": "learned" if external_write is None else "external",
            "candidate_id": ticket_id,
            "prediction_before_feedback": True,
            "state_was_present": state_was_present,
            "stream_id": key,
            "state_evicted": evicted_key is not None,
            "evicted_stream_id": evicted_key,
            "candidate_evicted": overflow_ticket is not None,
            "evicted_candidate_id": overflow_ticket,
            "promotion_applied": promotion_applied,
            "pending_candidates": len(self._pending_candidates),
            "pending_promotion": key in self._pending_promotions,
            "max_unitary_error": self.core.max_unitary_error(),
            "max_stochastic_error": self.core.max_stochastic_error(),
        }

    @staticmethod
    def _classification_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(target, torch.Tensor):
            raise TypeError("target must be a tensor")
        if target.dim() == 0:
            target = target.reshape(1)
        if target.shape != (logits.size(0),):
            raise ValueError(
                f"target must have shape [{logits.size(0)}], got {tuple(target.shape)}"
            )
        return F.cross_entropy(logits, target.detach().to(device=logits.device))

    @staticmethod
    def _project_entries(
        entries: Sequence[Tuple[str, torch.Tensor, float]],
        memory: OrthogonalGradientMemory,
        *,
        remember_gradient: bool,
        project_with_memory: bool,
        max_update_norm: Optional[float],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], float, float, bool, bool]:
        if not entries:
            return {}, {
                "raw_norm": 0.0,
                "projected_norm": 0.0,
                "retained_norm": 0.0,
                "max_abs_overlap": 0.0,
            }, 0.0, 1.0, False, False
        projection_applied = bool(project_with_memory and memory.max_rank > 0)
        if projection_applied:
            projected, stats = memory.project_preconditioned(entries)
        else:
            projected = {name: gradient for name, gradient, _step in entries}
            whitened_sq = sum(
                step * float(gradient.square().sum().item())
                for _name, gradient, step in entries
            )
            whitened_norm = math.sqrt(whitened_sq)
            stats = {
                "raw_norm": whitened_norm,
                "projected_norm": whitened_norm,
                "retained_norm": 1.0,
                "max_abs_overlap": 0.0,
            }
        memory_added = memory.observe(entries) if remember_gradient else False
        update_sq = sum(
            (step**2) * float(projected[name].square().sum().item())
            for name, _gradient, step in entries
        )
        unclipped_norm = math.sqrt(update_sq)
        clip_scale = 1.0
        if max_update_norm is not None and unclipped_norm > max_update_norm:
            clip_scale = max_update_norm / (unclipped_norm + 1e-12)
        return (
            projected,
            stats,
            unclipped_norm,
            clip_scale,
            bool(memory_added),
            projection_applied,
        )

    def _candidate_record(self, ticket_id: int) -> Dict[str, Any]:
        if not isinstance(ticket_id, int) or isinstance(ticket_id, bool) or ticket_id <= 0:
            raise TypeError("candidate_id must be a positive integer")
        record = self._pending_candidates.get(ticket_id)
        if record is not None:
            return record
        status = self._closed_candidates.get(ticket_id)
        if status is not None:
            raise RuntimeError(f"candidate {ticket_id} is closed: {status}")
        raise KeyError(f"unknown candidate: {ticket_id}")

    def _queue_promotion(
        self,
        key: Optional[Hashable],
        payload: torch.Tensor,
    ) -> None:
        combined = payload.detach().clone()
        if key in self._pending_promotions:
            combined = self._pending_promotions[key].to(
                device=combined.device,
                dtype=combined.dtype,
            ) + combined
        combined[:, 0, :] = 0.0
        if self.promotion_max_norm is not None:
            norms = torch.linalg.vector_norm(combined[:, 1:, :], dim=2, keepdim=True)
            scales = torch.clamp(
                self.promotion_max_norm / norms.clamp_min(1e-12),
                max=1.0,
            )
            combined[:, 1:, :] *= scales
        self._pending_promotions[key] = combined.detach()
        self.online_promotions_queued.add_(1)

    def _record_past_surprise(
        self,
        key: Optional[Hashable],
        loss: torch.Tensor,
    ) -> None:
        """Commit a normalized feedback loss for strictly future features."""

        history = self._feature_states.setdefault(key, self._empty_feature_state())
        normalizer = max(math.log(max(2, self.core.output_dim)), 1e-12)
        history["past_surprise"] = min(
            1.0,
            float(loss.detach().item()) / (2.0 * normalizer),
        )

    def resolve_utility(
        self,
        candidate_id: int,
        target: torch.Tensor,
        *,
        learn: bool = True,
        remember_gradient: bool = False,
        project_with_memory: bool = True,
        promote_missed_positive: bool = True,
    ) -> Dict[str, Any]:
        """Consume delayed feedback and learn future write advantage.

        The target is evaluated only after both shadow trajectories already
        exist.  Gate gradients are recomputed from the cached causal feature
        vector under current controller parameters (feature replay).
        """

        if remember_gradient and not learn:
            raise ValueError("remember_gradient requires learn=True")
        if remember_gradient and self.utility_gradient_memory.max_rank == 0:
            raise ValueError("remember_gradient requires utility_ogd_max_rank > 0")
        record = self._candidate_record(candidate_id)
        with torch.no_grad():
            no_logits = self.core.readout_state(record["no_write_state"])
            write_logits = self.core.readout_state(record["write_state"])
            no_loss = self._classification_loss(no_logits, target)
            write_loss = self._classification_loss(write_logits, target)
            current_record = self._last_outputs.get(record["stream_id"])
            current_task_loss: Optional[torch.Tensor] = None
            if current_record is not None:
                current_logits = current_record["logits"].to(
                    device=no_logits.device,
                    dtype=no_logits.dtype,
                )
                current_task_loss = self._classification_loss(
                    current_logits,
                    target,
                )
            advantage = float((no_loss - write_loss).item()) - self.memory_cost
            normalized_target = max(
                -self.advantage_target_clip,
                min(self.advantage_target_clip, advantage / self.advantage_scale),
            )

        active = [
            (f"utility_gate.{name}", parameter, self.utility_lr)
            for name, parameter in self.utility_gate.named_parameters()
            if parameter.requires_grad
        ]
        with torch.enable_grad():
            normalized_prediction = self.utility_gate(record["features"])
            target_value = normalized_prediction.new_full(
                normalized_prediction.shape,
                normalized_target,
            )
            utility_loss = F.mse_loss(normalized_prediction, target_value)
            gradients: Tuple[torch.Tensor, ...] = ()
            if learn and active:
                gradients = tuple(
                    torch.autograd.grad(
                        utility_loss,
                        [parameter for _name, parameter, _step in active],
                        create_graph=False,
                        retain_graph=False,
                    )
                )
        entries = [
            (name, gradient.detach(), step)
            for (name, _parameter, step), gradient in zip(active, gradients)
        ]
        projected, stats, unclipped_norm, clip_scale, memory_added, projected_flag = (
            self._project_entries(
                entries,
                self.utility_gradient_memory,
                remember_gradient=remember_gradient,
                project_with_memory=project_with_memory,
                max_update_norm=self.utility_max_update_norm,
            )
        )
        missed_positive = advantage > 0.0 and not bool(record["action_taken"])
        irreversible_false_positive = (
            advantage <= 0.0 and bool(record["action_taken"])
        )
        promotion_queued = bool(missed_positive and promote_missed_positive)
        version_before = int(self.online_utility_version.item())
        with torch.no_grad():
            for name, parameter, step in active:
                if name in projected:
                    parameter.add_(projected[name], alpha=-step * clip_scale)
            if entries:
                self.online_utility_updates.add_(1)
                self.online_utility_version.add_(1)
                self.online_parameter_version.add_(1)
            if promotion_queued:
                self._queue_promotion(record["stream_id"], record["payload"])
            if current_task_loss is not None:
                self._record_past_surprise(
                    record["stream_id"],
                    current_task_loss,
                )
            self.online_utility_feedback.add_(1)
            self._pending_candidates.pop(candidate_id)
            self._record_closed(candidate_id, "consumed")
        return {
            "candidate_id": candidate_id,
            "stream_id": record["stream_id"],
            "no_write_logits": no_logits.detach().clone(),
            "write_logits": write_logits.detach().clone(),
            "no_write_loss": float(no_loss.item()),
            "write_loss": float(write_loss.item()),
            "current_task_loss": (
                None
                if current_task_loss is None
                else float(current_task_loss.item())
            ),
            "memory_cost": self.memory_cost,
            "write_advantage": advantage,
            "normalized_utility_target": normalized_target,
            "predicted_advantage_at_issue": record[
                "predicted_advantage_at_issue"
            ],
            "predicted_advantage_before_update": float(
                self.advantage_scale * normalized_prediction.detach().item()
            ),
            "utility_loss": float(utility_loss.detach().item()),
            "action_taken": bool(record["action_taken"]),
            "learned_action_at_issue": bool(record["learned_action"]),
            "external_override_at_issue": bool(record["external_override"]),
            "utility_expert_at_issue": int(
                record.get("utility_expert_at_issue", 0)
            ),
            "missed_positive": missed_positive,
            "promotion_queued": promotion_queued,
            "irreversible_false_positive": irreversible_false_positive,
            "feedback_after_action": True,
            "causal_feature_replay": True,
            "did_update": bool(entries),
            "issued_utility_version": int(record["issued_utility_version"]),
            "utility_staleness": version_before
            - int(record["issued_utility_version"]),
            "candidate_age_observations": int(self.online_observations.item())
            - int(record["issued_observation"]),
            "utility_version": int(self.online_utility_version.item()),
            "parameter_version": int(self.online_parameter_version.item()),
            "ogd_rank": self.utility_gradient_memory.rank,
            "ogd_raw_norm": float(stats["raw_norm"]),
            "ogd_projected_norm": float(stats["projected_norm"]),
            "ogd_retained_norm": float(stats["retained_norm"]),
            "ogd_max_abs_overlap": float(stats["max_abs_overlap"]),
            "ogd_memory_added": memory_added,
            "ogd_projection_applied": projected_flag,
            "unclipped_update_norm": unclipped_norm,
            "update_clip_scale": clip_scale,
            "update_norm": unclipped_norm * clip_scale,
            "pending_candidates": len(self._pending_candidates),
        }

    def resolve_stream(
        self,
        target: torch.Tensor,
        *,
        stream_id: Any = None,
        learn: bool = True,
        promote_missed_positive: bool = True,
    ) -> List[Dict[str, Any]]:
        """Resolve all currently pending candidates for one stream in order."""

        key = self._stream_key(stream_id)
        candidate_ids = [
            ticket_id
            for ticket_id, record in self._pending_candidates.items()
            if record["stream_id"] == key
        ]
        return [
            self.resolve_utility(
                ticket_id,
                target,
                learn=learn,
                promote_missed_positive=promote_missed_positive,
            )
            for ticket_id in candidate_ids
        ]

    def learn_current(
        self,
        target: torch.Tensor,
        *,
        stream_id: Any = None,
        learn: bool = True,
        remember_gradient: bool = False,
        project_with_memory: bool = True,
    ) -> Dict[str, Any]:
        """Apply an issue-time exact readout update to the latest prediction."""

        if remember_gradient and not learn:
            raise ValueError("remember_gradient requires learn=True")
        if remember_gradient and self.readout_gradient_memory.max_rank == 0:
            raise ValueError("remember_gradient requires readout_ogd_max_rank > 0")
        key = self._stream_key(stream_id)
        record = self._last_outputs.get(key)
        if record is None:
            raise RuntimeError("stream has no prediction awaiting task feedback")
        if bool(record["feedback_consumed"]):
            raise RuntimeError("the latest prediction feedback was already consumed")
        logits = record["logits"].to(
            device=self.core.readout.weight.device,
            dtype=self.core.readout.weight.dtype,
        )
        state_vector = record["state_vector"].to(
            device=logits.device,
            dtype=logits.dtype,
        )
        loss = self._classification_loss(logits, target)
        target_work = target.reshape(1).detach().to(device=logits.device)
        error = F.softmax(logits, dim=1)
        error[0, int(target_work.item())] -= 1.0
        weight_gradient = error.transpose(0, 1) @ state_vector
        bias_gradient = error.squeeze(0) if self.core.readout.bias is not None else None
        active: List[Tuple[str, nn.Parameter, float]] = []
        gradients: List[torch.Tensor] = []
        if learn and self.core.readout.weight.requires_grad:
            active.append(("readout.weight", self.core.readout.weight, self.readout_lr))
            gradients.append(weight_gradient.detach())
        if (
            learn
            and self.core.readout.bias is not None
            and self.core.readout.bias.requires_grad
        ):
            assert bias_gradient is not None
            active.append(("readout.bias", self.core.readout.bias, self.readout_lr))
            gradients.append(bias_gradient.detach())
        entries = [
            (name, gradient, step)
            for (name, _parameter, step), gradient in zip(active, gradients)
        ]
        projected, stats, unclipped_norm, clip_scale, memory_added, projected_flag = (
            self._project_entries(
                entries,
                self.readout_gradient_memory,
                remember_gradient=remember_gradient,
                project_with_memory=project_with_memory,
                max_update_norm=self.readout_max_update_norm,
            )
        )
        version_before = int(self.online_readout_version.item())
        with torch.no_grad():
            for name, parameter, step in active:
                parameter.add_(projected[name], alpha=-step * clip_scale)
            if entries:
                self.online_readout_updates.add_(1)
                self.online_readout_version.add_(1)
                self.online_parameter_version.add_(1)
            record["feedback_consumed"] = True
            self._record_past_surprise(key, loss)
        return {
            "logits": logits.detach().clone(),
            "loss": float(loss.detach().item()),
            "prediction_before_update": True,
            "feedback_after_prediction": True,
            "did_update": bool(entries),
            "issued_readout_version": int(record["issued_readout_version"]),
            "readout_staleness": version_before
            - int(record["issued_readout_version"]),
            "readout_version": int(self.online_readout_version.item()),
            "parameter_version": int(self.online_parameter_version.item()),
            "ogd_rank": self.readout_gradient_memory.rank,
            "ogd_raw_norm": float(stats["raw_norm"]),
            "ogd_projected_norm": float(stats["projected_norm"]),
            "ogd_retained_norm": float(stats["retained_norm"]),
            "ogd_max_abs_overlap": float(stats["max_abs_overlap"]),
            "ogd_memory_added": memory_added,
            "ogd_projection_applied": projected_flag,
            "unclipped_update_norm": unclipped_norm,
            "update_clip_scale": clip_scale,
            "update_norm": unclipped_norm * clip_scale,
        }

    @property
    def pending_candidate_count(self) -> int:
        return len(self._pending_candidates)

    @property
    def state_bank_size(self) -> int:
        return len(self._stream_states)

    def pending_candidate_ids(self, *, stream_id: Any = None) -> Tuple[int, ...]:
        key = self._stream_key(stream_id)
        return tuple(
            ticket_id
            for ticket_id, record in self._pending_candidates.items()
            if record["stream_id"] == key
        )

    @torch.no_grad()
    def cancel_candidate(self, candidate_id: int) -> None:
        self._candidate_record(candidate_id)
        self._pending_candidates.pop(candidate_id)
        self._record_closed(candidate_id, "cancelled")

    @torch.no_grad()
    def reset_state(
        self,
        *,
        stream_id: Any = None,
        cancel_candidates: bool = True,
    ) -> None:
        key = self._stream_key(stream_id)
        self._stream_states.pop(key, None)
        if cancel_candidates:
            self._cancel_stream_runtime(key, "stream_reset")
        else:
            if any(
                record["stream_id"] == key
                for record in self._pending_candidates.values()
            ):
                raise RuntimeError("cannot retain candidates while resetting their stream")
            self._feature_states.pop(key, None)
            self._pending_promotions.pop(key, None)
            self._last_outputs.pop(key, None)

    @torch.no_grad()
    def reset_all_states(self) -> None:
        for key in list(self._stream_states):
            self._stream_states.pop(key)
            self._cancel_stream_runtime(key, "stream_reset")

    @torch.no_grad()
    def current_state(
        self,
        *,
        stream_id: Any = None,
        clone: bool = True,
    ) -> Optional[TemporalMQRState]:
        key = self._stream_key(stream_id)
        state = self._stream_states.get(key)
        if state is None:
            return None
        return self._clone_state(state) if clone else state
