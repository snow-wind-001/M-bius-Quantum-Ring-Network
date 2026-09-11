from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .online import OrthogonalGradientMemory
from .safety import residual_output_diagnostics
from .unitary import (
    CayleyUnistochasticParam,
    CyclicGivensUnistochasticParam,
    InactiveUnistochasticParam,
)


@dataclass(frozen=True)
class TemporalMQRState:
    """Per-timescale recurrent state for a temporal MQR cell.

    Every tensor has shape ``[batch, ring_dim]``.  Batch lanes are independent
    streams; callers must reset or replace the state when lane identities
    change.
    """

    rings: Tuple[torch.Tensor, ...]

    def detached(self) -> "TemporalMQRState":
        return TemporalMQRState(tuple(value.detach() for value in self.rings))


class MultiTimescaleMQR(nn.Module):
    """One-step, multi-timescale Möbius Quantum Ring dynamics.

    For external time ``t`` and ring ``j`` the state update is

    ``h[j,t] = sigma((1-lambda[j]) h[j,t-1] H[j]^T + kappa[j] J[j](x[t]))``.

    Unlike the equilibrium MQR, this module performs exactly one transition per
    external observation.  Consequently recurrent state has a measurable
    half-life instead of being almost erased by many internal relaxation steps.
    ``H`` is the identity, a unistochastic matrix, or a signed orthogonal
    cyclic-Givens operator. Orthogonal propagation preserves L2 before damping
    and activation; it does not satisfy stochastic or L-infinity constraints.
    ``write_scales=None`` selects unit-strength one-shot writes; pass the leak
    rates explicitly to recover conventional exponential-moving-average writes.
    """

    def __init__(
        self,
        input_dim: int,
        ring_dim: int,
        output_dim: int,
        *,
        leak_rates: Sequence[float] = (0.5, 0.1, 0.02, 0.005),
        write_scales: Optional[Sequence[float]] = None,
        injection_rank: int = 8,
        injection_activation: str = "tanh",
        state_activation: str = "tanh",
        transition_mode: str = "unistochastic",
        learn_transitions: bool = True,
        transition_structure: str = "dense_cayley",
        cyclic_givens_layers: int = 2,
        cayley_coordinate_mode: str = "projected",
        base_unitary_init: str = "identity",
        base_unitary_scale: float = 0.25,
        base_unitary_seed: Optional[int] = None,
        zero_init_policy_with_base: bool = True,
        freeze_full_leak_transitions: bool = True,
        readout_bias: bool = False,
        zero_init_readout: bool = False,
    ):
        super().__init__()
        if input_dim <= 0 or ring_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim, ring_dim, and output_dim must be positive")
        if injection_rank <= 0:
            raise ValueError("injection_rank must be positive")
        rates = tuple(float(value) for value in leak_rates)
        if not rates:
            raise ValueError("leak_rates must be non-empty")
        if any(not math.isfinite(value) or not (0.0 < value <= 1.0) for value in rates):
            raise ValueError("every leak rate must be finite and in (0, 1]")
        writes = (
            tuple(1.0 for _ in rates)
            if write_scales is None
            else tuple(float(value) for value in write_scales)
        )
        if len(writes) != len(rates):
            raise ValueError("write_scales must have the same length as leak_rates")
        if any(not math.isfinite(value) or value < 0.0 for value in writes):
            raise ValueError("every write scale must be finite and non-negative")
        if injection_activation not in ("none", "tanh", "relu", "gelu"):
            raise ValueError('injection_activation must be "none", "tanh", "relu", or "gelu"')
        if state_activation not in ("none", "tanh", "relu"):
            raise ValueError('state_activation must be "none", "tanh", or "relu"')
        if transition_mode not in ("identity", "unistochastic", "orthogonal"):
            raise ValueError(
                'transition_mode must be "identity", "unistochastic", or "orthogonal"'
            )
        if transition_mode == "orthogonal" and transition_structure != "cyclic_givens":
            raise ValueError('orthogonal transitions require transition_structure="cyclic_givens"')
        if transition_structure not in ("dense_cayley", "cyclic_givens"):
            raise ValueError(
                'transition_structure must be "dense_cayley" or "cyclic_givens"'
            )
        if (
            isinstance(cyclic_givens_layers, bool)
            or int(cyclic_givens_layers) != cyclic_givens_layers
            or int(cyclic_givens_layers) <= 0
        ):
            raise ValueError("cyclic_givens_layers must be a positive integer")
        if cayley_coordinate_mode not in ("projected", "minimal"):
            raise ValueError('cayley_coordinate_mode must be "projected" or "minimal"')
        if base_unitary_init not in ("identity", "random"):
            raise ValueError('base_unitary_init must be "identity" or "random"')
        if not math.isfinite(float(base_unitary_scale)) or base_unitary_scale <= 0.0:
            raise ValueError("base_unitary_scale must be finite and positive")
        if base_unitary_seed is not None and (
            isinstance(base_unitary_seed, bool)
            or int(base_unitary_seed) != base_unitary_seed
        ):
            raise ValueError("base_unitary_seed must be an integer or None")
        if transition_mode == "identity" and learn_transitions:
            # There is no transition parameter to learn in the identity control.
            learn_transitions = False

        self.input_dim = int(input_dim)
        self.ring_dim = int(ring_dim)
        self.output_dim = int(output_dim)
        self.num_timescales = len(rates)
        self.injection_rank = int(injection_rank)
        self.injection_activation = str(injection_activation)
        self.state_activation = str(state_activation)
        self.transition_mode = str(transition_mode)
        self.learn_transitions = bool(learn_transitions)
        self.transition_structure = str(transition_structure)
        self.cyclic_givens_layers = int(cyclic_givens_layers)
        self.cayley_coordinate_mode = str(cayley_coordinate_mode)
        self.base_unitary_init = str(base_unitary_init)
        self.base_unitary_scale = float(base_unitary_scale)
        self.base_unitary_seed = (
            None if base_unitary_seed is None else int(base_unitary_seed)
        )
        self.zero_init_policy_with_base = bool(zero_init_policy_with_base)
        self.freeze_full_leak_transitions = bool(freeze_full_leak_transitions)
        self._use_base_unitary = bool(
            self.transition_structure == "dense_cayley"
            and self.base_unitary_init == "random"
        )

        # Preserve user-declared decay constants in float64.  Forward steps cast
        # them to the input dtype, while analytical half-life diagnostics retain
        # the exact Python-float values after ``module.double()``.
        self.register_buffer("leak_rates", torch.tensor(rates, dtype=torch.float64))
        self.register_buffer("write_scales", torch.tensor(writes, dtype=torch.float64))
        self.input_down = nn.Linear(input_dim, injection_rank, bias=False)
        self.input_up = nn.ModuleList(
            [nn.Linear(injection_rank, ring_dim, bias=False) for _ in rates]
        )
        # Initialize shared input/readout blocks before topology-specific
        # parameters.  Reusing a seed therefore gives matched weights in
        # identity-versus-unistochastic ablations.
        self.readout = nn.Linear(
            len(rates) * ring_dim,
            output_dim,
            bias=bool(readout_bias),
        )
        if zero_init_readout:
            nn.init.zeros_(self.readout.weight)
            if self.readout.bias is not None:
                nn.init.zeros_(self.readout.bias)
        if transition_mode in ("unistochastic", "orthogonal"):
            if self.transition_structure == "dense_cayley":
                self.unitary_params = nn.ModuleList(
                    [
                        (
                            InactiveUnistochasticParam(ring_dim)
                            if self.freeze_full_leak_transitions and rate == 1.0
                            else CayleyUnistochasticParam(
                                ring_dim,
                                coordinate_mode=self.cayley_coordinate_mode,
                            )
                        )
                        for rate in rates
                    ]
                )
                base_unitaries = self._initialize_base_unitaries(
                    len(rates),
                    ring_dim,
                    scale=self.base_unitary_scale,
                    seed=self.base_unitary_seed,
                    enabled=self._use_base_unitary,
                )
            else:
                self.unitary_params = nn.ModuleList(
                    [
                        (
                            InactiveUnistochasticParam(ring_dim)
                            if self.freeze_full_leak_transitions and rate == 1.0
                            else CyclicGivensUnistochasticParam(
                                ring_dim,
                                layers=self.cyclic_givens_layers,
                                base_init=self.base_unitary_init,
                                base_scale=self.base_unitary_scale,
                                base_seed=(
                                    None
                                    if self.base_unitary_seed is None
                                    else self.base_unitary_seed + index
                                ),
                            )
                        )
                        for index, rate in enumerate(rates)
                    ]
                )
                base_unitaries = torch.empty(
                    0, ring_dim, ring_dim, dtype=torch.cdouble
                )
            # Register before constructing a frozen transition cache because
            # the cache is computed through ``_unitary_total`` below.
            self.register_buffer(
                "_base_unitaries",
                base_unitaries,
                persistent=self._use_base_unitary,
            )
            # A non-monomial fixed base supplies a generic first-order point for
            # A -> |Cayley(A) U_0|^2.  Starting the policy at A=0 makes the
            # distinction from the legacy near-identity initialization exact.
            if (
                self.transition_structure == "dense_cayley"
                and self._use_base_unitary
                and self.zero_init_policy_with_base
            ):
                for parameter in self.unitary_params:
                    if isinstance(parameter, CayleyUnistochasticParam):
                        parameter.reset_cayley_identity_()

            transition_active = torch.tensor(
                [
                    bool(self.learn_transitions)
                    and not (self.freeze_full_leak_transitions and rate == 1.0)
                    for rate in rates
                ],
                dtype=torch.bool,
            )
            for active, parameter in zip(transition_active.tolist(), self.unitary_params):
                if not active:
                    for coordinate in parameter.parameters():
                        coordinate.requires_grad_(False)
            if not self.learn_transitions:
                with torch.no_grad():
                    fixed_transitions = torch.stack(
                        [
                            self._transition_from_parameter(index)
                            for index in range(len(rates))
                        ]
                    )
                    fixed_unitary_error = max(
                        self._unitary_error(index) for index in range(len(rates))
                    )
            else:
                fixed_transitions = torch.empty(0, ring_dim, ring_dim)
                fixed_unitary_error = 0.0
        else:
            self.unitary_params = nn.ModuleList()
            base_unitaries = torch.empty(0, ring_dim, ring_dim, dtype=torch.cdouble)
            self.register_buffer(
                "_base_unitaries",
                base_unitaries,
                persistent=False,
            )
            transition_active = torch.zeros(len(rates), dtype=torch.bool)
            fixed_transitions = torch.empty(0, ring_dim, ring_dim)
            fixed_unitary_error = 0.0
        self.register_buffer(
            "_transition_active",
            transition_active,
            persistent=False,
        )
        # A frozen Cayley transition is a reservoir constant.  Cache it once so
        # inference does not repeat a complex matrix solve at every token.
        self.register_buffer("_fixed_transitions", fixed_transitions, persistent=True)
        self.register_buffer(
            "_fixed_unitary_error",
            torch.tensor(fixed_unitary_error, dtype=torch.float64),
            persistent=True,
        )
        # Inference commits run under no_grad and may reuse H until an in-place
        # coordinate update changes PyTorch's tensor version counter.  Trace
        # recomputation runs with gradients enabled and always rebuilds H.
        self._inference_transition_cache: Dict[
            int, Tuple[Tuple[int, ...], torch.Tensor]
        ] = {}

    @staticmethod
    def _initialize_base_unitaries(
        count: int,
        dim: int,
        *,
        scale: float,
        seed: Optional[int],
        enabled: bool,
    ) -> torch.Tensor:
        """Return frozen Cayley bases without consuming RNG when disabled."""

        identity = torch.eye(dim, dtype=torch.cdouble)
        if not enabled:
            return identity.unsqueeze(0).expand(count, -1, -1).clone()
        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
        values = []
        for _index in range(count):
            real = torch.randn(dim, dim, generator=generator, dtype=torch.double) * float(scale)
            imag = torch.randn(dim, dim, generator=generator, dtype=torch.double) * float(scale)
            A = 0.5 * torch.complex(real - real.T, imag + imag.T)
            values.append(torch.linalg.solve(identity + A, identity - A))
        return torch.stack(values)

    def _base_unitary(
        self,
        index: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._base_unitaries[int(index)].to(device=device, dtype=dtype)

    def _unitary_total(self, index: int) -> torch.Tensor:
        parameter = self.unitary_params[int(index)]
        policy = parameter.unitary()
        if isinstance(parameter, InactiveUnistochasticParam):
            return policy
        if not self._use_base_unitary:
            return policy
        return policy @ self._base_unitary(
            index,
            device=policy.device,
            dtype=policy.dtype,
        )

    def _transition_from_parameter(self, index: int) -> torch.Tensor:
        if self.transition_mode == "orthogonal":
            return self._unitary_total(index).real
        return self._unitary_total(index).abs().square()

    @torch.no_grad()
    def _unitary_error(self, index: int) -> float:
        unitary = self._unitary_total(index)
        identity = torch.eye(
            self.ring_dim,
            device=unitary.device,
            dtype=unitary.dtype,
        )
        error = torch.linalg.matrix_norm(
            unitary.conj().transpose(-2, -1) @ unitary - identity,
            ord="fro",
        )
        return float(error.item())

    @property
    def total_state_dim(self) -> int:
        return self.num_timescales * self.ring_dim

    def memory_half_lives(self) -> torch.Tensor:
        """Return amplitude half-lives in external observations.

        A leak of one has zero recurrent carry and therefore a zero half-life.
        """

        rates = self.leak_rates.detach().double()
        result = torch.zeros_like(rates)
        carried = rates < 1.0
        result[carried] = math.log(0.5) / torch.log1p(-rates[carried])
        return result

    def _inject(self, x: torch.Tensor) -> torch.Tensor:
        value = self.input_down(x)
        if self.injection_activation == "none":
            return value
        if self.injection_activation == "tanh":
            return torch.tanh(value)
        if self.injection_activation == "relu":
            return F.relu(value)
        if self.injection_activation == "gelu":
            return F.gelu(value)
        raise RuntimeError(f"unknown injection activation: {self.injection_activation}")

    def _activate_state(self, value: torch.Tensor) -> torch.Tensor:
        if self.state_activation == "none":
            return value
        if self.state_activation == "tanh":
            return torch.tanh(value)
        if self.state_activation == "relu":
            return F.relu(value)
        raise RuntimeError(f"unknown state activation: {self.state_activation}")

    def transition_matrix(
        self,
        index: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return the real contraction-preserving transition for one ring."""

        if not (0 <= int(index) < self.num_timescales):
            raise IndexError("timescale index out of range")
        reference = self.input_down.weight
        target_device = reference.device if device is None else device
        target_dtype = reference.dtype if dtype is None else dtype
        if self.transition_mode == "identity":
            return torch.eye(self.ring_dim, device=target_device, dtype=target_dtype)
        if not self.learn_transitions:
            return self._fixed_transitions[int(index)].to(
                device=target_device,
                dtype=target_dtype,
            )
        if not torch.is_grad_enabled():
            parameter = self.unitary_params[int(index)]
            versions = tuple(
                int(value._version) for value in parameter.parameters()
            )
            cached = self._inference_transition_cache.get(int(index))
            if cached is None or cached[0] != versions:
                value = self._transition_from_parameter(int(index)).detach()
                self._inference_transition_cache[int(index)] = (versions, value)
            else:
                value = cached[1]
        else:
            value = self._transition_from_parameter(int(index))
        return value.to(
            device=target_device,
            dtype=target_dtype,
        )

    @property
    def active_transition_indices(self) -> Tuple[int, ...]:
        """Indices whose Cayley coordinates can affect and update recurrence."""

        return tuple(
            index
            for index, active in enumerate(self._transition_active.tolist())
            if bool(active)
        )

    @torch.no_grad()
    def transition_diagnostics(
        self,
        *,
        include_jacobian: bool = False,
    ) -> List[Dict[str, Any]]:
        """Audit topology activity, base point, and local learnability per ring."""

        if self.transition_mode == "identity":
            return [
                {
                    "index": index,
                    "active": False,
                    "carry": 1.0 - float(self.leak_rates[index].item()),
                    "transition_mode": "identity",
                    "transition_distance_from_identity_fro": 0.0,
                    "modulus_square_jacobian_rank": 0,
                }
                for index in range(self.num_timescales)
            ]
        result: List[Dict[str, Any]] = []
        for index, parameter in enumerate(self.unitary_params):
            if isinstance(parameter, CayleyUnistochasticParam):
                right = (
                    self._base_unitary(
                        index,
                        device=parameter.A_real.device,
                        dtype=parameter.unitary().dtype,
                    )
                    if self._use_base_unitary
                    else None
                )
                item = parameter.coordinate_diagnostics(
                    include_jacobian=include_jacobian,
                    right_unitary=right,
                )
            else:
                item = parameter.coordinate_diagnostics(
                    include_jacobian=include_jacobian
                )
            item.update(
                {
                    "index": index,
                    "active": bool(self._transition_active[index].item()),
                    "carry": 1.0 - float(self.leak_rates[index].item()),
                    "base_unitary_init": self.base_unitary_init,
                }
            )
            if self.transition_mode == "orthogonal":
                # The parameter also exposes |U|² diagnostics, but that map is
                # not the operator used by this recurrent mode.
                item = {key: value for key, value in item.items()
                        if not key.startswith("modulus_square_")}
                item.update({
                    "transition_mode": "orthogonal",
                    "stochastic_constraints_applicable": False,
                    "contraction_norm": "l2",
                    "transition_distance_from_identity_fro":
                        item["unitary_distance_from_identity_fro"],
                })
            result.append(item)
        return result

    @torch.no_grad()
    def transition_drift_diagnostics(
        self,
        index: int,
        reference_A: torch.Tensor,
    ) -> Dict[str, float | bool]:
        """Audit the actual propagation operator, including signed rotations."""

        if self.transition_mode == "identity":
            return {
                "a_fro_drift": 0.0,
                "unitary_fro_drift": 0.0,
                "transition_fro_drift": 0.0,
                "unitary_fro_bound": 0.0,
                "transition_fro_bound": 0.0,
                "unitary_bound_violation": 0.0,
                "transition_bound_violation": 0.0,
                "bounds_certified": True,
            }
        if not (0 <= int(index) < self.num_timescales):
            raise IndexError("timescale index out of range")
        parameter = self.unitary_params[int(index)]
        if isinstance(parameter, CayleyUnistochasticParam):
            right = (
                self._base_unitary(
                    int(index),
                    device=parameter.A_real.device,
                    dtype=parameter.unitary().dtype,
                )
                if self._use_base_unitary
                else None
            )
            return parameter.drift_diagnostics(reference_A, right_unitary=right)
        result = parameter.drift_diagnostics(reference_A)
        if self.transition_mode == "orthogonal":
            result["transition_fro_drift"] = result["unitary_fro_drift"]
            result["transition_fro_bound"] = result["unitary_fro_bound"]
            result["transition_bound_violation"] = result["unitary_bound_violation"]
            result["bounds_certified"] = result["unitary_bound_violation"] == 0.0
        return result

    @torch.no_grad()
    def transition_references(self) -> List[torch.Tensor]:
        """Return exact parameter snapshots for atomic transition drift audits."""

        references: List[torch.Tensor] = []
        for parameter in self.unitary_params:
            if isinstance(parameter, CayleyUnistochasticParam):
                references.append(parameter.skew_hermitian_A().detach().clone())
            elif isinstance(parameter, InactiveUnistochasticParam):
                references.append(torch.empty(0))
            else:
                references.append(parameter.angles.detach().clone())
        return references

    def zero_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TemporalMQRState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return TemporalMQRState(
            tuple(
                torch.zeros(batch_size, self.ring_dim, device=device, dtype=dtype)
                for _ in range(self.num_timescales)
            )
        )

    def _validate_state(self, state: TemporalMQRState, batch_size: int) -> None:
        if len(state.rings) != self.num_timescales:
            raise ValueError(
                f"state must contain {self.num_timescales} rings, got {len(state.rings)}"
            )
        expected = (batch_size, self.ring_dim)
        for index, value in enumerate(state.rings):
            if value.shape != expected:
                raise ValueError(
                    f"state.rings[{index}] must have shape {expected}, got {tuple(value.shape)}"
                )
            if torch.is_complex(value) or not value.is_floating_point():
                raise TypeError(f"state.rings[{index}] must be real floating point")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"state.rings[{index}] contains NaN or Inf")

    def _prepare_write_gate(
        self,
        write_gate: Optional[torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Validate an external write gate and return shape ``[batch, scales]``.

        A rank-one gate applies the same decision to every timescale.  A
        rank-two gate can independently protect fast and slow rings.  Gates are
        controls supplied to the core.  The online wrapper may produce them
        with a separate auxiliary-supervised controller, but the core never
        propagates its task gradient into that controller.
        """

        if write_gate is None:
            return torch.ones(
                x.size(0),
                self.num_timescales,
                device=x.device,
                dtype=x.dtype,
            )
        if not isinstance(write_gate, torch.Tensor):
            raise TypeError("write_gate must be a tensor or None")
        if torch.is_complex(write_gate) or not write_gate.is_floating_point():
            raise TypeError("write_gate must be a real floating-point tensor")
        if write_gate.dim() == 1 and write_gate.shape == (x.size(0),):
            prepared = write_gate.unsqueeze(1).expand(-1, self.num_timescales)
        elif write_gate.dim() == 2 and write_gate.shape == (
            x.size(0),
            self.num_timescales,
        ):
            prepared = write_gate
        else:
            raise ValueError(
                "write_gate must have shape "
                f"[{x.size(0)}] or [{x.size(0)}, {self.num_timescales}], "
                f"got {tuple(write_gate.shape)}"
            )
        prepared = prepared.detach().to(device=x.device, dtype=x.dtype)
        if not bool(torch.isfinite(prepared).all()):
            raise ValueError("write_gate must contain only finite values")
        if not bool(((prepared >= 0.0) & (prepared <= 1.0)).all()):
            raise ValueError("write_gate values must lie in [0, 1]")
        return prepared

    def _prepare_promotion(
        self,
        promotion: Optional[torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Validate an external state forcing term.

        ``promotion`` is expressed in ring-state coordinates and has shape
        ``[batch, num_timescales, ring_dim]``.  It is deliberately detached:
        utility controllers may decide when to promote a cached candidate, but
        the recurrent core treats that decision as an external causal action.
        """

        expected = (x.size(0), self.num_timescales, self.ring_dim)
        if promotion is None:
            return torch.zeros(expected, device=x.device, dtype=x.dtype)
        if not isinstance(promotion, torch.Tensor):
            raise TypeError("promotion must be a tensor or None")
        if promotion.shape != expected:
            raise ValueError(
                f"promotion must have shape {expected}, got {tuple(promotion.shape)}"
            )
        if torch.is_complex(promotion) or not promotion.is_floating_point():
            raise TypeError("promotion must be a real floating-point tensor")
        prepared = promotion.detach().to(device=x.device, dtype=x.dtype)
        if not bool(torch.isfinite(prepared).all()):
            raise ValueError("promotion must contain only finite values")
        return prepared

    def input_forcing(self, x: torch.Tensor) -> torch.Tensor:
        """Return one-shot input forcing with shape ``[batch, scales, ring]``.

        The returned tensor already includes ``write_scales``.  It can be
        cached as a candidate payload and supplied later through ``promotion``
        without retaining an autograd graph.
        """

        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"x must be [batch, {self.input_dim}], got {tuple(x.shape)}")
        if not x.is_floating_point() or torch.is_complex(x):
            raise TypeError("x must be a real floating-point tensor")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("x must contain only finite values")
        encoded = self._inject(x)
        values = []
        for index, projection in enumerate(self.input_up):
            write = self.write_scales[index].to(device=x.device, dtype=x.dtype)
            values.append(write * projection(encoded))
        return torch.stack(values, dim=1)

    def readout_state(self, state: TemporalMQRState) -> torch.Tensor:
        """Read logits from an already advanced recurrent state."""

        if not isinstance(state, TemporalMQRState):
            raise TypeError("state must be a TemporalMQRState")
        if not state.rings:
            raise ValueError("state must contain at least one ring")
        batch_size = state.rings[0].size(0)
        self._validate_state(state, batch_size)
        reference = self.readout.weight
        vector = torch.cat(
            [
                value.to(device=reference.device, dtype=reference.dtype)
                for value in state.rings
            ],
            dim=1,
        )
        return self.readout(vector)

    def orthogonal_offsets(self, index: int, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Optional exogenous [batch, layers, pairs] rotation angles."""
        return None

    def forward_step(
        self,
        x: torch.Tensor,
        *,
        state: Optional[TemporalMQRState] = None,
        write_gate: Optional[torch.Tensor] = None,
        promotion: Optional[torch.Tensor] = None,
        return_certificate: bool = False,
    ) -> Any:
        """Advance independent batch streams by one external observation.

        ``write_gate`` has shape ``[batch]`` or ``[batch, num_timescales]``.
        ``promotion`` has shape ``[batch, num_timescales, ring_dim]``.  Both are
        external forcing terms and therefore do not alter the recurrent
        contraction factor.  ``return_certificate=True`` appends a detached
        signed-state bound certificate without changing the state dynamics.
        """

        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"x must be [batch, {self.input_dim}], got {tuple(x.shape)}")
        if not x.is_floating_point() or torch.is_complex(x):
            raise TypeError("x must be a real floating-point tensor")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("x must contain only finite values")
        if state is None:
            state = self.zero_state(x.size(0), device=x.device, dtype=x.dtype)
        else:
            self._validate_state(state, x.size(0))

        gates = self._prepare_write_gate(write_gate, x)
        promotions = self._prepare_promotion(promotion, x)
        forcing = self.input_forcing(x)
        next_rings: List[torch.Tensor] = []
        previous_linf: List[torch.Tensor] = []
        external_linf: List[torch.Tensor] = []
        transition_gains: List[torch.Tensor] = []
        analytic_bounds: List[torch.Tensor] = []
        observed_linf: List[torch.Tensor] = []
        for index, previous in enumerate(state.rings):
            previous = previous.to(device=x.device, dtype=x.dtype)
            leak = self.leak_rates[index].to(device=x.device, dtype=x.dtype)
            full_leak = float(self.leak_rates[index].item()) == 1.0
            if full_leak:
                # The recurrent transition is multiplied by exactly zero.  Do
                # not solve Cayley or allocate a dense matmul to a dead path.
                transition = None
                value = torch.zeros_like(previous)
            elif self.transition_mode == "identity":
                transition = None
                value = (1.0 - leak) * previous
            elif self.transition_mode == "orthogonal":
                parameter = self.unitary_params[index]
                offsets = self.orthogonal_offsets(index, x)
                value = (1.0 - leak) * parameter.apply_orthogonal(previous, angle_offsets=offsets)
                # Only the optional L-infinity certificate needs a dense
                # operator; ordinary signed propagation stays linear-cost.
                transition = (
                    self.transition_matrix(index, device=x.device, dtype=x.dtype)
                    if return_certificate else None
                )
                if return_certificate and offsets is not None:
                    identity = torch.eye(self.ring_dim, device=x.device, dtype=x.dtype)
                    transition = parameter.apply_orthogonal(
                        identity.expand(x.size(0), -1, -1),
                        angle_offsets=offsets[:, None],
                    ).transpose(-2, -1)
            else:
                transition = self.transition_matrix(
                    index, device=x.device, dtype=x.dtype
                )
                value = (1.0 - leak) * (previous @ transition.transpose(0, 1))
            external = (
                gates[:, index : index + 1] * forcing[:, index, :]
                + promotions[:, index, :]
            )
            value = value + external
            next_value = self._activate_state(value)
            next_rings.append(next_value)
            if return_certificate:
                previous_norm = previous.abs().amax(dim=1)
                external_norm = external.abs().amax(dim=1)
                # For row states transformed as h @ H^T, the relevant
                # l_infinity operator norm is max_i sum_j |H_ij|.
                gain = (
                    x.new_tensor(1.0)
                    if transition is None
                    else transition.abs().sum(dim=-1).amax(dim=-1)
                )
                bound = (1.0 - leak) * gain * previous_norm + external_norm
                previous_linf.append(previous_norm)
                external_linf.append(external_norm)
                transition_gains.append(gain.expand_as(previous_norm))
                analytic_bounds.append(bound)
                observed_linf.append(next_value.abs().amax(dim=1))

        next_state = TemporalMQRState(tuple(next_rings))
        state_vector = torch.cat(next_rings, dim=1)
        logits = self.readout(state_vector)
        if not return_certificate:
            return logits, next_state

        previous_tensor = torch.stack(previous_linf, dim=1).detach()
        external_tensor = torch.stack(external_linf, dim=1).detach()
        gain_tensor = torch.stack(transition_gains, dim=1).detach()
        bound_tensor = torch.stack(analytic_bounds, dim=1).detach()
        observed_tensor = torch.stack(observed_linf, dim=1).detach()
        raw_violation = observed_tensor - bound_tensor
        violation = raw_violation.clamp_min(0.0)
        scale = 1.0 + bound_tensor.abs().amax()
        numerical_tolerance = (
            64.0 * self.ring_dim * torch.finfo(x.dtype).eps * scale
        )
        certificate: Dict[str, Any] = {
            "previous_state_linf": previous_tensor,
            "external_forcing_linf": external_tensor,
            "transition_linf_gain": gain_tensor,
            "analytic_next_state_linf_bound": bound_tensor,
            "observed_next_state_linf": observed_tensor,
            "bound_violation": violation,
            "max_bound_violation": float(violation.amax().item()),
            "numerical_tolerance": float(numerical_tolerance.item()),
            "certified": bool(raw_violation.amax() <= numerical_tolerance),
            "requires_nonnegative_state": False,
        }
        return logits, next_state, certificate

    def forward(
        self,
        x: torch.Tensor,
        *,
        write_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Stateless prediction for independent examples."""

        logits, _ = self.forward_step(x, write_gate=write_gate)
        return logits

    @torch.no_grad()
    def max_unitary_error(self) -> float:
        if self.transition_mode == "identity":
            return 0.0
        if not self.learn_transitions:
            return float(self._fixed_unitary_error.item())
        return max(
            (self._unitary_error(index) for index in range(self.num_timescales)),
            default=0.0,
        )

    @torch.no_grad()
    def max_stochastic_error(self) -> float:
        if self.transition_mode == "identity":
            return 0.0
        if not self.learn_transitions:
            transitions = self._fixed_transitions
            row_error = (transitions.sum(dim=2) - 1.0).abs().max()
            column_error = (transitions.sum(dim=1) - 1.0).abs().max()
            return max(float(row_error.item()), float(column_error.item()))
        errors: List[float] = []
        for index in range(self.num_timescales):
            transition = self._transition_from_parameter(index)
            row_error = (transition.sum(dim=1) - 1.0).abs().max()
            column_error = (transition.sum(dim=0) - 1.0).abs().max()
            errors.extend((float(row_error.item()), float(column_error.item())))
        return max(errors, default=0.0)


class TemporalMQRSidecar(nn.Module):
    """Zero-disturbance residual adapter for frozen hidden representations.

    The wrapped temporal ring maps a hidden vector to a same-shaped residual.
    Its readout is initialized to exact zero, so ``adapted == hidden`` at
    construction while the readout retains a gradient path.  The recurrent
    state may be signed; diagnostics use the certified infinity-norm bound and
    never invoke the non-negative ``l1`` mass interpretation.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        ring_dim: int = 32,
        residual_clip_l2: Optional[float] = None,
        core_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or ring_dim <= 0:
            raise ValueError("hidden_dim and ring_dim must be positive")
        if residual_clip_l2 is not None and (
            not math.isfinite(float(residual_clip_l2))
            or float(residual_clip_l2) <= 0.0
        ):
            raise ValueError("residual_clip_l2 must be finite and positive or None")
        kwargs = dict(core_kwargs or {})
        reserved = {
            "input_dim",
            "ring_dim",
            "output_dim",
            "zero_init_readout",
        }.intersection(kwargs)
        if reserved:
            raise ValueError(f"core_kwargs must not redefine {sorted(reserved)}")
        self.hidden_dim = int(hidden_dim)
        self.residual_clip_l2 = (
            None if residual_clip_l2 is None else float(residual_clip_l2)
        )
        self.core = MultiTimescaleMQR(
            self.hidden_dim,
            int(ring_dim),
            self.hidden_dim,
            zero_init_readout=True,
            **kwargs,
        )

    def forward_step(
        self,
        hidden: torch.Tensor,
        *,
        state: Optional[TemporalMQRState] = None,
        write_gate: Optional[torch.Tensor] = None,
        promotion: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Any:
        """Return adapted hidden state, next ring state, and optional audit."""

        result = self.core.forward_step(
            hidden,
            state=state,
            write_gate=write_gate,
            promotion=promotion,
            return_certificate=return_diagnostics,
        )
        if return_diagnostics:
            raw_residual, next_state, state_certificate = result
        else:
            raw_residual, next_state = result
            state_certificate = None

        residual = raw_residual
        clip_scale = torch.ones(
            hidden.size(0), 1, device=hidden.device, dtype=hidden.dtype
        )
        if self.residual_clip_l2 is not None:
            flat = raw_residual.reshape(raw_residual.size(0), -1)
            norm = torch.linalg.vector_norm(flat, dim=1, keepdim=True)
            clip_scale = (
                float(self.residual_clip_l2)
                / norm.clamp_min(torch.finfo(raw_residual.dtype).eps)
            ).clamp(max=1.0)
            residual = raw_residual * clip_scale
        adapted = hidden + residual
        if not return_diagnostics:
            return adapted, next_state
        diagnostics: Dict[str, Any] = {
            **residual_output_diagnostics(hidden.detach(), residual.detach()),
            "raw_residual_l2_max": float(
                torch.linalg.vector_norm(
                    raw_residual.detach().reshape(raw_residual.size(0), -1), dim=1
                ).max().item()
            ),
            "residual_clip_scale_min": float(clip_scale.detach().min().item()),
            "state_certificate": state_certificate,
        }
        return adapted, next_state, diagnostics

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        adapted, _state = self.forward_step(hidden)
        return adapted


class TemporalWriteGate(nn.Module):
    """Low-rank continuous write controller for Temporal MQR.

    The controller maps a current observation or separate event feature vector
    to one gate per timescale.  It has no recurrent state and receives no task
    gradient through :class:`MultiTimescaleMQR`; online supervision is handled
    causally by :class:`OnlineTemporalMQRClassifier` after the current gate has
    already been used.
    """

    def __init__(
        self,
        input_dim: int,
        num_timescales: int,
        *,
        rank: int = 8,
        activation: str = "tanh",
        temperature: float = 1.0,
        minimum_gate: float = 0.0,
        initial_open_probability: float = 0.5,
    ):
        super().__init__()
        if input_dim <= 0 or num_timescales <= 0 or rank <= 0:
            raise ValueError("gate dimensions and rank must be positive")
        if activation not in ("none", "tanh", "relu", "gelu"):
            raise ValueError(
                'gate activation must be "none", "tanh", "relu", or "gelu"'
            )
        if not math.isfinite(float(temperature)) or temperature <= 0.0:
            raise ValueError("gate temperature must be finite and positive")
        if not math.isfinite(float(minimum_gate)) or not (
            0.0 <= minimum_gate < 1.0
        ):
            raise ValueError("minimum_gate must be finite and in [0, 1)")
        if not math.isfinite(float(initial_open_probability)) or not (
            minimum_gate < initial_open_probability < 1.0
        ):
            raise ValueError(
                "initial_open_probability must lie strictly between "
                "minimum_gate and one"
            )

        self.input_dim = int(input_dim)
        self.num_timescales = int(num_timescales)
        self.rank = int(rank)
        self.activation = str(activation)
        self.temperature = float(temperature)
        self.minimum_gate = float(minimum_gate)
        self.initial_open_probability = float(initial_open_probability)
        self.input_down = nn.Linear(self.input_dim, self.rank, bias=False)
        self.output = nn.Linear(self.rank, self.num_timescales, bias=True)

        # Zero output weights make the declared initial probability exact and
        # independent of x.  The first label trains the output layer; subsequent
        # labels then propagate into the low-rank input projection.
        base_probability = (
            (self.initial_open_probability - self.minimum_gate)
            / (1.0 - self.minimum_gate)
        )
        initial_logit = self.temperature * math.log(
            base_probability / (1.0 - base_probability)
        )
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.fill_(initial_logit)

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation == "none":
            return value
        if self.activation == "tanh":
            return torch.tanh(value)
        if self.activation == "relu":
            return F.relu(value)
        if self.activation == "gelu":
            return F.gelu(value)
        raise RuntimeError(f"unknown gate activation: {self.activation}")

    def raw_logits(self, gate_input: torch.Tensor) -> torch.Tensor:
        """Return unscaled logits with shape ``[batch, num_timescales]``."""

        if not isinstance(gate_input, torch.Tensor):
            raise TypeError("gate_input must be a tensor")
        if gate_input.dim() != 2 or gate_input.size(1) != self.input_dim:
            raise ValueError(
                f"gate_input must be [batch, {self.input_dim}], got "
                f"{tuple(gate_input.shape)}"
            )
        if torch.is_complex(gate_input) or not gate_input.is_floating_point():
            raise TypeError("gate_input must be a real floating-point tensor")
        if not bool(torch.isfinite(gate_input).all()):
            raise ValueError("gate_input must contain only finite values")
        reference = self.input_down.weight
        prepared = gate_input.to(device=reference.device, dtype=reference.dtype)
        return self.output(self._activate(self.input_down(prepared)))

    def base_probability(self, raw_logits: torch.Tensor) -> torch.Tensor:
        """Return the unconstrained Bernoulli probability used for BCE."""

        self._validate_raw_logits(raw_logits)
        return torch.sigmoid(raw_logits / self.temperature)

    def gate_from_logits(self, raw_logits: torch.Tensor) -> torch.Tensor:
        probability = self.base_probability(raw_logits)
        return self.minimum_gate + (1.0 - self.minimum_gate) * probability

    def forward(self, gate_input: torch.Tensor) -> torch.Tensor:
        return self.gate_from_logits(self.raw_logits(gate_input))

    def _validate_raw_logits(self, raw_logits: torch.Tensor) -> None:
        if not isinstance(raw_logits, torch.Tensor):
            raise TypeError("raw_logits must be a tensor")
        if raw_logits.dim() != 2 or raw_logits.size(1) != self.num_timescales:
            raise ValueError(
                "raw_logits must have shape [batch, "
                f"{self.num_timescales}], got {tuple(raw_logits.shape)}"
            )
        if torch.is_complex(raw_logits) or not raw_logits.is_floating_point():
            raise TypeError("raw_logits must be a real floating-point tensor")
        if not bool(torch.isfinite(raw_logits).all()):
            raise ValueError("raw_logits must contain only finite values")


class OnlineTemporalMQRClassifier(nn.Module):
    """Causal online wrapper for :class:`MultiTimescaleMQR`.

    ``online_step`` computes and returns the prediction under ``theta_t`` and
    only then commits an optional update to ``theta_(t+1)``.  Gradients are
    truncated at the previous recurrent state, so persistent activation memory
    and persistent parameter memory remain separately measurable.  Explicit
    ``stream_id`` values address an LRU-ordered state bank, while
    :meth:`infer_step` and :meth:`apply_feedback` provide bounded, single-use
    delayed-feedback transactions for readout-only adaptation.
    An optional low-rank event controller selects writes under old parameters;
    its auxiliary label is applied only after the current prediction and uses
    an independent OGD/clipping budget.
    """

    def __init__(
        self,
        input_dim: int,
        ring_dim: int,
        output_dim: int,
        *,
        core_kwargs: Optional[Dict[str, Any]] = None,
        lr: float = 1e-2,
        transition_lr_ratio: float = 0.1,
        injection_lr_ratio: float = 1.0,
        readout_lr_ratio: float = 1.0,
        ogd_max_rank: int = 0,
        gate_input_dim: Optional[int] = None,
        gate_kwargs: Optional[Dict[str, Any]] = None,
        gate_lr: Optional[float] = None,
        gate_ogd_max_rank: int = 0,
        gate_max_update_norm: Optional[float] = None,
        gate_positive_weight: float = 1.0,
        write_gate_controller: Optional[TemporalWriteGate] = None,
        carry_state: bool = True,
        max_update_norm: Optional[float] = None,
        max_streams: int = 64,
        stream_overflow_policy: str = "error",
        max_pending_feedback: int = 128,
        feedback_overflow_policy: str = "error",
        feedback_ttl_observations: Optional[int] = None,
    ):
        super().__init__()
        if not math.isfinite(float(lr)) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        ratios = (
            float(transition_lr_ratio),
            float(injection_lr_ratio),
            float(readout_lr_ratio),
        )
        if any(not math.isfinite(value) or value < 0.0 for value in ratios):
            raise ValueError("learning-rate ratios must be finite and non-negative")
        if max_update_norm is not None and (
            not math.isfinite(float(max_update_norm)) or max_update_norm <= 0
        ):
            raise ValueError("max_update_norm must be finite and positive or None")
        resolved_gate_lr = float(lr if gate_lr is None else gate_lr)
        if not math.isfinite(resolved_gate_lr) or resolved_gate_lr <= 0.0:
            raise ValueError("gate_lr must be finite and positive")
        if gate_ogd_max_rank < 0:
            raise ValueError("gate_ogd_max_rank must be non-negative")
        if gate_max_update_norm is not None and (
            not math.isfinite(float(gate_max_update_norm))
            or gate_max_update_norm <= 0
        ):
            raise ValueError(
                "gate_max_update_norm must be finite and positive or None"
            )
        if (
            not math.isfinite(float(gate_positive_weight))
            or gate_positive_weight <= 0.0
        ):
            raise ValueError("gate_positive_weight must be finite and positive")
        if max_streams <= 0:
            raise ValueError("max_streams must be positive")
        if stream_overflow_policy not in ("error", "lru"):
            raise ValueError('stream_overflow_policy must be "error" or "lru"')
        if max_pending_feedback <= 0:
            raise ValueError("max_pending_feedback must be positive")
        if feedback_overflow_policy not in ("error", "oldest"):
            raise ValueError(
                'feedback_overflow_policy must be "error" or "oldest"'
            )
        if feedback_ttl_observations is not None and feedback_ttl_observations <= 0:
            raise ValueError("feedback_ttl_observations must be positive or None")

        self.core = MultiTimescaleMQR(
            input_dim,
            ring_dim,
            output_dim,
            **dict(core_kwargs or {}),
        )
        if write_gate_controller is not None:
            if not isinstance(write_gate_controller, TemporalWriteGate):
                raise TypeError(
                    "write_gate_controller must be a TemporalWriteGate or None"
                )
            if gate_kwargs is not None:
                raise ValueError(
                    "gate_kwargs cannot be combined with write_gate_controller"
                )
            if (
                gate_input_dim is not None
                and int(gate_input_dim) != write_gate_controller.input_dim
            ):
                raise ValueError(
                    "gate_input_dim does not match write_gate_controller.input_dim"
                )
            controller: Optional[TemporalWriteGate] = write_gate_controller
        elif gate_input_dim is not None or gate_kwargs is not None:
            resolved_gate_input_dim = (
                input_dim if gate_input_dim is None else int(gate_input_dim)
            )
            if resolved_gate_input_dim <= 0:
                raise ValueError("gate_input_dim must be positive")
            controller_kwargs = dict(gate_kwargs or {})
            reserved_gate_arguments = {
                "input_dim",
                "num_timescales",
            }.intersection(controller_kwargs)
            if reserved_gate_arguments:
                raise ValueError(
                    "gate_kwargs must not redefine "
                    f"{sorted(reserved_gate_arguments)}"
                )
            controller = TemporalWriteGate(
                resolved_gate_input_dim,
                self.core.num_timescales,
                **controller_kwargs,
            )
        else:
            controller = None
        if (
            controller is not None
            and controller.num_timescales != self.core.num_timescales
        ):
            raise ValueError(
                "write_gate_controller.num_timescales must match the temporal core"
            )
        self.write_gate_controller = controller
        self.gate_input_dim = None if controller is None else controller.input_dim
        self.lr = float(lr)
        self.transition_lr_ratio = float(transition_lr_ratio)
        self.injection_lr_ratio = float(injection_lr_ratio)
        self.readout_lr_ratio = float(readout_lr_ratio)
        self.carry_state = bool(carry_state)
        self.max_update_norm = None if max_update_norm is None else float(max_update_norm)
        self.gradient_memory = OrthogonalGradientMemory(ogd_max_rank)
        # Delayed feedback touches only the readout.  Its OGD layout is kept
        # separate from the complete-parameter synchronous layout.
        self.feedback_gradient_memory = OrthogonalGradientMemory(ogd_max_rank)
        # Event-gate gradients live in a separate parameter space and therefore
        # require a separate OGD layout and clipping budget.
        self.gate_lr = resolved_gate_lr
        self.gate_max_update_norm = (
            None
            if gate_max_update_norm is None
            else float(gate_max_update_norm)
        )
        self.gate_positive_weight = float(gate_positive_weight)
        self.gate_gradient_memory = OrthogonalGradientMemory(gate_ogd_max_rank)
        self.max_streams = int(max_streams)
        self.stream_overflow_policy = str(stream_overflow_policy)
        self.max_pending_feedback = int(max_pending_feedback)
        self.feedback_overflow_policy = str(feedback_overflow_policy)
        self.feedback_ttl_observations = (
            None
            if feedback_ttl_observations is None
            else int(feedback_ttl_observations)
        )
        self._stream_states: OrderedDict[
            Optional[Hashable], TemporalMQRState
        ] = OrderedDict()
        self._stream_observations: Dict[Optional[Hashable], int] = {}
        self._pending_feedback: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self._closed_tickets: OrderedDict[int, str] = OrderedDict()
        self._ticket_history_limit = max(16, 2 * self.max_pending_feedback)
        self._next_ticket_id = 1
        self.register_buffer("online_observations", torch.zeros((), dtype=torch.long))
        self.register_buffer("online_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "online_parameter_version", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer("online_gate_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("online_gate_labels", torch.zeros((), dtype=torch.long))

    def get_extra_state(self) -> Dict[str, Any]:
        pending = []
        for ticket_id, record in self._pending_feedback.items():
            pending.append(
                (
                    int(ticket_id),
                    {
                        "logits": record["logits"].detach().clone(),
                        "state_vector": record["state_vector"].detach().clone(),
                        "issued_observation": int(record["issued_observation"]),
                        "issued_parameter_version": int(
                            record["issued_parameter_version"]
                        ),
                        "stream_id": record["stream_id"],
                    },
                )
            )
        return {
            "version": 3,
            "streams": [
                (
                    stream_id,
                    [value.detach().clone() for value in state.rings],
                    int(self._stream_observations.get(stream_id, 0)),
                )
                for stream_id, state in self._stream_states.items()
            ],
            "pending_feedback": pending,
            "closed_tickets": list(self._closed_tickets.items()),
            "next_ticket_id": int(self._next_ticket_id),
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        self._stream_states.clear()
        self._stream_observations.clear()
        self._pending_feedback.clear()
        self._closed_tickets.clear()
        self._next_ticket_id = 1
        if not state:
            return

        # Version-1 checkpoints contained one unnamed state slot.
        if int(state.get("version", 1)) <= 1:
            rings = state.get("state")
            if rings is not None:
                restored = self._restore_temporal_state(rings)
                self._stream_states[None] = restored
                self._stream_observations[None] = int(restored.rings[0].size(0))
            return

        streams = list(state.get("streams", []))
        if len(streams) > self.max_streams:
            raise RuntimeError(
                f"checkpoint contains {len(streams)} streams but max_streams="
                f"{self.max_streams}"
            )
        for stream_id, rings, observations in streams:
            key = self._stream_key(stream_id)
            if key in self._stream_states:
                raise RuntimeError(f"duplicate stream_id in checkpoint: {key!r}")
            self._stream_states[key] = self._restore_temporal_state(rings)
            self._stream_observations[key] = int(observations)

        pending = list(state.get("pending_feedback", []))
        if len(pending) > self.max_pending_feedback:
            raise RuntimeError(
                f"checkpoint contains {len(pending)} pending tickets but "
                f"max_pending_feedback={self.max_pending_feedback}"
            )
        for raw_ticket_id, record in pending:
            ticket_id = int(raw_ticket_id)
            if ticket_id <= 0 or ticket_id in self._pending_feedback:
                raise RuntimeError(f"invalid or duplicate ticket id: {ticket_id}")
            logits = record["logits"].detach()
            state_vector = record["state_vector"].detach()
            self._validate_ticket_tensors(logits, state_vector)
            self._pending_feedback[ticket_id] = {
                "logits": logits,
                "state_vector": state_vector,
                "issued_observation": int(record["issued_observation"]),
                "issued_parameter_version": int(
                    record["issued_parameter_version"]
                ),
                "stream_id": self._stream_key(record.get("stream_id")),
            }
        for raw_ticket_id, status in state.get("closed_tickets", []):
            ticket_id = int(raw_ticket_id)
            if ticket_id > 0:
                self._closed_tickets[ticket_id] = str(status)
        while len(self._closed_tickets) > self._ticket_history_limit:
            self._closed_tickets.popitem(last=False)
        largest_id = max(
            [0, *self._pending_feedback.keys(), *self._closed_tickets.keys()]
        )
        self._next_ticket_id = max(
            largest_id + 1,
            int(state.get("next_ticket_id", largest_id + 1)),
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load legacy temporal checkpoints without inventing runtime data."""

        version_key = prefix + "online_parameter_version"
        if version_key not in state_dict:
            state_dict[version_key] = self.online_parameter_version.detach().clone()
        for name in ("online_gate_updates", "online_gate_labels"):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        feedback_prefix = prefix + "feedback_gradient_memory."
        for name, value in self.feedback_gradient_memory.state_dict().items():
            key = feedback_prefix + name
            if key not in state_dict:
                state_dict[key] = value
        gate_memory_prefix = prefix + "gate_gradient_memory."
        for name, value in self.gate_gradient_memory.state_dict().items():
            key = gate_memory_prefix + name
            if key not in state_dict:
                state_dict[key] = value
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
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

    def _restore_temporal_state(
        self, rings: Sequence[torch.Tensor]
    ) -> TemporalMQRState:
        restored = tuple(value.detach() for value in rings)
        if len(restored) != self.core.num_timescales:
            raise RuntimeError(
                "checkpoint temporal state has the wrong number of timescales"
            )
        if not restored:
            raise RuntimeError("checkpoint temporal state is empty")
        batch_size = restored[0].size(0) if restored[0].dim() == 2 else -1
        expected = (batch_size, self.core.ring_dim)
        if batch_size <= 0 or any(value.shape != expected for value in restored):
            raise RuntimeError(
                f"checkpoint temporal rings must all have shape [batch, "
                f"{self.core.ring_dim}]"
            )
        if any(
            not value.is_floating_point()
            or torch.is_complex(value)
            or not bool(torch.isfinite(value).all())
            for value in restored
        ):
            raise RuntimeError(
                "checkpoint temporal rings must be finite real floating-point tensors"
            )
        return TemporalMQRState(restored)

    def _validate_ticket_tensors(
        self,
        logits: torch.Tensor,
        state_vector: torch.Tensor,
    ) -> None:
        if logits.dim() != 2 or logits.size(1) != self.core.output_dim:
            raise RuntimeError("feedback ticket logits have an invalid shape")
        expected = (logits.size(0), self.core.total_state_dim)
        if state_vector.shape != expected:
            raise RuntimeError(
                f"feedback ticket state_vector must have shape {expected}"
            )
        if not logits.is_floating_point() or not state_vector.is_floating_point():
            raise RuntimeError("feedback ticket tensors must be floating point")
        if torch.is_complex(logits) or torch.is_complex(state_vector):
            raise RuntimeError("feedback ticket tensors must be real")
        if not bool(torch.isfinite(logits).all()) or not bool(
            torch.isfinite(state_vector).all()
        ):
            raise RuntimeError("feedback ticket tensors must be finite")

    @staticmethod
    def _loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 1:
            return F.cross_entropy(logits, target, reduction="mean")
        if target.dim() == 2:
            if target.shape != logits.shape:
                raise ValueError(f"soft target must have shape {tuple(logits.shape)}")
            return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        raise ValueError("target must have shape [batch] or [batch, classes]")

    def _controller_input(
        self,
        x: torch.Tensor,
        gate_input: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        controller = self.write_gate_controller
        if controller is None:
            if gate_input is not None:
                raise ValueError(
                    "gate_input requires an enabled write_gate_controller"
                )
            return None
        if gate_input is None:
            if controller.input_dim != self.core.input_dim:
                raise ValueError(
                    f"gate_input is required because the controller expects "
                    f"{controller.input_dim} features but x has "
                    f"{self.core.input_dim}"
                )
            prepared = x
        else:
            if not isinstance(gate_input, torch.Tensor):
                raise TypeError("gate_input must be a tensor or None")
            prepared = gate_input
        if prepared.dim() != 2 or prepared.shape != (
            x.size(0),
            controller.input_dim,
        ):
            raise ValueError(
                "gate_input must have shape "
                f"[{x.size(0)}, {controller.input_dim}], got "
                f"{tuple(prepared.shape)}"
            )
        # Gate supervision is auxiliary.  Neither task gradients nor a caller's
        # upstream graph are allowed to train the controller implicitly.
        return prepared.detach()

    def _resolve_write_gate(
        self,
        x: torch.Tensor,
        *,
        write_gate: Optional[torch.Tensor],
        gate_input: Optional[torch.Tensor],
        controller_required: bool = False,
    ) -> Dict[str, Any]:
        """Resolve external/learned/default gates under the current parameters."""

        if self.write_gate_controller is None and gate_input is not None:
            raise ValueError("gate_input requires an enabled write_gate_controller")
        controller_input: Optional[torch.Tensor] = None
        should_evaluate_controller = self.write_gate_controller is not None and (
            write_gate is None
            or controller_required
            or gate_input is not None
            or self.write_gate_controller.input_dim == self.core.input_dim
        )
        if should_evaluate_controller:
            controller_input = self._controller_input(x, gate_input)
        raw_logits: Optional[torch.Tensor] = None
        base_probability: Optional[torch.Tensor] = None
        learned_gate: Optional[torch.Tensor] = None
        if should_evaluate_controller:
            assert self.write_gate_controller is not None
            assert controller_input is not None
            raw_logits = self.write_gate_controller.raw_logits(controller_input)
            base_probability = self.write_gate_controller.base_probability(raw_logits)
            learned_gate = self.write_gate_controller.minimum_gate + (
                1.0 - self.write_gate_controller.minimum_gate
            ) * base_probability

        if write_gate is not None:
            effective = self.core._prepare_write_gate(write_gate, x)
            source = "external"
        elif learned_gate is not None:
            # This detach is the architectural boundary: the task loss may
            # update the MQR core, but never the event controller.
            effective = self.core._prepare_write_gate(learned_gate.detach(), x)
            source = "learned"
        else:
            effective = self.core._prepare_write_gate(None, x)
            source = "default_open"
        return {
            "source": source,
            "effective": effective,
            "raw_logits": raw_logits,
            "base_probability": base_probability,
            "learned": learned_gate,
        }

    def _prepare_gate_target(
        self,
        gate_target: torch.Tensor,
        *,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if self.write_gate_controller is None:
            raise ValueError(
                "gate_target requires an enabled write_gate_controller"
            )
        if not isinstance(gate_target, torch.Tensor):
            raise TypeError("gate_target must be a tensor")
        if torch.is_complex(gate_target):
            raise TypeError("gate_target must be real")
        if gate_target.dim() == 1 and gate_target.shape == (batch_size,):
            prepared = gate_target.unsqueeze(1).expand(
                -1,
                self.core.num_timescales,
            )
        elif gate_target.dim() == 2 and gate_target.shape == (
            batch_size,
            self.core.num_timescales,
        ):
            prepared = gate_target
        else:
            raise ValueError(
                "gate_target must have shape "
                f"[{batch_size}] or [{batch_size}, "
                f"{self.core.num_timescales}], got {tuple(gate_target.shape)}"
            )
        prepared = prepared.detach().to(
            device=reference.device,
            dtype=reference.dtype,
        )
        if not bool(torch.isfinite(prepared).all()):
            raise ValueError("gate_target must contain only finite values")
        if not bool(((prepared >= 0.0) & (prepared <= 1.0)).all()):
            raise ValueError("gate_target values must lie in [0, 1]")
        return prepared

    @staticmethod
    def _gate_diagnostics(
        learned_gate: Optional[torch.Tensor],
        base_probability: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "gate_accuracy": None,
            "gate_false_open_rate": None,
            "gate_false_close_rate": None,
            "gate_cue_mean": None,
            "gate_distractor_mean": None,
        }
        if learned_gate is None or base_probability is None or target is None:
            return result
        with torch.no_grad():
            target_open = target >= 0.5
            target_closed = ~target_open
            predicted_open = base_probability >= 0.5
            result["gate_accuracy"] = float(
                (predicted_open == target_open).to(torch.float32).mean().item()
            )
            if bool(target_closed.any()):
                result["gate_false_open_rate"] = float(
                    predicted_open[target_closed].to(torch.float32).mean().item()
                )
                result["gate_distractor_mean"] = float(
                    learned_gate[target_closed].mean().item()
                )
            if bool(target_open.any()):
                result["gate_false_close_rate"] = float(
                    (~predicted_open[target_open]).to(torch.float32).mean().item()
                )
                result["gate_cue_mean"] = float(
                    learned_gate[target_open].mean().item()
                )
        return result

    @staticmethod
    def _public_gate_fields(gate_record: Dict[str, Any]) -> Dict[str, Any]:
        def detached(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if value is None else value.detach().clone()

        return {
            "write_gate_source": str(gate_record["source"]),
            "effective_write_gate": detached(gate_record["effective"]),
            "learned_write_gate": detached(gate_record["learned"]),
            "gate_base_probability": detached(gate_record["base_probability"]),
        }

    def _state_for(
        self,
        x: torch.Tensor,
        carry_state: bool,
        stream_id: Optional[Hashable],
    ) -> Tuple[Optional[TemporalMQRState], bool]:
        """Return a detached state and whether a shape mismatch reset it."""

        if not carry_state:
            return None, False
        state = self._stream_states.get(stream_id)
        if state is None:
            return None, False
        if any(
            value.shape != (x.size(0), self.core.ring_dim)
            for value in state.rings
        ):
            return None, True
        return state.detached(), False

    def _stream_eviction_candidate(
        self,
        stream_id: Optional[Hashable],
        carry_state: bool,
    ) -> Optional[Tuple[Optional[Hashable]]]:
        if not carry_state or stream_id in self._stream_states:
            return None
        if len(self._stream_states) < self.max_streams:
            return None
        if self.stream_overflow_policy == "error":
            raise RuntimeError(
                f"state bank is full ({self.max_streams} streams); reset a stream, "
                "increase max_streams, or explicitly choose "
                'stream_overflow_policy="lru"'
            )
        return (next(iter(self._stream_states)),)

    def _commit_stream_state(
        self,
        stream_id: Optional[Hashable],
        state: TemporalMQRState,
        eviction_candidate: Optional[Tuple[Optional[Hashable]]],
    ) -> Tuple[bool, Optional[Hashable]]:
        evicted = False
        evicted_id: Optional[Hashable] = None
        if eviction_candidate is not None:
            evicted_id = eviction_candidate[0]
            self._stream_states.pop(evicted_id)
            self._stream_observations.pop(evicted_id, None)
            evicted = True
        self._stream_states[stream_id] = state
        self._stream_states.move_to_end(stream_id)
        self._stream_observations[stream_id] = (
            self._stream_observations.get(stream_id, 0) + state.rings[0].size(0)
        )
        return evicted, evicted_id

    def _record_closed_ticket(self, ticket_id: int, status: str) -> None:
        self._closed_tickets.pop(int(ticket_id), None)
        self._closed_tickets[int(ticket_id)] = str(status)
        while len(self._closed_tickets) > self._ticket_history_limit:
            self._closed_tickets.popitem(last=False)

    def _expire_pending_feedback(self, *, now: Optional[int] = None) -> List[int]:
        if self.feedback_ttl_observations is None:
            return []
        current = int(self.online_observations.item()) if now is None else int(now)
        expired: List[int] = []
        for ticket_id, record in list(self._pending_feedback.items()):
            age = current - int(record["issued_observation"])
            if age > self.feedback_ttl_observations:
                self._pending_feedback.pop(ticket_id)
                self._record_closed_ticket(ticket_id, "expired")
                expired.append(ticket_id)
        return expired

    def _feedback_overflow_candidate(self, *, now: int) -> Optional[int]:
        self._expire_pending_feedback(now=now)
        if len(self._pending_feedback) < self.max_pending_feedback:
            return None
        if self.feedback_overflow_policy == "error":
            raise RuntimeError(
                f"feedback queue is full ({self.max_pending_feedback} tickets); "
                "consume/cancel feedback or explicitly choose "
                'feedback_overflow_policy="oldest"'
            )
        return next(iter(self._pending_feedback))

    def _issue_feedback_ticket(
        self,
        *,
        logits: torch.Tensor,
        state: TemporalMQRState,
        stream_id: Optional[Hashable],
        overflow_candidate: Optional[int],
    ) -> int:
        if overflow_candidate is not None:
            self._pending_feedback.pop(overflow_candidate)
            self._record_closed_ticket(overflow_candidate, "evicted")
        ticket_id = self._next_ticket_id
        self._next_ticket_id += 1
        self._pending_feedback[ticket_id] = {
            "logits": logits.detach().clone(),
            "state_vector": torch.cat(state.rings, dim=1).detach().clone(),
            "issued_observation": int(self.online_observations.item()),
            "issued_parameter_version": int(self.online_parameter_version.item()),
            "stream_id": stream_id,
        }
        return ticket_id

    def _feedback_record(self, ticket_id: int) -> Dict[str, Any]:
        if not isinstance(ticket_id, int) or isinstance(ticket_id, bool) or ticket_id <= 0:
            raise TypeError("ticket_id must be a positive integer")
        self._expire_pending_feedback()
        record = self._pending_feedback.get(ticket_id)
        if record is not None:
            return record
        status = self._closed_tickets.get(ticket_id)
        if status == "consumed":
            raise RuntimeError(f"feedback ticket {ticket_id} was already consumed")
        if status == "expired":
            raise RuntimeError(f"feedback ticket {ticket_id} has expired")
        if status == "evicted":
            raise RuntimeError(f"feedback ticket {ticket_id} was evicted")
        if status == "cancelled":
            raise RuntimeError(f"feedback ticket {ticket_id} was cancelled")
        raise KeyError(f"unknown feedback ticket: {ticket_id}")

    def _step_ratio(self, name: str) -> float:
        if name.startswith("readout."):
            return self.readout_lr_ratio
        if name.startswith("input_down.") or name.startswith("input_up."):
            return self.injection_lr_ratio
        if name.startswith("unitary_params."):
            return self.transition_lr_ratio
        raise RuntimeError(f"unclassified temporal MQR parameter: {name}")

    def _active_parameters(self) -> List[Tuple[str, nn.Parameter, float]]:
        active: List[Tuple[str, nn.Parameter, float]] = []
        for name, parameter in self.core.named_parameters():
            ratio = self._step_ratio(name)
            if parameter.requires_grad and ratio > 0.0:
                active.append((name, parameter, self.lr * ratio))
        return active

    def _active_gate_parameters(self) -> List[Tuple[str, nn.Parameter, float]]:
        if self.write_gate_controller is None:
            return []
        return [
            (name, parameter, self.gate_lr)
            for name, parameter in self.write_gate_controller.named_parameters()
            if parameter.requires_grad
        ]

    @staticmethod
    def _project_gradient_entries(
        gradient_entries: Sequence[Tuple[str, torch.Tensor, float]],
        memory: OrthogonalGradientMemory,
        *,
        remember_gradient: bool,
        project_with_memory: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], bool, bool]:
        projection_applied = bool(
            gradient_entries and project_with_memory and memory.max_rank > 0
        )
        if projection_applied:
            projected, stats = memory.project_preconditioned(gradient_entries)
        else:
            projected = {
                name: gradient for name, gradient, _step in gradient_entries
            }
            raw_norm_sq = sum(
                step * float(gradient.square().sum().item())
                for _name, gradient, step in gradient_entries
            )
            raw_norm = math.sqrt(raw_norm_sq)
            stats = {
                "raw_norm": raw_norm,
                "projected_norm": raw_norm,
                "retained_norm": 1.0 if gradient_entries else 0.0,
                "max_abs_overlap": 0.0,
            }
        memory_added = False
        if remember_gradient:
            if not gradient_entries:
                raise ValueError("remember_gradient requires active parameters")
            memory_added = memory.observe(gradient_entries)
        return projected, stats, memory_added, projection_applied

    def _update_geometry(
        self,
        gradient_entries: Sequence[Tuple[str, torch.Tensor, float]],
        projected: Dict[str, torch.Tensor],
    ) -> Tuple[float, float]:
        unclipped_sq = sum(
            (step**2) * float(projected[name].square().sum().item())
            for name, _gradient, step in gradient_entries
        )
        unclipped_norm = math.sqrt(unclipped_sq)
        clip_scale = 1.0
        if self.max_update_norm is not None and unclipped_norm > self.max_update_norm:
            clip_scale = self.max_update_norm / (unclipped_norm + 1e-12)
        return unclipped_norm, clip_scale

    def _gate_update_geometry(
        self,
        gradient_entries: Sequence[Tuple[str, torch.Tensor, float]],
        projected: Dict[str, torch.Tensor],
    ) -> Tuple[float, float]:
        unclipped_sq = sum(
            (step**2) * float(projected[name].square().sum().item())
            for name, _gradient, step in gradient_entries
        )
        unclipped_norm = math.sqrt(unclipped_sq)
        clip_scale = 1.0
        if (
            self.gate_max_update_norm is not None
            and unclipped_norm > self.gate_max_update_norm
        ):
            clip_scale = self.gate_max_update_norm / (unclipped_norm + 1e-12)
        return unclipped_norm, clip_scale

    @torch.no_grad()
    def preview_step(
        self,
        x: torch.Tensor,
        *,
        carry_state: Optional[bool] = None,
        stream_id: Any = None,
        write_gate: Optional[torch.Tensor] = None,
        gate_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Return the next prediction without mutating runtime state."""

        key = self._stream_key(stream_id)
        use_state = self.carry_state if carry_state is None else bool(carry_state)
        previous, reinitialized = self._state_for(x, use_state, key)
        gate_record = self._resolve_write_gate(
            x,
            write_gate=write_gate,
            gate_input=gate_input,
        )
        logits, next_state = self.core.forward_step(
            x,
            state=previous,
            write_gate=gate_record["effective"],
        )
        result = {
            "logits": logits,
            "state": next_state,
            "prediction_before_update": True,
            "preview_only": True,
            "state_carried": use_state,
            "state_was_present": previous is not None,
            "state_reinitialized": reinitialized,
            "stream_id": key,
            "state_bank_size": len(self._stream_states),
        }
        result.update(self._public_gate_fields(gate_record))
        return result

    @torch.no_grad()
    def infer_step(
        self,
        x: torch.Tensor,
        *,
        carry_state: Optional[bool] = None,
        stream_id: Any = None,
        write_gate: Optional[torch.Tensor] = None,
        gate_input: Optional[torch.Tensor] = None,
        issue_feedback_ticket: bool = True,
    ) -> Dict[str, Any]:
        """Commit inference immediately and optionally issue delayed feedback.

        The returned ticket caches the old-policy logits and the exact readout
        input.  :meth:`apply_feedback` can therefore reconstruct the issue-time
        readout gradient without retaining an autograd graph or consulting the
        stream's later recurrent state.
        """

        if x.dim() != 2 or x.size(1) != self.core.input_dim:
            raise ValueError(
                f"x must be [batch, {self.core.input_dim}], got {tuple(x.shape)}"
            )
        key = self._stream_key(stream_id)
        use_state = self.carry_state if carry_state is None else bool(carry_state)
        eviction_candidate = self._stream_eviction_candidate(key, use_state)
        previous, reinitialized = self._state_for(x, use_state, key)
        gate_record = self._resolve_write_gate(
            x,
            write_gate=write_gate,
            gate_input=gate_input,
        )
        logits, next_state = self.core.forward_step(
            x,
            state=previous,
            write_gate=gate_record["effective"],
        )
        committed_state = next_state.detached()

        future_observations = int(self.online_observations.item()) + x.size(0)
        ticket_overflow = (
            self._feedback_overflow_candidate(now=future_observations)
            if issue_feedback_ticket
            else None
        )
        evicted = False
        evicted_id: Optional[Hashable] = None
        if use_state:
            evicted, evicted_id = self._commit_stream_state(
                key,
                committed_state,
                eviction_candidate,
            )
        self.online_observations.add_(x.size(0))
        self._expire_pending_feedback()
        ticket_id = None
        if issue_feedback_ticket:
            ticket_id = self._issue_feedback_ticket(
                logits=logits,
                state=committed_state,
                stream_id=key,
                overflow_candidate=ticket_overflow,
            )
        result = {
            "logits": logits.detach().clone(),
            "state": committed_state,
            "ticket_id": ticket_id,
            "prediction_before_update": True,
            "preview_only": False,
            "did_update": False,
            "state_carried": use_state,
            "state_was_present": previous is not None,
            "state_reinitialized": reinitialized,
            "stream_id": key,
            "state_bank_size": len(self._stream_states),
            "state_evicted": evicted,
            "evicted_stream_id": evicted_id,
            "pending_feedback": len(self._pending_feedback),
            "parameter_version": int(self.online_parameter_version.item()),
        }
        result.update(self._public_gate_fields(gate_record))
        return result

    def online_step(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        *,
        learn: bool = True,
        remember_gradient: bool = False,
        project_with_memory: bool = True,
        carry_state: Optional[bool] = None,
        return_grad_x: bool = False,
        stream_id: Any = None,
        write_gate: Optional[torch.Tensor] = None,
        gate_input: Optional[torch.Tensor] = None,
        gate_target: Optional[torch.Tensor] = None,
        learn_gate: bool = True,
        remember_gate_gradient: bool = False,
        project_gate_with_memory: bool = True,
    ) -> Dict[str, Any]:
        """Predict first, then atomically commit optional task and gate updates.

        ``gate_target`` is auxiliary event supervision revealed after the gate
        has already selected the current write.  The effective learned gate is
        detached from the task graph, so this method is not an end-to-end
        delayed-reward estimator.
        """

        if x.dim() != 2 or x.size(1) != self.core.input_dim:
            raise ValueError(f"x must be [batch, {self.core.input_dim}], got {tuple(x.shape)}")
        if remember_gradient and (not learn or target is None):
            raise ValueError("remember_gradient requires a supervised update")
        if remember_gradient and self.gradient_memory.max_rank == 0:
            raise ValueError("remember_gradient requires ogd_max_rank > 0")
        if gate_target is not None and self.write_gate_controller is None:
            raise ValueError(
                "gate_target requires an enabled write_gate_controller"
            )
        if remember_gate_gradient and (not learn_gate or gate_target is None):
            raise ValueError(
                "remember_gate_gradient requires a supervised gate update"
            )
        if (
            remember_gate_gradient
            and self.gate_gradient_memory.max_rank == 0
        ):
            raise ValueError(
                "remember_gate_gradient requires gate_ogd_max_rank > 0"
            )

        key = self._stream_key(stream_id)
        use_state = self.carry_state if carry_state is None else bool(carry_state)
        eviction_candidate = self._stream_eviction_candidate(key, use_state)
        previous, reinitialized = self._state_for(x, use_state, key)
        x_work = x.detach().requires_grad_(bool(return_grad_x))
        active = self._active_parameters() if learn and target is not None else []
        active_gate = (
            self._active_gate_parameters()
            if learn_gate and gate_target is not None
            else []
        )

        with torch.enable_grad():
            # Resolve theta_t's gate before consulting its event label.  The
            # detached effective tensor prevents the task loss from crossing
            # into the controller.
            gate_record = self._resolve_write_gate(
                x,
                write_gate=write_gate,
                gate_input=gate_input,
                controller_required=gate_target is not None,
            )
            logits, next_state = self.core.forward_step(
                x_work,
                state=previous,
                write_gate=gate_record["effective"],
            )
            loss = None if target is None else self._loss_from_logits(logits, target.detach())
            gradients: Tuple[torch.Tensor, ...] = ()
            grad_x: Optional[torch.Tensor] = None
            if active:
                differentiation_targets: List[torch.Tensor] = [
                    parameter for _name, parameter, _step in active
                ]
                if return_grad_x:
                    differentiation_targets.append(x_work)
                assert loss is not None
                computed = torch.autograd.grad(
                    loss,
                    differentiation_targets,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                gradients = tuple(computed[: len(active)])
                if return_grad_x:
                    grad_x = computed[-1].detach()
            elif return_grad_x and loss is not None:
                grad_x = torch.autograd.grad(loss, x_work)[0].detach()

            gate_target_work: Optional[torch.Tensor] = None
            gate_loss: Optional[torch.Tensor] = None
            gate_gradients: Tuple[torch.Tensor, ...] = ()
            if gate_target is not None:
                raw_gate_logits = gate_record["raw_logits"]
                if raw_gate_logits is None or self.write_gate_controller is None:
                    raise RuntimeError("gate controller did not produce logits")
                gate_target_work = self._prepare_gate_target(
                    gate_target,
                    batch_size=x.size(0),
                    reference=raw_gate_logits,
                )
                gate_loss = F.binary_cross_entropy_with_logits(
                    raw_gate_logits / self.write_gate_controller.temperature,
                    gate_target_work,
                    reduction="mean",
                    pos_weight=raw_gate_logits.new_full(
                        (self.core.num_timescales,),
                        self.gate_positive_weight,
                    ),
                )
                if active_gate:
                    gate_gradients = tuple(
                        torch.autograd.grad(
                            gate_loss,
                            [
                                parameter
                                for _name, parameter, _step in active_gate
                            ],
                            retain_graph=False,
                            create_graph=False,
                            allow_unused=False,
                        )
                    )

        logits_before = logits.detach().clone()
        committed_state = next_state.detached()
        gradient_entries = [
            (name, gradient.detach(), step)
            for (name, _parameter, step), gradient in zip(active, gradients)
        ]
        projected, ogd_stats, memory_added, projection_applied = (
            self._project_gradient_entries(
                gradient_entries,
                self.gradient_memory,
                remember_gradient=remember_gradient,
                project_with_memory=project_with_memory,
            )
        )
        unclipped_norm, clip_scale = self._update_geometry(
            gradient_entries,
            projected,
        )
        gate_gradient_entries = [
            (name, gradient.detach(), step)
            for (name, _parameter, step), gradient in zip(
                active_gate,
                gate_gradients,
            )
        ]
        (
            gate_projected,
            gate_ogd_stats,
            gate_memory_added,
            gate_projection_applied,
        ) = self._project_gradient_entries(
            gate_gradient_entries,
            self.gate_gradient_memory,
            remember_gradient=remember_gate_gradient,
            project_with_memory=project_gate_with_memory,
        )
        gate_unclipped_norm, gate_clip_scale = self._gate_update_geometry(
            gate_gradient_entries,
            gate_projected,
        )
        gate_diagnostics = self._gate_diagnostics(
            gate_record["learned"],
            gate_record["base_probability"],
            gate_target_work,
        )
        any_parameter_update = bool(gradient_entries or gate_gradient_entries)

        with torch.no_grad():
            for (name, parameter, step), _gradient in zip(active, gradients):
                parameter.add_(projected[name], alpha=-step * clip_scale)
            for (name, parameter, step), _gradient in zip(
                active_gate,
                gate_gradients,
            ):
                parameter.add_(
                    gate_projected[name],
                    alpha=-step * gate_clip_scale,
                )
            evicted = False
            evicted_id: Optional[Hashable] = None
            if use_state:
                # The state belongs to the prediction under theta_t, not to the
                # just-updated theta_(t+1).
                evicted, evicted_id = self._commit_stream_state(
                    key,
                    committed_state,
                    eviction_candidate,
                )
            self.online_observations.add_(x.size(0))
            if gate_target_work is not None:
                self.online_gate_labels.add_(gate_target_work.numel())
            if gate_gradient_entries:
                self.online_gate_updates.add_(1)
            if any_parameter_update:
                self.online_updates.add_(1)
                self.online_parameter_version.add_(1)
            self._expire_pending_feedback()

        result = {
            "logits": logits_before,
            "loss": None if loss is None else float(loss.detach().item()),
            "state": committed_state,
            "prediction_before_update": True,
            "preview_only": False,
            "state_carried": use_state,
            "state_was_present": previous is not None,
            "state_reinitialized": reinitialized,
            "stream_id": key,
            "state_bank_size": len(self._stream_states),
            "state_evicted": evicted,
            "evicted_stream_id": evicted_id,
            "did_update": any_parameter_update,
            "core_did_update": bool(gradient_entries),
            "grad_x": grad_x,
            "ogd_rank": self.gradient_memory.rank,
            "ogd_retained_norm": float(ogd_stats["retained_norm"]),
            "ogd_raw_norm": float(ogd_stats["raw_norm"]),
            "ogd_projected_norm": float(ogd_stats["projected_norm"]),
            "ogd_max_abs_overlap": float(ogd_stats["max_abs_overlap"]),
            "ogd_first_order_decrease": -clip_scale
            * float(ogd_stats["projected_norm"]) ** 2,
            "ogd_memory_added": bool(memory_added),
            "ogd_projection_applied": projection_applied,
            "update_norm": unclipped_norm * clip_scale,
            "unclipped_update_norm": unclipped_norm,
            "update_clip_scale": clip_scale,
            "parameter_version": int(self.online_parameter_version.item()),
            "pending_feedback": len(self._pending_feedback),
            "max_unitary_error": self.core.max_unitary_error(),
            "max_stochastic_error": self.core.max_stochastic_error(),
            "gate_loss": (
                None if gate_loss is None else float(gate_loss.detach().item())
            ),
            "gate_did_update": bool(gate_gradient_entries),
            "gate_supervision_after_prediction": gate_target_work is not None,
            "gate_task_gradient_detached": True,
            "gate_label_count": (
                0 if gate_target_work is None else int(gate_target_work.numel())
            ),
            "gate_positive_weight": self.gate_positive_weight,
            "gate_ogd_rank": self.gate_gradient_memory.rank,
            "gate_ogd_retained_norm": float(
                gate_ogd_stats["retained_norm"]
            ),
            "gate_ogd_raw_norm": float(gate_ogd_stats["raw_norm"]),
            "gate_ogd_projected_norm": float(
                gate_ogd_stats["projected_norm"]
            ),
            "gate_ogd_max_abs_overlap": float(
                gate_ogd_stats["max_abs_overlap"]
            ),
            "gate_ogd_first_order_decrease": -gate_clip_scale
            * float(gate_ogd_stats["projected_norm"]) ** 2,
            "gate_ogd_memory_added": bool(gate_memory_added),
            "gate_ogd_projection_applied": gate_projection_applied,
            "gate_update_norm": gate_unclipped_norm * gate_clip_scale,
            "gate_unclipped_update_norm": gate_unclipped_norm,
            "gate_update_clip_scale": gate_clip_scale,
            "online_gate_updates": int(self.online_gate_updates.item()),
            "online_gate_labels": int(self.online_gate_labels.item()),
        }
        result.update(self._public_gate_fields(gate_record))
        result.update(gate_diagnostics)
        return result

    def _delayed_readout_gradient(
        self,
        record: Dict[str, Any],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        reference = self.core.readout.weight
        logits = record["logits"].to(
            device=reference.device,
            dtype=reference.dtype,
        )
        features = record["state_vector"].to(
            device=reference.device,
            dtype=reference.dtype,
        )
        if target.dim() == 1:
            if target.shape != (logits.size(0),):
                raise ValueError(
                    f"target must have shape [{logits.size(0)}], got "
                    f"{tuple(target.shape)}"
                )
            target_work = target.detach().to(device=logits.device)
            loss = F.cross_entropy(logits, target_work, reduction="mean")
            desired = F.one_hot(
                target_work,
                num_classes=self.core.output_dim,
            ).to(dtype=logits.dtype)
        elif target.dim() == 2:
            if target.shape != logits.shape:
                raise ValueError(
                    f"soft target must have shape {tuple(logits.shape)}"
                )
            target_work = target.detach().to(
                device=logits.device,
                dtype=logits.dtype,
            )
            loss = -(target_work * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            desired = target_work
        else:
            raise ValueError("target must have shape [batch] or [batch, classes]")

        # This also matches the declared soft-target loss when a caller uses
        # non-normalized weights: d[-y*log_softmax(a)]/da = p*sum(y)-y.
        target_mass = desired.sum(dim=1, keepdim=True)
        error = (F.softmax(logits, dim=1) * target_mass - desired) / logits.size(0)
        weight_gradient = error.transpose(0, 1) @ features
        bias_gradient = error.sum(dim=0) if self.core.readout.bias is not None else None
        return weight_gradient, logits, bias_gradient, loss

    def apply_feedback(
        self,
        ticket_id: int,
        target: torch.Tensor,
        *,
        learn: bool = True,
        remember_gradient: bool = False,
        project_with_memory: bool = True,
    ) -> Dict[str, Any]:
        """Consume one ticket and optionally apply its issue-time readout gradient.

        The cached gradient is exact for the readout parameters that produced
        the ticket.  If other updates happened meanwhile, it is intentionally a
        stale asynchronous gradient; the returned version gap makes that
        approximation observable.  Recurrent state and non-readout parameters
        are never consulted or changed by this method.
        """

        if not isinstance(target, torch.Tensor):
            raise TypeError("target must be a tensor")
        if remember_gradient and not learn:
            raise ValueError("remember_gradient requires a learned feedback update")
        if remember_gradient and self.feedback_gradient_memory.max_rank == 0:
            raise ValueError("remember_gradient requires ogd_max_rank > 0")
        record = self._feedback_record(ticket_id)
        weight_gradient, cached_logits, bias_gradient, loss = (
            self._delayed_readout_gradient(record, target)
        )
        issued_version = int(record["issued_parameter_version"])
        version_before = int(self.online_parameter_version.item())
        issued_observation = int(record["issued_observation"])
        current_observation = int(self.online_observations.item())
        if issued_version < 0 or issued_version > version_before:
            raise RuntimeError("feedback ticket has an invalid future parameter version")
        if issued_observation < 0 or issued_observation > current_observation:
            raise RuntimeError("feedback ticket has an invalid future observation index")

        active: List[Tuple[str, nn.Parameter, float]] = []
        gradients: List[torch.Tensor] = []
        step = self.lr * self.readout_lr_ratio
        if learn and step > 0.0:
            active.append(("readout.weight", self.core.readout.weight, step))
            gradients.append(weight_gradient.detach())
            if self.core.readout.bias is not None:
                assert bias_gradient is not None
                active.append(("readout.bias", self.core.readout.bias, step))
                gradients.append(bias_gradient.detach())
        gradient_entries = [
            (name, gradient, parameter_step)
            for (name, _parameter, parameter_step), gradient in zip(active, gradients)
        ]
        projected, ogd_stats, memory_added, projection_applied = (
            self._project_gradient_entries(
                gradient_entries,
                self.feedback_gradient_memory,
                remember_gradient=remember_gradient,
                project_with_memory=project_with_memory,
            )
        )
        unclipped_norm, clip_scale = self._update_geometry(
            gradient_entries,
            projected,
        )
        ticket_age = current_observation - issued_observation

        with torch.no_grad():
            for name, parameter, parameter_step in active:
                parameter.add_(
                    projected[name],
                    alpha=-parameter_step * clip_scale,
                )
            if gradient_entries:
                self.online_updates.add_(1)
                self.online_parameter_version.add_(1)
            self._pending_feedback.pop(ticket_id)
            self._record_closed_ticket(ticket_id, "consumed")

        return {
            "ticket_id": ticket_id,
            "logits": cached_logits.detach().clone(),
            "loss": float(loss.detach().item()),
            "loss_at_issue": float(loss.detach().item()),
            "prediction_before_update": True,
            "did_update": bool(gradient_entries),
            "readout_only_update": bool(gradient_entries),
            "state_mutated": False,
            "gradient_reference": "issue_time_readout",
            "stream_id": record["stream_id"],
            "issued_parameter_version": issued_version,
            "parameter_staleness": version_before - issued_version,
            "is_stale": version_before != issued_version,
            "parameter_version": int(self.online_parameter_version.item()),
            "ticket_age_observations": ticket_age,
            "pending_feedback": len(self._pending_feedback),
            "ogd_rank": self.feedback_gradient_memory.rank,
            "ogd_retained_norm": float(ogd_stats["retained_norm"]),
            "ogd_raw_norm": float(ogd_stats["raw_norm"]),
            "ogd_projected_norm": float(ogd_stats["projected_norm"]),
            "ogd_max_abs_overlap": float(ogd_stats["max_abs_overlap"]),
            "ogd_first_order_decrease": -clip_scale
            * float(ogd_stats["projected_norm"]) ** 2,
            "ogd_memory_added": bool(memory_added),
            "ogd_projection_applied": projection_applied,
            "update_norm": unclipped_norm * clip_scale,
            "unclipped_update_norm": unclipped_norm,
            "update_clip_scale": clip_scale,
        }

    @property
    def state_bank_size(self) -> int:
        return len(self._stream_states)

    @property
    def pending_feedback_count(self) -> int:
        self._expire_pending_feedback()
        return len(self._pending_feedback)

    def active_stream_ids(self) -> Tuple[Optional[Hashable], ...]:
        """Return stream IDs from least to most recently committed."""

        return tuple(self._stream_states.keys())

    def pending_feedback_ids(self) -> Tuple[int, ...]:
        self._expire_pending_feedback()
        return tuple(self._pending_feedback.keys())

    def stream_statistics(self) -> List[Dict[str, Any]]:
        return [
            {
                "stream_id": stream_id,
                "observations": int(self._stream_observations.get(stream_id, 0)),
                "lru_rank": index,
                "batch_size": int(state.rings[0].size(0)),
            }
            for index, (stream_id, state) in enumerate(self._stream_states.items())
        ]

    @torch.no_grad()
    def reset_state(self, stream_id: Any = None) -> None:
        key = self._stream_key(stream_id)
        self._stream_states.pop(key, None)
        self._stream_observations.pop(key, None)

    @torch.no_grad()
    def reset_all_states(self) -> None:
        self._stream_states.clear()
        self._stream_observations.clear()

    @torch.no_grad()
    def cancel_feedback(self, ticket_id: int) -> None:
        self._feedback_record(ticket_id)
        self._pending_feedback.pop(ticket_id)
        self._record_closed_ticket(ticket_id, "cancelled")

    @torch.no_grad()
    def clear_pending_feedback(self) -> None:
        for ticket_id in list(self._pending_feedback):
            self._pending_feedback.pop(ticket_id)
            self._record_closed_ticket(ticket_id, "cancelled")

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
        if clone:
            return TemporalMQRState(
                tuple(value.detach().clone() for value in state.rings)
            )
        return state.detached()
