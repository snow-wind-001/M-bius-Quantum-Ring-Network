"""Matched non-MQR temporal cores and resource audits for unified-agent tests."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import TemporalUtilityMQRAgent
from .baselines import SinkhornDoublyStochasticParam
from .temporal import MultiTimescaleMQR, TemporalMQRState


def _rates(values: Sequence[float]) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) or not (0.0 < value <= 1.0) for value in result):
        raise ValueError("leak rates must be finite and lie in (0, 1]")
    return result


def _prepare_gate(
    gate: Optional[torch.Tensor],
    x: torch.Tensor,
    count: int,
) -> torch.Tensor:
    if gate is None:
        return torch.ones(x.size(0), count, device=x.device, dtype=x.dtype)
    if gate.shape != (x.size(0), count):
        raise ValueError(f"write_gate must have shape [{x.size(0)}, {count}]")
    result = gate.detach().to(device=x.device, dtype=x.dtype)
    if not bool(torch.isfinite(result).all()) or not bool(
        ((result >= 0.0) & (result <= 1.0)).all()
    ):
        raise ValueError("write_gate must be finite and lie in [0, 1]")
    return result


class MultiTimescaleGRUCore(nn.Module):
    """GRU control with the same fast/slow write-gate contract as temporal MQR."""

    def __init__(
        self,
        input_dim: int,
        ring_dim: int,
        output_dim: int,
        *,
        leak_rates: Sequence[float] = (1.0, 0.10, 0.02),
        input_rank: Optional[int] = None,
        readout_bias: bool = True,
        zero_init_readout: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or ring_dim <= 0 or output_dim <= 0:
            raise ValueError("core dimensions must be positive")
        self.input_dim = int(input_dim)
        self.ring_dim = int(ring_dim)
        self.output_dim = int(output_dim)
        self.leak_rates_tuple = _rates(leak_rates)
        self.num_timescales = len(self.leak_rates_tuple)
        if input_rank is not None and input_rank <= 0:
            raise ValueError("input_rank must be positive or None")
        self.input_rank = None if input_rank is None else int(input_rank)
        self.cell_input_dim = self.input_dim if input_rank is None else int(input_rank)
        self.input_projection: nn.Module = (
            nn.Identity()
            if input_rank is None
            else nn.Linear(self.input_dim, int(input_rank), bias=False)
        )
        self.cells = nn.ModuleList(
            nn.GRUCell(self.cell_input_dim, self.ring_dim)
            for _ in self.leak_rates_tuple
        )
        self.readout = nn.Linear(
            self.num_timescales * self.ring_dim,
            self.output_dim,
            bias=bool(readout_bias),
        )
        if zero_init_readout:
            nn.init.zeros_(self.readout.weight)
            if self.readout.bias is not None:
                nn.init.zeros_(self.readout.bias)

    def zero_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> TemporalMQRState:
        return TemporalMQRState(
            tuple(
                torch.zeros(batch_size, self.ring_dim, device=device, dtype=dtype)
                for _ in self.leak_rates_tuple
            )
        )

    def readout_state(self, state: TemporalMQRState) -> torch.Tensor:
        return self.readout(torch.cat(state.rings, dim=1))

    def forward_step(
        self,
        x: torch.Tensor,
        *,
        state: Optional[TemporalMQRState] = None,
        write_gate: Optional[torch.Tensor] = None,
        promotion: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TemporalMQRState]:
        if promotion is not None:
            raise ValueError("GRU control does not accept MQR promotion forcing")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"x must have shape [batch, {self.input_dim}]")
        if state is None:
            state = self.zero_state(x.size(0), device=x.device, dtype=x.dtype)
        gates = _prepare_gate(write_gate, x, self.num_timescales)
        projected_input = self.input_projection(x)
        if self.input_rank is not None:
            projected_input = torch.tanh(projected_input)
        result = []
        for index, (cell, previous, leak) in enumerate(
            zip(self.cells, state.rings, self.leak_rates_tuple)
        ):
            carried = (1.0 - leak) * previous
            candidate = cell(projected_input, carried)
            gate = gates[:, index : index + 1]
            result.append(carried + gate * (candidate - carried))
        next_state = TemporalMQRState(tuple(result))
        return self.readout_state(next_state), next_state

    def max_unitary_error(self) -> float:
        return 0.0

    def max_stochastic_error(self) -> float:
        return 0.0

    def estimated_forward_macs(self) -> int:
        projection = (
            0 if self.input_rank is None else self.input_dim * self.cell_input_dim
        )
        recurrent = self.num_timescales * 3 * (
            self.cell_input_dim * self.ring_dim + self.ring_dim * self.ring_dim
        )
        output = self.num_timescales * self.ring_dim * self.output_dim
        return int(projection + recurrent + output)


class MultiTimescaleLSTMCore(nn.Module):
    """LSTM control with hidden and cell packed into each state ring."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        leak_rates: Sequence[float] = (1.0, 0.10, 0.02),
        input_rank: Optional[int] = None,
        readout_bias: bool = True,
        zero_init_readout: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("core dimensions must be positive")
        if input_rank is not None and input_rank <= 0:
            raise ValueError("input_rank must be positive or None")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.ring_dim = 2 * self.hidden_dim
        self.output_dim = int(output_dim)
        self.leak_rates_tuple = _rates(leak_rates)
        self.num_timescales = len(self.leak_rates_tuple)
        self.input_rank = None if input_rank is None else int(input_rank)
        self.cell_input_dim = self.input_dim if input_rank is None else int(input_rank)
        self.input_projection: nn.Module = (
            nn.Identity()
            if input_rank is None
            else nn.Linear(self.input_dim, int(input_rank), bias=False)
        )
        self.cells = nn.ModuleList(
            nn.LSTMCell(self.cell_input_dim, self.hidden_dim)
            for _ in self.leak_rates_tuple
        )
        self.readout = nn.Linear(
            self.num_timescales * self.hidden_dim,
            self.output_dim,
            bias=bool(readout_bias),
        )
        if zero_init_readout:
            nn.init.zeros_(self.readout.weight)
            if self.readout.bias is not None:
                nn.init.zeros_(self.readout.bias)

    def zero_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TemporalMQRState:
        return TemporalMQRState(
            tuple(
                torch.zeros(batch_size, self.ring_dim, device=device, dtype=dtype)
                for _ in self.leak_rates_tuple
            )
        )

    def _hidden(self, state: TemporalMQRState) -> Tuple[torch.Tensor, ...]:
        return tuple(value[:, : self.hidden_dim] for value in state.rings)

    def readout_state(self, state: TemporalMQRState) -> torch.Tensor:
        return self.readout(torch.cat(self._hidden(state), dim=1))

    def forward_step(
        self,
        x: torch.Tensor,
        *,
        state: Optional[TemporalMQRState] = None,
        write_gate: Optional[torch.Tensor] = None,
        promotion: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TemporalMQRState]:
        if promotion is not None:
            raise ValueError("LSTM control does not accept MQR promotion forcing")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"x must have shape [batch, {self.input_dim}]")
        if state is None:
            state = self.zero_state(x.size(0), device=x.device, dtype=x.dtype)
        gates = _prepare_gate(write_gate, x, self.num_timescales)
        projected_input = self.input_projection(x)
        if self.input_rank is not None:
            projected_input = torch.tanh(projected_input)
        packed = []
        for index, (cell, previous, leak) in enumerate(
            zip(self.cells, state.rings, self.leak_rates_tuple)
        ):
            previous_h, previous_c = previous.split(self.hidden_dim, dim=1)
            carried_h = (1.0 - leak) * previous_h
            carried_c = (1.0 - leak) * previous_c
            candidate_h, candidate_c = cell(
                projected_input,
                (carried_h, carried_c),
            )
            gate = gates[:, index : index + 1]
            next_h = carried_h + gate * (candidate_h - carried_h)
            next_c = carried_c + gate * (candidate_c - carried_c)
            packed.append(torch.cat((next_h, next_c), dim=1))
        next_state = TemporalMQRState(tuple(packed))
        return self.readout_state(next_state), next_state

    def max_unitary_error(self) -> float:
        return 0.0

    def max_stochastic_error(self) -> float:
        return 0.0

    def estimated_forward_macs(self) -> int:
        projection = (
            0 if self.input_rank is None else self.input_dim * self.cell_input_dim
        )
        recurrent = self.num_timescales * 4 * (
            self.cell_input_dim * self.hidden_dim
            + self.hidden_dim * self.hidden_dim
        )
        output = self.num_timescales * self.hidden_dim * self.output_dim
        return int(projection + recurrent + output)


class MultiTimescaleFastWeightCore(nn.Module):
    """Hebbian fast-weight control with gated multi-timescale matrix memory."""

    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        output_dim: int,
        *,
        leak_rates: Sequence[float] = (1.0, 0.10, 0.02),
        input_rank: Optional[int] = None,
        readout_bias: bool = True,
        zero_init_readout: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or memory_dim <= 0 or output_dim <= 0:
            raise ValueError("core dimensions must be positive")
        self.input_dim = int(input_dim)
        self.memory_dim = int(memory_dim)
        self.ring_dim = self.memory_dim * self.memory_dim
        self.output_dim = int(output_dim)
        self.leak_rates_tuple = _rates(leak_rates)
        self.num_timescales = len(self.leak_rates_tuple)
        if input_rank is not None and input_rank <= 0:
            raise ValueError("input_rank must be positive or None")
        self.input_rank = None if input_rank is None else int(input_rank)
        self.projection_dim = self.input_dim if input_rank is None else int(input_rank)
        self.input_projection: nn.Module = (
            nn.Identity()
            if input_rank is None
            else nn.Linear(self.input_dim, int(input_rank), bias=False)
        )
        self.keys = nn.ModuleList(
            nn.Linear(self.projection_dim, self.memory_dim, bias=False)
            for _ in self.leak_rates_tuple
        )
        self.values = nn.ModuleList(
            nn.Linear(self.projection_dim, self.memory_dim, bias=False)
            for _ in self.leak_rates_tuple
        )
        self.readout = nn.Linear(
            self.num_timescales * self.ring_dim,
            self.output_dim,
            bias=bool(readout_bias),
        )
        if zero_init_readout:
            nn.init.zeros_(self.readout.weight)
            if self.readout.bias is not None:
                nn.init.zeros_(self.readout.bias)

    def zero_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> TemporalMQRState:
        return TemporalMQRState(
            tuple(
                torch.zeros(batch_size, self.ring_dim, device=device, dtype=dtype)
                for _ in self.leak_rates_tuple
            )
        )

    def readout_state(self, state: TemporalMQRState) -> torch.Tensor:
        return self.readout(torch.cat(state.rings, dim=1))

    def forward_step(
        self,
        x: torch.Tensor,
        *,
        state: Optional[TemporalMQRState] = None,
        write_gate: Optional[torch.Tensor] = None,
        promotion: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TemporalMQRState]:
        if promotion is not None:
            raise ValueError("fast-weight control does not accept MQR promotion forcing")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"x must have shape [batch, {self.input_dim}]")
        if state is None:
            state = self.zero_state(x.size(0), device=x.device, dtype=x.dtype)
        gates = _prepare_gate(write_gate, x, self.num_timescales)
        projected_input = self.input_projection(x)
        if self.input_rank is not None:
            projected_input = torch.tanh(projected_input)
        result = []
        scale = 1.0 / math.sqrt(self.memory_dim)
        for index, (key_layer, value_layer, previous, leak) in enumerate(
            zip(self.keys, self.values, state.rings, self.leak_rates_tuple)
        ):
            memory = previous.reshape(x.size(0), self.memory_dim, self.memory_dim)
            carried = (1.0 - leak) * memory
            key = torch.tanh(key_layer(projected_input))
            value = torch.tanh(value_layer(projected_input))
            update = value.unsqueeze(2) * key.unsqueeze(1) * scale
            gate = gates[:, index].reshape(-1, 1, 1)
            next_memory = torch.tanh(carried + gate * update)
            result.append(next_memory.reshape(x.size(0), self.ring_dim))
        next_state = TemporalMQRState(tuple(result))
        return self.readout_state(next_state), next_state

    def max_unitary_error(self) -> float:
        return 0.0

    def max_stochastic_error(self) -> float:
        return 0.0

    def estimated_forward_macs(self) -> int:
        shared_projection = (
            0 if self.input_rank is None else self.input_dim * self.projection_dim
        )
        projections = (
            self.num_timescales
            * 2
            * self.projection_dim
            * self.memory_dim
        )
        outer_updates = self.num_timescales * self.memory_dim * self.memory_dim
        output = self.num_timescales * self.ring_dim * self.output_dim
        return int(shared_projection + projections + outer_updates + output)


class _InactiveSinkhornTransition(nn.Module):
    """Parameter-free full-leak transition with checkpoint-compatible logits.

    A ring with leak one has zero recurrent carry, so its transition is never
    evaluated and must not contribute trainable or frozen parameter capacity.
    Keeping ``logits`` as a persistent buffer preserves the historical state
    dictionary key without charging an unused parameter to Sinkhorn alone.
    """

    def __init__(self, dim: int, *, init_scale: float = 0.01) -> None:
        super().__init__()
        self.dim = int(dim)
        # Consume the same seeded draw as the historical active Sinkhorn
        # module.  This keeps all downstream active transitions and readouts
        # bitwise reproducible while moving the unused tensor out of the
        # parameter/resource accounting.
        self.register_buffer(
            "logits",
            torch.randn(self.dim, self.dim) * float(init_scale),
        )

    def doubly_stochastic_implicit(self) -> torch.Tensor:
        return torch.eye(self.dim, device=self.logits.device, dtype=self.logits.dtype)

    @torch.no_grad()
    def doubly_stochastic_errors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = self.logits.new_zeros(())
        return zero, zero


class MultiTimescaleSinkhornCore(nn.Module):
    """Temporal ring control with learned Sinkhorn-normalized transitions."""

    def __init__(
        self,
        input_dim: int,
        ring_dim: int,
        output_dim: int,
        *,
        leak_rates: Sequence[float] = (1.0, 0.10, 0.02),
        injection_rank: int = 8,
        sinkhorn_iterations: int = 30,
        readout_bias: bool = True,
        zero_init_readout: bool = False,
    ) -> None:
        super().__init__()
        if min(input_dim, ring_dim, output_dim, injection_rank) <= 0:
            raise ValueError("core dimensions and injection_rank must be positive")
        self.input_dim = int(input_dim)
        self.ring_dim = int(ring_dim)
        self.output_dim = int(output_dim)
        self.injection_rank = int(injection_rank)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.leak_rates_tuple = _rates(leak_rates)
        self.num_timescales = len(self.leak_rates_tuple)
        self.input_down = nn.Linear(self.input_dim, self.injection_rank, bias=False)
        self.input_up = nn.ModuleList(
            nn.Linear(self.injection_rank, self.ring_dim, bias=False)
            for _ in self.leak_rates_tuple
        )
        self.sinkhorn_params = nn.ModuleList(
            (
                _InactiveSinkhornTransition(self.ring_dim, init_scale=0.01)
                if leak == 1.0
                else SinkhornDoublyStochasticParam(
                    self.ring_dim,
                    iterations=self.sinkhorn_iterations,
                    init_scale=0.01,
                )
            )
            for leak in self.leak_rates_tuple
        )
        self.readout = nn.Linear(
            self.num_timescales * self.ring_dim,
            self.output_dim,
            bias=bool(readout_bias),
        )
        if zero_init_readout:
            nn.init.zeros_(self.readout.weight)
            if self.readout.bias is not None:
                nn.init.zeros_(self.readout.bias)

    def zero_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TemporalMQRState:
        return TemporalMQRState(
            tuple(
                torch.zeros(batch_size, self.ring_dim, device=device, dtype=dtype)
                for _ in self.leak_rates_tuple
            )
        )

    def readout_state(self, state: TemporalMQRState) -> torch.Tensor:
        return self.readout(torch.cat(state.rings, dim=1))

    def forward_step(
        self,
        x: torch.Tensor,
        *,
        state: Optional[TemporalMQRState] = None,
        write_gate: Optional[torch.Tensor] = None,
        promotion: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TemporalMQRState]:
        if promotion is not None:
            raise ValueError("Sinkhorn control does not accept MQR promotion forcing")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"x must have shape [batch, {self.input_dim}]")
        if state is None:
            state = self.zero_state(x.size(0), device=x.device, dtype=x.dtype)
        gates = _prepare_gate(write_gate, x, self.num_timescales)
        injection = torch.tanh(self.input_down(x))
        result = []
        for index, (up, transition, previous, leak) in enumerate(
            zip(
                self.input_up,
                self.sinkhorn_params,
                state.rings,
                self.leak_rates_tuple,
            )
        ):
            if leak == 1.0:
                carried = torch.zeros_like(previous)
            else:
                H = transition.doubly_stochastic_implicit().to(previous)
                carried = (1.0 - leak) * (previous @ H.transpose(0, 1))
            forcing = gates[:, index : index + 1] * up(injection)
            result.append(torch.tanh(carried + forcing))
        next_state = TemporalMQRState(tuple(result))
        return self.readout_state(next_state), next_state

    def max_unitary_error(self) -> float:
        return 0.0

    @torch.no_grad()
    def max_stochastic_error(self) -> float:
        values = []
        for parameter in self.sinkhorn_params:
            row, column = parameter.doubly_stochastic_errors()
            values.extend((float(row.item()), float(column.item())))
        return max(values, default=0.0)

    def estimated_forward_macs(self) -> int:
        injection = self.input_dim * self.injection_rank
        injection += self.num_timescales * self.injection_rank * self.ring_dim
        active_transitions = sum(leak < 1.0 for leak in self.leak_rates_tuple)
        transition = active_transitions * self.ring_dim * self.ring_dim
        normalization = (
            active_transitions
            * self.sinkhorn_iterations
            * 2
            * self.ring_dim
            * self.ring_dim
        )
        readout = self.num_timescales * self.ring_dim * self.output_dim
        return int(injection + transition + normalization + readout)


class MultiTimescaleLoRAResidualCore(nn.Module):
    """Stateless low-rank residual control with a matched state interface."""

    def __init__(
        self,
        input_dim: int,
        rank: int,
        output_dim: int,
        *,
        num_timescales: int = 3,
        state_dim_per_timescale: int = 4,
    ) -> None:
        super().__init__()
        if min(input_dim, rank, output_dim, num_timescales, state_dim_per_timescale) <= 0:
            raise ValueError("LoRA core dimensions must be positive")
        self.input_dim = int(input_dim)
        self.rank = int(rank)
        self.output_dim = int(output_dim)
        self.num_timescales = int(num_timescales)
        self.ring_dim = int(state_dim_per_timescale)
        self.down = nn.Linear(self.input_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.output_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def zero_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TemporalMQRState:
        return TemporalMQRState(
            tuple(
                torch.zeros(batch_size, self.ring_dim, device=device, dtype=dtype)
                for _ in range(self.num_timescales)
            )
        )

    def readout_state(self, state: TemporalMQRState) -> torch.Tensor:
        reference = state.rings[0]
        return reference.new_zeros((reference.size(0), self.output_dim))

    def forward_step(
        self,
        x: torch.Tensor,
        *,
        state: Optional[TemporalMQRState] = None,
        write_gate: Optional[torch.Tensor] = None,
        promotion: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TemporalMQRState]:
        if promotion is not None:
            raise ValueError("LoRA control does not accept MQR promotion forcing")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"x must have shape [batch, {self.input_dim}]")
        if state is None:
            state = self.zero_state(x.size(0), device=x.device, dtype=x.dtype)
        _prepare_gate(write_gate, x, self.num_timescales)
        return self.up(self.down(x)), state

    def max_unitary_error(self) -> float:
        return 0.0

    def max_stochastic_error(self) -> float:
        return 0.0

    def estimated_forward_macs(self) -> int:
        return int(self.input_dim * self.rank + self.rank * self.output_dim)


class FrozenResidualCoreWrapper(nn.Module):
    """Add one shared frozen base projection to a zero-initialized sidecar core."""

    def __init__(self, adaptive_core: nn.Module) -> None:
        super().__init__()
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
        missing = [name for name in required if not hasattr(adaptive_core, name)]
        if missing:
            raise TypeError(f"adaptive core is missing required attributes: {missing}")
        self.adaptive = adaptive_core
        self.input_dim = int(adaptive_core.input_dim)
        self.ring_dim = int(adaptive_core.ring_dim)
        self.output_dim = int(adaptive_core.output_dim)
        self.num_timescales = int(adaptive_core.num_timescales)
        self.frozen_base = nn.Linear(self.input_dim, self.output_dim, bias=True)
        for parameter in self.frozen_base.parameters():
            parameter.requires_grad_(False)

    @property
    def unitary_params(self):
        if not hasattr(self.adaptive, "unitary_params"):
            raise AttributeError("wrapped core has no unitary parameters")
        return self.adaptive.unitary_params

    def transition_references(self):
        if not hasattr(self.adaptive, "transition_references"):
            raise AttributeError("wrapped core has no transition reference API")
        return self.adaptive.transition_references()

    def transition_drift_diagnostics(self, index: int, reference: torch.Tensor):
        if not hasattr(self.adaptive, "transition_drift_diagnostics"):
            raise AttributeError("wrapped core has no transition drift API")
        return self.adaptive.transition_drift_diagnostics(index, reference)

    def zero_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TemporalMQRState:
        return self.adaptive.zero_state(batch_size, device=device, dtype=dtype)

    def readout_state(self, state: TemporalMQRState) -> torch.Tensor:
        return self.adaptive.readout_state(state)

    def forward_step(
        self,
        x: torch.Tensor,
        *,
        state: Optional[TemporalMQRState] = None,
        write_gate: Optional[torch.Tensor] = None,
        promotion: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TemporalMQRState]:
        residual, next_state = self.adaptive.forward_step(
            x,
            state=state,
            write_gate=write_gate,
            promotion=promotion,
        )
        return self.frozen_base(x) + residual, next_state

    def max_unitary_error(self) -> float:
        return float(self.adaptive.max_unitary_error())

    def max_stochastic_error(self) -> float:
        return float(self.adaptive.max_stochastic_error())

    def estimated_forward_macs(self) -> int:
        adaptive_macs = estimate_core_forward_macs(self.adaptive)
        return int(
            self.input_dim * self.output_dim
            + adaptive_macs
        )


@dataclass(frozen=True)
class AgentResourceAudit:
    method: str
    trainable_parameters: int
    persistent_state_scalars: int
    estimated_forward_macs: int
    utility_parameters: int
    task_parameters: int
    persistent_state_bytes: int = 0
    frozen_parameters: int = 0
    estimated_agent_forward_macs: int = 0
    estimated_task_update_macs: int = 0
    estimated_transition_refresh_macs: int = 0
    allocated_continual_memory_bytes: int = 0
    allocated_update_budget_macs: int = 0

    def to_dict(self) -> Dict[str, int | str]:
        return asdict(self)


def estimate_core_forward_macs(core: nn.Module) -> int:
    """Return a declared per-item multiply-accumulate estimate.

    For learned MQR transitions this includes a conservative cubic Cayley solve
    term.  The estimate is used as a rejection gate, not as a replacement for
    measured latency or profiler FLOPs.
    """

    if hasattr(core, "estimated_forward_macs"):
        return int(core.estimated_forward_macs())
    if not isinstance(core, MultiTimescaleMQR):
        raise TypeError("core does not expose an analytical MAC estimate")
    injection = core.input_dim * core.injection_rank
    injection += core.num_timescales * core.injection_rank * core.ring_dim
    active_carry = sum(float(value.item()) < 1.0 for value in core.leak_rates)
    transition = active_carry * core.ring_dim * core.ring_dim
    if core.transition_mode == "orthogonal":
        transition = active_carry * 2 * core.cyclic_givens_layers * core.ring_dim
    readout = core.total_state_dim * core.output_dim
    # H is cached for no-grad inference and invalidated by tensor version
    # counters after a coordinate update.  Refresh cost is charged separately
    # to the online update rather than to every prediction.
    return int(injection + transition + readout)


def estimate_transition_refresh_macs(core: nn.Module) -> int:
    """Conservative cost of rebuilding a learned transition after one update."""

    adaptive = getattr(core, "adaptive", None)
    if isinstance(adaptive, nn.Module):
        return estimate_transition_refresh_macs(adaptive)
    if not isinstance(core, MultiTimescaleMQR):
        return 0
    if core.transition_mode != "unistochastic" or not core.learn_transitions:
        return 0
    active = len(core.active_transition_indices)
    if core.transition_structure == "dense_cayley":
        return int(active * 8 * core.ring_dim**3)
    return int(
        active
        * (2 * core.cyclic_givens_layers + 1)
        * core.ring_dim**2
    )


def audit_agent_resources(
    agent: TemporalUtilityMQRAgent,
    method: str,
    *,
    allocated_continual_memory_bytes: int = 0,
) -> AgentResourceAudit:
    """Audit model, recurrent-state, and charged online-update resources.

    ``estimated_forward_macs`` retains the historical core-only definition so
    old result files remain reproducible.  ``estimated_agent_forward_macs``
    additionally charges the four Go heads and the utility critic.  The update
    estimate is a conservative three-forward charge.  When a common continual
    memory budget is supplied, one read and one write of every float32 slot are
    charged to ``allocated_update_budget_macs``.  This is an allocation gate;
    experiments must still report actual OGD/replay occupancy and latency.
    """

    if allocated_continual_memory_bytes < 0:
        raise ValueError("allocated_continual_memory_bytes must be non-negative")
    reference = next(agent.parameters())
    zero = agent.core.zero_state(1, device=reference.device, dtype=reference.dtype)
    state_scalars = sum(value.numel() for value in zero.rings)
    state_bytes = sum(value.numel() * value.element_size() for value in zero.rings)
    core_macs = estimate_core_forward_macs(agent.core)
    head_macs = agent.latent_dim * (2 * agent.points + 2)
    if agent.spatial_skip_heads is not None:
        channels = agent.spatial_skip_heads.channels
        head_macs += agent.points * 3 * channels * 9
        head_macs += 2 * agent.points * channels
        head_macs += 2 * (channels + agent.spatial_skip_heads.extra_dim)
    utility_macs = (
        agent.utility_gate.input_dim * agent.utility_gate.rank
        + agent.utility_gate.rank
    )
    agent_forward_macs = int(core_macs + head_macs + utility_macs)
    transition_refresh_macs = estimate_transition_refresh_macs(agent.core)
    update_macs = 3 * agent_forward_macs + transition_refresh_macs
    memory_slots = math.ceil(int(allocated_continual_memory_bytes) / 4)
    return AgentResourceAudit(
        method=str(method),
        trainable_parameters=sum(
            parameter.numel() for parameter in agent.parameters() if parameter.requires_grad
        ),
        persistent_state_scalars=int(state_scalars),
        estimated_forward_macs=core_macs,
        utility_parameters=agent.utility_parameter_count,
        task_parameters=agent.task_parameter_count,
        persistent_state_bytes=int(state_bytes),
        frozen_parameters=sum(
            parameter.numel()
            for parameter in agent.parameters()
            if not parameter.requires_grad
        ),
        estimated_agent_forward_macs=agent_forward_macs,
        estimated_task_update_macs=update_macs,
        estimated_transition_refresh_macs=transition_refresh_macs,
        allocated_continual_memory_bytes=int(allocated_continual_memory_bytes),
        allocated_update_budget_macs=int(update_macs + 2 * memory_slots),
    )


def matched_competitive_resource_gate(
    audits: Sequence[AgentResourceAudit],
    *,
    ratio_limit: float = 1.05,
) -> Dict[str, object]:
    """Fail-closed 5-axis resource gate for competitive online studies.

    The gate requires matched trainable parameters, exact recurrent-state
    bytes, matched total forward/update MAC allocations, identical frozen-base
    parameter counts, and an identical continual-memory capacity.  It does not
    treat unused padding as memory usage; actual OGD and replay occupancy must
    be recorded separately by the experiment.
    """

    if len(audits) < 2:
        raise ValueError("at least two resource audits are required")
    if ratio_limit < 1.0:
        raise ValueError("ratio_limit must be at least one")

    def ratio(values: Sequence[int]) -> float:
        low = min(values)
        high = max(values)
        if low == high == 0:
            return 1.0
        return float("inf") if low <= 0 else high / low

    fields = {
        "trainable_parameter_ratio": [item.trainable_parameters for item in audits],
        "recurrent_state_byte_ratio": [item.persistent_state_bytes for item in audits],
        "agent_forward_mac_ratio": [item.estimated_agent_forward_macs for item in audits],
        "allocated_update_mac_ratio": [
            item.allocated_update_budget_macs for item in audits
        ],
        "frozen_parameter_ratio": [item.frozen_parameters for item in audits],
        "continual_capacity_ratio": [
            item.allocated_continual_memory_bytes for item in audits
        ],
    }
    ratios = {name: ratio(values) for name, values in fields.items()}
    exact_fields = (
        "recurrent_state_byte_ratio",
        "frozen_parameter_ratio",
        "continual_capacity_ratio",
    )
    checks = {
        "trainable_parameters_matched": (
            ratios["trainable_parameter_ratio"] <= ratio_limit
        ),
        "recurrent_state_bytes_exact": all(
            math.isclose(ratios[name], 1.0, rel_tol=0.0, abs_tol=0.0)
            for name in exact_fields[:1]
        ),
        "agent_forward_macs_matched": (
            ratios["agent_forward_mac_ratio"] <= ratio_limit
        ),
        "allocated_update_macs_matched": (
            ratios["allocated_update_mac_ratio"] <= ratio_limit
        ),
        "frozen_parameters_exact": math.isclose(
            ratios["frozen_parameter_ratio"], 1.0, rel_tol=0.0, abs_tol=0.0
        ),
        "continual_capacity_exact": math.isclose(
            ratios["continual_capacity_ratio"], 1.0, rel_tol=0.0, abs_tol=0.0
        ),
    }
    return {
        **ratios,
        **checks,
        "ratio_limit": float(ratio_limit),
        "all_resources_matched": bool(all(checks.values())),
    }


def matched_resource_gate(
    audits: Sequence[AgentResourceAudit],
    *,
    parameter_ratio_limit: float = 1.05,
    state_ratio_limit: float = 1.05,
    mac_ratio_limit: float = 1.05,
) -> Dict[str, float | bool]:
    if len(audits) < 2:
        raise ValueError("at least two resource audits are required")
    if min(parameter_ratio_limit, state_ratio_limit, mac_ratio_limit) < 1.0:
        raise ValueError("resource ratio limits must be at least one")

    def ratio(values: Sequence[int]) -> float:
        low = min(values)
        high = max(values)
        return float("inf") if low <= 0 else high / low

    parameter_ratio = ratio([item.trainable_parameters for item in audits])
    state_ratio = ratio([item.persistent_state_scalars for item in audits])
    mac_ratio = ratio([item.estimated_forward_macs for item in audits])
    return {
        "parameter_ratio": parameter_ratio,
        "state_ratio": state_ratio,
        "estimated_mac_ratio": mac_ratio,
        "parameter_ratio_limit": float(parameter_ratio_limit),
        "state_ratio_limit": float(state_ratio_limit),
        "estimated_mac_ratio_limit": float(mac_ratio_limit),
        "parameters_matched": parameter_ratio <= parameter_ratio_limit,
        "state_matched": state_ratio <= state_ratio_limit,
        "estimated_macs_matched": mac_ratio <= mac_ratio_limit,
        "all_resources_matched": bool(
            parameter_ratio <= parameter_ratio_limit
            and state_ratio <= state_ratio_limit
            and mac_ratio <= mac_ratio_limit
        ),
    }
