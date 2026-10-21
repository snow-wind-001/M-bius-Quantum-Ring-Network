"""Experimental Go write/read layouts with identical learned tensor budgets.

Spatial states use one 16-dimensional ring per board point and timescale.
This increases state memory, not parameter count. The previous-observation
control stores visible observations only and is not an orthogonal memory.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from .go_memory import GoResearchCore
from .go_policy import SignedQueryGoAgent
from .temporal import TemporalMQRState
from .constraint_transport import TerminalGradient


class ReadableGoCore(GoResearchCore):
    """Preserve parameter shapes while changing where observation writes go."""

    def __init__(self, *args: Any, write_layout: str = "shared",
                 geometry: str = "orth", **kwargs: Any) -> None:
        if write_layout not in ("shared", "grouped", "spatial", "convolution", "previous"):
            raise ValueError("unknown write layout")
        if geometry not in ("orth", "similar"):
            raise ValueError("unknown recurrent geometry")
        super().__init__(*args, **kwargs)
        if self.transition_mode != "orthogonal" or self.state_activation != "none":
            raise ValueError("readable study requires signed linear state transitions")
        if self.injection_activation != "tanh" or self.num_timescales != 6:
            raise ValueError("readable study requires six rings with tanh writes")
        self.write_layout, self.geometry = write_layout, geometry
        self.points = self.board_size ** 2
        self.spatial_slots = self.points if write_layout in ("spatial", "convolution") else 1
        if write_layout == "grouped" and self.board_dim % 3:
            raise ValueError("grouped input needs three equal feature partitions")
        scale = torch.linspace(-math.log(2) / 2, math.log(2) / 2,
                               self.ring_dim, dtype=torch.float64).exp()
        self.register_buffer("coordinate_scale", scale)

    @property
    def state_widths(self) -> Tuple[int, ...]:
        widths = (self.spatial_slots * self.ring_dim,) * self.num_timescales
        return widths + ((self.board_dim + 1,) * 2 if self.write_layout == "previous" else ())

    def get_extra_state(self) -> Dict[str, Any]:
        return {**super().get_extra_state(), "readable_version": 1,
                "write_layout": self.write_layout, "geometry": self.geometry}

    def zero_state(self, batch_size: int, *, device: torch.device,
                   dtype: torch.dtype) -> TemporalMQRState:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        return TemporalMQRState(tuple(torch.zeros(batch_size, width, device=device, dtype=dtype)
                                      for width in self.state_widths))

    def _validate_state(self, state: TemporalMQRState, batch_size: int) -> None:
        if len(state.rings) != len(self.state_widths):
            raise ValueError("state layout differs from the configured memory")
        for value, width in zip(state.rings, self.state_widths):
            if (value.shape != (batch_size, width) or not value.is_floating_point()
                    or torch.is_complex(value) or not bool(torch.isfinite(value).all())):
                raise ValueError("state must have finite real values in the configured coordinates")

    def _validate_input(self, x: torch.Tensor) -> None:
        if (x.ndim != 2 or x.size(1) != self.input_dim or not x.is_floating_point()
                or torch.is_complex(x) or not bool(torch.isfinite(x).all())):
            raise ValueError("observations must be finite [batch, input_dim] tensors")

    def input_forcing(self, x: torch.Tensor) -> torch.Tensor:
        """Return writes [B,6,slots*16]; no query or feedback enters a write."""
        self._validate_input(x)
        board = self.board_features(x)
        weight = self.input_down.weight
        if self.write_layout == "grouped":
            # Position-major planes, then metadata. Three 102-feature blocks
            # use disjoint pieces of the same 8x306 learned tensor.
            position_major = board[:, :3 * self.points].reshape(-1, 3, self.points).transpose(1, 2).flatten(1)
            inputs = torch.cat((position_major, board[:, 3 * self.points:]), dim=1).reshape(-1, 3, self.board_dim // 3)
            weights = weight.reshape(weight.size(0), 3, self.board_dim // 3)
            encoded = torch.einsum("bgi,kgi->bgk", inputs, weights).tanh()
            values = [self.write_scales[j].to(x) * projection(encoded[:, j // 2])
                      for j, projection in enumerate(self.input_up)]
        elif self.write_layout in ("spatial", "convolution"):
            planes = board[:, :3 * self.points].reshape(-1, 3, self.points)
            if self.write_layout == "spatial":
                local = torch.einsum("bcp,kcp->bpk", planes,
                                     weight[:, :3 * self.points].reshape(weight.size(0), 3, self.points))
            else:
                # The same 8x(3P+6) tensor becomes a shared relative-position
                # filter plus metadata coefficients, with no dummy parameters.
                size = self.board_size
                low, high = (size - 1) // 2, size // 2
                padded = F.pad(planes.reshape(-1, 3, size, size), (low, high, low, high))
                local = F.conv2d(padded, weight[:, :3 * self.points].reshape(
                    weight.size(0), 3, size, size)).flatten(2).transpose(1, 2)
            metadata = F.linear(board[:, 3 * self.points:], weight[:, 3 * self.points:])
            encoded = (local + metadata[:, None]).tanh()
            values = [(self.write_scales[j].to(x) * projection(encoded)).flatten(1)
                      for j, projection in enumerate(self.input_up)]
        else:
            encoded = F.linear(board, weight).tanh()
            values = [self.write_scales[j].to(x) * projection(encoded)
                      for j, projection in enumerate(self.input_up)]
        return torch.stack(values, dim=1)

    def carry(self, index: int, value: torch.Tensor, x: torch.Tensor,
              *, transpose: bool = False) -> torch.Tensor:
        """Apply the actual state Jacobian (or transpose) to [B,slots*16]."""
        rho = (1 - self.leak_rates[index]).to(value)
        if float(rho) == 0:
            return torch.zeros_like(value)
        reshaped = value.reshape(x.size(0), self.spatial_slots, self.ring_dim)
        offsets = self.orthogonal_offsets(index, x)
        if offsets is not None:
            offsets = offsets[:, None]
        scale = self.coordinate_scale.to(value)
        if self.geometry == "similar":
            reshaped = reshaped * scale if transpose else reshaped / scale
        rotated = self.unitary_params[index].apply_orthogonal(
            reshaped, transpose=transpose, angle_offsets=offsets)
        if self.geometry == "similar":
            rotated = rotated / scale if transpose else rotated * scale
        return (rho * rotated).flatten(1)

    def readout_state(self, state: TemporalMQRState) -> torch.Tensor:
        """Use the unchanged 96-to-16 global head after spatial averaging."""
        self._validate_state(state, state.rings[0].size(0))
        vector = torch.cat([value.reshape(value.size(0), self.spatial_slots, self.ring_dim).mean(1)
                            for value in state.rings[:self.num_timescales]], dim=1)
        return self.readout(vector)

    def forward_step(self, x: torch.Tensor, *, state: Optional[TemporalMQRState] = None,
                     write_gate: Optional[torch.Tensor] = None,
                     promotion: Optional[torch.Tensor] = None,
                     return_certificate: bool = False):
        if return_certificate:
            raise ValueError("use the explicit readable-layout Jacobian checks")
        if promotion is not None:
            raise ValueError("promotion is outside this write-layout study")
        forcing = self.input_forcing(x)
        if state is None:
            state = self.zero_state(x.size(0), device=x.device, dtype=x.dtype)
        self._validate_state(state, x.size(0))
        gates = self._prepare_write_gate(write_gate, x)
        rings = tuple(self.carry(j, previous.to(x), x) + gates[:, j:j + 1] * forcing[:, j]
                      for j, previous in enumerate(state.rings[:self.num_timescales]))
        if self.write_layout == "previous":
            # Two raw registers: the previous and the just-observed board.
            # A validity flag distinguishes a real empty board from no history.
            current = torch.cat((self.board_features(x), x.new_ones(x.size(0), 1)), dim=1)
            rings += (state.rings[-1].to(x), current)
        result = TemporalMQRState(rings)
        return self.readout_state(result), result


class ReadableGoAgent(SignedQueryGoAgent):
    """Read the ring at each queried position or pool it as an ablation."""

    def __init__(self, *args: Any, spatial_read: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.core, ReadableGoCore) or self.history_slots:
            raise ValueError("readable queries require their explicit state layout")
        self.spatial_read = bool(spatial_read)

    def get_extra_state(self) -> Dict[str, Any]:
        return {**super().get_extra_state(), "readable_query": {"version": 1, "spatial_read": self.spatial_read}}

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        if state.get("readable_query") != {"version": 1, "spatial_read": self.spatial_read}:
            raise ValueError("checkpoint query layout differs")
        super().set_extra_state(state)

    def _state_for(self, key, x: torch.Tensor) -> TemporalMQRState:
        state = self._stream_states.get(key)
        if state is None:
            return self._zero_state(x)
        self.core._validate_state(state, 1)
        return state.detached()

    def _restore_state(self, values: Sequence[torch.Tensor]) -> TemporalMQRState:
        reference = next(self.parameters())
        state = TemporalMQRState(tuple(value.detach().to(reference) for value in values))
        self.core._validate_state(state, 1)
        return state

    def _query_features(self, latent: torch.Tensor, state: TemporalMQRState,
                        x: torch.Tensor):
        self.core._validate_state(state, x.size(0))
        if self.core.spatial_slots == 1 or not self.spatial_read:
            pooled = TemporalMQRState(tuple(
                h.reshape(x.size(0), self.core.spatial_slots, self.ring_dim).mean(1)
                for h in state.rings[:self.core.num_timescales]))
            hidden, correction, addressing, memory_read = super()._query_features(latent, pooled, x)
            if self.core.write_layout == "previous":
                previous = state.rings[-2]
                colors = previous[:, :2 * self.points].reshape(-1, 2, self.points)
                black, white = colors[:, 0], colors[:, 1]
                geometry = self.spatial_skip_heads.geometry_planes.to(x).expand(x.size(0), -1, -1, -1)
                coordinates = geometry[:, 1:3].flatten(2)
                tokens = torch.stack((black, white, black - white,
                    previous[:, 2 * self.points:3 * self.points], coordinates[:, 0], coordinates[:, 1],
                    black + white, torch.ones_like(black)), dim=-1) * previous[:, -1:, None]
                added = F.linear(tokens, self.point_correction.weight[:, self.query_dim:])
                added = torch.stack((added[..., 0] - added[..., 0].mean(1, keepdim=True), added[..., 1]), dim=-1)
                correction = correction + added
                memory_read = memory_read + tokens
            return hidden, correction, addressing, memory_read
        base = self.spatial_skip_heads
        hidden = base.spatial_features(x[:, :self.board_dim])
        hidden = hidden * (1 + self.ring_channel_gain(latent.tanh())[:, :, None, None])
        geometry = base.geometry_planes.to(x).expand(x.size(0), -1, -1, -1)
        query = (self.point_query(hidden) + self.geometry_query(geometry)).flatten(2).transpose(1, 2)
        query = F.normalize(query, dim=-1, eps=1e-6)
        keys = F.normalize(self.ring_keys, dim=-1, eps=1e-6)
        addressing = (query @ keys.T).tanh()
        rings = torch.stack([h.reshape(x.size(0), self.points, self.ring_dim) for h in state.rings], dim=2)
        contents = torch.stack([rings.roll(shift, dims=-1) for shift in range(4)], dim=-1).flatten(2, 3)
        values = self.ring_value(contents)
        ring_read = torch.einsum("bps,bpsq->bpq", addressing, values) / math.sqrt(keys.size(0))
        correction = self.point_correction(torch.cat((ring_read, torch.zeros_like(ring_read)), dim=-1))
        correction = torch.stack((correction[..., 0] - correction[..., 0].mean(1, keepdim=True),
                                  correction[..., 1]), dim=-1)
        return hidden, correction, addressing, ring_read


@torch.enable_grad()
def readable_terminal_gradients(agent: ReadableGoAgent, features: torch.Tensor,
                               objective: Callable, *, backend: str = "transport") -> TerminalGradient:
    """Current-parameter terminal gradients through complete finite histories.

    The state pullback is rho*R.T, or rho*S^-T*R.T*S.T for the similarity
    control, separately at each spatial point. Raw memory registers use their
    exact shift Jacobian. Local parameter VJPs include every write and rotation.
    """
    if type(agent) is not ReadableGoAgent or type(agent.core) is not ReadableGoCore:
        raise TypeError("this adjoint is derived only for the readable study")
    if backend not in ("transport", "autograd"):
        raise ValueError("unknown derivative backend")
    if features.ndim == 2:
        features = features[:, None]
    if features.ndim != 3 or min(features.shape[:2]) < 1 or features.size(2) != agent.input_dim:
        raise ValueError("features must be nonempty [time,batch,input_dim]")
    active = [(name, p) for name, p in agent._named_task_parameters() if p.requires_grad]
    if len({id(p) for _, p in active}) != len(active):
        raise ValueError("shared parameters must appear exactly once")
    parameters = [p for _, p in active]
    reference = parameters[0]
    xs = features.detach().to(reference)
    version = int(agent.online_parameter_version)
    state = agent.core.zero_state(xs.size(1), device=xs.device, dtype=xs.dtype)
    prefix = []
    if backend == "transport":
        with torch.no_grad():
            for x in xs:
                prefix.append(state.detached())
                _, state = agent.core.forward_step(x, state=state)
        terminal = TemporalMQRState(tuple(h.detach().requires_grad_(True) for h in state.rings))
    else:
        for x in xs:
            _, state = agent.core.forward_step(x, state=state)
        terminal = state
    values = objective(agent.readout_state(terminal, features=xs[-1])).reshape(-1)
    if not values.numel() or not bool(torch.isfinite(values).all()):
        raise ValueError("terminal objectives must be nonempty and finite")
    widths = [p.numel() for p in parameters]
    offsets = [0]
    for width in widths:
        offsets.append(offsets[-1] + width)
    jacobian = reference.new_zeros(values.numel(), offsets[-1])
    q = [reference.new_zeros(values.numel(), *h.shape) for h in terminal.rings]
    inputs = parameters + (list(terminal.rings) if backend == "transport" else [])
    for row in range(values.numel()):
        gradients = torch.autograd.grad(values[row], inputs, allow_unused=True,
                                        retain_graph=row + 1 < values.numel())
        for index, gradient in enumerate(gradients[:len(parameters)]):
            if gradient is not None:
                jacobian[row, offsets[index]:offsets[index + 1]] = gradient.detach().flatten()
        for index, gradient in enumerate(gradients[len(parameters):]):
            if gradient is not None:
                q[index][row] = gradient.detach()
    result_values = values.detach().clone()
    del values, terminal, state, inputs, gradients
    error = 0.
    if backend == "transport":
        core_indices = [i for i, (name, _) in enumerate(active) if name.startswith("core.")]
        core_parameters = [parameters[i] for i in core_indices]
        for step in range(xs.size(0) - 1, -1, -1):
            _, current = agent.core.forward_step(xs[step], state=prefix[step])
            differentiable = [i for i, h in enumerate(current.rings) if h.requires_grad]
            for row in range(result_values.numel()):
                gradients = torch.autograd.grad(
                    [current.rings[i] for i in differentiable], core_parameters,
                    grad_outputs=[q[i][row] for i in differentiable],
                    allow_unused=True, retain_graph=row + 1 < result_values.numel())
                for index, gradient in zip(core_indices, gradients):
                    if gradient is not None:
                        jacobian[row, offsets[index]:offsets[index + 1]].add_(gradient.detach().flatten())
            with torch.no_grad():
                pulled = []
                repeated_x = xs[step].repeat(result_values.numel(), 1)
                for index in range(agent.core.num_timescales):
                    value = q[index]
                    previous = agent.core.carry(index, value.flatten(0, 1), repeated_x,
                                                transpose=True).reshape_as(value)
                    if agent.core.geometry == "orth":
                        rho = (1 - agent.core.leak_rates[index]).to(value)
                        error = max(error, float((previous.norm(dim=-1) - rho * value.norm(dim=-1)).abs().max()))
                    pulled.append(previous)
                if agent.core.write_layout == "previous":
                    pulled += [torch.zeros_like(q[-2]), q[-2]]
                q = pulled
            del current
    if int(agent.online_parameter_version) != version or not bool(torch.isfinite(jacobian).all()):
        raise RuntimeError("invalid current-parameter behavior derivative")
    return TerminalGradient(result_values, jacobian, tuple(n for n, _ in active),
        {"backend": backend, "history_steps": xs.size(0), "objectives": len(result_values),
         "state_trace_bytes": sum(h.numel() * h.element_size() for s in prefix for h in s.rings),
         "max_adjoint_norm_error": error, "parameter_version": version})
