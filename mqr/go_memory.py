"""Experimental spatial queries over signed rings and bounded raw Go history.

External slots store observations, never labels, legal masks or simulator
position_history. Their black/white coordinates are independent of parameters.
Queries are learned; stored values are not silently reinterpreted after SGD.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .agent import GoMultiHeadOutput, GoSpatialSkipHeads, TemporalUtilityMQRAgent
from .go import GoBoard
from .go_agent import GoVectorEncoder
from .go_online import GoOnlineSession
from .temporal import MultiTimescaleMQR, TemporalMQRState


def fixed_color_features(x: torch.Tensor, board_size: int) -> torch.Tensor:
    """Convert [own, opponent, previous, scalars, ...] to black/white coordinates."""
    points = board_size ** 2
    if x.ndim != 2 or x.size(1) < 3 * points + 6:
        raise ValueError("fixed colors require the exact, unprojected Go representation")
    black_turn = x[:, 3 * points:3 * points + 1]
    own, opponent = x[:, :points], x[:, points:2 * points]
    black = black_turn * own + (1 - black_turn) * opponent
    white = black_turn * opponent + (1 - black_turn) * own
    return torch.cat((black, white, x[:, 2 * points:]), dim=1)


class GoHistoryEncoder(GoVectorEncoder):
    """Pure board encoder; callers explicitly supply oldest-first memory slots."""

    def __init__(self, board_size: int, *, history_slots: int = 0) -> None:
        if isinstance(history_slots, bool) or not isinstance(history_slots, int) or history_slots < 0:
            raise ValueError("history_slots must be a non-negative integer")
        base_dim = 3 * board_size ** 2 + 6
        super().__init__(board_size, base_dim, projection_mode="identity")
        self.history_slots = history_slots
        self.slot_dim = 2 * self.points + 2  # fixed-color board, validity, observation time
        self.output_dim = base_dim + history_slots * self.slot_dim

    def encode_board(self, board: GoBoard, *, require_encoder_grad: bool = False) -> torch.Tensor:
        if require_encoder_grad:
            raise ValueError("history observations have no encoder parameters")
        x = self.exact_features(board)
        return torch.cat((x, x.new_zeros(1, self.history_slots * self.slot_dim)), dim=1)

    def with_history(self, board: GoBoard, slots: torch.Tensor) -> torch.Tensor:
        if slots.shape != (self.history_slots, self.slot_dim):
            raise ValueError("history slots do not match the encoder")
        x = self.exact_features(board)
        return torch.cat((x, slots.to(x).reshape(1, -1)), dim=1)


class GoObservationMemory:
    """FIFO raw observations with a fixed byte budget and explicit eviction."""

    def __init__(self, encoder: GoHistoryEncoder) -> None:
        self.encoder = encoder
        self.slots = encoder.projection.new_zeros(encoder.history_slots, encoder.slot_dim)
        self.writes = 0
        self.evictions = 0

    def write(self, board: GoBoard) -> None:
        if self.encoder.history_slots == 0:
            return
        exact = fixed_color_features(self.encoder.exact_features(board), board.size)
        points = board.size ** 2
        row = torch.cat((exact[0, :2 * points], exact.new_tensor([
            1.0, len(board.move_history) / (2 * points),
        ])))
        if self.writes >= self.encoder.history_slots:
            self.slots[:-1] = self.slots[1:].clone()
            self.evictions += 1
        index = min(self.writes, self.encoder.history_slots - 1)
        self.slots[index].copy_(row)
        self.writes += 1

    def clear(self) -> None:
        self.slots.zero_()
        self.writes = self.evictions = 0

    @property
    def storage_bytes(self) -> int:
        return self.slots.numel() * self.slots.element_size()

    def state_dict(self) -> Dict[str, Any]:
        return {"slots": self.slots.clone(), "writes": self.writes, "evictions": self.evictions}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        slots = state["slots"]
        if slots.shape != self.slots.shape or not bool(torch.isfinite(slots).all()):
            raise ValueError("invalid observation memory slots")
        writes, evictions = int(state["writes"]), int(state["evictions"])
        if writes < 0 or evictions != max(0, writes - self.encoder.history_slots):
            raise ValueError("invalid observation memory counters")
        self.slots.copy_(slots.to(self.slots))
        self.writes, self.evictions = writes, evictions


class GoResearchCore(MultiTimescaleMQR):
    """Multi-ring core with stable forcing and optional input-only rotations.

    External history is read separately, so it cannot bypass a ring-only
    ablation through the input projection. R(x) depends on the current raw
    board, never on the recurrent state or any future feedback.
    """

    def __init__(
        self, input_dim: int, ring_dim: int, output_dim: int, *,
        board_size: int, fixed_colors: bool = True, conditional_scale: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if not math.isfinite(conditional_scale) or conditional_scale < 0:
            raise ValueError("conditional_scale must be finite and non-negative")
        base_dim = 3 * board_size ** 2 + 6
        if input_dim < base_dim:
            raise ValueError("GoResearchCore requires exact board planes")
        super().__init__(base_dim, ring_dim, output_dim, **kwargs)
        self.input_dim = int(input_dim)
        self.board_dim = base_dim
        self.board_size = int(board_size)
        self.fixed_colors = bool(fixed_colors)
        self.conditional_scale = float(conditional_scale)
        if conditional_scale and self.transition_mode != "orthogonal":
            raise ValueError("conditional rotations require signed orthogonal propagation")
        self.angle_controllers = nn.ModuleDict()
        if conditional_scale:
            for index in self.active_transition_indices:
                controller = nn.Linear(base_dim, self.cyclic_givens_layers * (ring_dim // 2), bias=False)
                nn.init.zeros_(controller.weight)
                self.angle_controllers[str(index)] = controller

    def board_features(self, x: torch.Tensor) -> torch.Tensor:
        board = x[:, :self.board_dim]
        return fixed_color_features(board, self.board_size) if self.fixed_colors else board

    def _inject(self, x: torch.Tensor) -> torch.Tensor:
        return super()._inject(self.board_features(x))

    def orthogonal_offsets(self, index: int, x: torch.Tensor) -> Optional[torch.Tensor]:
        if str(index) not in self.angle_controllers:
            return None
        angles = self.angle_controllers[str(index)](self.board_features(x)).tanh()
        return self.conditional_scale * angles.reshape(x.size(0), self.cyclic_givens_layers, self.ring_dim // 2)

    def get_extra_state(self) -> Dict[str, Any]:
        return {"version": 1, "board_size": self.board_size, "input_dim": self.input_dim,
                "fixed_colors": self.fixed_colors, "conditional_scale": self.conditional_scale}

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        if state != self.get_extra_state():
            raise ValueError("checkpoint memory coordinates or conditional rotation configuration differ")

    def conditioned_transition(self, index: int, x: torch.Tensor) -> torch.Tensor:
        """Return actual [batch, N, N] operator for auditing at supplied inputs."""
        offsets = self.orthogonal_offsets(index, x)
        if offsets is None:
            return self.transition_matrix(index, device=x.device, dtype=x.dtype).expand(x.size(0), -1, -1)
        identity = torch.eye(self.ring_dim, device=x.device, dtype=x.dtype).expand(x.size(0), -1, -1)
        return self.unitary_params[index].apply_orthogonal(
            identity, angle_offsets=offsets[:, None],
        ).transpose(-2, -1)

    def transition_references(self) -> List[torch.Tensor]:
        if self.conditional_scale:
            raise ValueError("conditional R(x) requires input-indexed drift probes; a fixed-matrix guard is invalid")
        return super().transition_references()


class PositionQueryGoAgent(TemporalUtilityMQRAgent):
    """Let each board point query raw ring slots and past observation tokens.

    Neighboring ring scalars form four-channel contents. Learned slot keys
    expose their addresses to a geometry-aware spatial query. External tokens
    contain black, white, row, column, and observation time. All attention is
    a readout operation; it is not asserted to be orthogonal.
    """

    def __init__(
        self, input_dim: int, *, channels: int = 12, query_dim: int = 8,
        history_slots: int = 0, spatial_geometry: bool = True, spatial_depth: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_dim, spatial_skip_channels=0, **kwargs)
        if query_dim <= 0 or history_slots < 0:
            raise ValueError("query_dim must be positive and history_slots non-negative")
        self.board_dim = 3 * self.points + 6
        self.history_slots = int(history_slots)
        if input_dim != self.board_dim + history_slots * (2 * self.points + 2):
            raise ValueError("input dimension does not match history slot layout")
        self.spatial_skip_heads = GoSpatialSkipHeads(
            self.board_dim, self.board_size, channels, geometry=spatial_geometry, depth=spatial_depth,
        )
        self.query_dim = int(query_dim)
        self.point_query = nn.Conv2d(channels, query_dim, 1)
        self.ring_channel_gain = nn.Linear(self.latent_dim, channels, bias=False)
        nn.init.zeros_(self.ring_channel_gain.weight)
        self.ring_keys = nn.Parameter(torch.randn(self.core.num_timescales * self.ring_dim, query_dim) / math.sqrt(query_dim))
        self.ring_value = nn.Linear(4, query_dim, bias=False)
        self.external_key = nn.Linear(5, query_dim, bias=False)
        self.external_value = nn.Linear(5, query_dim, bias=False)
        self.point_correction = nn.Linear(2 * query_dim, 2, bias=False)
        nn.init.zeros_(self.point_correction.weight)
        for module in (self.placement_head, self.legality_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for module in (self.pass_head, self.value_head):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        if not history_slots:
            for module in (self.external_key, self.external_value):
                module.requires_grad_(False)

    def _query_output(self, latent: torch.Tensor, state: TemporalMQRState, x: torch.Tensor) -> GoMultiHeadOutput:
        base = self.spatial_skip_heads
        board = x[:, :self.board_dim]
        hidden = base.spatial_features(board)
        hidden = hidden * (1 + self.ring_channel_gain(latent.tanh())[:, :, None, None])
        query = self.point_query(hidden).flatten(2).transpose(1, 2)
        ring = torch.stack(state.rings, dim=1)
        contents = torch.stack([ring.roll(shift, dims=-1) for shift in range(4)], dim=-1)
        values = self.ring_value(contents.flatten(1, 2))
        attention = (query @ self.ring_keys.T / math.sqrt(self.query_dim)).softmax(dim=-1)
        ring_read = attention @ values
        external_read = torch.zeros_like(ring_read)
        if self.history_slots:
            slots = x[:, self.board_dim:].reshape(x.size(0), self.history_slots, 2 * self.points + 2)
            colors = slots[..., :2 * self.points].reshape(x.size(0), self.history_slots, 2, self.points).transpose(-1, -2)
            coords = base.geometry_planes[0, 1:3].to(x).flatten(1).T
            coords = coords.expand(x.size(0), self.history_slots, -1, -1)
            age = slots[..., -1:].unsqueeze(-2).expand(-1, -1, self.points, -1)
            tokens = torch.cat((colors, coords, age), dim=-1).flatten(1, 2)
            valid = slots[..., -2:-1].expand(-1, -1, self.points).reshape(x.size(0), -1) > 0
            scores = query @ self.external_key(tokens).transpose(-1, -2) / math.sqrt(self.query_dim)
            # All-empty memory produces exactly zero, with no NaNs or fake slot.
            weights = scores.masked_fill(~valid[:, None], torch.finfo(scores.dtype).min).softmax(dim=-1) * valid[:, None]
            external_read = weights @ self.external_value(tokens)
        correction = self.point_correction(torch.cat((ring_read, external_read), dim=-1))
        pooled = torch.cat((hidden.mean(dim=(2, 3)), board[:, 3 * self.points:]), dim=1)
        latent = latent.tanh()
        return GoMultiHeadOutput(
            base.placement(hidden).flatten(1) + correction[..., 0],
            base.legality(hidden).flatten(1) + correction[..., 1],
            base.pass_decision(pooled).squeeze(1) + self.pass_head(latent).squeeze(1),
            (base.value(pooled).squeeze(1) + self.value_head(latent).squeeze(1)).tanh(),
            latent, legality_policy_scale=self.legality_policy_scale,
        )

    def _transition(self, x: torch.Tensor, state: TemporalMQRState, *, slow_write: bool) -> Tuple[GoMultiHeadOutput, TemporalMQRState]:
        gates = x.new_ones(x.size(0), self.core.num_timescales)
        if not slow_write:
            gates[:, 1:] = 0
        latent, next_state = self.core.forward_step(x, state=state, write_gate=gates)
        return self._query_output(latent, next_state, x), next_state

    def _causal_state_output(self, state: TemporalMQRState, previous_input: Optional[torch.Tensor]) -> GoMultiHeadOutput:
        ref = state.rings[0]
        x = ref.new_zeros(ref.size(0), self.input_dim) if previous_input is None else previous_input.to(ref)
        return self._query_output(self.core.readout_state(state), state, x)

    def readout_state(self, state: TemporalMQRState, *, features: Optional[torch.Tensor] = None) -> GoMultiHeadOutput:
        """Position queries require the corresponding board observation."""
        if features is None:
            raise ValueError("position queries require features for the current board")
        return self._query_output(self.core.readout_state(state), state, features)

    def _named_task_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        result = super()._named_task_parameters() + [("ring_keys", self.ring_keys)]
        for name in ("point_query", "ring_channel_gain", "ring_value", "external_key", "external_value", "point_correction"):
            result.extend((f"{name}.{key}", value) for key, value in getattr(self, name).named_parameters())
        return result


class HistoryGoSession(GoOnlineSession):
    """One game owns one raw memory; queries precede writes and teacher labels."""

    def __init__(self, agent: TemporalUtilityMQRAgent, encoder: GoHistoryEncoder, **kwargs: Any) -> None:
        super().__init__(agent, encoder, **kwargs)
        self.observation_memory = GoObservationMemory(encoder)

    def _encode_observation(self, board: GoBoard) -> torch.Tensor:
        return self.encoder.with_history(board, self.observation_memory.slots)

    def _after_observation(self, board: GoBoard) -> None:
        self.observation_memory.write(board)

    def reset_game(self) -> None:
        super().reset_game()
        self.observation_memory.clear()

    @property
    def online_tensor_bytes(self) -> int:
        return super().online_tensor_bytes + self.observation_memory.storage_bytes

    def state_dict(self) -> Dict[str, Any]:
        result = super().state_dict()
        result["observation_memory"] = self.observation_memory.state_dict()
        return result

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # Validate memory first, so a dimension error cannot partially load weights.
        candidate = GoObservationMemory(self.encoder)
        candidate.load_state_dict(state["observation_memory"])
        super().load_state_dict(state)
        self.observation_memory = candidate
