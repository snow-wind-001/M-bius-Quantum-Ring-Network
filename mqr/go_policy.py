"""Signed position queries over the unchanged orthogonal multi-ring core."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .agent import GoMultiHeadOutput
from .go_memory import PositionQueryGoAgent
from .temporal import TemporalMQRState


class SignedQueryGoAgent(PositionQueryGoAgent):
    """Read signed ring contents with geometry-dependent, non-convex weights.

    For P points and S ring slots, A = tanh(normalize(Q) normalize(K)^T),
    and the read is A V / sqrt(S). Unlike uniform softmax attention this can
    express signed position contrasts. Orthogonality concerns the recurrent
    transition and constraint transport, not this deliberately nonlinear head.
    """

    def __init__(self, input_dim: int, *, adapt_spatial: bool = False,
                 spatial_head_only: bool = False, **kwargs: Any) -> None:
        if spatial_head_only and not adapt_spatial:
            raise ValueError("spatial_head_only requires adapt_spatial")
        super().__init__(input_dim, **kwargs)
        self.adapt_spatial = bool(adapt_spatial)
        self.spatial_head_only = bool(spatial_head_only)
        self.geometry_query = nn.Conv2d(4, self.query_dim, 1, bias=False)
        # A zero output matrix blocks all query/key gradients at initialization.
        nn.init.normal_(self.point_correction.weight, std=0.05)
        self.spatial_skip_heads.requires_grad_(self.adapt_spatial)
        if self.spatial_head_only:
            self.spatial_skip_heads.local.requires_grad_(False)
            if self.spatial_skip_heads.local_second is not None:
                self.spatial_skip_heads.local_second.requires_grad_(False)

    def query_components(
        self, latent: torch.Tensor, state: TemporalMQRState, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return spatial features [B,C,H,W], corrections [B,P,2], weights [B,P,S]."""
        base = self.spatial_skip_heads
        hidden = base.spatial_features(x[:, :self.board_dim])
        hidden = hidden * (1 + self.ring_channel_gain(latent.tanh())[:, :, None, None])
        geometry = base.geometry_planes.to(x).expand(x.size(0), -1, -1, -1)
        query = (self.point_query(hidden) + self.geometry_query(geometry)).flatten(2).transpose(1, 2)
        query = F.normalize(query, dim=-1, eps=1e-6)
        keys = F.normalize(self.ring_keys, dim=-1, eps=1e-6)
        addressing = (query @ keys.T).tanh()
        rings = torch.stack(state.rings, dim=1)
        contents = torch.stack([rings.roll(shift, dims=-1) for shift in range(4)], dim=-1)
        ring_read = addressing @ self.ring_value(contents.flatten(1, 2)) / math.sqrt(keys.size(0))
        external_read = torch.zeros_like(ring_read)
        if self.history_slots:
            slots = x[:, self.board_dim:].reshape(x.size(0), self.history_slots, 2 * self.points + 2)
            colors = slots[..., :2 * self.points].reshape(
                x.size(0), self.history_slots, 2, self.points,
            ).transpose(-1, -2)
            coords = geometry[:, 1:3].flatten(2).transpose(1, 2)
            coords = coords[:, None].expand(-1, self.history_slots, -1, -1)
            time = slots[..., -1:].unsqueeze(-2).expand(-1, -1, self.points, -1)
            tokens = torch.cat((colors, coords, time), dim=-1).flatten(1, 2)
            valid = slots[..., -2:-1].expand(-1, -1, self.points).flatten(1) > 0
            scores = query @ self.external_key(tokens).transpose(-1, -2) / math.sqrt(self.query_dim)
            weights = scores.masked_fill(~valid[:, None], torch.finfo(scores.dtype).min).softmax(-1)
            external_read = (weights * valid[:, None]) @ self.external_value(tokens)
        correction = self.point_correction(torch.cat((ring_read, external_read), dim=-1))
        # Placement's common mode is unidentifiable under conditional softmax.
        correction = torch.stack((
            correction[..., 0] - correction[..., 0].mean(dim=1, keepdim=True),
            correction[..., 1],
        ), dim=-1)
        return hidden, correction, addressing

    def _query_output(
        self, latent: torch.Tensor, state: TemporalMQRState, x: torch.Tensor,
        *, include_query: bool = True,
    ) -> GoMultiHeadOutput:
        base = self.spatial_skip_heads
        hidden, correction, _ = self.query_components(latent, state, x)
        if not include_query:
            correction = torch.zeros_like(correction)
        pooled = torch.cat((hidden.mean(dim=(2, 3)), x[:, 3 * self.points:self.board_dim]), dim=1)
        latent = latent.tanh()
        return GoMultiHeadOutput(
            base.placement(hidden).flatten(1) + correction[..., 0],
            base.legality(hidden).flatten(1) + correction[..., 1],
            base.pass_decision(pooled).squeeze(1) + self.pass_head(latent).squeeze(1),
            (base.value(pooled).squeeze(1) + self.value_head(latent).squeeze(1)).tanh(),
            latent, legality_policy_scale=self.legality_policy_scale,
        )

    def _named_task_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return super()._named_task_parameters() + [
            (f"geometry_query.{name}", p) for name, p in self.geometry_query.named_parameters()
        ]

    def get_extra_state(self) -> Dict[str, Any]:
        result = super().get_extra_state()
        result["signed_query"] = {"version": 1, "adapt_spatial": self.adapt_spatial}
        if self.spatial_head_only:
            result["signed_query"]["spatial_head_only"] = True
        return result

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        expected = {"version": 1, "adapt_spatial": self.adapt_spatial}
        if self.spatial_head_only:
            expected["spatial_head_only"] = True
        if state.get("signed_query") != expected:
            raise ValueError("checkpoint signed query or spatial adaptation configuration differs")
        super().set_extra_state(state)
