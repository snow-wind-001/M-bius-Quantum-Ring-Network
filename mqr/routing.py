"""Residual MQR address routing with content-preserving slot memory.

This module separates *what* is remembered from *where* it is read.  Content
is written to physical slots without recurrent averaging.  Only a probability
distribution over slot addresses is transported by a residual doubly
stochastic operator

``T_epsilon = (1 - epsilon) I + epsilon T``.

The sparse cyclic implementation composes local two-slot unistochastic
factors.  Each factor is induced by a real Givens rotation, while their product
is guaranteed doubly stochastic (a product of unistochastic matrices need not
itself be unistochastic).  Dense and block Cayley variants retain the exact
``|U|^2`` construction where that distinction matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unitary import CayleyUnistochasticParam


@dataclass(frozen=True)
class RoutedMemoryState:
    """Physical content and logical address state.

    ``content`` has shape ``[batch, slots, content_dim]`` and ``address`` has
    shape ``[batch, slots]``.  Routing never changes ``content``; an explicit
    write transaction is the only operation allowed to mutate a content slot.
    """

    content: torch.Tensor
    address: torch.Tensor

    def detached(self) -> "RoutedMemoryState":
        return RoutedMemoryState(
            content=self.content.detach(),
            address=self.address.detach(),
        )


@dataclass(frozen=True)
class WriteBudgetState:
    """Per-stream cumulative write ledger.

    The hard allowance after ``t`` observations is ``ceil(rate * t)``.  This
    permits one initial write without allowing the long-run budget to drift.
    """

    steps: torch.Tensor
    writes: torch.Tensor

    def detached(self) -> "WriteBudgetState":
        return WriteBudgetState(self.steps.detach(), self.writes.detach())


@dataclass(frozen=True)
class AddressRoutingCost:
    """Analytic operation and workspace audit for one routing configuration."""

    route_family: str
    trainable_parameters: int
    address_state_bytes: int
    content_state_bytes: int
    forward_macs: int
    refresh_macs: int
    amortized_refresh_macs: float
    peak_workspace_bytes: int
    refresh_interval: int
    estimate_scope: str = "analytic_upper_bound_not_wall_clock"

    @property
    def total_state_bytes(self) -> int:
        return self.address_state_bytes + self.content_state_bytes

    @property
    def amortized_total_macs(self) -> float:
        return float(self.forward_macs) + float(self.amortized_refresh_macs)


class ResidualAddressRouter(nn.Module):
    """Route slot addresses while retaining an explicit identity channel.

    Args:
        slots: Number of addresses.
        route_family: ``identity``, ``cyclic_givens``, ``block_cayley``,
            ``dense_cayley``, ``local_permutation``, or
            ``learnable_sparse_permutation``.
        epsilon: Fraction of the routed path in ``T_epsilon``.
        learn_epsilon: Learn a bounded epsilon.  ``epsilon_floor`` prevents an
            exactly closed route from also closing its gradient.
        givens_layers: Alternating nearest-neighbour matchings on the cycle.
        block_size: Cayley solve size for ``block_cayley``.
        permutation_shift: Signed cyclic shift for ``local_permutation``.
        sparse_permutation_probability: Initial move probability for each
            learnable local swap factor.  This control has the same factor
            graph and parameter count as ``cyclic_givens``; it replaces
            ``sin(theta)^2`` by ``sigmoid(logit)``.

    Address tensors are row distributions and advance as ``a_next = a T``.
    ``transpose=True`` applies ``T^T`` for context-reversal ablations; it is not
    claimed to invert a diffuse stochastic transition.
    """

    _FAMILIES = {
        "identity",
        "cyclic_givens",
        "block_cayley",
        "dense_cayley",
        "local_permutation",
        "learnable_sparse_permutation",
    }

    def __init__(
        self,
        slots: int,
        *,
        route_family: str = "cyclic_givens",
        epsilon: float = 0.9,
        learn_epsilon: bool = False,
        epsilon_floor: float = 1e-3,
        givens_layers: int = 2,
        givens_base_angle: float = 0.35,
        block_size: int = 8,
        permutation_shift: int = 1,
        sparse_permutation_probability: float = 0.25,
        cayley_coordinate_mode: str = "minimal",
    ) -> None:
        super().__init__()
        if isinstance(slots, bool) or int(slots) != slots or int(slots) < 2:
            raise ValueError("slots must be an integer of at least two")
        route_family = str(route_family)
        if route_family not in self._FAMILIES:
            raise ValueError(f"unknown route_family: {route_family}")
        if not math.isfinite(float(epsilon)) or not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError("epsilon must be finite and in [0, 1]")
        if not math.isfinite(float(epsilon_floor)) or not 0.0 <= float(epsilon_floor) < 0.5:
            raise ValueError("epsilon_floor must be finite and in [0, 0.5)")
        if learn_epsilon and not epsilon_floor < float(epsilon) < 1.0 - epsilon_floor:
            raise ValueError(
                "a learned epsilon must lie strictly inside its bounded interval"
            )
        if isinstance(givens_layers, bool) or int(givens_layers) != givens_layers or int(givens_layers) <= 0:
            raise ValueError("givens_layers must be a positive integer")
        if not math.isfinite(float(givens_base_angle)):
            raise ValueError("givens_base_angle must be finite")
        if isinstance(block_size, bool) or int(block_size) != block_size or int(block_size) <= 0:
            raise ValueError("block_size must be a positive integer")
        if isinstance(permutation_shift, bool) or int(permutation_shift) != permutation_shift:
            raise ValueError("permutation_shift must be an integer")
        if not math.isfinite(float(sparse_permutation_probability)) or not (
            0.0 < float(sparse_permutation_probability) < 1.0
        ):
            raise ValueError(
                "sparse_permutation_probability must lie strictly in (0, 1)"
            )
        if route_family in {"cyclic_givens", "learnable_sparse_permutation"} and int(slots) % 2:
            raise ValueError(
                f"{route_family} requires an even number of slots"
            )
        if route_family == "block_cayley" and int(slots) % int(block_size):
            raise ValueError("slots must be divisible by block_size")

        self.slots = int(slots)
        self.route_family = route_family
        self.learn_epsilon = bool(learn_epsilon)
        self.epsilon_floor = float(epsilon_floor)
        self.givens_layers = int(givens_layers)
        self.block_size = int(block_size)
        self.permutation_shift = int(permutation_shift) % self.slots
        self.sparse_permutation_probability = float(
            sparse_permutation_probability
        )

        if self.learn_epsilon:
            span = 1.0 - 2.0 * self.epsilon_floor
            normalized = (float(epsilon) - self.epsilon_floor) / span
            logit = math.log(normalized) - math.log1p(-normalized)
            self.epsilon_logit = nn.Parameter(torch.tensor(logit))
            self.register_buffer("_fixed_epsilon", torch.empty(0), persistent=False)
        else:
            self.register_parameter("epsilon_logit", None)
            self.register_buffer(
                "_fixed_epsilon", torch.tensor(float(epsilon), dtype=torch.float32)
            )

        if self.route_family in {
            "cyclic_givens",
            "learnable_sparse_permutation",
        }:
            pair_i, pair_j = self._cyclic_pairs()
            self.register_buffer("_pair_i", pair_i, persistent=False)
            self.register_buffer("_pair_j", pair_j, persistent=False)
        else:
            self.register_buffer("_pair_i", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("_pair_j", torch.empty(0, dtype=torch.long), persistent=False)

        if self.route_family == "cyclic_givens":
            angles = torch.full(
                (self.givens_layers, self.slots // 2),
                float(givens_base_angle),
            )
            self.givens_angles = nn.Parameter(angles)
        else:
            self.register_parameter("givens_angles", None)

        if self.route_family == "learnable_sparse_permutation":
            probability = self.sparse_permutation_probability
            logit = math.log(probability) - math.log1p(-probability)
            self.sparse_permutation_logits = nn.Parameter(
                torch.full((self.givens_layers, self.slots // 2), logit)
            )
        else:
            self.register_parameter("sparse_permutation_logits", None)

        if self.route_family == "block_cayley":
            self.cayley_blocks = nn.ModuleList(
                [
                    CayleyUnistochasticParam(
                        self.block_size, coordinate_mode=cayley_coordinate_mode
                    )
                    for _ in range(self.slots // self.block_size)
                ]
            )
        elif self.route_family == "dense_cayley":
            self.cayley_blocks = nn.ModuleList(
                [
                    CayleyUnistochasticParam(
                        self.slots, coordinate_mode=cayley_coordinate_mode
                    )
                ]
            )
        else:
            self.cayley_blocks = nn.ModuleList()

    def _cyclic_pairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        left_layers = []
        right_layers = []
        for layer in range(self.givens_layers):
            if layer % 2 == 0:
                left = list(range(0, self.slots, 2))
                right = list(range(1, self.slots, 2))
            else:
                left = list(range(1, self.slots, 2))
                right = list(range(2, self.slots, 2)) + [0]
            left_layers.append(left)
            right_layers.append(right)
        return (
            torch.tensor(left_layers, dtype=torch.long),
            torch.tensor(right_layers, dtype=torch.long),
        )

    def epsilon(self, reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the bounded residual routing coefficient."""

        if self.learn_epsilon:
            span = 1.0 - 2.0 * self.epsilon_floor
            value = self.epsilon_floor + span * torch.sigmoid(self.epsilon_logit)
        else:
            value = self._fixed_epsilon
        if reference is not None:
            value = value.to(device=reference.device, dtype=reference.dtype)
        return value

    @staticmethod
    def _validate_address(address: torch.Tensor, slots: int) -> None:
        if not isinstance(address, torch.Tensor) or address.dim() != 2:
            raise ValueError("address must be a rank-two tensor")
        if address.size(1) != slots:
            raise ValueError(f"address must have shape [batch, {slots}]")
        if torch.is_complex(address) or not address.is_floating_point():
            raise TypeError("address must be real floating point")
        if not bool(torch.isfinite(address).all()):
            raise ValueError("address must contain only finite values")

    def _apply_cyclic_givens(
        self, address: torch.Tensor, *, transpose: bool
    ) -> torch.Tensor:
        layers = (
            range(self.givens_layers - 1, -1, -1)
            if transpose
            else range(self.givens_layers)
        )
        result = address
        for layer in layers:
            left = self._pair_i[layer]
            right = self._pair_j[layer]
            theta = self.givens_angles[layer].to(address)
            keep = torch.cos(theta).square()
            move = torch.sin(theta).square()
            left_values = result.index_select(1, left)
            right_values = result.index_select(1, right)
            next_left = keep * left_values + move * right_values
            next_right = move * left_values + keep * right_values
            result = result.index_copy(1, left, next_left)
            result = result.index_copy(1, right, next_right)
        return result

    def _apply_sparse_permutation(
        self, address: torch.Tensor, *, transpose: bool
    ) -> torch.Tensor:
        """Apply differentiable local stay/swap factors on the ring graph.

        Each factor is a two-by-two doubly stochastic matrix.  At a hard
        endpoint it is a permutation; during learning it is the direct
        transition-space control for the Givens relation
        ``move = sin(theta)^2``.
        """

        layers = (
            range(self.givens_layers - 1, -1, -1)
            if transpose
            else range(self.givens_layers)
        )
        result = address
        for layer in layers:
            left = self._pair_i[layer]
            right = self._pair_j[layer]
            move = torch.sigmoid(
                self.sparse_permutation_logits[layer].to(address)
            )
            keep = 1.0 - move
            left_values = result.index_select(1, left)
            right_values = result.index_select(1, right)
            next_left = keep * left_values + move * right_values
            next_right = move * left_values + keep * right_values
            result = result.index_copy(1, left, next_left)
            result = result.index_copy(1, right, next_right)
        return result

    def _apply_cayley(self, address: torch.Tensor, *, transpose: bool) -> torch.Tensor:
        if self.route_family == "dense_cayley":
            transition = self.cayley_blocks[0].unistochastic().to(address)
            return address @ (transition.T if transpose else transition)
        pieces = []
        for index, parameter in enumerate(self.cayley_blocks):
            start = index * self.block_size
            block = address[:, start : start + self.block_size]
            transition = parameter.unistochastic().to(address)
            pieces.append(block @ (transition.T if transpose else transition))
        return torch.cat(pieces, dim=1)

    def base_route(
        self, address: torch.Tensor, *, transpose: bool = False
    ) -> torch.Tensor:
        """Apply the non-residual route ``T`` to a row distribution."""

        self._validate_address(address, self.slots)
        if self.route_family == "identity":
            return address
        if self.route_family == "local_permutation":
            shift = -self.permutation_shift if transpose else self.permutation_shift
            return torch.roll(address, shifts=shift, dims=1)
        if self.route_family == "cyclic_givens":
            return self._apply_cyclic_givens(address, transpose=transpose)
        if self.route_family == "learnable_sparse_permutation":
            return self._apply_sparse_permutation(address, transpose=transpose)
        return self._apply_cayley(address, transpose=transpose)

    def forward(
        self,
        address: torch.Tensor,
        *,
        transpose: bool = False,
        identity_override: bool = False,
    ) -> torch.Tensor:
        """Apply ``(1-epsilon) I + epsilon T`` to an address distribution."""

        self._validate_address(address, self.slots)
        epsilon = self.epsilon(address)
        routed = (
            address
            if identity_override
            else self.base_route(address, transpose=transpose)
        )
        return (1.0 - epsilon) * address + epsilon * routed

    def transition_matrix(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        transpose: bool = False,
        identity_override: bool = False,
    ) -> torch.Tensor:
        """Materialize the effective row transition for diagnostics only."""

        reference = next(self.parameters(), self._fixed_epsilon)
        target_device = reference.device if device is None else device
        target_dtype = reference.dtype if dtype is None else dtype
        identity = torch.eye(self.slots, device=target_device, dtype=target_dtype)
        return self.forward(
            identity,
            transpose=transpose,
            identity_override=identity_override,
        )

    @torch.no_grad()
    def stochastic_diagnostics(self) -> Dict[str, Any]:
        transition = self.transition_matrix(dtype=torch.float64)
        row_error = (transition.sum(dim=1) - 1.0).abs().amax()
        column_error = (transition.sum(dim=0) - 1.0).abs().amax()
        minimum = transition.amin()
        identity = torch.eye(self.slots, device=transition.device, dtype=transition.dtype)
        return {
            "route_family": self.route_family,
            "epsilon": float(self.epsilon().item()),
            "row_sum_max_error": float(row_error.item()),
            "column_sum_max_error": float(column_error.item()),
            "minimum_entry": float(minimum.item()),
            "identity_channel_weight": 1.0 - float(self.epsilon().item()),
            "transition_distance_from_identity_fro": float(
                torch.linalg.matrix_norm(transition - identity, ord="fro").item()
            ),
            "globally_unistochastic_certified": self.route_family
            in {"identity", "block_cayley", "dense_cayley", "local_permutation"},
            "local_unistochastic_factors": (
                self.givens_layers
                if self.route_family
                in {"cyclic_givens", "learnable_sparse_permutation"}
                else 0
            ),
            "direct_transition_parameterization": self.route_family
            == "learnable_sparse_permutation",
        }

    def cost_profile(
        self,
        *,
        batch_size: int = 1,
        content_dim: int = 0,
        dtype_bytes: int = 4,
        refresh_interval: int = 1,
    ) -> AddressRoutingCost:
        """Return explicit forward, refresh, amortized, and peak estimates.

        MAC counts cover address transport and route materialization/solve, not
        the task encoder, utility critic, or optimizer.  Complex Cayley solves
        use a conservative real-equivalent factor of eight.  These analytic
        estimates must be accompanied by measured latency in formal studies.
        """

        for name, value in (
            ("batch_size", batch_size),
            ("dtype_bytes", dtype_bytes),
            ("refresh_interval", refresh_interval),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(content_dim, bool) or int(content_dim) != content_dim or int(content_dim) < 0:
            raise ValueError("content_dim must be a non-negative integer")
        batch = int(batch_size)
        scalar_bytes = int(dtype_bytes)
        interval = int(refresh_interval)
        parameters = sum(value.numel() for value in self.parameters() if value.requires_grad)

        if self.route_family == "identity":
            forward_macs = 0
            refresh_macs = 0
            workspace_scalars = batch * self.slots
        elif self.route_family == "local_permutation":
            # ``roll`` is an index/data movement rather than a multiply-add.
            forward_macs = 0
            refresh_macs = 0
            workspace_scalars = batch * self.slots
        elif self.route_family in {
            "cyclic_givens",
            "learnable_sparse_permutation",
        }:
            forward_macs = 4 * batch * self.slots * self.givens_layers
            refresh_macs = 4 * self.slots * self.givens_layers
            workspace_scalars = 3 * batch * self.slots + 2 * self.slots
        elif self.route_family == "block_cayley":
            blocks = self.slots // self.block_size
            forward_macs = batch * self.slots * self.block_size
            refresh_macs = 8 * blocks * self.block_size**3
            workspace_scalars = 4 * self.block_size**2 + 2 * batch * self.block_size
        else:
            forward_macs = batch * self.slots**2
            refresh_macs = 8 * self.slots**3
            workspace_scalars = 4 * self.slots**2 + 2 * batch * self.slots
        # Residual interpolation costs one multiply-add per address entry.
        forward_macs += 2 * batch * self.slots
        return AddressRoutingCost(
            route_family=self.route_family,
            trainable_parameters=int(parameters),
            address_state_bytes=batch * self.slots * scalar_bytes,
            content_state_bytes=batch * self.slots * int(content_dim) * scalar_bytes,
            forward_macs=int(forward_macs),
            refresh_macs=int(refresh_macs),
            amortized_refresh_macs=float(refresh_macs) / float(interval),
            peak_workspace_bytes=int(workspace_scalars * scalar_bytes),
            refresh_interval=interval,
        )


class ContextualAddressRouterBank(nn.Module):
    """A bounded bank of independently updated address routes.

    Context identifiers are opaque integer keys.  They select a route but do
    not reveal its topology.  Gradients reach only routers used by the current
    batch, which makes A -> B -> A retention an auditable structural property
    instead of relying on optimizer luck.

    Args:
        contexts: Number of preallocated context routes.
        slots: Address count in every route.
        router_kwargs: Keyword arguments forwarded to
            :class:`ResidualAddressRouter`.
    """

    def __init__(
        self,
        contexts: int,
        slots: int,
        **router_kwargs: Any,
    ) -> None:
        super().__init__()
        if isinstance(contexts, bool) or int(contexts) != contexts or int(contexts) <= 0:
            raise ValueError("contexts must be a positive integer")
        self.contexts = int(contexts)
        self.slots = int(slots)
        self.router_kwargs: Mapping[str, Any] = dict(router_kwargs)
        self.routers = nn.ModuleList(
            [
                ResidualAddressRouter(self.slots, **dict(router_kwargs))
                for _ in range(self.contexts)
            ]
        )

    def _validate_context_ids(
        self, context_ids: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        if not isinstance(context_ids, torch.Tensor) or context_ids.dim() != 1:
            raise ValueError("context_ids must be a rank-one tensor")
        if context_ids.numel() != batch_size:
            raise ValueError("context_ids must align with the address batch")
        if context_ids.dtype == torch.bool or context_ids.is_floating_point() or torch.is_complex(context_ids):
            raise TypeError("context_ids must use an integer dtype")
        parsed = context_ids.to(dtype=torch.long)
        if bool(((parsed < 0) | (parsed >= self.contexts)).any()):
            raise ValueError("context_ids contain an out-of-range route index")
        return parsed

    def forward(
        self,
        address: torch.Tensor,
        context_ids: torch.Tensor,
        *,
        transpose: bool = False,
        identity_override: bool = False,
    ) -> torch.Tensor:
        ResidualAddressRouter._validate_address(address, self.slots)
        parsed = self._validate_context_ids(context_ids, address.size(0))
        result = torch.empty_like(address)
        for context_index, router in enumerate(self.routers):
            selected = parsed == context_index
            if bool(selected.any()):
                result[selected] = router(
                    address[selected],
                    transpose=transpose,
                    identity_override=identity_override,
                )
        return result

    def transition_matrix(
        self,
        context_index: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not 0 <= int(context_index) < self.contexts:
            raise ValueError("context_index is out of range")
        return self.routers[int(context_index)].transition_matrix(**kwargs)

    def cost_profile(
        self,
        *,
        active_contexts_per_token: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Report persistent-bank and active-route costs separately."""

        if active_contexts_per_token != 1:
            raise ValueError("exactly one route may be active per token")
        active = self.routers[0].cost_profile(**kwargs)
        return {
            "contexts": self.contexts,
            "persistent_trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "active_route": active,
        }


class RoutedSlotMemory(nn.Module):
    """Content-preserving slots controlled by a residual address router.

    ``route_projection="straight_through_top1"`` commits a discrete address
    after scoring it with the residual stochastic route.  This prevents the
    address itself from asymptotically diffusing while retaining a surrogate
    gradient through the route scores.  The proposal remains doubly
    stochastic; the committed projection is explicitly nonlinear and is not
    covered by linear mass-preservation claims.
    """

    def __init__(
        self,
        slots: int,
        content_dim: int,
        *,
        router: Optional[ResidualAddressRouter] = None,
        read_mode: str = "straight_through_top1",
        route_projection: str = "straight_through_top1",
    ) -> None:
        super().__init__()
        if isinstance(content_dim, bool) or int(content_dim) != content_dim or int(content_dim) <= 0:
            raise ValueError("content_dim must be a positive integer")
        if read_mode not in {"soft", "hard", "straight_through_top1"}:
            raise ValueError('read_mode must be "soft", "hard", or "straight_through_top1"')
        if route_projection not in {"none", "hard", "straight_through_top1"}:
            raise ValueError(
                'route_projection must be "none", "hard", or '
                '"straight_through_top1"'
            )
        self.slots = int(slots)
        self.content_dim = int(content_dim)
        self.router = router if router is not None else ResidualAddressRouter(slots)
        if self.router.slots != self.slots:
            raise ValueError("router and memory must use the same slot count")
        self.read_mode = str(read_mode)
        self.route_projection = str(route_projection)

    def zero_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        initial_slot: int = 0,
    ) -> RoutedMemoryState:
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not 0 <= int(initial_slot) < self.slots:
            raise ValueError("initial_slot is out of range")
        content = torch.zeros(
            int(batch_size), self.slots, self.content_dim, device=device, dtype=dtype
        )
        address = F.one_hot(
            torch.full(
                (int(batch_size),), int(initial_slot), device=device, dtype=torch.long
            ),
            num_classes=self.slots,
        ).to(dtype=dtype)
        return RoutedMemoryState(content, address)

    def _validate_state(self, state: RoutedMemoryState) -> None:
        if not isinstance(state, RoutedMemoryState):
            raise TypeError("state must be a RoutedMemoryState")
        if state.content.dim() != 3 or state.content.shape[1:] != (
            self.slots,
            self.content_dim,
        ):
            raise ValueError(
                f"content must be [batch, {self.slots}, {self.content_dim}]"
            )
        if state.address.shape != (state.content.size(0), self.slots):
            raise ValueError(f"address must be [batch, {self.slots}]")
        if state.content.device != state.address.device or state.content.dtype != state.address.dtype:
            raise ValueError("content and address must share device and dtype")
        if torch.is_complex(state.content) or not state.content.is_floating_point():
            raise TypeError("memory state must be real floating point")
        if not bool(torch.isfinite(state.content).all()) or not bool(
            torch.isfinite(state.address).all()
        ):
            raise ValueError("memory state contains NaN or Inf")

    def _address_weights(
        self, address: torch.Tensor, *, mode: Optional[str] = None
    ) -> torch.Tensor:
        selected = self.read_mode if mode is None else str(mode)
        if selected not in {"soft", "hard", "straight_through_top1"}:
            raise ValueError("unknown address read mode")
        if selected == "soft":
            return address
        hard = F.one_hot(address.argmax(dim=1), num_classes=self.slots).to(address)
        if selected == "hard":
            return hard
        return hard + address - address.detach()

    def read(
        self,
        state: RoutedMemoryState,
        *,
        address: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Read one physical slot; straight-through mode keeps route gradients."""

        self._validate_state(state)
        query = state.address if address is None else address.to(state.address)
        if query.shape != state.address.shape:
            raise ValueError("read address has an incompatible shape")
        weights = self._address_weights(query, mode=mode)
        return torch.einsum("bs,bsd->bd", weights, state.content)

    def write(
        self,
        state: RoutedMemoryState,
        value: torch.Tensor,
        *,
        gate: Optional[torch.Tensor] = None,
        address: Optional[torch.Tensor] = None,
    ) -> RoutedMemoryState:
        """Write a value to exactly one forward slot.

        The forward pass uses a hard slot, so unrelated contents are bitwise
        unchanged.  A straight-through address supplies gradients to the route.
        """

        self._validate_state(state)
        expected = (state.content.size(0), self.content_dim)
        if value.shape != expected:
            raise ValueError(f"value must have shape {expected}")
        value = value.to(state.content)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("value must contain only finite values")
        if gate is None:
            strength = torch.ones(
                state.content.size(0), 1, device=value.device, dtype=value.dtype
            )
        else:
            if gate.dim() == 1:
                strength = gate.unsqueeze(1)
            elif gate.shape == (state.content.size(0), 1):
                strength = gate
            else:
                raise ValueError("gate must have shape [batch] or [batch, 1]")
            strength = strength.to(value)
            if not bool(torch.isfinite(strength).all()) or not bool(
                ((strength >= 0.0) & (strength <= 1.0)).all()
            ):
                raise ValueError("gate values must be finite and in [0, 1]")
        target = state.address if address is None else address.to(state.address)
        if target.shape != state.address.shape:
            raise ValueError("write address has an incompatible shape")
        weights = self._address_weights(target, mode="straight_through_top1")
        update = strength.unsqueeze(2) * weights.unsqueeze(2)
        next_content = (1.0 - update) * state.content + update * value.unsqueeze(1)
        return RoutedMemoryState(next_content, state.address)

    def route(
        self,
        state: RoutedMemoryState,
        *,
        transpose: bool = False,
        identity_override: bool = False,
    ) -> RoutedMemoryState:
        """Advance only the address; content is returned without modification."""

        self._validate_state(state)
        next_address = self.router(
            state.address,
            transpose=transpose,
            identity_override=identity_override,
        )
        if self.route_projection != "none":
            hard = F.one_hot(
                next_address.argmax(dim=1), num_classes=self.slots
            ).to(next_address)
            if self.route_projection == "hard":
                next_address = hard
            else:
                next_address = hard + next_address - next_address.detach()
        return RoutedMemoryState(state.content, next_address)

    def forward_step(
        self,
        state: RoutedMemoryState,
        *,
        write_value: Optional[torch.Tensor] = None,
        write_gate: Optional[torch.Tensor] = None,
        transpose: bool = False,
        identity_override: bool = False,
    ) -> Tuple[torch.Tensor, RoutedMemoryState]:
        """Predict from ``state`` before an optional write and route commit."""

        prediction = self.read(state)
        committed = state
        if write_value is not None:
            committed = self.write(committed, write_value, gate=write_gate)
        elif write_gate is not None:
            raise ValueError("write_gate requires write_value")
        committed = self.route(
            committed,
            transpose=transpose,
            identity_override=identity_override,
        )
        return prediction, committed

    def state_bytes(self, *, batch_size: int = 1, dtype_bytes: int = 4) -> int:
        if batch_size <= 0 or dtype_bytes <= 0:
            raise ValueError("batch_size and dtype_bytes must be positive")
        return int(batch_size) * self.slots * (self.content_dim + 1) * int(dtype_bytes)


class BalancedAdvantageReplay:
    """Bounded replay that samples positive and non-positive utility equally."""

    def __init__(self, feature_dim: int, horizons: int, *, capacity_per_sign: int = 512) -> None:
        if feature_dim <= 0 or horizons <= 0 or capacity_per_sign <= 0:
            raise ValueError("feature_dim, horizons, and capacity_per_sign must be positive")
        self.feature_dim = int(feature_dim)
        self.horizons = int(horizons)
        self.capacity_per_sign = int(capacity_per_sign)
        self._positive: list[Tuple[torch.Tensor, torch.Tensor]] = []
        self._negative: list[Tuple[torch.Tensor, torch.Tensor]] = []

    def add(self, features: torch.Tensor, advantages: torch.Tensor) -> None:
        if features.dim() != 2 or features.size(1) != self.feature_dim:
            raise ValueError(f"features must be [batch, {self.feature_dim}]")
        if advantages.dim() == 1 and self.horizons == 1:
            advantages = advantages.unsqueeze(1)
        if advantages.shape != (features.size(0), self.horizons):
            raise ValueError(f"advantages must be [batch, {self.horizons}]")
        for feature, target in zip(features.detach().cpu(), advantages.detach().cpu()):
            destination = self._positive if float(target.mean().item()) > 0.0 else self._negative
            destination.append((feature.clone(), target.clone()))
            if len(destination) > self.capacity_per_sign:
                del destination[0 : len(destination) - self.capacity_per_sign]

    def sample(
        self,
        batch_size: int,
        *,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 1:
            raise ValueError("batch_size must be greater than one")
        if not self._positive or not self._negative:
            raise RuntimeError("both advantage signs are required before balanced sampling")
        positive_count = batch_size // 2
        negative_count = batch_size - positive_count

        def choose(values: list[Tuple[torch.Tensor, torch.Tensor]], count: int) -> list[Tuple[torch.Tensor, torch.Tensor]]:
            indices = torch.randint(len(values), (count,), generator=generator).tolist()
            return [values[index] for index in indices]

        selected = choose(self._positive, positive_count) + choose(self._negative, negative_count)
        permutation = torch.randperm(len(selected), generator=generator).tolist()
        selected = [selected[index] for index in permutation]
        features = torch.stack([item[0] for item in selected])
        targets = torch.stack([item[1] for item in selected])
        if device is not None:
            features = features.to(device)
            targets = targets.to(device)
        return features, targets

    @property
    def counts(self) -> Dict[str, int]:
        return {"positive": len(self._positive), "non_positive": len(self._negative)}

    @property
    def byte_size(self) -> int:
        scalar_count = (len(self._positive) + len(self._negative)) * (
            self.feature_dim + self.horizons
        )
        return 4 * scalar_count


class BudgetedMultiTimescaleUtilityGate(nn.Module):
    """Causal write critic with multi-horizon utility and a hard write budget."""

    def __init__(
        self,
        feature_dim: int,
        *,
        horizons: Sequence[int] = (1, 4, 16),
        hidden_dim: int = 16,
        write_budget: float = 0.2,
        temperature: float = 1.0,
        horizon_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        parsed_horizons = tuple(int(value) for value in horizons)
        if not parsed_horizons or any(value <= 0 for value in parsed_horizons):
            raise ValueError("horizons must contain positive integers")
        if len(set(parsed_horizons)) != len(parsed_horizons):
            raise ValueError("horizons must be unique")
        if not math.isfinite(float(write_budget)) or not 0.0 < float(write_budget) <= 1.0:
            raise ValueError("write_budget must be finite and in (0, 1]")
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if horizon_weights is None:
            weights = torch.tensor(
                [1.0 / math.sqrt(value) for value in parsed_horizons],
                dtype=torch.float32,
            )
        else:
            if len(horizon_weights) != len(parsed_horizons):
                raise ValueError("horizon_weights must match horizons")
            weights = torch.tensor(tuple(float(value) for value in horizon_weights))
            if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0.0).all()) or not bool(weights.sum() > 0.0):
                raise ValueError("horizon_weights must be finite, non-negative, and nonzero")
        weights = weights / weights.sum()

        self.feature_dim = int(feature_dim)
        self.horizons = parsed_horizons
        self.write_budget = float(write_budget)
        self.temperature = float(temperature)
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), len(self.horizons)),
        )
        self.register_buffer("horizon_weights", weights)
        self.register_buffer("dual_threshold", torch.tensor(0.0))

    def horizon_advantages(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2 or features.size(1) != self.feature_dim:
            raise ValueError(f"features must be [batch, {self.feature_dim}]")
        if not features.is_floating_point() or torch.is_complex(features):
            raise TypeError("features must be real floating point")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features must contain only finite values")
        return self.network(features)

    def aggregate_advantage(self, features: torch.Tensor) -> torch.Tensor:
        values = self.horizon_advantages(features)
        weights = self.horizon_weights.to(values)
        return values @ weights

    def write_probability(self, features: torch.Tensor) -> torch.Tensor:
        advantage = self.aggregate_advantage(features)
        threshold = self.dual_threshold.to(advantage)
        return torch.sigmoid((advantage - threshold) / self.temperature)

    def zero_budget_state(
        self, batch_size: int, *, device: torch.device
    ) -> WriteBudgetState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return WriteBudgetState(
            steps=torch.zeros(batch_size, device=device, dtype=torch.long),
            writes=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def decide(
        self,
        features: torch.Tensor,
        budget_state: WriteBudgetState,
        *,
        probability_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, WriteBudgetState, Dict[str, torch.Tensor]]:
        """Make a causal decision and enforce the cumulative hard budget."""

        if not 0.0 <= float(probability_threshold) <= 1.0:
            raise ValueError("probability_threshold must lie in [0, 1]")
        if not isinstance(budget_state, WriteBudgetState):
            raise TypeError("budget_state must be a WriteBudgetState")
        if budget_state.steps.shape != (features.size(0),) or budget_state.writes.shape != (features.size(0),):
            raise ValueError("budget state has an incompatible batch size")
        probability = self.write_probability(features)
        next_steps = budget_state.steps + 1
        allowance = torch.ceil(
            next_steps.to(probability.dtype) * self.write_budget
        ).to(torch.long)
        eligible = probability >= float(probability_threshold)
        budget_available = budget_state.writes < allowance
        decision = eligible & budget_available
        next_writes = budget_state.writes + decision.to(torch.long)
        state = WriteBudgetState(next_steps, next_writes)
        diagnostics = {
            "probability": probability,
            "eligible": eligible,
            "budget_available": budget_available,
            "allowance": allowance,
            "budget_slack": allowance - next_writes,
        }
        return decision, state, diagnostics

    def balanced_loss(
        self,
        features: torch.Tensor,
        advantage_targets: torch.Tensor,
        *,
        calibration_weight: float = 0.25,
        budget_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Fit horizon returns with equal total mass for positive/negative rows."""

        predictions = self.horizon_advantages(features)
        if advantage_targets.dim() == 1 and len(self.horizons) == 1:
            advantage_targets = advantage_targets.unsqueeze(1)
        if advantage_targets.shape != predictions.shape:
            raise ValueError(
                f"advantage_targets must have shape {tuple(predictions.shape)}"
            )
        targets = advantage_targets.to(predictions)
        weights = self.horizon_weights.to(predictions)
        aggregate_target = targets @ weights
        positive = aggregate_target > 0.0
        negative = ~positive
        row_weight = torch.ones_like(aggregate_target)
        if bool(positive.any()) and bool(negative.any()):
            row_weight[positive] = 0.5 / positive.sum().to(predictions.dtype)
            row_weight[negative] = 0.5 / negative.sum().to(predictions.dtype)
            row_weight = row_weight * row_weight.numel()
        regression_per_row = F.smooth_l1_loss(
            predictions, targets, reduction="none"
        ).mean(dim=1)
        regression = (row_weight * regression_per_row).mean()
        aggregate_prediction = predictions @ weights
        logits = (aggregate_prediction - self.dual_threshold.to(predictions)) / self.temperature
        class_target = positive.to(predictions.dtype)
        calibration_per_row = F.binary_cross_entropy_with_logits(
            logits, class_target, reduction="none"
        )
        calibration = (row_weight * calibration_per_row).mean()
        mean_probability = torch.sigmoid(logits).mean()
        budget_penalty = (mean_probability - self.write_budget).square()
        total = regression + float(calibration_weight) * calibration + float(budget_weight) * budget_penalty
        return {
            "loss": total,
            "regression_loss": regression,
            "calibration_loss": calibration,
            "budget_penalty": budget_penalty,
            "mean_probability": mean_probability,
            "positive_fraction": positive.to(predictions.dtype).mean(),
        }

    @torch.no_grad()
    def update_dual_threshold_(
        self, observed_soft_write_rate: float, *, learning_rate: float = 0.05
    ) -> float:
        if not math.isfinite(float(observed_soft_write_rate)) or not 0.0 <= float(observed_soft_write_rate) <= 1.0:
            raise ValueError("observed_soft_write_rate must be in [0, 1]")
        if not math.isfinite(float(learning_rate)) or float(learning_rate) < 0.0:
            raise ValueError("learning_rate must be finite and non-negative")
        self.dual_threshold.add_(
            float(learning_rate) * (float(observed_soft_write_rate) - self.write_budget)
        )
        return float(self.dual_threshold.item())


def utility_calibration_metrics(
    probabilities: torch.Tensor,
    advantages: torch.Tensor,
    *,
    decisions: Optional[torch.Tensor] = None,
    event_mask: Optional[torch.Tensor] = None,
    write_budget: Optional[float] = None,
    bins: int = 10,
) -> Dict[str, float]:
    """Return AUPRC, Brier, ECE, write gap, and budget diagnostics."""

    probability = probabilities.detach().flatten().double()
    target_value = advantages.detach().flatten().double()
    if probability.numel() == 0 or probability.shape != target_value.shape:
        raise ValueError("probabilities and advantages must be non-empty and aligned")
    if not bool(torch.isfinite(probability).all()) or not bool(
        torch.isfinite(target_value).all()
    ):
        raise ValueError("calibration inputs must be finite")
    if not bool(((probability >= 0.0) & (probability <= 1.0)).all()):
        raise ValueError("probabilities must lie in [0, 1]")
    if bins <= 0:
        raise ValueError("bins must be positive")
    target = target_value > 0.0
    target_float = target.double()
    brier = (probability - target_float).square().mean()

    order = torch.argsort(probability, descending=True)
    ordered_target = target_float[order]
    true_positive = ordered_target.cumsum(0)
    rank = torch.arange(1, probability.numel() + 1, dtype=torch.double)
    precision = true_positive / rank
    positive_count = ordered_target.sum()
    if bool(positive_count > 0.0):
        recall = true_positive / positive_count
        previous = torch.cat([torch.zeros(1, dtype=torch.double), recall[:-1]])
        auprc = ((recall - previous) * precision).sum()
    else:
        auprc = torch.zeros((), dtype=torch.double)

    ece = torch.zeros((), dtype=torch.double)
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.double)
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        if bool(selected.any()):
            weight = selected.double().mean()
            ece = ece + weight * (
                probability[selected].mean() - target_float[selected].mean()
            ).abs()

    result = {
        "auprc": float(auprc.item()),
        "brier": float(brier.item()),
        "ece": float(ece.item()),
        "positive_rate": float(target_float.mean().item()),
    }
    if decisions is not None:
        decision = decisions.detach().flatten().bool()
        if decision.shape != target.shape:
            raise ValueError("decisions must align with probabilities")
        write_rate = decision.double().mean()
        result["write_rate"] = float(write_rate.item())
        if write_budget is not None:
            result["budget_violation"] = max(
                0.0, float(write_rate.item()) - float(write_budget)
            )
        if event_mask is not None:
            events = event_mask.detach().flatten().bool()
            if events.shape != target.shape or bool(events.all()) or bool((~events).all()):
                raise ValueError("event_mask must align and contain both classes")
            event_rate = decision[events].double().mean()
            distractor_rate = decision[~events].double().mean()
            result.update(
                {
                    "event_write_rate": float(event_rate.item()),
                    "distractor_write_rate": float(distractor_rate.item()),
                    "event_distractor_write_gap": float(
                        (event_rate - distractor_rate).item()
                    ),
                }
            )
    return result
