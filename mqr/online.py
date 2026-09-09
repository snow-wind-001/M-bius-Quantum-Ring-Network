from __future__ import annotations

import math
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ring import MoebiusQuantumRing, RingState


GradientEntry = Tuple[str, torch.Tensor, float]


class OrthogonalGradientMemory(nn.Module):
    """Low-rank memory for preconditioned orthogonal gradient descent.

    For per-parameter step sizes collected in the diagonal matrix ``D``, the
    memory stores an orthonormal basis of whitened historical gradients
    ``sqrt(D) g_old``.  A new gradient is projected in the same coordinates:

        w = sqrt(D) g,  w_perp = (I - Q^T Q) w,
        delta_theta = -sqrt(D) w_perp.

    Consequently every remembered gradient has zero first-order inner product
    with the update, while the current loss decreases to first order by
    ``-||w_perp||^2``.  The guarantee assumes that relative step-size ratios do
    not change while the memory is active.
    """

    def __init__(self, max_rank: int = 16, *, tolerance: float = 1e-8):
        super().__init__()
        if max_rank < 0:
            raise ValueError("max_rank must be non-negative")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.max_rank = int(max_rank)
        self.tolerance = float(tolerance)
        self.register_buffer("_basis", torch.empty(0, 0), persistent=True)
        self._layout: Optional[List[Tuple[str, Tuple[int, ...]]]] = None
        self._relative_steps: Optional[List[float]] = None

    @property
    def rank(self) -> int:
        return int(self._basis.size(0)) if self._basis.dim() == 2 else 0

    @property
    def dimension(self) -> int:
        if self._basis.dim() == 2 and self._basis.size(1) > 0:
            return int(self._basis.size(1))
        if self._layout is None:
            return 0
        return sum(math.prod(shape) for _, shape in self._layout)

    @property
    def is_full(self) -> bool:
        return self.rank >= self.max_rank

    @property
    def storage_bytes(self) -> int:
        """Bytes occupied by the dense gradient basis (excluding small metadata)."""

        return int(self._basis.numel() * self._basis.element_size())

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "layout": self._layout,
            "relative_steps": self._relative_steps,
        }

    @torch.no_grad()
    def snapshot(self) -> Dict[str, Any]:
        """Return an exact in-memory snapshot for an atomic candidate update."""

        return {
            "version": 1,
            "basis": self._basis.detach().clone(),
            "layout": None if self._layout is None else list(self._layout),
            "relative_steps": (
                None
                if self._relative_steps is None
                else list(self._relative_steps)
            ),
        }

    @torch.no_grad()
    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore a snapshot created by :meth:`snapshot`."""

        if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
            raise ValueError("invalid OrthogonalGradientMemory snapshot")
        basis = snapshot.get("basis")
        if not isinstance(basis, torch.Tensor) or basis.dim() != 2:
            raise ValueError("gradient-memory snapshot has no rank-two basis")
        self._basis = basis.detach().clone().to(device=self._basis.device)
        layout = snapshot.get("layout")
        self._layout = (
            None
            if layout is None
            else [(str(name), tuple(shape)) for name, shape in layout]
        )
        steps = snapshot.get("relative_steps")
        self._relative_steps = None if steps is None else [float(value) for value in steps]

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        layout = state.get("layout")
        self._layout = None if layout is None else [(str(n), tuple(s)) for n, s in layout]
        steps = state.get("relative_steps")
        self._relative_steps = None if steps is None else [float(v) for v in steps]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The basis width is discovered on the first gradient. Resize the
        # placeholder buffer before loading a checkpoint with an active memory.
        key = prefix + "_basis"
        if key in state_dict:
            self._basis = torch.empty_like(state_dict[key])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def clear(self, *, reset_layout: bool = False) -> None:
        dim = 0 if reset_layout else self.dimension
        self._basis = self._basis.new_empty((0, dim))
        if reset_layout:
            self._layout = None
            self._relative_steps = None

    def _validate_and_whiten(
        self, entries: Sequence[GradientEntry]
    ) -> Tuple[torch.Tensor, List[Tuple[str, torch.Tensor, float]]]:
        prepared: List[Tuple[str, torch.Tensor, float]] = []
        seen = set()
        for name, gradient, step_size in entries:
            name = str(name)
            if name in seen:
                raise ValueError(f"Duplicate gradient name: {name}")
            seen.add(name)
            if not isinstance(gradient, torch.Tensor):
                raise TypeError(f"Gradient {name!r} must be a tensor")
            if torch.is_complex(gradient) or not gradient.is_floating_point():
                raise TypeError(f"Gradient {name!r} must be a real floating-point tensor")
            step = float(step_size)
            if not math.isfinite(step) or step <= 0:
                raise ValueError(f"Step size for {name!r} must be finite and positive")
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError(f"Gradient {name!r} contains NaN or Inf")
            prepared.append((name, gradient.detach(), step))

        if not prepared:
            raise ValueError("At least one active gradient is required")

        layout = [(name, tuple(gradient.shape)) for name, gradient, _ in prepared]
        max_step = max(step for _, _, step in prepared)
        relative_steps = [step / max_step for _, _, step in prepared]
        if self._layout is None:
            self._layout = layout
            self._relative_steps = relative_steps
        else:
            if layout != self._layout:
                raise ValueError(
                    "Gradient layout changed while orthogonal memory is active; "
                    "call clear(reset_layout=True) before changing trainable parameter groups"
                )
            assert self._relative_steps is not None
            if any(
                not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
                for a, b in zip(relative_steps, self._relative_steps)
            ):
                raise ValueError(
                    "Relative learning-rate ratios changed while orthogonal memory is active; "
                    "clear the memory before changing the preconditioner"
                )

        reference = prepared[0][1]
        dtype = reference.dtype
        device = reference.device
        parts = [
            gradient.to(device=device, dtype=dtype).reshape(-1) * math.sqrt(step)
            for _, gradient, step in prepared
        ]
        whitened = torch.cat(parts)

        if self._basis.numel() == 0:
            self._basis = torch.empty(
                (0, whitened.numel()), device=device, dtype=dtype
            )
        elif self._basis.size(1) != whitened.numel():
            raise RuntimeError("Stored basis width does not match the current gradient layout")
        elif self._basis.device != device or self._basis.dtype != dtype:
            self._basis = self._basis.to(device=device, dtype=dtype)
        return whitened, prepared

    @torch.no_grad()
    def project_preconditioned(
        self,
        entries: Sequence[GradientEntry],
        *,
        allowed_names: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Project a named gradient and return gradients in original coordinates.

        ``allowed_names`` restricts the candidate update to complete parameter
        blocks while retaining exact orthogonality to the stored full-layout
        gradients.  This is used for slow Cayley updates: transition gradients
        stay in the stable memory layout but are exactly zero between scheduled
        ticks.  The constrained projection is performed in the allowed
        coordinate subspace, rather than projecting first and masking later.
        """

        whitened, prepared = self._validate_and_whiten(entries)
        prepared_names = [name for name, _gradient, _step in prepared]
        if allowed_names is None:
            allowed = set(prepared_names)
        else:
            allowed = {str(name) for name in allowed_names}
            unknown = sorted(allowed.difference(prepared_names))
            if unknown:
                raise ValueError(f"allowed_names contains unknown gradients: {unknown}")

        mask_parts = [
            torch.full(
                (gradient.numel(),),
                name in allowed,
                device=whitened.device,
                dtype=torch.bool,
            )
            for name, gradient, _step in prepared
        ]
        mask = torch.cat(mask_parts)
        candidate = torch.where(mask, whitened, torch.zeros_like(whitened))
        projected = torch.zeros_like(candidate)
        if bool(mask.any()):
            allowed_candidate = candidate[mask]
            if self.rank == 0:
                allowed_projected = allowed_candidate
            else:
                restricted_basis = self._basis[:, mask]
                gram = restricted_basis @ restricted_basis.transpose(0, 1)
                gram_pinv = torch.linalg.pinv(gram, hermitian=True)
                allowed_projected = allowed_candidate.clone()
                # Two passes suppress numerical overlap when the restricted
                # historical directions are nearly linearly dependent.
                for _ in range(2):
                    overlap = restricted_basis @ allowed_projected
                    correction = restricted_basis.transpose(0, 1) @ (
                        gram_pinv @ overlap
                    )
                    allowed_projected -= correction
            projected[mask] = allowed_projected

        raw_norm = float(torch.linalg.vector_norm(candidate).item())
        projected_norm = float(torch.linalg.vector_norm(projected).item())
        overlap = 0.0
        if self.rank > 0 and projected.numel() > 0:
            overlap = float((self._basis @ projected).abs().max().item())

        result: Dict[str, torch.Tensor] = {}
        offset = 0
        for name, gradient, step in prepared:
            count = gradient.numel()
            part = projected[offset : offset + count].reshape(gradient.shape)
            result[name] = (part / math.sqrt(step)).to(
                device=gradient.device, dtype=gradient.dtype
            )
            offset += count

        retained = projected_norm / raw_norm if raw_norm > 0 else 0.0
        return result, {
            "rank": float(self.rank),
            "raw_norm": raw_norm,
            "projected_norm": projected_norm,
            "retained_norm": retained,
            "max_abs_overlap": overlap,
        }

    @torch.no_grad()
    def observe(self, entries: Sequence[GradientEntry]) -> bool:
        """Add the novel component of a historical gradient to the memory."""

        whitened, _ = self._validate_and_whiten(entries)
        if self.is_full:
            return False
        raw_norm = torch.linalg.vector_norm(whitened)
        if float(raw_norm.item()) == 0.0:
            return False

        residual = whitened.clone()
        if self.rank > 0:
            for _ in range(2):
                residual -= self._basis.transpose(0, 1) @ (self._basis @ residual)
        residual_norm = torch.linalg.vector_norm(residual)
        threshold = self.tolerance * max(1.0, float(raw_norm.item()))
        if float(residual_norm.item()) <= threshold:
            return False

        new_direction = (residual / residual_norm).unsqueeze(0)
        self._basis = torch.cat([self._basis, new_direction], dim=0)
        return True

    @torch.no_grad()
    def orthogonality_error(self) -> float:
        if self.rank == 0:
            return 0.0
        identity = torch.eye(self.rank, device=self._basis.device, dtype=self._basis.dtype)
        return float(torch.linalg.matrix_norm(self._basis @ self._basis.T - identity).item())


class OnlineMultiRingClassifier(nn.Module):
    """Prequential online learner with a sparse bank of independent MQR rings.

    ``online_step`` first computes and returns a prediction using parameters
    ``theta_t`` and only then commits an optional supervised update to
    ``theta_{t+1}``.  Explicit context IDs receive stable ring assignments;
    therefore an update to one context cannot modify another context's ring as
    long as capacity is available and routing is kept fixed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        num_rings: int = 4,
        ring_kwargs: Optional[Dict[str, Any]] = None,
        lr: float = 1e-2,
        unitary_lr_ratio: float = 0.5,
        injection_lr_ratio: float = 1.0,
        readout_lr_ratio: float = 1.0,
        state_target_weight: float = 0.0,
        state_target_lr_ratio: float = 1.0,
        h_mix_beta_lr_ratio: float = 1.0,
        adjoint_steps: Optional[int] = None,
        ogd_max_rank: int = 16,
        carry_state: bool = True,
        overflow_policy: str = "error",
        novelty_threshold: Optional[float] = None,
        routing_key_momentum: float = 0.95,
        max_update_norm: Optional[float] = None,
    ):
        super().__init__()
        if num_rings <= 0:
            raise ValueError("num_rings must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if min(
            unitary_lr_ratio,
            injection_lr_ratio,
            readout_lr_ratio,
            state_target_lr_ratio,
            h_mix_beta_lr_ratio,
        ) < 0:
            raise ValueError("all learning-rate ratios must be non-negative")
        if state_target_weight < 0:
            raise ValueError("state_target_weight must be non-negative")
        if overflow_policy not in ("error", "least_used"):
            raise ValueError('overflow_policy must be "error" or "least_used"')
        if novelty_threshold is not None and not (-1.0 <= novelty_threshold <= 1.0):
            raise ValueError("novelty_threshold must be in [-1, 1]")
        if not (0.0 <= routing_key_momentum < 1.0):
            raise ValueError("routing_key_momentum must be in [0, 1)")
        if max_update_norm is not None and max_update_norm <= 0:
            raise ValueError("max_update_norm must be positive or None")

        kwargs = dict(ring_kwargs or {})
        reserved = {"input_dim", "hidden_dim", "output_dim"}.intersection(kwargs)
        if reserved:
            raise ValueError(f"ring_kwargs must not redefine {sorted(reserved)}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_rings = int(num_rings)
        self.lr = float(lr)
        self.unitary_lr_ratio = float(unitary_lr_ratio)
        self.injection_lr_ratio = float(injection_lr_ratio)
        self.readout_lr_ratio = float(readout_lr_ratio)
        self.state_target_weight = float(state_target_weight)
        self.state_target_lr_ratio = float(state_target_lr_ratio)
        self.h_mix_beta_lr_ratio = float(h_mix_beta_lr_ratio)
        self.carry_state = bool(carry_state)
        self.overflow_policy = str(overflow_policy)
        self.novelty_threshold = novelty_threshold
        self.routing_key_momentum = float(routing_key_momentum)
        self.max_update_norm = None if max_update_norm is None else float(max_update_norm)

        self.rings = nn.ModuleList(
            [
                MoebiusQuantumRing(input_dim, hidden_dim, output_dim, **kwargs)
                for _ in range(self.num_rings)
            ]
        )
        default_adjoint = self.rings[0].relaxation_steps
        self.adjoint_steps = int(default_adjoint if adjoint_steps is None else adjoint_steps)
        if self.adjoint_steps <= 0:
            raise ValueError("adjoint_steps must be positive")

        self.gradient_memories = nn.ModuleList(
            [OrthogonalGradientMemory(ogd_max_rank) for _ in range(self.num_rings)]
        )
        self._states: List[Optional[RingState]] = [None] * self.num_rings
        self._context_to_ring: Dict[Hashable, int] = {}
        self.register_buffer("ring_usage", torch.zeros(self.num_rings, dtype=torch.long))
        self.register_buffer(
            "routing_keys", torch.zeros(self.num_rings, self.input_dim, dtype=torch.float32)
        )
        self.register_buffer("routing_key_counts", torch.zeros(self.num_rings, dtype=torch.long))

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "context_to_ring": dict(self._context_to_ring),
            "states": [None if state is None else state.h.detach().clone() for state in self._states],
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        mapping = state.get("context_to_ring", {})
        self._context_to_ring = {key: int(value) for key, value in mapping.items()}
        saved_states = state.get("states", [None] * self.num_rings)
        self._states = [
            None if value is None else RingState(h=value.detach()) for value in saved_states
        ]
        if len(self._states) != self.num_rings:
            self._states = [None] * self.num_rings

    @staticmethod
    def _context_key(context_id: Any) -> Hashable:
        if isinstance(context_id, torch.Tensor):
            if context_id.numel() != 1:
                raise ValueError("context_id tensor must be scalar")
            context_id = context_id.item()
        try:
            hash(context_id)
        except TypeError as exc:
            raise TypeError("context_id must be hashable") from exc
        return context_id

    def route_context(self, context_id: Any, *, allocate: bool = True) -> int:
        """Return the stable explicit-context assignment, allocating if needed."""

        if context_id is None:
            return 0
        key = self._context_key(context_id)
        if key in self._context_to_ring:
            return self._context_to_ring[key]
        if not allocate:
            raise KeyError(f"Unknown context_id: {context_id!r}")

        assigned = set(self._context_to_ring.values())
        free = [index for index in range(self.num_rings) if index not in assigned]
        if free:
            ring_index = free[0]
        elif self.overflow_policy == "least_used":
            ring_index = int(torch.argmin(self.ring_usage).item())
        else:
            raise RuntimeError(
                f"No free ring for context {context_id!r}; increase num_rings or explicitly "
                'choose overflow_policy="least_used" (which forfeits exact isolation)'
            )
        self._context_to_ring[key] = ring_index
        return ring_index

    def forward(self, x: torch.Tensor, *, context_id: Any = None) -> torch.Tensor:
        """Stateless inference through one routed ring without learning.

        Use :meth:`online_step` when recurrent state should advance or feedback
        should update parameters.  This method is intended for ordinary
        evaluation and leaves usage counters and ring states unchanged.
        """

        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"x must be [B, {self.input_dim}], got {tuple(x.shape)}")
        ring_index = (
            self.route_context(context_id)
            if context_id is not None
            else self._route_by_novelty(x)
        )
        return self.rings[ring_index](x)

    @torch.no_grad()
    def preview_step(
        self,
        x: torch.Tensor,
        *,
        context_id: Any = None,
        carry_state: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Preview the next online prediction without committing mutable state.

        This method uses the same routed ring and warm-start state as
        :meth:`online_step`, but changes neither parameters, recurrent state,
        usage counters, routing keys, nor gradient memory.  An explicit context
        may receive its deterministic ring assignment.  Calling
        ``online_step`` afterward with unchanged parameters and the same input
        therefore returns the same pre-update logits and can commit feedback
        only after an external action has already been taken.
        """

        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"x must be [B, {self.input_dim}], got {tuple(x.shape)}")
        ring_index = (
            self.route_context(context_id)
            if context_id is not None
            else self._route_by_novelty(x)
        )
        use_state = self.carry_state if carry_state is None else bool(carry_state)
        state = self._states[ring_index] if use_state else None
        if state is not None and state.h.shape != (x.size(0), self.hidden_dim):
            state = None
        info = self.rings[ring_index].eqprop_update_step(
            x,
            None,
            lr=self.lr,
            state=state,
        )
        info["grad_x"] = None
        info.update(
            {
                "ring_index": ring_index,
                "context_id": context_id,
                "prediction_before_update": True,
                "state_carried": use_state,
                "preview_only": True,
            }
        )
        return info

    @torch.no_grad()
    def _route_by_novelty(self, x: torch.Tensor) -> int:
        if self.novelty_threshold is None:
            return 0
        active = torch.nonzero(self.routing_key_counts > 0, as_tuple=False).flatten()
        if active.numel() == 0:
            return 0

        key = x.detach().float().mean(dim=0)
        key_norm = torch.linalg.vector_norm(key)
        if float(key_norm.item()) == 0.0:
            return int(active[0].item())
        key = key / key_norm
        active_keys = self.routing_keys.index_select(0, active).to(device=key.device)
        similarities = active_keys @ key
        best_position = int(torch.argmax(similarities).item())
        best_ring = int(active[best_position].item())
        best_similarity = float(similarities[best_position].item())

        if best_similarity < self.novelty_threshold:
            unused = torch.nonzero(self.routing_key_counts == 0, as_tuple=False).flatten()
            if unused.numel() > 0:
                return int(unused[0].item())
        return best_ring

    @torch.no_grad()
    def _update_routing_key(self, ring_index: int, x: torch.Tensor) -> None:
        key = x.detach().float().mean(dim=0)
        norm = torch.linalg.vector_norm(key)
        if float(norm.item()) == 0.0:
            return
        key = (key / norm).to(device=self.routing_keys.device)
        if int(self.routing_key_counts[ring_index].item()) == 0:
            updated = key
        else:
            momentum = self.routing_key_momentum
            updated = momentum * self.routing_keys[ring_index] + (1.0 - momentum) * key
            updated_norm = torch.linalg.vector_norm(updated)
            if float(updated_norm.item()) > 0.0:
                updated = updated / updated_norm
        self.routing_keys[ring_index].copy_(updated)
        self.routing_key_counts[ring_index].add_(x.size(0))

    @staticmethod
    def _loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 1:
            return F.cross_entropy(logits, target, reduction="mean")
        if target.dim() == 2:
            if target.shape != logits.shape:
                raise ValueError(f"Soft target must have shape {tuple(logits.shape)}")
            return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        raise ValueError("target must be [B] or [B, C]")

    @torch.no_grad()
    def online_step(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        *,
        context_id: Any = None,
        learn: bool = True,
        remember_gradient: bool = False,
        project_with_memory: bool = True,
        carry_state: Optional[bool] = None,
        return_grad_x: bool = False,
    ) -> Dict[str, Any]:
        """Predict with ``theta_t``, then optionally learn and return that prediction.

        Set ``project_with_memory=False`` while collecting gradients from the
        current task, then enable it after a context boundary.  This stages a
        protection basis without suppressing within-task plasticity.
        """

        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"x must be [B, {self.input_dim}], got {tuple(x.shape)}")
        if remember_gradient and (not learn or target is None):
            raise ValueError("remember_gradient requires a supervised update")

        ring_index = (
            self.route_context(context_id)
            if context_id is not None
            else self._route_by_novelty(x)
        )
        use_state = self.carry_state if carry_state is None else bool(carry_state)
        state = self._states[ring_index] if use_state else None
        if state is not None and state.h.shape != (x.size(0), self.hidden_dim):
            # Batch lanes no longer align; a stale state would mix unrelated samples.
            state = None

        ring = self.rings[ring_index]
        if learn and target is not None:
            memory = self.gradient_memories[ring_index]
            memory_for_update = memory if memory.max_rank > 0 else None
            info = ring.eqprop_update_step(
                x,
                target,
                lr=self.lr,
                unitary_lr_ratio=self.unitary_lr_ratio,
                injection_lr_ratio=self.injection_lr_ratio,
                readout_lr_ratio=self.readout_lr_ratio,
                adjoint_steps=self.adjoint_steps,
                state=state,
                state_target_weight=self.state_target_weight,
                state_target_lr_ratio=self.state_target_lr_ratio,
                h_mix_beta_lr_ratio=self.h_mix_beta_lr_ratio,
                orthogonal_memory=memory_for_update,
                remember_gradient=remember_gradient,
                project_with_memory=project_with_memory,
                return_grad_x=return_grad_x,
                max_update_norm=self.max_update_norm,
            )
        else:
            info = ring.eqprop_update_step(x, None, lr=self.lr, state=state)
            # Keep the output schema stable for external encoders.  Without a
            # feedback target there is no supervised input gradient to return.
            info["grad_x"] = None
            if target is not None:
                loss = self._loss_from_logits(info["logits"], target)
                info["loss"] = float(loss.item())
                info["loss_cls"] = float(loss.item())

        if use_state:
            self._states[ring_index] = RingState(h=info["h_star"].detach())
        self.ring_usage[ring_index].add_(x.size(0))
        self._update_routing_key(ring_index, x)

        info.update(
            {
                "ring_index": ring_index,
                "context_id": context_id,
                "prediction_before_update": True,
                "state_carried": use_state,
            }
        )
        return info

    @torch.no_grad()
    def reset_state(self, ring_index: Optional[int] = None) -> None:
        if ring_index is None:
            self._states = [None] * self.num_rings
            return
        if not (0 <= int(ring_index) < self.num_rings):
            raise IndexError("ring_index out of range")
        self._states[int(ring_index)] = None

    @torch.no_grad()
    def reset_routing(self) -> None:
        self._context_to_ring.clear()
        self.ring_usage.zero_()
        self.routing_keys.zero_()
        self.routing_key_counts.zero_()
        self.reset_state()
