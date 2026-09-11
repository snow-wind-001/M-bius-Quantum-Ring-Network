"""Current-parameter behavior constraints built by explicit ring adjoints."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import torch

from .constraint_transport import terminal_gradients
from .go_memory import PositionQueryGoAgent
from .go_protection import GoBehaviorMemory, ProtectedGoSession
from .online import OrthogonalGradientMemory


class ConstraintGoBehaviorMemory(GoBehaviorMemory):
    """Raw-prefix anchors with interchangeable exact gradient backends.

    Near ties in normalized residual coverage choose the earliest candidate.
    This keeps algebraically equivalent float32 backends from spending their
    rank budget differently just because all initial normalized norms are one.
    """

    def __init__(
        self, capacity: int = 8, *, strata: int = 4, backend: str = "transport",
        selection_tolerance: float = 1e-6,
    ) -> None:
        super().__init__(capacity, strata=strata)
        if backend not in ("transport", "autograd"):
            raise ValueError("unknown constraint gradient backend")
        if not math.isfinite(selection_tolerance) or not 0 <= selection_tolerance < 1e-3:
            raise ValueError("selection_tolerance must be finite and in [0, 1e-3)")
        self.backend = backend
        self.selection_tolerance = float(selection_tolerance)
        self.refresh_count = 0
        self.refresh_seconds = 0.0
        self.peak_state_trace_bytes = 0
        self.max_adjoint_norm_error = 0.0
        self.last_diagnostics: Optional[Dict[str, Any]] = None

    def refresh(self, agent: PositionQueryGoAgent) -> Dict[str, Any]:
        if not self.frozen or agent.pending_ticket_count:
            raise RuntimeError("refresh requires frozen anchors and no pending feedback")
        active = [(name, p) for name, p in agent._named_task_parameters() if p.requires_grad]
        if not active or agent.task_gradient_memory.max_rank == 0:
            raise ValueError("refresh requires trainable parameters and OGD capacity")
        started = time.perf_counter()
        candidates, audits = [], []
        for record in self.anchors:
            if not 1 <= len(record["features"]) <= agent.max_trace_horizon:
                raise ValueError("anchor prefix exceeds the agent's declared trace horizon")
            target = record["action"]
            if not 0 <= target < agent.action_size:
                raise ValueError("anchor action is out of range")

            def scores(output):
                policy = output.policy_logits[0]
                alternative = policy.detach().clone()
                alternative[target] = -torch.inf
                return torch.stack((policy[target], policy[int(alternative.argmax())]))

            result = terminal_gradients(agent, record["features"], scores, backend=self.backend)
            if result.parameter_names != tuple(name for name, _ in active):
                raise RuntimeError("constraint gradient layout changed")
            candidates.append(result.jacobian)
            audits.append(result.diagnostics)
        matrix = torch.cat(candidates)
        normalized = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
        memory = OrthogonalGradientMemory(
            agent.task_gradient_memory.max_rank, tolerance=agent.task_gradient_memory.tolerance,
        ).to(matrix)
        selected = []
        while memory.rank < memory.max_rank and len(selected) < len(matrix):
            residual = normalized if not memory.rank else normalized - (normalized @ memory._basis.T) @ memory._basis
            novelty = residual.square().sum(dim=1)
            if selected:
                novelty[selected] = -1
            best = float(novelty.max())
            if best < 1e-10:
                break
            eligible = (novelty >= best - self.selection_tolerance) & (novelty > 1e-10)
            index = int(eligible.nonzero()[0, 0])
            selected.append(index)
            cursor, entries = 0, []
            for name, p in active:
                entries.append((name, matrix[index, cursor:cursor + p.numel()].reshape_as(p), agent.task_lr))
                cursor += p.numel()
            memory.observe(entries)
        coverage = (normalized @ memory._basis.T).square().sum(dim=1) if memory.rank else matrix.new_zeros(len(matrix))
        diagnostics = {
            "backend": self.backend, "rank": memory.rank, "gradient_directions": len(matrix),
            "selected_candidates": selected, "mean_gradient_energy_coverage": float(coverage.mean()),
            "min_gradient_energy_coverage": float(coverage.min()),
            "orthogonality_error": memory.orthogonality_error(),
            "parameter_version": int(agent.online_parameter_version),
            "max_adjoint_norm_error": max(a["max_adjoint_norm_error"] for a in audits),
            "peak_state_trace_bytes": max(a["state_trace_bytes"] for a in audits),
            "max_credit_steps": max(a["credit_steps"] for a in audits),
            "refresh_seconds": time.perf_counter() - started,
        }
        # Commit the new basis only after every prefix and diagnostic succeeds.
        agent.task_gradient_memory.restore(memory.snapshot())
        self.last_refresh_version = diagnostics["parameter_version"]
        self.refresh_count += 1
        self.refresh_seconds += diagnostics["refresh_seconds"]
        self.peak_state_trace_bytes = max(self.peak_state_trace_bytes, diagnostics["peak_state_trace_bytes"])
        self.max_adjoint_norm_error = max(self.max_adjoint_norm_error, diagnostics["max_adjoint_norm_error"])
        self.last_diagnostics = diagnostics
        return diagnostics

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["constraint_transport"] = {
            "version": 1, "backend": self.backend, "selection_tolerance": self.selection_tolerance,
            "refresh_count": self.refresh_count, "refresh_seconds": self.refresh_seconds,
            "peak_state_trace_bytes": self.peak_state_trace_bytes,
            "max_adjoint_norm_error": self.max_adjoint_norm_error,
            "last_diagnostics": self.last_diagnostics,
        }
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        config = state.get("constraint_transport", {})
        if (config.get("version") != 1 or config.get("backend") != self.backend
                or config.get("selection_tolerance") != self.selection_tolerance):
            raise ValueError("constraint transport checkpoint configuration differs")
        for name in ("refresh_count", "refresh_seconds", "peak_state_trace_bytes", "max_adjoint_norm_error"):
            value = config[name]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid constraint transport counter")
        super().load_state_dict(state)
        for name in ("refresh_count", "refresh_seconds", "peak_state_trace_bytes",
                     "max_adjoint_norm_error", "last_diagnostics"):
            setattr(self, name, config[name])


class ConstraintGoSession(ProtectedGoSession):
    """Existing causal online transactions with transport-aware restoration."""

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        raw = state["protection"]["memory"]
        candidate = None
        if raw is not None:
            config = raw.get("constraint_transport", {})
            candidate = ConstraintGoBehaviorMemory(
                raw["capacity"], strata=raw["strata"], backend=config.get("backend", ""),
                selection_tolerance=config.get("selection_tolerance", -1),
            )
            candidate.load_state_dict(raw)
            existing = self.behavior_memory
            if existing is not None and (
                not isinstance(existing, ConstraintGoBehaviorMemory)
                or existing.backend != candidate.backend
                or existing.selection_tolerance != candidate.selection_tolerance
            ):
                raise ValueError("restored constraint backend differs from configured memory")
        super().load_state_dict(state)
        self.behavior_memory = candidate
