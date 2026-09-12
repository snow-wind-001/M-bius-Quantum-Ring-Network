"""Task-margin halfspaces with current orthogonal recurrent adjoints."""
from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, Tuple

import torch

from .constraint_transport import terminal_gradients
from .go_outcome import PolicyBehaviorMemory


@torch.no_grad()
def project_halfspaces(
    delta: torch.Tensor, normals: torch.Tensor, bounds: torch.Tensor, *,
    tolerance: float = 1e-8, max_sweeps: int = 1024,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Project [P] onto G step >= b with [K,P] G, b<=0 (zero is feasible).

    Dual coordinate ascent solves min 1/2||step-delta||². Computation is float64;
    the returned step has the input dtype. Failure of the KKT check returns
    zero, rather than silently reporting an approximate step as a projection.
    """
    if (delta.ndim != 1 or normals.ndim != 2 or normals.size(1) != delta.numel()
            or bounds.shape != (normals.size(0),) or not 0 < tolerance < 1
            or max_sweeps < 1 or not all(bool(torch.isfinite(x).all()) for x in (delta, normals, bounds))
            or bool((bounds > 0).any())):
        raise ValueError("finite compatible halfspaces with a feasible zero step are required")
    if not normals.size(0):
        return delta.clone(), {"constraints": 0, "converged": True, "active": 0,
                               "sweeps": 0, "norm_retention": 1.0, "correction_norm": 0.0}
    d, g, b = delta.double(), normals.double(), bounds.double()
    lengths = g.norm(dim=1)
    keep = lengths > 1e-12
    g, b = g[keep] / lengths[keep, None], b[keep] / lengths[keep]
    gram, residual = g @ g.T, b - g @ d
    multipliers = torch.zeros_like(b)
    sweeps = 0
    for sweeps in range(max_sweeps + 1):
        slack = gram @ multipliers - residual
        primal_error = float((-slack).clamp_min(0).max()) if len(slack) else 0.0
        kkt_error = float(torch.where(multipliers > tolerance, slack.abs(), (-slack).clamp_min(0)).max()) if len(slack) else 0.0
        if max(primal_error, kkt_error) <= tolerance or sweeps == max_sweeps:
            break
        for index in range(len(b)):
            correction = (residual[index] - gram[index] @ multipliers) / gram[index, index]
            multipliers[index] = (multipliers[index] + correction).clamp_min(0)
    converged = max(primal_error, kkt_error) <= tolerance
    candidate = (d + g.T @ multipliers).to(delta) if converged else torch.zeros_like(delta)
    # A projection onto a convex set containing zero cannot enlarge its norm.
    if float(candidate.norm()) > float(delta.norm()) + 1e-6:
        converged = False
        candidate.zero_()
    return candidate, {"constraints": normals.size(0), "converged": converged,
        "active": int((multipliers > tolerance).sum()), "sweeps": sweeps,
        "primal_error": primal_error, "kkt_error": kkt_error,
        "norm_retention": float(candidate.norm() / delta.norm().clamp_min(1e-30)),
        "correction_norm": float((candidate - delta).norm())}


class TaskMarginMemory(PolicyBehaviorMemory):
    """Keep reference action margins, allowing improvements and slack usage.

    Each anchor contributes gradients against its strongest current competitors.
    The nonlinear all-action margin check still runs after projection. No KL
    equality is imposed and old probabilities may improve or redistribute.
    """

    def __init__(self, *args: Any, competitors: int = 2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.reliable_only or self.margin_fraction is None or competitors < 1:
            raise ValueError("task constraints require reliable margins and competitors")
        self.competitors = int(competitors)
        self.normals = torch.empty(0, 0)
        self.bounds = torch.empty(0)
        self.parameter_names = ()

    def refresh(self, agent) -> Dict[str, Any]:
        if not self.frozen or agent.pending_ticket_count:
            raise RuntimeError("refresh requires frozen anchors and resolved feedback")
        started = time.perf_counter()
        active = [(n, p) for n, p in agent._named_task_parameters() if p.requires_grad]
        reference = active[0][1]
        rows, bounds, audits = [], [], []
        for record in self.anchors:
            with torch.no_grad():
                current = self._output(agent, record)
                alternatives = current.clone()
                alternatives[record["action"]] = -torch.inf
                choices = alternatives.topk(min(self.competitors, agent.action_size - 1)).indices
            def margins(output):
                scores = output.policy_logits[0]
                return scores[record["action"]] - scores[choices]
            result = terminal_gradients(agent, record["features"], margins, backend=self.backend)
            if result.parameter_names != tuple(n for n, _ in active):
                raise RuntimeError("margin parameter coordinates changed")
            slack = result.values - record["margin_floor"]
            if bool((slack < -1e-6).any()):
                raise RuntimeError("current anchor is outside its declared feasible margin")
            rows.append(result.jacobian)
            bounds.append(-slack.clamp_min(0))
            audits.append(result.diagnostics)
        self.normals = torch.cat(rows) if rows else reference.new_empty(0, sum(p.numel() for _, p in active))
        self.bounds = torch.cat(bounds) if bounds else reference.new_empty(0)
        self.parameter_names = tuple(n for n, _ in active)
        agent.task_gradient_memory.clear()
        self.last_refresh_version = int(agent.online_parameter_version)
        elapsed = time.perf_counter() - started
        self.refresh_count += 1
        self.refresh_seconds += elapsed
        trace_bytes = max((a["state_trace_bytes"] for a in audits), default=0)
        error = max((a["max_adjoint_norm_error"] for a in audits), default=0.0)
        self.peak_state_trace_bytes = max(self.peak_state_trace_bytes, trace_bytes)
        self.max_adjoint_norm_error = max(self.max_adjoint_norm_error, error)
        self.last_diagnostics = {"backend": self.backend, "constraint_kind": "margin_halfspaces",
            "rank": 0, "gradient_directions": len(self.bounds), "parameter_version": self.last_refresh_version,
            "max_adjoint_norm_error": error, "peak_state_trace_bytes": trace_bytes,
            "refresh_seconds": elapsed}
        return self.last_diagnostics

    def project_update(self, delta: torch.Tensor, *, parameter_version: int):
        if parameter_version != self.last_refresh_version:
            raise RuntimeError("halfspace normals are not from the candidate's parameter version")
        return project_halfspaces(delta, self.normals.to(delta), self.bounds.to(delta))

    @property
    def storage_bytes(self) -> int:
        return super().storage_bytes + self.normals.numel() * self.normals.element_size() + self.bounds.numel() * self.bounds.element_size()

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["task_margin"] = copy.deepcopy({"version": 1, "competitors": self.competitors,
            "normals": self.normals, "bounds": self.bounds, "parameter_names": self.parameter_names})
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        saved = state.get("task_margin", {})
        if saved.get("version") != 1 or saved.get("competitors") != self.competitors:
            raise ValueError("task-margin configuration differs")
        normals, bounds = saved["normals"], saved["bounds"]
        if (normals.ndim != 2 or bounds.shape != (normals.size(0),)
                or not bool(torch.isfinite(normals).all()) or not bool(torch.isfinite(bounds).all())
                or bool((bounds > 0).any())):
            raise ValueError("invalid saved halfspaces")
        super().load_state_dict(state)
        self.normals, self.bounds = normals.clone(), bounds.clone()
        self.parameter_names = tuple(saved["parameter_names"])
