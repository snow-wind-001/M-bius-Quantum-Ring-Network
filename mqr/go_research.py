"""Controlled online rule learning with behavior constraints over Go histories.

This module is an explicit research API. The existing game sessions and their
defaults are unchanged. A feedback window contains completed independent
histories; predictions are recorded before its labels enter an update.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.nn import functional as F

from .constraint_transport import terminal_gradients
from .go_cone import project_halfspaces
from .go_memory import GoResearchCore
from .temporal import TemporalMQRState


class SimilarityGoCore(GoResearchCore):
    """Stable nonorthogonal control A(x)=S R(x) S^-1, with fixed cond(S).

    Every parameter and input write is shared in form with GoResearchCore.
    Products telescope: ||A_t ... A_1||_2 <= cond(S), so this control does not
    introduce extra exponential forgetting. It is a coordinate-conditioned
    family, not a representative of every nonorthogonal recurrent model.
    """

    def __init__(self, *args: Any, similarity_condition: float = 2.0, **kwargs: Any) -> None:
        if not math.isfinite(similarity_condition) or similarity_condition < 1:
            raise ValueError("similarity condition must be finite and >= 1")
        super().__init__(*args, **kwargs)
        if self.transition_mode != "orthogonal" or self.state_activation != "none":
            raise ValueError("similarity control requires linear signed rotations")
        self.similarity_condition = float(similarity_condition)
        self.register_buffer("coordinate_scale", torch.linspace(
            -math.log(similarity_condition) / 2, math.log(similarity_condition) / 2,
            self.ring_dim, dtype=torch.float64,
        ).exp())

    def get_extra_state(self) -> Dict[str, Any]:
        return {**super().get_extra_state(), "similarity_condition": self.similarity_condition}

    def conditioned_transition(self, index: int, x: torch.Tensor) -> torch.Tensor:
        rotation = super().conditioned_transition(index, x)
        scale = self.coordinate_scale.to(x)
        return scale[None, :, None] * rotation / scale[None, None, :]

    def forward_step(
        self, x: torch.Tensor, *, state: Optional[TemporalMQRState] = None,
        write_gate: Optional[torch.Tensor] = None, promotion: Optional[torch.Tensor] = None,
        return_certificate: bool = False,
    ):
        if return_certificate:
            raise ValueError("use the similarity-product bound; the orthogonal certificate does not apply")
        # input_forcing validates the exact input shape, type and finiteness.
        forcing = self.input_forcing(x)
        state = self.zero_state(x.size(0), device=x.device, dtype=x.dtype) if state is None else state
        self._validate_state(state, x.size(0))
        gates = self._prepare_write_gate(write_gate, x)
        promotions = self._prepare_promotion(promotion, x)
        scale = self.coordinate_scale.to(x)
        rings = []
        for index, previous in enumerate(state.rings):
            previous = previous.to(x)
            rho = 1 - self.leak_rates[index].to(x)
            if float(rho) == 0:
                carried = torch.zeros_like(previous)
            else:
                carried = rho * scale * self.unitary_params[index].apply_orthogonal(
                    previous / scale, angle_offsets=self.orthogonal_offsets(index, x),
                )
            rings.append(carried + gates[:, index:index + 1] * forcing[:, index] + promotions[:, index])
        next_state = TemporalMQRState(tuple(rings))
        return self.readout_state(next_state), next_state


def task_parameters(agent) -> List[torch.nn.Parameter]:
    """Return the same ordered shared parameter coordinates used by transport."""
    return [p for _, p in agent._named_task_parameters() if p.requires_grad]


def parameter_vector(agent) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in task_parameters(agent)])


@torch.no_grad()
def set_parameter_vector(agent, vector: torch.Tensor) -> None:
    parameters = task_parameters(agent)
    if vector.ndim != 1 or vector.numel() != sum(p.numel() for p in parameters) or not bool(torch.isfinite(vector).all()):
        raise ValueError("invalid research parameter vector")
    offset = 0
    for p in parameters:
        p.copy_(vector[offset:offset + p.numel()].reshape_as(p))
        offset += p.numel()


def history_output(agent, features: torch.Tensor, *, reset: bool = False, credit: Optional[int] = None):
    """Read [T,B,D] complete histories, with a fresh state for each batch lane."""
    if features.ndim != 3 or min(features.shape[:2]) < 1:
        raise ValueError("histories must be nonempty [time,batch,features]")
    state = agent.core.zero_state(features.size(1), device=features.device, dtype=features.dtype)
    for index, x in enumerate(features):
        if reset:
            state = agent.core.zero_state(x.size(0), device=x.device, dtype=x.dtype)
        elif credit and index and index % credit == 0:
            state = state.detached()
        _, state = agent.core.forward_step(x, state=state)
    return agent.readout_state(state, features=features[-1])


def queried_logits(output, queries: torch.Tensor) -> torch.Tensor:
    return output.legality_logits.gather(1, queries[:, None]).squeeze(1)


@dataclass
class RuleBatch:
    """A batch of completed histories with feedback unavailable to forward()."""
    features: torch.Tensor
    queries: torch.Tensor
    labels: torch.Tensor

    def subset(self, indices) -> "RuleBatch":
        return RuleBatch(self.features[:, indices], self.queries[indices], self.labels[indices])


class RuleConstraint:
    """Fixed correct-behavior margins with current or explicitly stale normals.

    Stale experiments refresh the margins (bounds) at every update, cache only
    the normals, and retain the identical nonlinear finite-step guard.
    """
    def __init__(self, anchors: RuleBatch, floors: torch.Tensor, *, backend: str = "transport", refresh_every: int = 1):
        if backend not in ("transport", "autograd") or refresh_every < 1:
            raise ValueError("invalid constraint backend or refresh interval")
        if floors.shape != anchors.labels.shape or not bool((floors > 0).all()):
            raise ValueError("every protected correct behavior needs a positive margin floor")
        self.anchors = RuleBatch(anchors.features.detach().clone(), anchors.queries.detach().clone(), anchors.labels.detach().clone())
        self.floors = floors.detach().clone()
        self.backend, self.refresh_every = backend, refresh_every
        self.normals = None
        self.last_refresh = -1
        self.refreshes = 0

    def margins(self, agent) -> torch.Tensor:
        out = history_output(agent, self.anchors.features)
        return (2 * self.anchors.labels - 1) * queried_logits(out, self.anchors.queries)

    def refresh(self, agent, update: int) -> None:
        if self.normals is not None and update - self.last_refresh < self.refresh_every:
            return
        if not self.floors.numel():
            self.normals = parameter_vector(agent).new_empty(0, parameter_vector(agent).numel())
        elif self.backend == "transport":
            def objective(out):
                return (2 * self.anchors.labels - 1) * queried_logits(out, self.anchors.queries)
            self.normals = terminal_gradients(agent, self.anchors.features, objective, backend="transport").jacobian
        else:
            values = self.margins(agent)
            parameters = task_parameters(agent)
            rows = []
            for index, value in enumerate(values):
                grads = torch.autograd.grad(value, parameters, allow_unused=True, retain_graph=index + 1 < len(values))
                rows.append(torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1)
                                       for p, g in zip(parameters, grads)]))
            self.normals = torch.stack(rows)
        self.last_refresh, self.refreshes = update, self.refreshes + 1

    def state_dict(self) -> Dict[str, Any]:
        result = copy.deepcopy(vars(self))
        result["anchors"] = {"features": self.anchors.features.clone(), "queries": self.anchors.queries.clone(),
                             "labels": self.anchors.labels.clone()}
        return result


class OnlineRuleLearner:
    """One Adam proposal per feedback window, followed by optional constraints.

    Adam moments advance once per observed gradient even if a proposal is
    rejected; moments are never recomputed from unseen labels. A hard cap is
    applied before projection. Norm control uses its own hypothetical projected
    step length, without a guard: it is not a matched trajectory comparator.
    """
    def __init__(self, agent, *, mode: str = "free", lr: float = .003,
                 max_step: float = .5, constraint: Optional[RuleConstraint] = None):
        if mode not in ("free", "current", "stale", "norm", "frozen"):
            raise ValueError("unknown online research mode")
        if mode in ("current", "stale", "norm") and constraint is None:
            raise ValueError("this mode requires an explicit constraint object")
        self.agent, self.mode, self.max_step, self.constraint = agent, mode, float(max_step), constraint
        self.optimizer = torch.optim.Adam(task_parameters(agent), lr=lr)
        self.updates = 0

    def step(self, batch: RuleBatch) -> Dict[str, Any]:
        output = history_output(self.agent, batch.features)
        logits = queried_logits(output, batch.queries)
        predictions = logits.detach().clone()
        labels = batch.labels.detach().to(logits)
        if labels.shape != logits.shape or not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("rule feedback must be binary and match predictions")
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        result = {"pre_logits": predictions.tolist(), "pre_loss": float(loss.detach()),
                  "pre_accuracy": float(((predictions >= 0) == labels.bool()).float().mean()),
                  "feedback": len(labels), "accepted": False, "active": 0, "backtracks": 0,
                  "raw_step_norm": 0., "step_norm": 0., "raw_anchor_violations": 0,
                  "actual_anchor_violations": 0, "first_order_max_error": 0., "normal_age": 0}
        if self.mode == "frozen":
            return result
        before = parameter_vector(self.agent)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        raw = parameter_vector(self.agent) - before
        raw *= min(1., self.max_step / max(float(raw.norm()), 1e-30))
        set_parameter_vector(self.agent, before)
        delta = raw
        c = self.constraint
        if c is not None and c.floors.numel():
            c.refresh(self.agent, self.updates)
            with torch.no_grad():
                margins = c.margins(self.agent)
            # Norm-only controls may already violate old bounds; their shadow
            # projection keeps zero feasible and is never claimed as protection.
            bounds = -(margins - c.floors).clamp_min(0)
            projected, audit = project_halfspaces(raw, c.normals, bounds)
            result.update(active=audit["active"], projection_converged=audit["converged"],
                          normal_age=self.updates - c.last_refresh)
            set_parameter_vector(self.agent, before + raw)
            with torch.no_grad():
                raw_margins = c.margins(self.agent)
            result["raw_anchor_violations"] = int((raw_margins < c.floors - 1e-5).sum())
            result["first_order_max_error"] = float((raw_margins - margins - c.normals @ raw).abs().max())
            set_parameter_vector(self.agent, before)
            delta = raw * (projected.norm() / raw.norm().clamp_min(1e-30)) if self.mode == "norm" else projected
            if self.mode in ("current", "stale"):
                for attempt in range(9):
                    set_parameter_vector(self.agent, before + delta)
                    with torch.no_grad():
                        valid = bool((c.margins(self.agent) >= c.floors - 1e-5).all())
                    if valid:
                        break
                    if attempt == 8:
                        delta = torch.zeros_like(delta)
                        break
                    result["backtracks"] += 1
                    delta = delta * .5
                set_parameter_vector(self.agent, before)
        set_parameter_vector(self.agent, before + delta)
        if c is not None and c.floors.numel():
            with torch.no_grad():
                result["actual_anchor_violations"] = int((c.margins(self.agent) < c.floors - 1e-5).sum())
        result.update(raw_step_norm=float(raw.norm()), step_norm=float(delta.norm()),
                      accepted=bool(delta.norm() > 0), refreshes=0 if c is None else c.refreshes)
        self.updates += 1
        if result["accepted"]:
            self.agent.online_parameter_version += 1
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {"agent": copy.deepcopy(self.agent.state_dict()), "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "updates": self.updates, "mode": self.mode, "max_step": self.max_step,
                "constraint": None if self.constraint is None else self.constraint.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state["mode"] != self.mode or state["max_step"] != self.max_step:
            raise ValueError("research learner configuration differs")
        self.agent.load_state_dict(state["agent"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])
        if (state["constraint"] is None) != (self.constraint is None):
            raise ValueError("constraint configuration differs")
        if self.constraint is not None:
            saved = state["constraint"]
            if saved["backend"] != self.constraint.backend or saved["refresh_every"] != self.constraint.refresh_every:
                raise ValueError("constraint refresh contract differs")
            restored = copy.deepcopy(saved)
            restored["anchors"] = RuleBatch(**restored["anchors"])
            self.constraint.__dict__.update(restored)
