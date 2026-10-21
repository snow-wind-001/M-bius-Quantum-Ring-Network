"""Research updates that reserve margin slack and reproject shorter Adam steps."""
from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, Optional

import torch
from torch.nn import functional as F

from .go_cone import project_halfspaces
from .go_readable import readable_terminal_gradients
from .go_research import (OnlineRuleLearner, RuleBatch, RuleConstraint, history_output,
                          parameter_vector, queried_logits, set_parameter_vector)


class ReadableBehaviorConstraint(RuleConstraint):
    """Protect rule margins and optional fixed-context, legal policy choices.

    Policy anchors have [T,B,D] observations, chosen actions, and legal masks.
    Their margin uses the strongest current alternative; the actual finite-step
    guard therefore checks all legal alternatives, including pass when legal.
    """
    def __init__(self, anchors: RuleBatch, floors: torch.Tensor, *,
                 policy: Optional[Dict[str, torch.Tensor]] = None,
                 backend: str = "transport"):
        super().__init__(anchors, floors, backend=backend)
        self.policy = copy.deepcopy(policy)
        self.rule_count = len(floors)
        if policy is not None:
            if policy["features"].shape[0] != anchors.features.shape[0]:
                raise ValueError("rule and policy anchors need the same finite context")
            if not bool((policy["floors"] > 0).all()):
                raise ValueError("policy margins must be positive at consolidation")
            self.floors = torch.cat((self.floors, policy["floors"].detach().clone()))

    def features(self) -> torch.Tensor:
        if self.policy is None:
            return self.anchors.features
        return torch.cat((self.anchors.features, self.policy["features"]), dim=1)

    def objectives(self, output) -> torch.Tensor:
        signs = 2 * self.anchors.labels - 1
        rule_logits = output.legality_logits[:self.rule_count].gather(
            1, self.anchors.queries[:, None]).squeeze(1)
        values = signs * rule_logits
        if self.policy is not None:
            scores = output.policy_logits[self.rule_count:]
            chosen = scores.gather(1, self.policy["actions"][:, None]).squeeze(1)
            alternatives = self.policy["legal"].clone()
            alternatives.scatter_(1, self.policy["actions"][:, None], False)
            rival = scores.masked_fill(~alternatives, -torch.inf).max(1).values
            values = torch.cat((values, chosen - rival))
        return values

    def margins(self, agent) -> torch.Tensor:
        return self.objectives(history_output(agent, self.features()))

    def refresh(self, agent, update: int) -> None:
        if self.normals is not None and update == self.last_refresh:
            return
        result = readable_terminal_gradients(agent, self.features(), self.objectives, backend=self.backend)
        self.normals = result.jacobian
        self.last_refresh, self.refreshes = update, self.refreshes + 1
        self.last_diagnostics = result.diagnostics


class FeasibleRuleLearner(OnlineRuleLearner):
    """Compare unchanged guards with slack reservation and curvature estimates.

    Variants reserve/adaptive/rollback keep half the present linear slack.
    Every rejected trial starts from the same parameters and reprojects the
    shorter original Adam proposal. A measured local remainder estimates
    curvature but is never trusted without the actual behavior check.
    """
    VARIANTS = ("free", "current", "reserve", "adaptive", "rollback", "frozen")

    def __init__(self, agent, *, variant: str = "current", lr: float = .003,
                 max_step: float = .5, constraint: Optional[RuleConstraint] = None):
        if variant not in self.VARIANTS:
            raise ValueError("unknown finite-step variant")
        super().__init__(agent, mode=variant if variant in ("free", "frozen") else "current",
                         lr=lr, max_step=max_step, constraint=constraint)
        self.variant = variant
        self.trust_radius = float(max_step)
        self.curvature = 0.

    def step(self, batch: RuleBatch) -> Dict[str, Any]:
        started = time.perf_counter()
        if self.variant in ("free", "current", "frozen"):
            result = super().step(batch)
            result.update(variant=self.variant, optimizer_rolled_back=False,
                          projection_trials=0 if self.variant != "current" else 1,
                          trust_radius=self.max_step, curvature_estimate=0.,
                          seconds=time.perf_counter() - started)
            return result
        c = self.constraint
        if c is None or not c.floors.numel():
            raise ValueError("feasible-step study needs nonempty mastered-behavior anchors")
        output = history_output(self.agent, batch.features)
        logits = queried_logits(output, batch.queries)
        predictions = logits.detach().clone()
        labels = batch.labels.detach().to(logits)
        if labels.shape != logits.shape or not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("binary feedback must follow matching predictions")
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        before = parameter_vector(self.agent)
        optimizer_before = copy.deepcopy(self.optimizer.state_dict()) if self.variant == "rollback" else None
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        raw = parameter_vector(self.agent) - before
        raw *= min(1., self.max_step / max(float(raw.norm()), 1e-30))
        set_parameter_vector(self.agent, before)
        c.refresh(self.agent, self.updates)
        with torch.no_grad():
            margins = c.margins(self.agent)
        slack = margins - c.floors
        # Use the identical float32 predicate at entry and commit. Subtracting
        # first can round a previously accepted boundary outside the tolerance.
        if bool((margins < c.floors - 1e-5).any()):
            raise RuntimeError("a protected behavior was already outside its feasible floor")
        bounds = -.5 * slack.clamp_min(0)
        radius = self.max_step if self.variant == "reserve" else self.trust_radius
        accepted = torch.zeros_like(raw)
        raw_violations = 0
        max_remainder = 0.
        trials, backtracks, active, converged = 0, 0, 0, True
        set_parameter_vector(self.agent, before + raw)
        with torch.no_grad():
            raw_margins = c.margins(self.agent)
        raw_violations = int((raw_margins < c.floors - 1e-5).sum())
        raw_remainder = float((raw_margins - margins - c.normals @ raw).abs().max())
        set_parameter_vector(self.agent, before)
        for attempt in range(9):
            proposed = raw * min(1., radius / max(float(raw.norm()), 1e-30))
            candidate, audit = project_halfspaces(proposed, c.normals, bounds)
            trials += 1
            active = max(active, audit["active"])
            converged = converged and audit["converged"]
            set_parameter_vector(self.agent, before + candidate)
            with torch.no_grad():
                actual = c.margins(self.agent)
            remainder = (margins + c.normals @ candidate - actual).clamp_min(0)
            max_remainder = max(max_remainder, float(remainder.max()))
            norm_squared = float(candidate.square().sum())
            if norm_squared > 1e-20:
                estimate = 2 * float(remainder.max()) / norm_squared
                self.curvature = max(.9 * self.curvature, estimate)
            if bool((actual >= c.floors - 1e-5).all()):
                accepted = candidate
                break
            if attempt == 8:
                break
            backtracks += 1
            radius *= .5
            if self.variant != "reserve" and self.curvature > 0:
                safe_radius = math.sqrt(max(float(slack.min()), 1e-10) / (1.25 * self.curvature))
                radius = min(radius, max(safe_radius, self.max_step / 65536))
            set_parameter_vector(self.agent, before)
        set_parameter_vector(self.agent, before + accepted)
        changed = bool(accepted.norm() > 0)
        rolled_back = self.variant == "rollback" and not changed
        if rolled_back:
            self.optimizer.load_state_dict(optimizer_before)
        if self.variant != "reserve":
            self.trust_radius = min(self.max_step, max(radius * (1.25 if changed else .5),
                                                      self.max_step / 65536))
        with torch.no_grad():
            violations = int((c.margins(self.agent) < c.floors - 1e-5).sum())
        if violations:
            raise RuntimeError("committed update violated an actual behavior floor")
        self.updates += 1
        if changed:
            self.agent.online_parameter_version += 1
        return {"pre_logits": predictions.tolist(), "pre_loss": float(loss.detach()),
                "pre_accuracy": float(((predictions >= 0) == labels.bool()).float().mean()),
                "feedback": len(labels), "accepted": changed, "active": active,
                "backtracks": backtracks, "raw_step_norm": float(raw.norm()),
                "step_norm": float(accepted.norm()), "raw_anchor_violations": raw_violations,
                "actual_anchor_violations": violations, "first_order_max_error": raw_remainder,
                "normal_age": 0, "refreshes": c.refreshes, "projection_converged": converged,
                "projection_trials": trials, "variant": self.variant,
                "optimizer_rolled_back": rolled_back, "curvature_estimate": self.curvature,
                "trust_radius": self.trust_radius, "maximum_trial_remainder": max_remainder,
                "seconds": time.perf_counter() - started}

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "finite_step": {"version": 1, "variant": self.variant,
                "trust_radius": self.trust_radius, "curvature": self.curvature}}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        saved = state.get("finite_step", {})
        if saved.get("version") != 1 or saved.get("variant") != self.variant:
            raise ValueError("finite-step optimizer policy differs")
        super().load_state_dict(state)
        self.trust_radius = float(saved["trust_radius"])
        self.curvature = float(saved["curvature"])
