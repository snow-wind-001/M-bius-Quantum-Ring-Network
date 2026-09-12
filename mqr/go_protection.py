"""Train-only behavioral anchors, refreshed OGD, and measured finite-step drift.

The anchor budget is separate from inference memory. The guarantee is limited
to the saved anchor policies under their explicitly replayed raw contexts.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence

import torch

from .agent import TemporalUtilityMQRAgent
from .go_memory import HistoryGoSession
from .online import OrthogonalGradientMemory


class GoBehaviorMemory:
    """Bounded round-robin strata of seen prefixes, with immutable reference policies."""

    def __init__(self, capacity: int = 8, *, strata: int = 4) -> None:
        if capacity < strata or strata < 1 or capacity % strata:
            raise ValueError("capacity must be a positive multiple of strata")
        self.capacity, self.strata = int(capacity), int(strata)
        self.records: List[List[Dict[str, Any]]] = [[] for _ in range(strata)]
        self.seen = [0] * strata
        self.frozen = False
        self.last_refresh_version: Optional[int] = None

    @property
    def anchors(self) -> List[Dict[str, Any]]:
        # Interleave strata so a small OGD rank does not spend its entire
        # budget on the first stratum. No held-out metric selects anchors.
        return [group[index] for index in range(self.capacity // self.strata)
                for group in self.records if index < len(group)]

    def project_update(self, delta: torch.Tensor, *, parameter_version: int):
        """Optional finite-slack projection of a flat proposed parameter step."""
        return delta, {}

    def add(self, features: Sequence[torch.Tensor], action: int, *, stratum: int) -> None:
        """Add an already observed full prefix; freeze() ends collection."""
        if self.frozen:
            raise RuntimeError("reference anchor collection is frozen")
        if not features or not 0 <= stratum < self.strata:
            raise ValueError("a nonempty prefix and valid stratum are required")
        shape = features[0].shape
        if len(shape) != 2 or shape[0] != 1 or any(x.shape != shape or not bool(torch.isfinite(x).all()) for x in features):
            raise ValueError("anchor inputs must be finite [1, input_dim] tensors")
        record = {"features": torch.cat([x.detach().cpu().clone() for x in features]),
                  "action": int(action)}
        group = self.records[stratum]
        quota = self.capacity // self.strata
        if len(group) < quota:
            group.append(record)
        else:
            group[self.seen[stratum] % quota] = record
        self.seen[stratum] += 1

    def _output(self, agent: TemporalUtilityMQRAgent, record: Dict[str, Any]) -> torch.Tensor:
        reference = next(agent.parameters())
        features = record["features"].to(reference)
        if features.size(1) != agent.input_dim or not 1 <= features.size(0) <= agent.max_trace_horizon:
            raise ValueError("anchor dimensions or prefix length do not match the agent")
        if not 0 <= record["action"] < agent.action_size:
            raise ValueError("anchor action is out of range")
        state = agent.core.zero_state(1, device=reference.device, dtype=reference.dtype)
        for x in features.split(1):
            output, state = agent._transition(x, state, slow_write=True)
        return output.policy_logits[0]

    @torch.no_grad()
    def freeze(self, agent: TemporalUtilityMQRAgent) -> None:
        if agent.pending_ticket_count or not self.anchors:
            raise RuntimeError("freeze needs resolved feedback and nonempty training anchors")
        if self.frozen:
            raise RuntimeError("anchor reference policies are immutable once frozen")
        for record in self.anchors:
            record["reference_log_probs"] = self._output(agent, record).detach().cpu()
        self.frozen = True

    @torch.no_grad()
    def drift(self, agent: TemporalUtilityMQRAgent) -> Dict[str, float]:
        if not self.frozen:
            raise RuntimeError("freeze reference behaviors before measuring drift")
        kls, scores = [], []
        for record in self.anchors:
            current = self._output(agent, record)
            reference = record["reference_log_probs"].to(current)
            kls.append(float((reference.exp() * (reference - current)).sum().clamp_min(0)))
            action = record["action"]
            scores.append(float((current[action] - reference[action]).abs()))
        return {"max_policy_kl": max(kls), "mean_policy_kl": sum(kls) / len(kls),
                "max_target_log_probability_drift": max(scores)}

    @torch.enable_grad()
    def refresh(self, agent: TemporalUtilityMQRAgent) -> Dict[str, Any]:
        """Rebuild the basis at current parameters, maximizing uncovered directions.

        Each anchor proposes its target and strongest alternative log-probability
        gradients. Greedy normalized residual coverage avoids first-anchor bias.
        This changes neither model parameters nor committed live state.
        """
        if not self.frozen or agent.pending_ticket_count:
            raise RuntimeError("refresh requires frozen anchors and no pending feedback")
        active = [(name, p) for name, p in agent._named_task_parameters() if p.requires_grad]
        if not active or agent.task_gradient_memory.max_rank == 0:
            raise ValueError("refresh requires trainable parameters and OGD capacity")
        candidates = []
        for record in self.anchors:
            output = self._output(agent, record)
            other = output.detach().clone()
            other[record["action"]] = -torch.inf
            for index, action in enumerate((record["action"], int(other.argmax()))):
                gradients = torch.autograd.grad(output[action], [p for _, p in active],
                                                allow_unused=True, retain_graph=index == 0)
                flat = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                  for (_, p), g in zip(active, gradients)])
                candidates.append(flat)
        matrix = torch.stack(candidates)
        normalized = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
        memory = OrthogonalGradientMemory(agent.task_gradient_memory.max_rank,
                                         tolerance=agent.task_gradient_memory.tolerance).to(matrix)
        selected = []
        for _ in range(min(memory.max_rank, len(candidates))):
            residual = normalized if not memory.rank else normalized - (normalized @ memory._basis.T) @ memory._basis
            novelty = residual.square().sum(dim=1)
            if selected:
                novelty[selected] = -1
            index = int(novelty.argmax())
            if float(novelty[index]) < 1e-10:
                break
            selected.append(index)
            cursor, entries = 0, []
            for name, p in active:
                entries.append((name, matrix[index, cursor:cursor + p.numel()].reshape_as(p), agent.task_lr))
                cursor += p.numel()
            memory.observe(entries)
        agent.task_gradient_memory.restore(memory.snapshot())
        coverage = (normalized @ memory._basis.T).square().sum(dim=1) if memory.rank else matrix.new_zeros(len(candidates))
        self.last_refresh_version = int(agent.online_parameter_version)
        return {"rank": memory.rank, "gradient_directions": len(candidates),
                "mean_gradient_energy_coverage": float(coverage.mean()),
                "min_gradient_energy_coverage": float(coverage.min()),
                "orthogonality_error": memory.orthogonality_error(),
                "parameter_version": self.last_refresh_version}

    @property
    def storage_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for record in self.anchors
                   for value in record.values() if isinstance(value, torch.Tensor))

    def state_dict(self) -> Dict[str, Any]:
        return copy.deepcopy({"capacity": self.capacity, "strata": self.strata,
                              "records": self.records, "seen": self.seen, "frozen": self.frozen,
                              "last_refresh_version": self.last_refresh_version})

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state["capacity"] != self.capacity or state["strata"] != self.strata:
            raise ValueError("anchor memory budget does not match")
        groups = state["records"]
        if len(groups) != self.strata or any(len(group) > self.capacity // self.strata for group in groups):
            raise ValueError("checkpoint exceeds the anchor capacity")
        self.records = copy.deepcopy(groups)
        self.seen = list(state["seen"])
        self.frozen = bool(state["frozen"])
        self.last_refresh_version = state["last_refresh_version"]


class ProtectedGoSession(HistoryGoSession):
    """Refresh OGD at boundaries, then backtrack a candidate's global step size.

    Finite-step KL is measured against immutable phase-A reference policies.
    A scalar shrink preserves Q delta = 0. It makes no promise about unseen
    positions or gradients beyond the finite reference bank.
    """

    def __init__(self, *args: Any, behavior_memory: Optional[GoBehaviorMemory] = None,
                 refresh_every: int = 8, max_anchor_kl: Optional[float] = 0.01,
                 norm_matched_sgd: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if refresh_every < 1 or (max_anchor_kl is not None and (not math.isfinite(max_anchor_kl) or max_anchor_kl < 0)):
            raise ValueError("invalid refresh interval or anchor KL budget")
        if norm_matched_sgd and self.project_with_memory:
            raise ValueError("norm-matched SGD must disable directional OGD projection")
        self.behavior_memory = behavior_memory
        self.refresh_every = int(refresh_every)
        self.max_anchor_kl = max_anchor_kl
        self.norm_matched_sgd = bool(norm_matched_sgd)

    def refresh_protection(self) -> Optional[Dict[str, Any]]:
        memory = self.behavior_memory
        if memory is None or not memory.frozen:
            return None
        return memory.refresh(self.agent)

    def flush(self) -> Optional[Dict[str, Any]]:
        if not self._pending:
            return None
        memory = self.behavior_memory
        protected = memory is not None and memory.frozen
        # Refresh must occur before a prediction window starts, not here:
        # pending same-version tickets are owned by that window.
        active = [p for _, p in self.agent._named_task_parameters() if p.requires_grad]
        before = [p.detach().clone() for p in active] if protected else []
        old_version = int(self.agent.online_parameter_version)
        old_updates = int(self.agent.online_task_updates)
        result = super().flush()
        if protected and result["did_update"]:
            delta = [p.detach() - old for p, old in zip(active, before)]
            flat = torch.cat([d.flatten() for d in delta])
            projected, projection = memory.project_update(flat, parameter_version=old_version)
            if projection:
                if projected.shape != flat.shape or not bool(torch.isfinite(projected).all()):
                    raise RuntimeError("invalid constraint projection")
                delta = list(projected.split([p.numel() for p in active]))
                delta = [d.reshape_as(p) for d, p in zip(delta, active)]
                result["constraint_projection"] = projection
                result["update_norm"] = float(projected.norm())
            scale = 1.0
            if projection and float(projected.norm()) == 0:
                scale = 0.0
            if self.norm_matched_sgd and self.agent.task_gradient_memory.rank:
                flat = torch.cat([d.flatten() for d in delta])
                basis = self.agent.task_gradient_memory._basis.to(flat)
                projected = flat - basis.T @ (basis @ flat)
                # Recover the un-clipped candidate to match min(||P delta||, cap).
                unclipped_norm = float(projected.norm()) / max(result["update_clip_scale"], 1e-12)
                cap = self.agent.max_update_norm
                target_norm = unclipped_norm if cap is None else min(unclipped_norm, cap)
                scale = min(1.0, target_norm / max(float(flat.norm()), 1e-12))
            with torch.no_grad():
                for p, old, d in zip(active, before, delta):
                    p.copy_(old + scale * d)
                drift = memory.drift(self.agent)
                backtracks = 0
                guard_reasons = set()
                while ((self.max_anchor_kl is not None and drift["max_policy_kl"] > self.max_anchor_kl + 1e-7)
                       or drift.get("max_margin_violation", 0.0) > 1e-7):
                    if self.max_anchor_kl is not None and drift["max_policy_kl"] > self.max_anchor_kl + 1e-7:
                        guard_reasons.add("anchor_policy_kl")
                    if drift.get("max_margin_violation", 0.0) > 1e-7:
                        guard_reasons.add("anchor_action_margin")
                    backtracks += 1
                    scale = 0.0 if backtracks >= 12 else scale * 0.5
                    for p, old, d in zip(active, before, delta):
                        p.copy_(old + scale * d)
                    drift = memory.drift(self.agent)
                    if scale == 0:
                        break
                if scale == 0:
                    self.agent.online_parameter_version.fill_(old_version)
                    self.agent.online_task_updates.fill_(old_updates)
                    result.update(did_update=False, rolled_back=True, parameter_version=old_version,
                                  safety_passed=False, safety_violations=result["safety_violations"] + sorted(guard_reasons))
            result.update(anchor_drift=drift, anchor_step_scale=scale, anchor_backtracks=backtracks,
                          update_norm=result["update_norm"] * scale)
        return result

    def observe(self, *args: Any, learn: bool = True, **kwargs: Any) -> Dict[str, Any]:
        memory = self.behavior_memory
        if learn and not self._pending and memory is not None and memory.frozen:
            last = memory.last_refresh_version
            if last is None or int(self.agent.online_parameter_version) - last >= self.refresh_every:
                self.refresh_protection()
        return super().observe(*args, learn=learn, **kwargs)

    @property
    def online_tensor_bytes(self) -> int:
        return super().online_tensor_bytes + (0 if self.behavior_memory is None else self.behavior_memory.storage_bytes)

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["protection"] = {"refresh_every": self.refresh_every, "max_anchor_kl": self.max_anchor_kl,
                               "norm_matched_sgd": self.norm_matched_sgd,
                               "memory": None if self.behavior_memory is None else self.behavior_memory.state_dict()}
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        record = state["protection"]
        for key in ("refresh_every", "max_anchor_kl", "norm_matched_sgd"):
            if record[key] != getattr(self, key):
                raise ValueError(f"protection protocol differs: {key}")
        raw = record["memory"]
        candidate = None
        if raw is not None:
            candidate = GoBehaviorMemory(raw["capacity"], strata=raw["strata"])
            candidate.load_state_dict(raw)
        super().load_state_dict(state)
        self.behavior_memory = candidate
