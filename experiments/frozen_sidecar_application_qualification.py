#!/usr/bin/env python3
"""Qualify a frozen-backbone MQR sidecar on controlled online corrections.

This is a deterministic synthetic mechanism test.  It asks whether a zero-op
temporal sidecar can learn two conflicting, context-dependent preferences while
the frozen model and one protected old-domain probe remain unchanged.  It is
not a benchmark of MQR against other online learners and is not evidence of
general capability improvement.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mqr.safety import categorical_policy_kl
from mqr.temporal import (
    OnlineTemporalMQRClassifier,
    TemporalMQRSidecar,
    TemporalMQRState,
)


DEFAULT_SEEDS = tuple(20260828 + index for index in range(10))


class FrozenToyModel(nn.Module):
    """A fixed feature map and head used only to test sidecar contracts."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4, bias=False, dtype=torch.float64)
        self.head = nn.Linear(4, 2, bias=True, dtype=torch.float64)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.eye(4, dtype=torch.float64))
            self.head.weight.zero_()
            self.head.weight[0, 0] = 1.0
            self.head.weight[1, 1] = 1.0
            self.head.bias.copy_(torch.tensor([0.15, -0.15], dtype=torch.float64))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def hidden(self, token: torch.Tensor) -> torch.Tensor:
        return self.backbone(token)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)


def _tokens() -> Tuple[Mapping[str, torch.Tensor], torch.Tensor]:
    dtype = torch.float64
    profiles = {
        "session_a": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype),
        "session_b": torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=dtype),
        "protected_old_domain": torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]], dtype=dtype
        ),
    }
    query = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=dtype)
    return profiles, query


def _snapshot_named_parameters(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def _max_parameter_drift(
    module: nn.Module,
    snapshot: Mapping[str, torch.Tensor],
    *,
    exclude: Iterable[str] = (),
) -> float:
    excluded = set(exclude)
    values = [
        float((parameter.detach() - snapshot[name]).abs().max().item())
        for name, parameter in module.named_parameters()
        if name not in excluded
    ]
    return max(values, default=0.0)


def _state_vector(state: TemporalMQRState) -> torch.Tensor:
    return torch.cat(state.rings, dim=1)


def _state_max_abs_difference(
    left: TemporalMQRState,
    right: TemporalMQRState,
) -> float:
    return max(
        float((a - b).abs().max().item())
        for a, b in zip(left.rings, right.rings)
    )


def _session_forward(
    frozen: FrozenToyModel,
    sidecar: TemporalMQRSidecar,
    profile: torch.Tensor,
    query: torch.Tensor,
    *,
    retain_context: bool = True,
    diagnostics: bool = False,
) -> Tuple[torch.Tensor, TemporalMQRState, Dict[str, Any] | None]:
    with torch.no_grad():
        profile_hidden = frozen.hidden(profile)
        _adapted_profile, profile_state = sidecar.forward_step(profile_hidden)
    query_hidden = frozen.hidden(query)
    state = profile_state.detached() if retain_context else None
    if diagnostics:
        adapted, query_state, audit = sidecar.forward_step(
            query_hidden,
            state=state,
            return_diagnostics=True,
        )
        return frozen.logits(adapted), query_state, audit
    adapted, query_state = sidecar.forward_step(query_hidden, state=state)
    return frozen.logits(adapted), query_state, None


def _base_logits(frozen: FrozenToyModel, query: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return frozen.logits(frozen.hidden(query)).detach()


def _build_sidecar() -> TemporalMQRSidecar:
    sidecar = TemporalMQRSidecar(
        4,
        ring_dim=5,
        residual_clip_l2=2.0,
        core_kwargs={
            "leak_rates": (1.0, 0.2, 0.05),
            "injection_rank": 4,
            "transition_mode": "unistochastic",
            "learn_transitions": False,
            "cayley_coordinate_mode": "minimal",
        },
    ).double()
    # This qualification deliberately isolates the cheapest safe adaptation
    # path: a fixed/cached ring reservoir and one online residual readout.
    for parameter in sidecar.parameters():
        parameter.requires_grad_(False)
    sidecar.core.readout.weight.requires_grad_(True)
    return sidecar


def _project_away_protected_feature(
    gradient: torch.Tensor,
    protected_feature: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """Project each readout row into the exact nullspace of one probe feature."""

    denominator = protected_feature.square().sum()
    if float(denominator.item()) <= torch.finfo(gradient.dtype).eps:
        raise RuntimeError("protected feature has zero norm")
    projected = gradient - (
        (gradient @ protected_feature.transpose(0, 1)) / denominator
    ) * protected_feature
    overlap = float(
        (projected @ protected_feature.transpose(0, 1)).abs().max().item()
    )
    return projected, overlap


def _causal_api_check(seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed + 10_000)
    template = OnlineTemporalMQRClassifier(
        4,
        5,
        2,
        core_kwargs={
            "leak_rates": (1.0, 0.2),
            "injection_rank": 3,
            "transition_mode": "identity",
        },
        lr=0.2,
        transition_lr_ratio=0.0,
        injection_lr_ratio=0.0,
        readout_lr_ratio=1.0,
        carry_state=False,
    ).double()
    target_zero = copy.deepcopy(template)
    target_one = copy.deepcopy(template)
    x = torch.tensor([[0.2, -0.4, 0.7, 0.1]], dtype=torch.float64)
    preview = template.preview_step(x, carry_state=False)["logits"]
    left = target_zero.online_step(x, torch.tensor([0]), carry_state=False)
    right = target_one.online_step(x, torch.tensor([1]), carry_state=False)
    issued_difference = float((left["logits"] - right["logits"]).abs().max().item())
    preview_difference = max(
        float((preview - left["logits"]).abs().max().item()),
        float((preview - right["logits"]).abs().max().item()),
    )
    post_left = target_zero.preview_step(x, carry_state=False)["logits"]
    post_right = target_one.preview_step(x, carry_state=False)["logits"]
    post_update_difference = float((post_left - post_right).abs().max().item())
    passed = bool(
        issued_difference == 0.0
        and preview_difference == 0.0
        and post_update_difference > 0.0
        and left["prediction_before_update"]
        and right["prediction_before_update"]
        and left["did_update"]
        and right["did_update"]
    )
    return {
        "counterfactual_issue_logits_linf_difference": issued_difference,
        "preview_to_issue_linf_difference": preview_difference,
        "post_update_logits_linf_difference": post_update_difference,
        "both_predictions_before_update": bool(
            left["prediction_before_update"] and right["prediction_before_update"]
        ),
        "passed": passed,
    }


def _session_isolation_check(seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed + 20_000)
    template = OnlineTemporalMQRClassifier(
        4,
        5,
        2,
        core_kwargs={
            "leak_rates": (1.0, 0.2, 0.05),
            "injection_rank": 4,
            "transition_mode": "unistochastic",
            "learn_transitions": False,
            "cayley_coordinate_mode": "minimal",
        },
        carry_state=True,
        max_streams=2,
    ).double()
    interleaved = copy.deepcopy(template)
    reference_a = copy.deepcopy(template)
    reference_b = copy.deepcopy(template)
    profiles, query = _tokens()

    interleaved.infer_step(
        profiles["session_a"], stream_id="a", issue_feedback_ticket=False
    )
    interleaved.infer_step(
        profiles["session_b"], stream_id="b", issue_feedback_ticket=False
    )
    interleaved_a = interleaved.infer_step(
        query, stream_id="a", issue_feedback_ticket=False
    )
    interleaved_b = interleaved.infer_step(
        query, stream_id="b", issue_feedback_ticket=False
    )

    reference_a.infer_step(
        profiles["session_a"], stream_id="a", issue_feedback_ticket=False
    )
    separate_a = reference_a.infer_step(
        query, stream_id="a", issue_feedback_ticket=False
    )
    reference_b.infer_step(
        profiles["session_b"], stream_id="b", issue_feedback_ticket=False
    )
    separate_b = reference_b.infer_step(
        query, stream_id="b", issue_feedback_ticket=False
    )

    logits_error = max(
        float((interleaved_a["logits"] - separate_a["logits"]).abs().max().item()),
        float((interleaved_b["logits"] - separate_b["logits"]).abs().max().item()),
    )
    state_error = max(
        _state_max_abs_difference(interleaved_a["state"], separate_a["state"]),
        _state_max_abs_difference(interleaved_b["state"], separate_b["state"]),
    )
    return {
        "interleaved_to_separate_logits_linf_error": logits_error,
        "interleaved_to_separate_state_linf_error": state_error,
        "state_bank_size": interleaved.state_bank_size,
        "active_stream_ids": list(interleaved.active_stream_ids()),
        "passed": bool(
            logits_error == 0.0
            and state_error == 0.0
            and interleaved.state_bank_size == 2
            and set(interleaved.active_stream_ids()) == {"a", "b"}
        ),
    }


def run_seed(seed: int, *, updates: int, learning_rate: float) -> Dict[str, Any]:
    torch.manual_seed(seed)
    frozen = FrozenToyModel()
    sidecar = _build_sidecar()
    profiles, query = _tokens()
    frozen_snapshot = _snapshot_named_parameters(frozen)
    sidecar_snapshot = _snapshot_named_parameters(sidecar)
    transition_snapshot = [
        sidecar.core.transition_matrix(index).detach().clone()
        for index in range(sidecar.core.num_timescales)
    ]
    base_logits = _base_logits(frozen, query)

    initial_logits: Dict[str, torch.Tensor] = {}
    initial_noop_error = 0.0
    protected_feature: torch.Tensor | None = None
    for name, profile in profiles.items():
        logits, state, _audit = _session_forward(
            frozen, sidecar, profile, query, diagnostics=False
        )
        initial_logits[name] = logits.detach().clone()
        initial_noop_error = max(
            initial_noop_error,
            float((logits.detach() - base_logits).abs().max().item()),
        )
        if name == "protected_old_domain":
            protected_feature = _state_vector(state).detach()
    assert protected_feature is not None

    first_b_nll = math.nan
    max_projection_overlap = 0.0
    predictions_before_update = True
    feedback_count = 0
    for step in range(updates):
        session = "session_a" if step % 2 == 0 else "session_b"
        target = torch.tensor([0 if session == "session_a" else 1])
        logits, _state, _audit = _session_forward(
            frozen,
            sidecar,
            profiles[session],
            query,
            diagnostics=False,
        )
        issued_logits = logits.detach().clone()
        loss = F.cross_entropy(logits, target)
        if session == "session_b" and math.isnan(first_b_nll):
            first_b_nll = float(loss.detach().item())
        gradient = torch.autograd.grad(loss, sidecar.core.readout.weight)[0]
        projected, overlap = _project_away_protected_feature(
            gradient.detach(), protected_feature
        )
        max_projection_overlap = max(max_projection_overlap, overlap)
        with torch.no_grad():
            sidecar.core.readout.weight.add_(projected, alpha=-learning_rate)
        predictions_before_update = predictions_before_update and bool(
            torch.equal(issued_logits, logits.detach())
        )
        feedback_count += 1

    final_logits: Dict[str, torch.Tensor] = {}
    final_audits: Dict[str, Dict[str, Any]] = {}
    for name, profile in profiles.items():
        logits, _state, audit = _session_forward(
            frozen, sidecar, profile, query, diagnostics=True
        )
        final_logits[name] = logits.detach().clone()
        assert audit is not None
        final_audits[name] = audit

    targets = {"session_a": 0, "session_b": 1}
    personalized_accuracy = sum(
        int(final_logits[name].argmax(dim=1).item() == target)
        for name, target in targets.items()
    ) / len(targets)
    reset_accuracy = 0.0
    for name, target in targets.items():
        logits, _state, _audit = _session_forward(
            frozen,
            sidecar,
            profiles[name],
            query,
            retain_context=False,
        )
        reset_accuracy += int(logits.argmax(dim=1).item() == target)
    reset_accuracy /= len(targets)
    frozen_base_accuracy = sum(
        int(base_logits.argmax(dim=1).item() == target)
        for target in targets.values()
    ) / len(targets)
    final_b_nll = float(
        F.cross_entropy(final_logits["session_b"], torch.tensor([1])).item()
    )
    protected_kl = categorical_policy_kl(
        initial_logits["protected_old_domain"],
        final_logits["protected_old_domain"],
    )

    fixed_parameter_drift = _max_parameter_drift(
        sidecar,
        sidecar_snapshot,
        exclude=("core.readout.weight",),
    )
    transition_drift = max(
        float(
            (
                sidecar.core.transition_matrix(index) - transition_snapshot[index]
            ).abs().max().item()
        )
        for index in range(sidecar.core.num_timescales)
    )
    readout_drift = float(
        torch.linalg.vector_norm(
            sidecar.core.readout.weight.detach()
            - sidecar_snapshot["core.readout.weight"]
        ).item()
    )
    residual_l2_max = max(
        float(audit["residual_l2_max"]) for audit in final_audits.values()
    )
    state_certified = all(
        bool(audit["state_certificate"]["certified"])
        for audit in final_audits.values()
    )

    # Exercise a fail-closed candidate transaction at the application layer.
    # The generic sidecar exposes numerical diagnostics; the caller owns this
    # snapshot/probe/rollback transaction (the Go agent has a built-in version).
    readout_before_candidate = sidecar.core.readout.weight.detach().clone()
    protected_logits_before_candidate = final_logits[
        "protected_old_domain"
    ].detach().clone()
    denominator = protected_feature.square().sum()
    unsafe_delta = torch.zeros_like(sidecar.core.readout.weight)
    unsafe_delta[1] = 10.0 * protected_feature.squeeze(0) / denominator
    with torch.no_grad():
        sidecar.core.readout.weight.add_(unsafe_delta)
    unsafe_logits, _state, _audit = _session_forward(
        frozen,
        sidecar,
        profiles["protected_old_domain"],
        query,
    )
    unsafe_kl = categorical_policy_kl(
        protected_logits_before_candidate,
        unsafe_logits.detach(),
    )
    candidate_limit = 1e-3
    candidate_rejected = unsafe_kl > candidate_limit
    if candidate_rejected:
        with torch.no_grad():
            sidecar.core.readout.weight.copy_(readout_before_candidate)
    restored_logits, _state, _audit = _session_forward(
        frozen,
        sidecar,
        profiles["protected_old_domain"],
        query,
    )
    rollback_parameter_error = float(
        (
            sidecar.core.readout.weight.detach() - readout_before_candidate
        ).abs().max().item()
    )
    rollback_output_error = float(
        (restored_logits.detach() - protected_logits_before_candidate)
        .abs()
        .max()
        .item()
    )

    return {
        "seed": seed,
        "initial_noop_logits_linf_error": initial_noop_error,
        "frozen_base_personalized_accuracy": frozen_base_accuracy,
        "mqr_context_personalized_accuracy": personalized_accuracy,
        "reset_before_query_accuracy": reset_accuracy,
        "session_b_first_prequential_nll": first_b_nll,
        "session_b_final_nll": final_b_nll,
        "session_b_corrected": bool(
            final_logits["session_b"].argmax(dim=1).item() == 1
            and final_b_nll < first_b_nll
        ),
        "protected_old_domain_policy_kl": protected_kl,
        "protected_projection_max_abs_overlap": max_projection_overlap,
        "frozen_model_parameter_linf_drift": _max_parameter_drift(
            frozen, frozen_snapshot
        ),
        "fixed_sidecar_parameter_linf_drift": fixed_parameter_drift,
        "transition_linf_drift": transition_drift,
        "readout_l2_drift": readout_drift,
        "residual_l2_max": residual_l2_max,
        "state_bound_certified": state_certified,
        "feedback_count": feedback_count,
        "prediction_before_update": predictions_before_update,
        "unsafe_candidate": {
            "policy_kl": unsafe_kl,
            "policy_kl_limit": candidate_limit,
            "rejected": candidate_rejected,
            "parameter_linf_error_after_rollback": rollback_parameter_error,
            "output_linf_error_after_rollback": rollback_output_error,
        },
    }


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def _aggregate(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    all_session_success = all(
        float(row["mqr_context_personalized_accuracy"]) == 1.0 for row in rows
    )
    all_corrections = all(bool(row["session_b_corrected"]) for row in rows)
    all_rollbacks = all(
        bool(row["unsafe_candidate"]["rejected"])
        and float(
            row["unsafe_candidate"]["parameter_linf_error_after_rollback"]
        )
        == 0.0
        and float(row["unsafe_candidate"]["output_linf_error_after_rollback"])
        == 0.0
        for row in rows
    )
    return {
        "seeds": len(rows),
        "initial_noop_logits_linf_error_max": max(
            float(row["initial_noop_logits_linf_error"]) for row in rows
        ),
        "frozen_model_parameter_linf_drift_max": max(
            float(row["frozen_model_parameter_linf_drift"]) for row in rows
        ),
        "fixed_sidecar_parameter_linf_drift_max": max(
            float(row["fixed_sidecar_parameter_linf_drift"]) for row in rows
        ),
        "transition_linf_drift_max": max(
            float(row["transition_linf_drift"]) for row in rows
        ),
        "readout_l2_drift_min": min(float(row["readout_l2_drift"]) for row in rows),
        "frozen_base_personalized_accuracy_mean": _mean(
            rows, "frozen_base_personalized_accuracy"
        ),
        "mqr_context_personalized_accuracy_mean": _mean(
            rows, "mqr_context_personalized_accuracy"
        ),
        "mqr_context_personalized_accuracy_min": min(
            float(row["mqr_context_personalized_accuracy"]) for row in rows
        ),
        "reset_before_query_accuracy_mean": _mean(
            rows, "reset_before_query_accuracy"
        ),
        "session_b_first_prequential_nll_mean": _mean(
            rows, "session_b_first_prequential_nll"
        ),
        "session_b_final_nll_mean": _mean(rows, "session_b_final_nll"),
        "session_b_correction_successes": sum(
            bool(row["session_b_corrected"]) for row in rows
        ),
        "protected_old_domain_policy_kl_max": max(
            float(row["protected_old_domain_policy_kl"]) for row in rows
        ),
        "protected_projection_max_abs_overlap": max(
            float(row["protected_projection_max_abs_overlap"]) for row in rows
        ),
        "residual_l2_max": max(float(row["residual_l2_max"]) for row in rows),
        "state_bound_certified_seeds": sum(
            bool(row["state_bound_certified"]) for row in rows
        ),
        "feedback_count_per_seed": sorted(
            {int(row["feedback_count"]) for row in rows}
        ),
        "all_predictions_before_update": all(
            bool(row["prediction_before_update"]) for row in rows
        ),
        "unsafe_candidate_policy_kl_min": min(
            float(row["unsafe_candidate"]["policy_kl"]) for row in rows
        ),
        "atomic_rollback_successes": sum(
            bool(row["unsafe_candidate"]["rejected"])
            and float(
                row["unsafe_candidate"]["parameter_linf_error_after_rollback"]
            )
            == 0.0
            and float(row["unsafe_candidate"]["output_linf_error_after_rollback"])
            == 0.0
            for row in rows
        ),
        "all_session_memory_success": all_session_success,
        "all_error_corrections_success": all_corrections,
        "all_atomic_rollbacks_success": all_rollbacks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "analysis/results/frozen_sidecar_application_qualification.json",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    if args.updates <= 0 or args.updates % 2 != 0:
        raise ValueError("updates must be a positive even integer")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be finite and positive")

    started = time.perf_counter()
    rows = [
        run_seed(seed, updates=args.updates, learning_rate=args.learning_rate)
        for seed in args.seeds
    ]
    aggregate = _aggregate(rows)
    causal = _causal_api_check(args.seeds[0])
    isolation = _session_isolation_check(args.seeds[0])
    criteria = {
        "initial_sidecar_is_exact_noop": aggregate[
            "initial_noop_logits_linf_error_max"
        ]
        == 0.0,
        "frozen_model_has_zero_parameter_drift": aggregate[
            "frozen_model_parameter_linf_drift_max"
        ]
        == 0.0,
        "only_online_readout_changes": bool(
            aggregate["fixed_sidecar_parameter_linf_drift_max"] == 0.0
            and aggregate["transition_linf_drift_max"] == 0.0
            and aggregate["readout_l2_drift_min"] > 0.0
        ),
        "prediction_precedes_feedback": bool(
            aggregate["all_predictions_before_update"] and causal["passed"]
        ),
        "sessions_are_state_isolated": bool(isolation["passed"]),
        "context_memory_is_necessary_and_sufficient_here": bool(
            aggregate["all_session_memory_success"]
            and aggregate["reset_before_query_accuracy_mean"] == 0.5
        ),
        "online_feedback_corrects_repeated_error": bool(
            aggregate["all_error_corrections_success"]
        ),
        "protected_probe_is_constant": bool(
            aggregate["protected_old_domain_policy_kl_max"] <= 1e-12
        ),
        "state_and_residual_budgets_hold": bool(
            aggregate["state_bound_certified_seeds"] == len(rows)
            and aggregate["residual_l2_max"] <= 2.0 + 1e-12
        ),
        "unsafe_candidates_are_atomically_rejected": bool(
            aggregate["all_atomic_rollbacks_success"]
        ),
    }
    payload = {
        "schema_version": 1,
        "experiment": "frozen_sidecar_application_qualification",
        "evidence_scope": (
            "deterministic synthetic mechanism qualification; not real-model "
            "task efficacy and not algorithm-superiority evidence"
        ),
        "protocol": {
            "seeds": args.seeds,
            "updates_per_seed": args.updates,
            "feedback_budget_per_seed": args.updates,
            "learning_rate": args.learning_rate,
            "prediction_before_update": True,
            "backbone_and_native_head_frozen": True,
            "sidecar_trainable_block": "residual readout weight only",
            "transition": "frozen cached minimal-Cayley unistochastic",
            "residual_l2_limit": 2.0,
            "constancy_control": (
                "exact readout-gradient nullspace projection for one "
                "pre-registered protected probe"
            ),
            "task": (
                "two sessions share an identical query but require opposite "
                "labels after distinct profile tokens"
            ),
            "network_used": False,
        },
        "seed_results": rows,
        "aggregate": aggregate,
        "causal_api_check": causal,
        "session_isolation_check": isolation,
        "qualification_gate": {
            "criteria": criteria,
            "application_mechanism_qualified": all(criteria.values()),
            "real_frozen_model_task_efficacy_proven": False,
            "mqr_specific_advantage_proven": False,
            "general_capability_enhancement_proven": False,
            "animal_like_learning_proven": False,
        },
        "limitations": [
            "The frozen model and data are synthetic and deliberately small.",
            "The test compares context memory with frozen/no-memory ablations, not GRU, LoRA, replay, or Sinkhorn.",
            "Constancy protects one declared probe and does not guarantee unmeasured-domain behavior.",
            "Only the residual readout learns; Cayley transitions remain frozen and cached.",
            "The generic TemporalMQRSidecar relies on its caller for candidate-update transactions.",
        ],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float64",
        },
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if not payload["qualification_gate"]["application_mechanism_qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
