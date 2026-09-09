#!/usr/bin/env python3
"""Fail-closed audit of MQR's application positioning and evidence gaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APPLICATION = (
    ROOT / "analysis/results/frozen_sidecar_application_qualification.json"
)
DEFAULT_GO_RESULTS = (
    ROOT
    / "analysis/results/unified_temporal_mqr_go_competitive_10x10_formal_10seed.json",
    ROOT
    / "analysis/results/unified_temporal_mqr_go_competitive_13x13_formal_10seed.json",
)
DEFAULT_GO_JOINT = (
    ROOT
    / "analysis/results/unified_temporal_mqr_go_competitive_joint_verify.json"
)
DEFAULT_MINICPM = (
    ROOT / "analysis/results/unified_temporal_mqr_minicpm_go_smoke.json"
)
DEFAULT_OUTPUT = ROOT / "analysis/results/mqr_application_positioning_audit.json"

REQUIRED_BASELINES = (
    "identity",
    "GRU",
    "LSTM",
    "fast-weight",
    "LoRA",
    "OGD-LoRA",
    "equal-byte replay",
    "Sinkhorn",
)
METHOD_ALIASES = {
    "mqr_identity_ogd": "identity",
    "gru_ogd": "GRU",
    "lstm_ogd": "LSTM",
    "fast_weight_ogd": "fast-weight",
    "lora": "LoRA",
    "ogd_lora": "OGD-LoRA",
    "replay_lora": "equal-byte replay",
    "sinkhorn_ogd": "Sinkhorn",
}


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"non-finite value at {path}: {value}")


def verify_application(payload: Dict[str, Any]) -> Dict[str, Any]:
    assert payload["schema_version"] == 1
    assert payload["experiment"] == "frozen_sidecar_application_qualification"
    assert "not algorithm-superiority evidence" in payload["evidence_scope"]
    assert len(payload["protocol"]["seeds"]) >= 10
    assert payload["protocol"]["prediction_before_update"] is True
    assert payload["protocol"]["backbone_and_native_head_frozen"] is True
    aggregate = payload["aggregate"]
    assert aggregate["initial_noop_logits_linf_error_max"] == 0.0
    assert aggregate["frozen_model_parameter_linf_drift_max"] == 0.0
    assert aggregate["fixed_sidecar_parameter_linf_drift_max"] == 0.0
    assert aggregate["transition_linf_drift_max"] == 0.0
    assert aggregate["readout_l2_drift_min"] > 0.0
    assert aggregate["mqr_context_personalized_accuracy_min"] == 1.0
    assert aggregate["reset_before_query_accuracy_mean"] == 0.5
    assert aggregate["session_b_final_nll_mean"] < aggregate[
        "session_b_first_prequential_nll_mean"
    ]
    assert aggregate["protected_old_domain_policy_kl_max"] <= 1e-12
    assert aggregate["state_bound_certified_seeds"] == aggregate["seeds"]
    assert aggregate["residual_l2_max"] <= 2.0 + 1e-12
    assert aggregate["atomic_rollback_successes"] == aggregate["seeds"]
    assert payload["causal_api_check"]["passed"] is True
    assert payload["session_isolation_check"]["passed"] is True
    gate = payload["qualification_gate"]
    assert gate["application_mechanism_qualified"] is True
    assert gate["real_frozen_model_task_efficacy_proven"] is False
    assert gate["mqr_specific_advantage_proven"] is False
    assert gate["general_capability_enhancement_proven"] is False
    assert gate["animal_like_learning_proven"] is False
    _finite_tree(payload)
    return {
        "seeds": aggregate["seeds"],
        "context_accuracy_mean": aggregate[
            "mqr_context_personalized_accuracy_mean"
        ],
        "no_memory_accuracy_mean": aggregate["reset_before_query_accuracy_mean"],
        "first_b_nll_mean": aggregate["session_b_first_prequential_nll_mean"],
        "final_b_nll_mean": aggregate["session_b_final_nll_mean"],
        "protected_policy_kl_max": aggregate[
            "protected_old_domain_policy_kl_max"
        ],
        "application_mechanism_qualified": True,
    }


def verify_minicpm(payload: Dict[str, Any]) -> Dict[str, Any]:
    assert payload["schema_version"] == 1
    assert payload["experiment"] == "unified_temporal_mqr_minicpm_go"
    assert payload["invariants"]["prediction_before_update"] is True
    assert payload["invariants"]["backbone_frozen"] is True
    assert payload["lora"]["frozen_parameter_probe_max_abs_drift"] == 0.0
    assert payload["mqr_effective"] is False
    return {
        "checkpoint_path": payload["config"].get("model_path"),
        "frozen_parameter_probe_max_abs_drift": payload["lora"][
            "frozen_parameter_probe_max_abs_drift"
        ],
        "prediction_before_update": True,
        "connectivity_smoke_passed": True,
        "task_efficacy_proven": False,
        "mqr_effective": False,
    }


def audit_go_competitiveness(
    payloads: Sequence[Dict[str, Any]],
    joint: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit completion separately from the negative competitiveness verdict."""

    assert len(payloads) == 2
    scales: Dict[str, Any] = {}
    all_methods = set()
    for payload in payloads:
        assert payload["schema_version"] == 2
        assert payload["experiment"] == "unified_temporal_mqr_go_competitive"
        assert payload["complete"] is True
        config = payload["config"]
        board_size = int(config["board_size"])
        assert board_size in (10, 13)
        methods = tuple(payload["methods"])
        all_methods.update(methods)
        present = sorted(
            {METHOD_ALIASES[name] for name in methods if name in METHOD_ALIASES}
        )
        missing = [name for name in REQUIRED_BASELINES if name not in present]
        protocol = payload["protocol_gate"]
        strict = payload["resources"]["strict_gate"]
        gate = payload["effectiveness_gate"]
        seed_count = len(config["seeds"])
        assert 10 <= seed_count <= 20
        assert bool(config["formal"]) is True
        assert protocol["formal_protocol_pass"] is True
        assert protocol["feedback_and_gradient_budgets_equal"] is True
        assert strict["all_resources_matched"] is True
        assert strict["recurrent_state_byte_ratio"] == 1.0
        assert not missing
        assert gate["online_learning"]["supported"] is True
        assert gate["mqr_effective_on_this_scale"] is False
        assert gate["mqr_effective"] is False
        scales[str(board_size)] = {
            "board_size": board_size,
            "seed_count": seed_count,
            "train_exposures_per_task": int(config["train_exposures"]),
            "probe_exposures_per_task": int(config["probe_exposures"]),
            "unmasked_games_per_stage": int(config["eval_games"]),
            "present_same_protocol_baselines": present,
            "missing_same_protocol_baselines": missing,
            "formal_protocol_pass": True,
            "strict_resource_gate": strict,
            "online_learning_supported": True,
            "mqr_effective_on_this_scale": False,
        }

    assert set(scales) == {"10", "13"}
    assert joint["same_protocol_except_board_resources"] is True
    assert joint["mqr_effective"] is False
    assert len(joint["scales"]) == 2
    assert all(item["formal_protocol_pass"] for item in joint["scales"])
    assert all(item["resource_gate_pass"] for item in joint["scales"])
    assert all(item["online_learning_supported"] for item in joint["scales"])
    assert not any(item["mqr_effective_on_this_scale"] for item in joint["scales"])
    present_all = sorted(
        {METHOD_ALIASES[name] for name in all_methods if name in METHOD_ALIASES}
    )
    return {
        "artifact_kind": "formal_10x10_and_13x13_replication",
        "scales": scales,
        "present_same_protocol_baselines": present_all,
        "missing_same_protocol_baselines": [],
        "requirements": {
            "10_to_20_seeds": True,
            "primary_board_at_least_10x10": True,
            "13x13_scaling_replication": True,
            "all_required_baselines": True,
            "same_parameter_state_byte_flop_budget": True,
            "same_feedback_budget": True,
            "mqr_joint_performance_gate": False,
        },
        "same_protocol_except_board_resources": True,
        "formal_experiment_complete": True,
        "online_learning_supported": True,
        "mqr_effective": False,
        "independent_competitiveness_established": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application", type=Path, default=DEFAULT_APPLICATION)
    parser.add_argument(
        "--go-results",
        type=Path,
        nargs=2,
        default=list(DEFAULT_GO_RESULTS),
    )
    parser.add_argument("--go-joint", type=Path, default=DEFAULT_GO_JOINT)
    parser.add_argument("--minicpm-result", type=Path, default=DEFAULT_MINICPM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    application = verify_application(_load(args.application))
    minicpm = verify_minicpm(_load(args.minicpm_result))
    competition = audit_go_competitiveness(
        [_load(path) for path in args.go_results],
        _load(args.go_joint),
    )
    payload = {
        "schema_version": 2,
        "purpose": "audit_current_mqr_application_positioning_and_competitive_evidence",
        "source_artifacts": {
            "application_qualification": {
                "path": str(args.application.relative_to(ROOT)),
                "sha256": _sha256(args.application),
            },
            "minicpm_connectivity_smoke": {
                "path": str(args.minicpm_result.relative_to(ROOT)),
                "sha256": _sha256(args.minicpm_result),
            },
            "formal_go_results": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": _sha256(path),
                }
                for path in args.go_results
            ],
            "formal_go_joint_verification": {
                "path": str(args.go_joint.relative_to(ROOT)),
                "sha256": _sha256(args.go_joint),
            },
        },
        "application_mechanism": application,
        "real_frozen_backbone_connectivity": minicpm,
        "formal_competitiveness": competition,
        "verdict": {
            "positioning_is_consistent_with_available_evidence": True,
            "best_current_scope": (
                "bounded frozen-model online personalization, session memory, "
                "and feedback-driven correction sidecar"
            ),
            "generic_sidecar_transaction_is_caller_owned": True,
            "real_model_application_gain_established": False,
            "general_capability_enhancement_established": False,
            "animal_like_learning_established": False,
            "independent_mqr_algorithm_advantage_established": False,
            "formal_competitive_experiment_complete": True,
        },
        "interpretation": (
            "The positioning is a justified engineering/research target, not a "
            "demonstrated deployment benefit. Synthetic application contracts "
            "pass and a real MiniCPM backbone remains frozen in smoke testing. "
            "Formal 10x10/13x13 experiments now pass protocol and allocation "
            "gates and demonstrate online loss reduction, but the joint MQR "
            "performance gate remains negative."
        ),
    }
    assert payload["verdict"]["positioning_is_consistent_with_available_evidence"]
    assert not payload["verdict"]["independent_mqr_algorithm_advantage_established"]
    assert payload["verdict"]["formal_competitive_experiment_complete"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
