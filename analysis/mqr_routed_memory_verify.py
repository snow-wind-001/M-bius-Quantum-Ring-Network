#!/usr/bin/env python3
"""Fail-closed verifier for routed-memory qualification JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def _mean(values: Iterable[float]) -> float:
    parsed = [float(value) for value in values]
    if not parsed:
        raise ValueError("cannot summarize an empty sample")
    return sum(parsed) / len(parsed)


def _t95(count: int) -> float:
    values = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
    }
    return values.get(count, 1.96)


def _summary(values: Sequence[float]) -> Dict[str, float | int | str]:
    parsed = [float(value) for value in values]
    mean = _mean(parsed)
    if len(parsed) == 1:
        radius = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in parsed) / (len(parsed) - 1)
        radius = _t95(len(parsed)) * math.sqrt(variance / len(parsed))
    return {
        "mean": mean,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
        "n": len(parsed),
        "ci_method": "two_sided_student_t_95",
    }


def _assert_summary(recorded: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    if int(recorded["n"]) != int(expected["n"]):
        raise AssertionError(f"{label}: sample count mismatch")
    if recorded["ci_method"] != expected["ci_method"]:
        raise AssertionError(f"{label}: CI method mismatch")
    for field in ("mean", "ci95_low", "ci95_high"):
        if not math.isclose(
            float(recorded[field]),
            float(expected[field]),
            rel_tol=1e-11,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"{label}: {field} mismatch")


def verify(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise AssertionError("unsupported schema_version")
    protocol = payload["protocol"]
    runs = payload["runs"]
    aggregate = payload["aggregate"]
    if len(runs) != int(protocol["seeds"]):
        raise AssertionError("run count does not match protocol seeds")
    if len({int(run["seed"]) for run in runs}) != len(runs):
        raise AssertionError("seed identifiers must be unique")
    expected_status = (
        "formal"
        if len(runs) >= 10 and int(protocol["eval_episodes"]) >= 64
        else "smoke_or_pilot"
    )
    if payload["status"] != expected_status:
        raise AssertionError("status does not match sample size")
    if protocol["gate_has_explicit_event_marker"] is not False:
        raise AssertionError("formal gate may not receive an event marker")
    if protocol["gate_has_query_or_phase_feature"] is not False:
        raise AssertionError("formal gate may not receive query/phase features")
    if protocol["critic_target"] != "exact paired finite-horizon write/no-write return":
        raise AssertionError("critic target contract changed")
    if protocol["address_commit_projection"] != "straight_through_top1":
        raise AssertionError("address projection contract changed")

    methods = (
        "mqr",
        "identity",
        "gru",
        "fast_weight",
        "mqr_identity_replacement",
        "fixed_ring_buffer",
    )
    metrics = (
        "accuracy",
        "forward_accuracy",
        "context_reversal_accuracy",
        "repeated_query_accuracy",
        "runtime_ms_per_episode",
    )
    for method in methods:
        for metric in metrics:
            expected = _summary([run["methods"][method][metric] for run in runs])
            recorded = aggregate["method_summaries"][method][metric]
            _assert_summary(recorded, expected, f"{method}.{metric}")

    paired = {}
    for control in ("identity", "gru", "fast_weight", "fixed_ring_buffer"):
        expected = _summary(
            [
                run["methods"]["mqr"]["accuracy"]
                - run["methods"][control]["accuracy"]
                for run in runs
            ]
        )
        recorded = aggregate["paired_mqr_minus_control_accuracy"][control]
        _assert_summary(recorded, expected, f"mqr-minus-{control}")
        paired[control] = expected
    replacement = _summary(
        [
            run["methods"]["mqr"]["accuracy"]
            - run["methods"]["mqr_identity_replacement"]["accuracy"]
            for run in runs
        ]
    )
    _assert_summary(
        aggregate["identity_replacement_accuracy_drop"],
        replacement,
        "identity replacement",
    )

    gate_names = (
        "auprc",
        "brier",
        "ece",
        "write_rate",
        "event_write_rate",
        "distractor_write_rate",
        "event_distractor_write_gap",
        "budget_violation",
        "positive_rate",
    )
    gate = {}
    for name in gate_names:
        expected = _summary(
            [run["methods"]["mqr"]["gate_calibration"][name] for run in runs]
        )
        _assert_summary(
            aggregate["gate_calibration_summaries"][name],
            expected,
            f"gate.{name}",
        )
        gate[name] = expected

    resources_pass = True
    for run in runs:
        audit = run["resource_audit"]
        methods_audit = audit["methods"]
        for method in methods_audit.values():
            if int(method["state_bytes"]) != int(protocol["same_state_floats_all_cores"]) * 4:
                raise AssertionError("state byte budget mismatch")
        for control, ratios in audit["mqr_to_control_ratios"].items():
            for field, recorded_ratio in ratios.items():
                expected_ratio = float(methods_audit["mqr"][field]) / max(
                    float(methods_audit[control][field]), 1.0
                )
                if not math.isclose(
                    float(recorded_ratio), expected_ratio, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise AssertionError(f"resource ratio mismatch: {control}.{field}")
                resources_pass = resources_pass and expected_ratio <= 1.05 + 1e-12
        if bool(audit["one_sided_all_resources_within_1_05"]) != resources_pass:
            raise AssertionError("per-run resource gate mismatch")

    mqr = aggregate["method_summaries"]["mqr"]
    cyclic = _summary(
        [run["methods"]["mqr"]["cyclic_address_accuracy"] for run in runs]
    )
    expected_gates = {
        "formal_seed_count_at_least_10": len(runs) >= 10,
        "formal_episode_count_at_least_64": int(protocol["eval_episodes"]) >= 64,
        "long_delay_at_least_four_slot_cycles": int(protocol["eval_delay"])
        >= 4 * int(protocol["slots"]),
        "residual_identity_channel_present": 0.0 < float(protocol["epsilon"]) < 1.0,
        "no_explicit_event_or_query_marker_in_gate": True,
        "mqr_accuracy_ci_above_0_75": float(mqr["accuracy"]["ci95_low"]) > 0.75,
        "long_delay_forward_ci_above_0_75": float(mqr["forward_accuracy"]["ci95_low"])
        > 0.75,
        "context_reversal_ci_above_0_75": float(
            mqr["context_reversal_accuracy"]["ci95_low"]
        )
        > 0.75,
        "repeated_query_ci_above_0_75": float(
            mqr["repeated_query_accuracy"]["ci95_low"]
        )
        > 0.75,
        "beats_identity": float(paired["identity"]["ci95_low"]) > 0.0,
        "beats_gru_or_fast_weight": float(paired["gru"]["ci95_low"]) > 0.0
        or float(paired["fast_weight"]["ci95_low"]) > 0.0,
        "identity_replacement_hurts": float(replacement["ci95_low"]) > 0.0,
        "event_distractor_write_gap_positive": float(
            gate["event_distractor_write_gap"]["ci95_low"]
        )
        > 0.0,
        "write_budget_never_violated": float(gate["budget_violation"]["ci95_high"])
        <= 1e-12,
        "utility_auprc_above_prevalence": float(gate["auprc"]["ci95_low"])
        > float(gate["positive_rate"]["ci95_high"]),
        "cyclic_address_accuracy_above_0_99": float(cyclic["ci95_low"]) > 0.99,
        "content_zero_drift": all(
            float(run["methods"]["mqr"]["no_write_content_drift_linf"]) == 0.0
            for run in runs
        ),
        "all_peak_and_amortized_resources_within_1_05": resources_pass,
    }
    expected_gates["mqr_route_mechanism_qualified"] = all(expected_gates.values())
    if aggregate["decision_gates"] != expected_gates:
        raise AssertionError("decision gates do not match raw evidence")
    expected_independent = {
        "route_mechanism_qualified": expected_gates["mqr_route_mechanism_qualified"],
        "beats_direct_fixed_ring_buffer": float(
            paired["fixed_ring_buffer"]["ci95_low"]
        )
        > 0.0,
    }
    expected_independent["mqr_independent_algorithm_advantage"] = all(
        expected_independent.values()
    )
    if aggregate["independent_advantage_gates"] != expected_independent:
        raise AssertionError("independent-advantage gates mismatch")
    if bool(aggregate["mqr_route_mechanism_qualified"]) != bool(
        expected_gates["mqr_route_mechanism_qualified"]
    ):
        raise AssertionError("route qualification flag mismatch")
    if bool(aggregate["mqr_independent_algorithm_advantage"]) != bool(
        expected_independent["mqr_independent_algorithm_advantage"]
    ):
        raise AssertionError("independent advantage flag mismatch")
    return {
        "verified": True,
        "status": payload["status"],
        "seeds": len(runs),
        "route_mechanism_qualified": expected_gates["mqr_route_mechanism_qualified"],
        "independent_algorithm_advantage": expected_independent[
            "mqr_independent_algorithm_advantage"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.result), indent=2))


if __name__ == "__main__":
    main()
