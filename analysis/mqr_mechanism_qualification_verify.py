#!/usr/bin/env python3
"""Fail-closed verifier for the Temporal MQR mechanism qualification JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METHODS = (
    "legacy_dense_truncated",
    "dense_nonflat_truncated",
    "dense_nonflat_trace",
    "cyclic_trace",
    "cyclic_trace_ogd",
    "identity_trace",
    "gru_trace",
    "lstm_trace",
    "fast_weight_trace",
    "sinkhorn_trace",
)
COMPETITIVE_METHODS = (
    "cyclic_trace",
    "identity_trace",
    "gru_trace",
    "lstm_trace",
    "fast_weight_trace",
    "sinkhorn_trace",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


_T975_BY_DF = (
    12.706205, 4.302653, 3.182446, 2.776445, 2.570582,
    2.446912, 2.364624, 2.306004, 2.262157, 2.228139,
    2.200985, 2.178813, 2.160369, 2.144787, 2.131450,
    2.119905, 2.109816, 2.100922, 2.093024, 2.085963,
    2.079614, 2.073873, 2.068658, 2.063899, 2.059539,
    2.055529, 2.051831, 2.048407, 2.045230, 2.042272,
)


def _summary(values: Sequence[float]) -> dict[str, float | int | str]:
    mean = _mean(values)
    if len(values) <= 1:
        radius = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        df = min(len(values) - 1, len(_T975_BY_DF))
        radius = _T975_BY_DF[df - 1] * math.sqrt(variance / len(values))
    return {
        "mean": mean,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
        "n": len(values),
        "ci_method": "two_sided_student_t_95",
    }


def _assert_summary(stored: Mapping[str, Any], raw: Sequence[float], name: str) -> None:
    expected = _summary(raw)
    for field in ("mean", "ci95_low", "ci95_high"):
        _require(
            math.isclose(
                float(stored[field]), float(expected[field]), rel_tol=0.0, abs_tol=1e-12
            ),
            f"{name} {field} mismatch",
        )
    _require(int(stored["n"]) == len(raw), f"{name} n mismatch")
    _require(stored["ci_method"] == expected["ci_method"], f"{name} CI method mismatch")


def _ratio(values: Sequence[float]) -> float:
    low, high = min(values), max(values)
    if low == high == 0.0:
        return 1.0
    return float("inf") if low <= 0.0 else high / low


def verify(result: Mapping[str, Any]) -> None:
    _require(int(result.get("schema_version", 0)) == 2, "unsupported schema")
    protocol = result["protocol"]
    _require(protocol["predict_before_update"] is True, "causal transaction disabled")
    _require(protocol["future_label_in_gate_features"] is False, "gate label leakage")
    _require(protocol["query_identical_across_hidden_bits"] is True, "query is not isolated")
    _require(protocol["explicit_event_marker"] is False, "explicit marker is enabled")
    _require(protocol["ci_method"] == "two_sided_student_t_95", "wrong CI method")
    _require(
        protocol["independent_advantage_requires_all_non_mqr_controls"] is True,
        "non-MQR gate is not intersectional",
    )
    _require(protocol["forgetting_improvement_required"] is True, "forgetting gate missing")
    _require(
        tuple(protocol["competitive_methods"]) == COMPETITIVE_METHODS,
        "competitive method set mismatch",
    )
    query_indices = tuple(int(value) for value in protocol["repeated_query_indices"])
    _require(len(query_indices) >= 2, "repeated-query task is missing")
    runs = list(result["runs"])
    _require(len(runs) == int(protocol["seeds"]), "seed count mismatch")
    _require(len({int(run["seed"]) for run in runs}) == len(runs), "duplicate seeds")

    for run in runs:
        methods = run["methods"]
        _require(set(methods) == set(METHODS), "method set mismatch")
        diagnostics = run["initial_transition_diagnostics"]
        for method in ("dense_nonflat_trace", "cyclic_trace"):
            slow = diagnostics[method][1]
            _require(slow["active"] is True, f"{method} slow transition is inactive")
            _require(
                int(slow["modulus_square_jacobian_rank"]) > 0,
                f"{method} has a zero transition Jacobian",
            )
            _require(
                float(slow["modulus_square_jacobian_largest_singular_value"]) > 0.0,
                f"{method} Jacobian scale is zero",
            )
        cyclic = diagnostics["cyclic_trace"][1]
        _require(cyclic["literal_cycle_adjacency"] is True, "cyclic core is not literal")
        _require(
            int(cyclic["raw_parameter_count"])
            < int(diagnostics["dense_nonflat_trace"][1]["raw_parameter_count"]),
            "cyclic transition is not lower parameter than dense Cayley",
        )

        for method, record in methods.items():
            _require(record["max_unitary_error"] < 1e-4, f"{method} unitary error")
            _require(record["max_stochastic_error"] < 1e-4, f"{method} stochastic error")
            scalar_values = (
                record["paired_history"]["accuracy"],
                record["paired_history"]["probability_separation"],
                record["task_a_forgetting"],
                record["mean_train_loss_first_quarter"],
                record["mean_train_loss_last_quarter"],
                record["runtime_seconds"],
            )
            _require(_finite(scalar_values), f"{method} contains nonfinite metrics")
            _require(
                0.0 <= record["paired_history"]["accuracy"] <= 1.0,
                f"{method} accuracy is out of range",
            )
            paired_episode_count = 2 * int(protocol["paired_episodes"])
            history = record["paired_history"]
            _require(
                int(history["event_observations"]) == paired_episode_count,
                f"{method} event count mismatch",
            )
            _require(
                int(history["query_observations"])
                == paired_episode_count * len(query_indices),
                f"{method} repeated query count mismatch",
            )
            _require(
                int(history["distractor_observations"])
                == paired_episode_count
                * (int(protocol["sequence_length"]) - 1 - len(query_indices)),
                f"{method} distractor count mismatch",
            )
        _require(
            methods["legacy_dense_truncated"]["mean_future_injection_gradient"] == 0.0,
            "truncated control unexpectedly reports future credit",
        )
        _require(
            methods["dense_nonflat_truncated"]["mean_future_transition_gradient"] == 0.0,
            "truncated non-flat control unexpectedly reports future credit",
        )
        _require(
            methods["cyclic_trace"]["mean_future_injection_gradient"] > 0.0,
            "trace credit does not reach injection",
        )
        _require(
            methods["cyclic_trace"]["mean_future_transition_gradient"] > 0.0,
            "trace credit does not reach transition",
        )
        _require(
            methods["cyclic_trace"]["mean_earliest_input_future_gradient"] > 0.0,
            "future loss does not reach earliest observation",
        )

    aggregate = result["aggregate"]
    for method in METHODS:
        stored = aggregate["method_summaries"][method]["history_accuracy"]
        raw = [
            float(run["methods"][method]["paired_history"]["accuracy"])
            for run in runs
        ]
        _assert_summary(stored, raw, f"{method} history accuracy")

    recurrent_controls = ("gru_trace", "lstm_trace", "fast_weight_trace", "sinkhorn_trace")
    paired_accuracy = aggregate["paired_cyclic_minus_control_accuracy"]
    paired_forgetting = aggregate["paired_control_minus_cyclic_forgetting"]
    for control in ("identity_trace", *recurrent_controls):
        accuracy_raw = [
            float(run["methods"]["cyclic_trace"]["paired_history"]["accuracy"])
            - float(run["methods"][control]["paired_history"]["accuracy"])
            for run in runs
        ]
        forgetting_raw = [
            float(run["methods"][control]["task_a_forgetting"])
            - float(run["methods"]["cyclic_trace"]["task_a_forgetting"])
            for run in runs
        ]
        _assert_summary(paired_accuracy[control], accuracy_raw, f"accuracy vs {control}")
        _assert_summary(paired_forgetting[control], forgetting_raw, f"forgetting vs {control}")

    for run in runs:
        audits = list(run["resource_audits"])
        ratio_fields = {
            "trainable_parameter_ratio": "trainable_parameters",
            "recurrent_state_byte_ratio": "persistent_state_bytes",
            "agent_forward_mac_ratio": "estimated_agent_forward_macs",
            "allocated_update_mac_ratio": "allocated_update_budget_macs",
            "frozen_parameter_ratio": "frozen_parameters",
            "continual_capacity_ratio": "allocated_continual_memory_bytes",
        }
        for gate_name, selected in (
            ("diagnostic_resource_gate", audits),
            (
                "competitive_resource_gate",
                [row for row in audits if row["method"] in COMPETITIVE_METHODS],
            ),
        ):
            gate = run[gate_name]
            for ratio_name, audit_name in ratio_fields.items():
                expected = _ratio([float(row[audit_name]) for row in selected])
                _require(
                    math.isclose(
                        float(gate[ratio_name]), expected, rel_tol=0.0, abs_tol=1e-12
                    ),
                    f"{gate_name} ratio mismatch: {ratio_name}",
                )

    gates = aggregate["decision_gates"]
    summaries = aggregate["method_summaries"]
    mqr = summaries["cyclic_trace"]
    expected_gates = {
        "seed_count_at_least_10": len(runs) >= 10,
        "formal_protocol_satisfied": bool(
            len(runs) >= 10
            and int(protocol["sequence_length"]) >= 10
            and int(protocol["train_episodes"]) >= 160
            and int(protocol["reversal_episodes"]) >= 40
            and int(protocol["eval_episodes"]) >= 40
            and int(protocol["paired_episodes"]) >= 40
        ),
        "state_budget_is_192_bytes": (
            int(protocol["ring_dim"]) * 2 * 4 == 192
        ),
        "competitive_resources_within_1_05": all(
            bool(run["competitive_resource_gate"]["all_resources_matched"])
            for run in runs
        ),
        "history_ci_above_chance": float(mqr["history_accuracy"]["ci95_low"]) > 0.5,
        "history_separation_ci_positive": (
            float(mqr["history_probability_separation"]["ci95_low"]) > 0.0
        ),
        "reversal_accuracy_ci_above_chance": (
            float(mqr["task_b_reversed_accuracy"]["ci95_low"]) > 0.5
        ),
        "reversal_gain_ci_positive": float(mqr["task_b_reversal_gain"]["ci95_low"]) > 0.0,
        "learned_write_not_collapsed": (
            0.05 < float(mqr["learned_write_rate"]["mean"]) < 0.95
        ),
        "future_credit_reaches_injection": (
            float(mqr["future_injection_gradient"]["mean"]) > 1e-8
        ),
        "future_credit_reaches_transition": (
            float(mqr["future_transition_gradient"]["mean"]) > 1e-8
        ),
        "identity_replacement_hurts": (
            float(mqr["identity_replacement_accuracy_drop"]["ci95_low"]) > 0.0
        ),
        "beats_identity": float(paired_accuracy["identity_trace"]["ci95_low"]) > 0.0,
        "beats_all_non_mqr_controls": all(
            float(paired_accuracy[control]["ci95_low"]) > 0.0
            for control in recurrent_controls
        ),
        "reduces_forgetting_vs_identity": (
            float(paired_forgetting["identity_trace"]["ci95_low"]) > 0.0
        ),
        "reduces_forgetting_vs_all_non_mqr_controls": all(
            float(paired_forgetting[control]["ci95_low"]) > 0.0
            for control in recurrent_controls
        ),
    }
    _require(
        set(gates) == {*expected_gates, "mqr_independent_advantage"},
        "decision gate set contains stale or missing criteria",
    )
    for name, expected in expected_gates.items():
        _require(bool(gates[name]) == bool(expected), f"decision gate mismatch: {name}")
    derived_effective = all(
        bool(value) for name, value in gates.items() if name != "mqr_independent_advantage"
    )
    _require(
        bool(gates["mqr_independent_advantage"]) == derived_effective,
        "decision gate is internally inconsistent",
    )
    _require(
        bool(aggregate["mqr_effective"]) == derived_effective,
        "mqr_effective does not equal preregistered gates",
    )
    formal = bool(
        len(runs) >= 10
        and int(protocol["sequence_length"]) >= 10
        and int(protocol["train_episodes"]) >= 160
        and int(protocol["reversal_episodes"]) >= 40
        and int(protocol["eval_episodes"]) >= 40
        and int(protocol["paired_episodes"]) >= 40
    )
    expected_status = "formal_mechanism_qualification" if formal else "smoke_or_pilot"
    _require(result["status"] == expected_status, "evidence status is overstated")
    _require(
        len(result["preserved_historical_evidence"]) == 3,
        "historical negative evidence references are missing",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("analysis/results/mqr_mechanism_qualification_smoke.json"),
    )
    args = parser.parse_args()
    result = json.loads(args.path.read_text(encoding="utf-8"))
    verify(result)
    print(
        json.dumps(
            {
                "verified": True,
                "path": str(args.path),
                "status": result["status"],
                "seeds": len(result["runs"]),
                "mqr_effective": result["aggregate"]["mqr_effective"],
                "decision_gates": result["aggregate"]["decision_gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
