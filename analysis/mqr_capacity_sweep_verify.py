#!/usr/bin/env python3
"""Fail-closed verifier for the cyclic-MQR recurrent-capacity sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METHODS = ("cyclic_trace", "identity_trace")
STATE_BUDGETS = (48, 192, 768)


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


def verify(payload: Mapping[str, Any]) -> None:
    _require(int(payload.get("schema_version", 0)) == 2, "unsupported schema")
    _require(payload.get("experiment") == "mqr_cyclic_capacity_sweep", "wrong experiment")
    protocol = payload["protocol"]
    _require(protocol["predict_before_update"] is True, "noncausal update order")
    _require(protocol["explicit_event_marker"] is False, "event marker enabled")
    _require(protocol["future_label_in_gate_features"] is False, "gate label leakage")
    _require(protocol["ci_method"] == "two_sided_student_t_95", "wrong CI method")
    _require(tuple(protocol["methods"]) == METHODS, "method set mismatch")
    budgets = tuple(int(value) for value in protocol["state_bytes"])
    _require(len(set(budgets)) == len(budgets), "duplicate state budget")
    seeds = int(protocol["seeds"])
    runs = list(payload["runs"])
    _require(len(runs) == seeds * len(budgets), "run count mismatch")
    identities = {(int(run["state_bytes"]), int(run["seed"])) for run in runs}
    _require(len(identities) == len(runs), "duplicate budget/seed run")

    for run in runs:
        state_bytes = int(run["state_bytes"])
        _require(state_bytes in budgets, "unexpected state budget")
        _require(int(run["ring_dim"]) * 2 * 4 == state_bytes, "state layout mismatch")
        _require(set(run["methods"]) == set(METHODS), "run method set mismatch")
        for method, record in run["methods"].items():
            history = record["history"]
            _require(
                _finite(
                    (
                        history["accuracy"],
                        history["probability_separation"],
                        record["mean_first_quarter_loss"],
                        record["mean_last_quarter_loss"],
                        record["runtime_seconds"],
                    )
                ),
                f"{method} has nonfinite metrics",
            )
            _require(0.0 <= float(history["accuracy"]) <= 1.0, "accuracy out of range")
            _require(float(record["max_unitary_error"]) < 1e-4, "unitarity failure")
            _require(float(record["max_stochastic_error"]) < 1e-4, "stochasticity failure")
        _require(
            float(run["methods"]["cyclic_trace"]["mean_future_transition_gradient"]) > 0.0,
            "cyclic future loss does not reach transition",
        )
        _require(
            float(run["methods"]["identity_trace"]["mean_future_transition_gradient"]) == 0.0,
            "identity unexpectedly has transition gradients",
        )
        expected_refreshes = int(protocol["train_episodes"]) // int(
            protocol["transition_update_interval"]
        )
        _require(
            int(run["methods"]["cyclic_trace"]["transition_refreshes"])
            == expected_refreshes,
            "cyclic transition refresh schedule mismatch",
        )
        _require(
            int(run["methods"]["identity_trace"]["transition_refreshes"]) == 0,
            "identity reports transition refreshes",
        )

        resource = run["resources"]
        audits = resource["audits"]
        _require(tuple(item["method"] for item in audits) == METHODS, "audit order mismatch")
        _require(
            all(int(item["persistent_state_bytes"]) == state_bytes for item in audits),
            "resource state bytes mismatch",
        )
        expected_ratios = {
            "trainable_parameter_ratio": _ratio(
                [float(item["trainable_parameters"]) for item in audits]
            ),
            "recurrent_state_byte_ratio": _ratio(
                [float(item["persistent_state_bytes"]) for item in audits]
            ),
            "agent_forward_mac_ratio": _ratio(
                [float(item["estimated_agent_forward_macs"]) for item in audits]
            ),
            "amortized_update_mac_ratio": _ratio(
                [float(item["estimated_amortized_update_macs"]) for item in audits]
            ),
            "peak_update_mac_ratio": _ratio(
                [float(item["estimated_task_update_macs"]) for item in audits]
            ),
        }
        for name, expected in expected_ratios.items():
            _require(
                math.isclose(
                    float(resource["ratios"][name]), expected, rel_tol=0.0, abs_tol=1e-12
                ),
                f"resource ratio mismatch: {name}",
            )
        expected_average = bool(
            expected_ratios["trainable_parameter_ratio"] <= 1.05
            and expected_ratios["recurrent_state_byte_ratio"] == 1.0
            and expected_ratios["agent_forward_mac_ratio"] <= 1.05
            and expected_ratios["amortized_update_mac_ratio"] <= 1.05
        )
        _require(resource["average_resource_gate"] == expected_average, "average gate mismatch")
        _require(
            resource["peak_update_resource_gate"]
            == bool(expected_average and expected_ratios["peak_update_mac_ratio"] <= 1.05),
            "peak gate mismatch",
        )

    aggregate = payload["aggregate"]
    all_average_independent = []
    all_peak_independent = []
    for state_bytes in budgets:
        key = str(state_bytes)
        selected = [run for run in runs if int(run["state_bytes"]) == state_bytes]
        item = aggregate["budgets"][key]
        for method in METHODS:
            raw = [float(run["methods"][method]["history"]["accuracy"]) for run in selected]
            stored = item["methods"][method]["history_accuracy"]
            _assert_summary(stored, raw, f"{state_bytes}/{method} accuracy")
        paired = [
            float(run["methods"]["cyclic_trace"]["history"]["accuracy"])
            - float(run["methods"]["identity_trace"]["history"]["accuracy"])
            for run in selected
        ]
        _assert_summary(
            item["paired_cyclic_minus_identity_accuracy"],
            paired,
            f"{state_bytes} paired accuracy",
        )
        expected_average = all(
            bool(value)
            for name, value in item["decision_gates"].items()
            if name != "peak_update_resources_within_1_05"
        )
        expected_peak = all(bool(value) for value in item["decision_gates"].values())
        _require(
            bool(item["average_budget_advantage_at_budget"]) == expected_average,
            "average-budget decision is inconsistent",
        )
        _require(
            bool(item["peak_qualified_advantage_at_budget"]) == expected_peak,
            "peak-qualified decision is inconsistent",
        )
        all_average_independent.append(expected_average)
        all_peak_independent.append(expected_peak)
    _require(
        bool(aggregate["mqr_capacity_advantage"]) == all(all_average_independent),
        "cross-capacity average-budget decision is inconsistent",
    )
    _require(
        bool(aggregate["mqr_peak_qualified_capacity_advantage"])
        == all(all_peak_independent),
        "cross-capacity peak-qualified decision is inconsistent",
    )

    formal = bool(
        seeds >= 10
        and int(protocol["sequence_length"]) >= 10
        and int(protocol["train_episodes"]) >= 160
        and int(protocol["paired_episodes"]) >= 40
        and budgets == STATE_BUDGETS
    )
    _require(
        payload["status"] == ("formal_capacity_sweep" if formal else "smoke_or_pilot"),
        "evidence status is overstated",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("analysis/results/mqr_cyclic_capacity_sweep_formal_10seed.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    verify(payload)
    print(json.dumps({
        "verified": True,
        "path": str(args.path),
        "status": payload["status"],
        "mqr_capacity_advantage": payload["aggregate"]["mqr_capacity_advantage"],
    }, indent=2))


if __name__ == "__main__":
    main()
