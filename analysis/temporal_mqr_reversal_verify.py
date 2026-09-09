#!/usr/bin/env python3
"""Independent structural and arithmetic checks for the reversal benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence


METHODS = (
    "mqr_utility",
    "mqr_utility_ogd",
    "mqr_context_utility_ogd",
    "mqr_expert_utility",
    "mqr_expert_utility_ogd",
    "gru",
    "fast_weight",
    "lora",
    "ogd_lora",
    "replay_lora",
)
SEEDS = (43, 71, 101, 131, 173)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("analysis/results/temporal_mqr_reversal_matched.json"),
    )
    return parser.parse_args()


def _finite(value: Any) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)
    elif isinstance(value, float):
        assert math.isfinite(value), value


def _close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    assert abs(float(actual) - float(expected)) <= tolerance, (actual, expected)


def _mean(runs: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float:
    values = []
    for run in runs:
        value: Any = run
        for key in path:
            value = value[key]
        values.append(float(value))
    return statistics.fmean(values)


def _exact_bootstrap_ci(values: Sequence[float]) -> tuple[float, float]:
    """Enumerate all n**n paired bootstrap means and use linear quantiles."""

    count = len(values)
    means = sorted(
        statistics.fmean(values[index] for index in indices)
        for indices in product(range(count), repeat=count)
    )

    def quantile(probability: float) -> float:
        position = (len(means) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        return means[lower] * (1.0 - fraction) + means[upper] * fraction

    return quantile(0.025), quantile(0.975)


def main() -> None:
    path = _args().path
    raw = path.read_bytes()
    payload = json.loads(raw)
    _finite(payload)
    assert payload["schema_version"] == 1
    assert payload["configuration"]["seeds"] == list(SEEDS)
    assert payload["configuration"]["methods"] == list(METHODS)
    assert payload["protocol"]["feedback_order"].startswith("predict with theta_t")
    assert payload["protocol"]["cross_task_forgetting_interpretable"] is True
    assert payload["protocol"]["reversal_context"] == "warmup-shift"
    assert payload["protocol"]["resource_matching"]["adaptive_parameter_range"] == [
        288,
        320,
    ]
    assert payload["protocol"]["resource_matching"]["tensor_online_state_cap_bytes"] == 4096

    audits = payload["seed_audits"]
    assert [item["seed"] for item in audits] == list(SEEDS)
    for item in audits:
        assert all(item["initialization_audit"].values())
        assert len(item["stream_sha256"]) == 5
        assert len(set(item["stream_sha256"].values())) == 5

    runs = payload["runs"]
    assert len(runs) == len(SEEDS) * len(METHODS)
    indexed = {(int(run["seed"]), str(run["method"])): run for run in runs}
    assert set(indexed) == {(seed, method) for seed in SEEDS for method in METHODS}
    for run in runs:
        resource = run["resource"]
        assert 288 <= resource["adaptive_parameters_including_calibration"] <= 320
        assert resource["peak_tensor_online_state_bytes"] <= 4096
        assert resource["parameter_cap_satisfied"] is True
        assert resource["tensor_byte_cap_satisfied"] is True
        for boundary in run["boundaries"].values():
            for task in boundary.values():
                assert task["parameter_max_drift"] == 0.0
        if run["method"].startswith("mqr_"):
            assert float(run["max_unitary_error"]) < 1e-4
            assert float(run["max_stochastic_error"]) < 1e-4

    for seed in SEEDS:
        for method in ("mqr_expert_utility", "mqr_expert_utility_ogd"):
            run = indexed[(seed, method)]
            isolation = run["expert_isolation"]
            assert isolation["exact_old_expert_parameter_isolation"] is True
            assert isolation["new_expert_did_learn"] is True
            assert isolation["expert_zero_max_abs_drift_during_task_b"] == 0.0
            assert isolation["expert_one_max_abs_drift_during_task_b"] > 0.0
            assert run["phases"]["task_a"]["repeat_expert_one_rate"] == 0.0
            assert run["phases"]["task_b_reversal"]["repeat_expert_one_rate"] == 1.0
        assert (
            indexed[(seed, "mqr_expert_utility_ogd")]["derived"][
                "task_a_probe_drop_after_b"
            ]
            == 0.0
        )
        assert indexed[(seed, "lora")]["derived"]["task_a_probe_drop_after_b"] > 0.30

    aggregate = payload["aggregate"]["by_method"]
    aggregate_paths = {
        "task_a_probe_drop_after_b": ("derived", "task_a_probe_drop_after_b"),
        "task_a_equal_training_forgetting": (
            "derived",
            "task_a_equal_training_forgetting",
        ),
        "mean_post_task_accuracy": ("derived", "mean_post_task_accuracy"),
        "final_balanced_accuracy": ("derived", "final_balanced_accuracy"),
        "task_b_reversal_last_quarter": (
            "phases",
            "task_b_reversal",
            "last_quarter_accuracy",
        ),
    }
    for method in METHODS:
        selected = [indexed[(seed, method)] for seed in SEEDS]
        for aggregate_name, run_path in aggregate_paths.items():
            _close(
                aggregate[method][aggregate_name]["mean"],
                _mean(selected, run_path),
            )

    best = max(
        ("gru", "fast_weight", "lora", "ogd_lora", "replay_lora"),
        key=lambda method: aggregate[method]["mean_post_task_accuracy"]["mean"],
    )
    gates = payload["claim_gates"]
    assert best == gates["best_baseline_by_mean_post_task_accuracy"] == "lora"
    contrast = payload["aggregate"]["paired_contrasts"][
        "mqr_expert_utility_ogd_vs_lora"
    ]
    accuracy_lower = contrast["mean_post_task_accuracy_delta"][
        "paired_bootstrap_95ci"
    ][0]
    forgetting_lower = contrast["task_a_probe_drop_reduction"][
        "paired_bootstrap_95ci"
    ][0]
    assert gates["mqr_noninferior_to_best_baseline_at_5pp_margin"] == (
        accuracy_lower > -0.05
    )
    assert gates["mqr_strict_accuracy_superiority_to_best_baseline"] == (
        accuracy_lower > 0.0
    )
    assert gates["mqr_strict_forgetting_reduction_vs_best_baseline"] == (
        forgetting_lower > 0.0
    )
    assert gates["independent_competitive_advantage_established"] is False
    assert aggregate["mqr_expert_utility_ogd"]["resource"][
        "total_sidecar_parameters"
    ] == 1354
    assert aggregate["lora"]["resource"]["total_sidecar_parameters"] == 300
    assert (
        aggregate["mqr_expert_utility_ogd"]["milliseconds_per_episode"]["mean"]
        / aggregate["lora"]["milliseconds_per_episode"]["mean"]
        > 70.0
    )
    final_balance_values = [
        float(indexed[(seed, "mqr_expert_utility_ogd")]["derived"][
            "final_balanced_accuracy"
        ])
        - float(indexed[(seed, "lora")]["derived"]["final_balanced_accuracy"])
        for seed in SEEDS
    ]
    final_balance = _exact_bootstrap_ci(final_balance_values)
    _close(statistics.fmean(final_balance_values), 0.11625)
    _close(final_balance[0], 0.04625)
    _close(final_balance[1], 0.17375)
    mean_accuracy = contrast["mean_post_task_accuracy_delta"][
        "paired_bootstrap_95ci"
    ]
    assert final_balance[0] > 0.0
    assert mean_accuracy[1] < 0.0

    digest = hashlib.sha256(raw).hexdigest()
    print(f"verified {path}")
    print(f"sha256={digest}")
    print(
        "claim: expert MQR has strict forgetting reduction but fails the "
        "pre-registered overall accuracy/non-inferiority gate"
    )


if __name__ == "__main__":
    main()
