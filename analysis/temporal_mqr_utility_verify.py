#!/usr/bin/env python3
"""Independent consistency and claim-boundary verifier for Utility MQR JSON."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "analysis/results/temporal_mqr_utility_digits.json"
EXPECTED_VARIANTS = (
    "learned_utility",
    "learned_utility_ogd",
    "learned_no_updates",
    "causal_novelty",
    "clairvoyant_oracle",
    "ungated",
    "random_sparse",
)


def close(left: float, right: float, *, tolerance: float = 1e-12) -> None:
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{left!r} != {right!r}")


def summary(values: Iterable[float]) -> Dict[str, float]:
    samples = list(values)
    return {
        "mean": statistics.fmean(samples),
        "sample_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def verify_summary(payload: Mapping[str, object], runs: Sequence[Mapping[str, object]]) -> None:
    metrics = (
        "accuracy",
        "first_quarter_accuracy",
        "last_quarter_accuracy",
        "target_write_rate",
        "distractor_write_rate",
        "positive_utility_write_rate",
        "utility_auprc",
        "semantic_target_auprc_diagnostic",
        "mean_realized_advantage",
        "mean_target_advantage",
        "mean_distractor_advantage",
        "mean_predicted_target_advantage",
        "mean_predicted_distractor_advantage",
        "mean_utility_loss",
        "milliseconds_per_episode",
    )
    reported = payload["summary"]
    assert isinstance(reported, dict)
    for variant in EXPECTED_VARIANTS:
        selected = [run for run in runs if run["variant"] == variant]
        entry = reported[variant]
        assert isinstance(entry, dict)
        for split in ("training", "evaluation"):
            for metric in metrics:
                expected = summary(
                    float(run[split][metric])  # type: ignore[index]
                    for run in selected
                )
                actual = entry[f"{split}_{metric}"]
                close(float(actual["mean"]), expected["mean"])
                close(float(actual["sample_std"]), expected["sample_std"])


def verify_contrasts(payload: Mapping[str, object], runs: Sequence[Mapping[str, object]]) -> None:
    indexed = {
        (int(run["seed"]), str(run["variant"])): float(
            run["evaluation"]["accuracy"]  # type: ignore[index]
        )
        for run in runs
    }
    definitions = {
        "learned_vs_no_updates": ("learned_utility", "learned_no_updates"),
        "learned_vs_ungated": ("learned_utility", "ungated"),
        "ogd_vs_plain": ("learned_utility_ogd", "learned_utility"),
        "ogd_vs_no_updates": ("learned_utility_ogd", "learned_no_updates"),
        "ogd_vs_ungated": ("learned_utility_ogd", "ungated"),
        "ogd_vs_random_sparse": ("learned_utility_ogd", "random_sparse"),
        "causal_heuristic_vs_learned": ("causal_novelty", "learned_utility"),
        "causal_heuristic_vs_ogd": ("causal_novelty", "learned_utility_ogd"),
        "clairvoyant_gap": ("clairvoyant_oracle", "learned_utility"),
        "clairvoyant_vs_ogd": ("clairvoyant_oracle", "learned_utility_ogd"),
        "random_sparse_vs_ungated": ("random_sparse", "ungated"),
    }
    reported = payload["paired_contrasts"]
    assert isinstance(reported, dict)
    seeds = sorted({seed for seed, _variant in indexed})
    for name, (left, right) in definitions.items():
        expected_values = [
            indexed[(seed, left)] - indexed[(seed, right)] for seed in seeds
        ]
        actual = reported[name]
        assert actual["left"] == left and actual["right"] == right
        assert len(actual["paired_differences"]) == len(expected_values)
        for reported_value, expected_value in zip(
            actual["paired_differences"], expected_values
        ):
            close(float(reported_value), expected_value)
        expected = summary(expected_values)
        close(float(actual["mean"]), expected["mean"])
        close(float(actual["sample_std"]), expected["sample_std"])


def main() -> None:
    raw = RESULT.read_bytes()
    payload = json.loads(raw)
    assert payload["schema_version"] == 1
    protocol = payload["protocol"]
    assert protocol["marker_free"] is True
    assert protocol["target_position_randomized"] is True
    assert "CE(no-write shadow" in protocol["utility_target"]
    configuration = payload["configuration"]
    seeds = [int(value) for value in configuration["seeds"]]
    assert seeds == [7, 17, 29]
    assert configuration["calibration_samples"] == 500
    assert configuration["train_episodes"] == 200
    assert configuration["eval_episodes"] == 100
    assert configuration["candidate_count"] == 5
    assert configuration["warmup_count"] == 2

    runs: List[Mapping[str, object]] = list(payload["runs"])
    assert len(runs) == len(seeds) * len(EXPECTED_VARIANTS)
    keys = {(int(run["seed"]), str(run["variant"])) for run in runs}
    assert keys == {(seed, variant) for seed in seeds for variant in EXPECTED_VARIANTS}
    for seed in seeds:
        assert payload["initialization_audit"][str(seed)][
            "all_variant_parameters_matched"
        ]
        calibration = payload["calibration"][str(seed)]
        assert calibration["count"] == 500
        assert calibration["last_quarter_accuracy"] >= 0.70

    verify_summary(payload, runs)
    verify_contrasts(payload, runs)

    for run in runs:
        assert run["total_parameters"] == 5521
        assert run["utility_gate_parameters"] == 49
        assert run["readout_parameters"] == 480
        assert run["total_state_dim"] == 48
        assert run["shadow_state_elements_per_candidate"] == 96
        assert float(run["evaluation_parameter_max_drift"]) == 0.0
        assert float(run["max_unitary_error"]) < 2e-6
        assert float(run["max_stochastic_error"]) < 1e-6
        evaluation = run["evaluation"]
        counts = evaluation["target_position_counts"]
        assert sum(int(value) for value in counts.values()) == 100
        assert all(int(counts[str(position)]) > 0 for position in range(5))
        if run["variant"] == "learned_utility_ogd":
            assert 0 < int(run["utility_ogd_rank"]) <= 8
            assert int(run["utility_ogd_bytes"]) > 0

    aggregate = payload["summary"]
    learned = aggregate["learned_utility"]
    learned_ogd = aggregate["learned_utility_ogd"]
    no_updates = aggregate["learned_no_updates"]
    ungated = aggregate["ungated"]
    causal = aggregate["causal_novelty"]
    oracle = aggregate["clairvoyant_oracle"]
    random_sparse = aggregate["random_sparse"]

    learned_accuracy = float(learned["evaluation_accuracy"]["mean"])
    ogd_accuracy = float(learned_ogd["evaluation_accuracy"]["mean"])
    assert learned_accuracy >= 0.50
    assert ogd_accuracy >= 0.65
    assert ogd_accuracy - float(no_updates["evaluation_accuracy"]["mean"]) >= 0.55
    assert ogd_accuracy - float(ungated["evaluation_accuracy"]["mean"]) >= 0.55
    assert ogd_accuracy - float(random_sparse["evaluation_accuracy"]["mean"]) >= 0.45
    close(ogd_accuracy, float(causal["evaluation_accuracy"]["mean"]))
    close(ogd_accuracy, float(oracle["evaluation_accuracy"]["mean"]))
    assert ogd_accuracy - learned_accuracy >= 0.10
    assert (
        float(learned_ogd["training_accuracy"]["mean"])
        - float(learned["training_accuracy"]["mean"])
        >= 0.15
    )
    close(float(learned_ogd["evaluation_target_write_rate"]["mean"]), 1.0)
    close(float(learned_ogd["evaluation_distractor_write_rate"]["mean"]), 0.0)
    close(
        float(learned_ogd["evaluation_semantic_target_auprc_diagnostic"]["mean"]),
        1.0,
    )
    assert float(learned_ogd["evaluation_mean_target_advantage"]["mean"]) > 0.5
    assert float(learned_ogd["evaluation_mean_distractor_advantage"]["mean"]) < -0.2
    assert (
        float(learned_ogd["evaluation_mean_predicted_target_advantage"]["mean"])
        > 0.5
    )
    assert (
        float(learned_ogd["evaluation_mean_predicted_distractor_advantage"]["mean"])
        < -0.2
    )
    assert float(learned_ogd["evaluation_utility_auprc"]["mean"]) >= 0.70

    digest = hashlib.sha256(raw).hexdigest()
    print("Utility MQR canonical result verified")
    print(f"sha256={digest}")
    print(f"learned_accuracy={100.0 * learned_accuracy:.2f}%")
    print(f"learned_ogd_accuracy={100.0 * ogd_accuracy:.2f}%")
    print(
        "ogd_vs_no_updates_pp="
        f"{100.0 * float(payload['paired_contrasts']['ogd_vs_no_updates']['mean']):.2f}"
    )
    print(
        "ogd_vs_ungated_pp="
        f"{100.0 * float(payload['paired_contrasts']['ogd_vs_ungated']['mean']):.2f}"
    )


if __name__ == "__main__":
    main()
