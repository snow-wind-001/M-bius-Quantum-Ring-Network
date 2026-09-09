#!/usr/bin/env python3
"""Independent arithmetic and integrity audit for learned-gate Digits results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


EXPECTED_VARIANTS = (
    "multiscale_ungated",
    "multiscale_oracle_gate",
    "multiscale_learned_gate",
    "multiscale_learned_no_gate_updates",
    "multiscale_oracle_false_open_10",
    "multiscale_oracle_false_open_25",
    "multiscale_oracle_false_close_10",
    "single_slow_learned_gate",
)

GATE_LEARNING_GROUP = (
    "multiscale_ungated",
    "multiscale_oracle_gate",
    "multiscale_learned_gate",
    "multiscale_oracle_false_open_10",
    "multiscale_oracle_false_open_25",
    "multiscale_oracle_false_close_10",
)

CONTRASTS = {
    "learned_vs_ungated": (
        "multiscale_learned_gate",
        "multiscale_ungated",
    ),
    "learned_gate_update_effect": (
        "multiscale_learned_gate",
        "multiscale_learned_no_gate_updates",
    ),
    "oracle_vs_learned": (
        "multiscale_oracle_gate",
        "multiscale_learned_gate",
    ),
    "oracle_false_open_10_cost": (
        "multiscale_oracle_false_open_10",
        "multiscale_oracle_gate",
    ),
    "oracle_false_open_25_cost": (
        "multiscale_oracle_false_open_25",
        "multiscale_oracle_gate",
    ),
    "oracle_false_close_10_cost": (
        "multiscale_oracle_false_close_10",
        "multiscale_oracle_gate",
    ),
    "multiscale_vs_single_slow_learned": (
        "multiscale_learned_gate",
        "single_slow_learned_gate",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path(
            "analysis/results/temporal_mqr_learned_gate_digits.json"
        ),
    )
    return parser.parse_args()


def close(actual: float, expected: float, *, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{actual} != {expected}")


def summarize(values: Sequence[float]) -> Tuple[float, float]:
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def verify_reported_summaries(
    payload: Mapping[str, object],
    indexed: Mapping[Tuple[int, str], Mapping[str, object]],
    seeds: Sequence[int],
) -> None:
    for variant in EXPECTED_VARIANTS:
        summary = payload["summary"][variant]  # type: ignore[index]
        for name, reported in summary.items():
            if not isinstance(reported, dict) or set(reported) != {
                "mean",
                "sample_std",
            }:
                continue
            if name in (
                "training_gate_max_drift",
                "evaluation_parameter_max_drift",
            ):
                samples = [
                    float(indexed[(seed, variant)][name]) for seed in seeds
                ]
            else:
                split, metric = name.split("_", 1)
                samples = [
                    float(indexed[(seed, variant)][split][metric])  # type: ignore[index]
                    for seed in seeds
                ]
            mean, sample_std = summarize(samples)
            close(mean, float(reported["mean"]))
            close(sample_std, float(reported["sample_std"]))


def main() -> None:
    args = parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    if int(payload["schema_version"]) != 1:
        raise AssertionError("unsupported schema version")
    seeds = [int(value) for value in payload["configuration"]["seeds"]]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise AssertionError("verification requires at least two unique seeds")

    runs: List[Mapping[str, object]] = list(payload["runs"])
    indexed: Dict[Tuple[int, str], Mapping[str, object]] = {}
    for run in runs:
        key = (int(run["seed"]), str(run["variant"]))
        if key in indexed:
            raise AssertionError(f"duplicate run: {key}")
        indexed[key] = run
    expected_keys = {
        (seed, variant) for seed in seeds for variant in EXPECTED_VARIANTS
    }
    if set(indexed) != expected_keys:
        raise AssertionError("run matrix is incomplete or contains unknown variants")

    verify_reported_summaries(payload, indexed, seeds)

    for name, (left, right) in CONTRASTS.items():
        differences = [
            float(indexed[(seed, left)]["evaluation"]["accuracy"])  # type: ignore[index]
            - float(indexed[(seed, right)]["evaluation"]["accuracy"])  # type: ignore[index]
            for seed in seeds
        ]
        reported = payload["paired_contrasts"][name]
        if reported["left"] != left or reported["right"] != right:
            raise AssertionError(f"contrast definition mismatch: {name}")
        if [float(v) for v in reported["paired_differences"]] != differences:
            raise AssertionError(f"paired samples mismatch: {name}")
        mean, sample_std = summarize(differences)
        close(mean, float(reported["mean"]))
        close(sample_std, float(reported["sample_std"]))

    multiscale = tuple(v for v in EXPECTED_VARIANTS if v.startswith("multiscale_"))
    controller_metrics = (
        "controller_gate_accuracy",
        "controller_false_open_rate",
        "controller_false_close_rate",
        "controller_cue_mean",
        "controller_distractor_mean",
    )
    for seed in seeds:
        if len(
            {
                str(indexed[(seed, variant)]["initial_parameter_sha256"])
                for variant in multiscale
            }
        ) != 1:
            raise AssertionError(f"multiscale initialization mismatch: seed {seed}")
        if len(
            {
                str(indexed[(seed, variant)]["final_controller_sha256"])
                for variant in GATE_LEARNING_GROUP
            }
        ) != 1:
            raise AssertionError(
                f"background controller trajectories mismatch: seed {seed}"
            )

        reference = indexed[(seed, GATE_LEARNING_GROUP[0])]["evaluation"]
        for variant in GATE_LEARNING_GROUP[1:]:
            candidate = indexed[(seed, variant)]["evaluation"]
            for metric in controller_metrics:
                close(float(candidate[metric]), float(reference[metric]))

        frozen = indexed[(seed, "multiscale_learned_no_gate_updates")]
        close(float(frozen["training_gate_max_drift"]), 0.0)
        if frozen["initial_controller_sha256"] != frozen["final_controller_sha256"]:
            raise AssertionError("no-update controller hash changed")
        if int(frozen["online_gate_updates"]) != 0:
            raise AssertionError("no-update controller reports gate updates")

        for variant in EXPECTED_VARIANTS:
            run = indexed[(seed, variant)]
            close(float(run["evaluation_parameter_max_drift"]), 0.0)
            scale_count = len(run["leak_rates"])
            expected_labels = scale_count * (
                int(run["training"]["frame_count"])  # type: ignore[index]
                + int(run["evaluation"]["frame_count"])  # type: ignore[index]
            )
            if int(run["online_gate_labels"]) != expected_labels:
                raise AssertionError("gate label counter mismatch")
            expected_gate_updates = (
                int(run["training"]["frame_count"])  # type: ignore[index]
                if bool(run["gate_learning_enabled"])
                else 0
            )
            if int(run["online_gate_updates"]) != expected_gate_updates:
                raise AssertionError("gate update counter mismatch")

        oracle = indexed[(seed, "multiscale_oracle_gate")]["evaluation"]
        close(float(oracle["effective_false_open_rate"]), 0.0)
        close(float(oracle["effective_false_close_rate"]), 0.0)
        ungated = indexed[(seed, "multiscale_ungated")]["evaluation"]
        close(float(ungated["effective_false_open_rate"]), 1.0)
        close(float(ungated["effective_false_close_rate"]), 0.0)
        false_open_10 = indexed[
            (seed, "multiscale_oracle_false_open_10")
        ]["evaluation"]
        false_open_25 = indexed[
            (seed, "multiscale_oracle_false_open_25")
        ]["evaluation"]
        close(float(false_open_10["effective_false_close_rate"]), 0.0)
        close(float(false_open_25["effective_false_close_rate"]), 0.0)
        if float(false_open_25["effective_false_open_rate"]) < float(
            false_open_10["effective_false_open_rate"]
        ):
            raise AssertionError("paired false-open corruption is not monotone")
        false_close = indexed[
            (seed, "multiscale_oracle_false_close_10")
        ]["evaluation"]
        close(float(false_close["effective_false_open_rate"]), 0.0)

    audit = payload["initialization_audit"]
    if not all(
        bool(value)
        for value in audit["multiscale_initial_parameters_matched_by_seed"].values()
    ):
        raise AssertionError("reported initialization audit failed")
    if not all(
        bool(value)
        for value in audit[
            "multiscale_learned_controller_final_matched_by_seed"
        ].values()
    ):
        raise AssertionError("reported controller trajectory audit failed")

    print(f"verified {len(runs)} runs across {len(seeds)} seeds")
    print("multiscale initialization and background gate trajectories: exact match")
    print("held-out parameter drift and no-gate-update drift: exactly zero")
    print("gate label/update counters and corruption ordering: verified")
    for name in CONTRASTS:
        reported = payload["paired_contrasts"][name]
        print(
            f"{name}: {100.0 * float(reported['mean']):+.2f} "
            f"± {100.0 * float(reported['sample_std']):.2f} pp"
        )


if __name__ == "__main__":
    main()
