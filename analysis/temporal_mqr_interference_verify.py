#!/usr/bin/env python3
"""Independent arithmetic and integrity audit for interference-Digits results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


EXPECTED_VARIANTS = (
    "memoryless_query_linear",
    "single_slow_ungated",
    "single_slow_oracle_gate",
    "multiscale_ungated",
    "multiscale_random_gate",
    "multiscale_oracle_gate",
    "multiscale_oracle_no_learning",
)

CONTRASTS = {
    "multiscale_oracle_gate_effect": (
        "multiscale_oracle_gate",
        "multiscale_ungated",
    ),
    "single_slow_oracle_gate_effect": (
        "single_slow_oracle_gate",
        "single_slow_ungated",
    ),
    "oracle_vs_random_gate": (
        "multiscale_oracle_gate",
        "multiscale_random_gate",
    ),
    "oracle_gate_learning_effect": (
        "multiscale_oracle_gate",
        "multiscale_oracle_no_learning",
    ),
    "multiscale_vs_single_slow_oracle": (
        "multiscale_oracle_gate",
        "single_slow_oracle_gate",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("analysis/results/temporal_mqr_interference_digits.json"),
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


def main() -> None:
    args = parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    if int(payload["schema_version"]) != 1:
        raise AssertionError("unsupported schema version")
    configured_seeds = [int(value) for value in payload["configuration"]["seeds"]]
    if len(configured_seeds) < 2 or len(set(configured_seeds)) != len(configured_seeds):
        raise AssertionError("verification requires unique independent seeds")

    runs: List[Mapping[str, object]] = list(payload["runs"])
    indexed: Dict[Tuple[int, str], Mapping[str, object]] = {}
    for run in runs:
        key = (int(run["seed"]), str(run["variant"]))
        if key in indexed:
            raise AssertionError(f"duplicate run: {key}")
        indexed[key] = run
    expected_keys = {
        (seed, variant) for seed in configured_seeds for variant in EXPECTED_VARIANTS
    }
    if set(indexed) != expected_keys:
        raise AssertionError("run matrix is incomplete or contains unexpected variants")

    for variant in EXPECTED_VARIANTS:
        heldout = [
            float(indexed[(seed, variant)]["evaluation"]["accuracy"])  # type: ignore[index]
            for seed in configured_seeds
        ]
        mean, sample_std = summarize(heldout)
        reported = payload["summary"][variant]["heldout_accuracy"]
        close(mean, float(reported["mean"]))
        close(sample_std, float(reported["sample_std"]))
        for seed in configured_seeds:
            run = indexed[(seed, variant)]
            close(float(run["evaluation_parameter_max_drift"]), 0.0)
            if int(run["evaluation"]["count"]) != int(  # type: ignore[index]
                payload["configuration"]["eval_samples"]
            ):
                raise AssertionError("evaluation count mismatch")

    for name, (left, right) in CONTRASTS.items():
        differences = [
            float(indexed[(seed, left)]["evaluation"]["accuracy"])  # type: ignore[index]
            - float(indexed[(seed, right)]["evaluation"]["accuracy"])  # type: ignore[index]
            for seed in configured_seeds
        ]
        mean, sample_std = summarize(differences)
        reported = payload["paired_contrasts"][name]
        if [float(value) for value in reported["paired_differences"]] != differences:
            raise AssertionError(f"paired samples mismatch for {name}")
        close(mean, float(reported["mean"]))
        close(sample_std, float(reported["sample_std"]))

    for seed in configured_seeds:
        single_hashes = {
            str(indexed[(seed, variant)]["initial_parameter_sha256"])
            for variant in ("single_slow_ungated", "single_slow_oracle_gate")
        }
        multiscale_hashes = {
            str(indexed[(seed, variant)]["initial_parameter_sha256"])
            for variant in (
                "multiscale_ungated",
                "multiscale_random_gate",
                "multiscale_oracle_gate",
                "multiscale_oracle_no_learning",
            )
        }
        if len(single_hashes) != 1 or len(multiscale_hashes) != 1:
            raise AssertionError(f"initial parameters are not matched for seed {seed}")
        no_learning = indexed[(seed, "multiscale_oracle_no_learning")]
        close(float(no_learning["training_parameter_max_drift"]), 0.0)
        memoryless = indexed[(seed, "memoryless_query_linear")]
        if memoryless["learn_enabled"] is not True:
            raise AssertionError("memoryless query control must attempt online learning")
        close(float(memoryless["training_parameter_max_drift"]), 0.0)
        close(float(memoryless["training"]["mean_update_norm"]), 0.0)  # type: ignore[index]
        for split in ("training", "evaluation"):
            oracle_fraction = float(
                indexed[(seed, "multiscale_oracle_gate")][split][  # type: ignore[index]
                    "gate_open_fraction"
                ]
            )
            random_fraction = float(
                indexed[(seed, "multiscale_random_gate")][split][  # type: ignore[index]
                    "gate_open_fraction"
                ]
            )
            close(oracle_fraction, random_fraction)
            close(
                float(
                    indexed[(seed, "multiscale_ungated")][split][  # type: ignore[index]
                        "gate_open_fraction"
                    ]
                ),
                1.0,
            )

    print(f"verified {len(runs)} runs across {len(configured_seeds)} seeds")
    print("initialization hashes: matched within every gate ablation")
    print("oracle/random write counts: exactly matched")
    print("evaluation/no-learning drift: exactly zero")
    print("zero-query learning gradient: exactly zero")
    for name in CONTRASTS:
        reported = payload["paired_contrasts"][name]
        print(
            f"{name}: {100.0 * float(reported['mean']):+.2f} "
            f"± {100.0 * float(reported['sample_std']):.2f} pp"
        )


if __name__ == "__main__":
    main()
