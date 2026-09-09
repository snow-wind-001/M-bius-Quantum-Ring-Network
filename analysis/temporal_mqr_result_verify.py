#!/usr/bin/env python3
"""Independent integrity checks for the canonical temporal-MQR result JSON."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Dict, Mapping, Tuple


RESULT_PATH = Path(__file__).resolve().parent / "results" / "temporal_mqr_delayed_digits.json"


def close(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    config = payload["configuration"]
    seeds = [int(seed) for seed in config["seeds"]]
    assert len(seeds) == len(set(seeds)) == 3
    train_count = int(config["train_samples"])
    eval_count = int(config["eval_samples"])

    indexed: Dict[Tuple[int, str], Mapping[str, object]] = {}
    for run in payload["runs"]:
        key = (int(run["seed"]), str(run["variant"]))
        assert key not in indexed
        indexed[key] = run
        assert int(run["training"]["count"]) == train_count
        assert int(run["evaluation"]["count"]) == eval_count
        assert float(run["evaluation_parameter_max_drift"]) == 0.0
        if "max_unitary_error" in run:
            assert float(run["max_unitary_error"]) < 5e-6
            assert float(run["max_stochastic_error"]) < 2e-6

    variants = {variant for _seed, variant in indexed}
    assert len(indexed) == len(seeds) * len(variants)
    no_learning = "multiscale_unistochastic_no_learning"
    for seed in seeds:
        frozen = indexed[(seed, no_learning)]
        assert float(frozen["training_parameter_max_drift"]) == 0.0
        assert int(frozen["online_parameters"]) == 0

    definitions = {
        "multiscale_learning_effect": (
            "multiscale_unistochastic",
            no_learning,
        ),
        "temporal_vs_equilibrium": (
            "multiscale_unistochastic",
            "equilibrium_k24_control",
        ),
        "slow_vs_fast": ("single_slow_mqr", "single_fast_mqr"),
        "multiscale_vs_single_slow": (
            "multiscale_unistochastic",
            "single_slow_mqr",
        ),
        "decoupled_write_effect": (
            "multiscale_unistochastic",
            "multiscale_coupled_write",
        ),
        "unistochastic_vs_identity": (
            "multiscale_unistochastic",
            "multiscale_identity",
        ),
    }
    recomputed: Dict[str, list[float]] = {}
    for name, (left, right) in definitions.items():
        values = []
        for seed in seeds:
            left_accuracy = float(indexed[(seed, left)]["evaluation"]["accuracy"])
            right_accuracy = float(indexed[(seed, right)]["evaluation"]["accuracy"])
            values.append(left_accuracy - right_accuracy)
        recomputed[name] = values
        recorded = payload["paired_contrasts"][name]
        assert len(recorded["paired_differences"]) == len(values)
        assert all(
            close(actual, expected)
            for actual, expected in zip(recorded["paired_differences"], values)
        )
        assert close(recorded["mean"], statistics.fmean(values))
        assert close(recorded["sample_std"], statistics.stdev(values))

    # These are audits of this immutable result record, not universal model tests.
    assert all(value > 0.0 for value in recomputed["multiscale_learning_effect"])
    assert all(value > 0.0 for value in recomputed["temporal_vs_equilibrium"])
    assert all(value > 0.0 for value in recomputed["decoupled_write_effect"])
    assert all(value < 0.0 for value in recomputed["multiscale_vs_single_slow"])
    assert all(value < 0.0 for value in recomputed["unistochastic_vs_identity"])

    print(
        "temporal result verified:",
        f"{len(indexed)} runs, {train_count} train and {eval_count} eval sequences/run,",
        "all paired summaries and zero-drift audits exact",
    )


if __name__ == "__main__":
    main()
