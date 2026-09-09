#!/usr/bin/env python3
"""Independent verifier for the five-seed MiniCPM/Sayuri Go v2 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


SEEDS = (2031, 2032, 2033, 2034, 2035)
CONDITIONS = (
    "rules-online",
    "rules-mqr-only",
    "rules-lora-only",
    "rules-no-learning",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("analysis/results/minicpm_go_real_games_v2_5seed.json"),
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


def _close(actual: float, expected: float, tolerance: float = 1e-10) -> None:
    assert abs(float(actual) - float(expected)) <= tolerance, (actual, expected)


def _mean_ci(values: Sequence[float]) -> tuple[float, float, tuple[float, float]]:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    # df=4, two-sided 95%.
    half = 2.7764451051977987 * sd / math.sqrt(len(values))
    return mean, sd, (mean - half, mean + half)


def _curve_trend(
    curve: Sequence[Mapping[str, Any]], metric: str, favorable: str
) -> Mapping[str, float | str]:
    points = [
        (float(item["after_train_games"]), float(item[metric]))
        for item in curve
        if item.get(metric) is not None
    ]
    x_mean = statistics.fmean(point[0] for point in points)
    y_mean = statistics.fmean(point[1] for point in points)
    denominator = sum((point[0] - x_mean) ** 2 for point in points)
    slope = sum(
        (point[0] - x_mean) * (point[1] - y_mean) for point in points
    ) / denominator
    favorable_steps = sum(
        (right[1] <= left[1] if favorable == "lower" else right[1] >= left[1])
        for left, right in zip(points[:-1], points[1:])
    )
    return {
        "direction": favorable,
        "slope_per_training_game": slope,
        "first": points[0][1],
        "last": points[-1][1],
        "favorable_change": (
            points[0][1] - points[-1][1]
            if favorable == "lower"
            else points[-1][1] - points[0][1]
        ),
        "favorable_step_fraction": favorable_steps / (len(points) - 1),
    }


def main() -> None:
    path = _args().path
    raw = path.read_bytes()
    payload = json.loads(raw)
    _finite(payload)
    assert payload["format"] == "mqr-minicpm-go-real-games-v2"
    config = payload["config"]
    assert config["seeds"] == list(SEEDS)
    assert config["conditions"] == list(CONDITIONS)
    assert config["train_games"] == config["eval_games"] == 8
    assert config["probe_positions"] == 32
    assert config["probe_pass_fraction"] == 0.25
    assert config["evaluation_teacher"] == "native-policy"
    assert config["min_pass_occupancy"] == 0.6
    assert config["pass_update_scale"] == 0.0
    assert config["curve_every_games"] == 2
    assert payload["model"]["path"].endswith("MiniCPM5-1B-AWQ-INT4")
    assert payload["model"]["lora_parameters"] == 43_008
    assert payload["peak_cuda_memory_bytes"] < 3 * 2**30

    runs = payload["runs"]
    assert len(runs) == len(SEEDS) * len(CONDITIONS)
    indexed = {(int(run["seed"]), str(run["condition"])): run for run in runs}
    assert set(indexed) == {
        (seed, condition) for seed in SEEDS for condition in CONDITIONS
    }
    for run in runs:
        assert run["before_probe"]["placement_target_count"] == 24
        assert run["before_probe"]["pass_target_count"] == 8
        assert run["after_probe"]["placement_target_count"] == 24
        assert run["after_probe"]["pass_target_count"] == 8
        curve = run["training"]["probe_learning_curve"]
        assert [point["after_train_games"] for point in curve] == [0, 2, 4, 6, 8]
        assert run["before_probe"] == {k: v for k, v in curve[0].items() if k != "after_train_games"}
        assert run["training"]["max_mqr_unitary_error"] < 1e-4
        directions = {
            "loss": "lower",
            "conditional_placement_loss": "lower",
            "conditional_placement_agreement": "higher",
            "false_pass_rate_on_placement_targets": "lower",
            "pass_brier": "lower",
            "raw_legal_rate": "higher",
            "raw_teacher_agreement": "higher",
        }
        for metric, favorable in directions.items():
            expected = _curve_trend(curve, metric, favorable)
            actual = run["training"]["probe_trends"][metric]
            assert actual["direction"] == expected["direction"]
            for field in (
                "slope_per_training_game",
                "first",
                "last",
                "favorable_change",
                "favorable_step_fraction",
            ):
                _close(actual[field], expected[field])

    for seed in SEEDS:
        combined = indexed[(seed, "rules-online")]
        mqr = indexed[(seed, "rules-mqr-only")]
        lora = indexed[(seed, "rules-lora-only")]
        frozen = indexed[(seed, "rules-no-learning")]

        assert all(frozen["no_learning_audit"].values())
        assert frozen["parameter_drift"]["mqr"]["all"]["delta_l2"] == 0.0
        assert frozen["parameter_drift"]["lora"]["all"]["delta_l2"] == 0.0
        assert lora["parameter_drift"]["mqr"]["all"]["delta_l2"] == 0.0
        assert lora["parameter_drift"]["lora"]["all"]["delta_l2"] > 0.0
        assert mqr["parameter_drift"]["mqr"]["all"]["delta_l2"] > 0.0
        assert mqr["parameter_drift"]["lora"]["all"]["delta_l2"] == 0.0
        assert combined["parameter_drift"]["mqr"]["all"]["delta_l2"] > 0.0
        assert combined["parameter_drift"]["lora"]["all"]["delta_l2"] > 0.0

        # LoRA updates are real but do not cross an argmax/game-behavior boundary.
        assert abs(float(lora["probe_changes"]["loss_improvement"])) < 2e-5
        assert lora["game_changes"]["paired_student_margin_changes"] == [0.0] * 8
        assert lora["game_changes"]["student_win_rate_change"] == 0.0

        # Combined and MQR-only behavior is identical; tiny FP16 probe loss
        # differences are allowed because LoRA tensors do move.
        assert combined["game_changes"] == mqr["game_changes"]
        assert abs(
            float(combined["probe_changes"]["loss_improvement"])
            - float(mqr["probe_changes"]["loss_improvement"])
        ) < 1e-4

        before = combined["before_probe"]
        after = combined["after_probe"]
        _close(
            combined["probe_changes"]["loss_improvement"],
            float(before["loss"]) - float(after["loss"]),
        )
        _close(
            combined["probe_changes"]["conditional_placement_loss_improvement"],
            float(before["conditional_placement_loss"])
            - float(after["conditional_placement_loss"]),
        )
        _close(
            combined["probe_changes"]["pass_brier_improvement"],
            float(before["pass_brier"]) - float(after["pass_brier"]),
        )

    online = [indexed[(seed, "rules-online")] for seed in SEEDS]
    metrics = {
        "probe_loss_improvement": [
            float(run["probe_changes"]["loss_improvement"]) for run in online
        ],
        "conditional_placement_loss_improvement": [
            float(run["probe_changes"]["conditional_placement_loss_improvement"])
            for run in online
        ],
        "pass_brier_improvement": [
            float(run["probe_changes"]["pass_brier_improvement"]) for run in online
        ],
        "student_margin_improvement": [
            float(run["game_changes"]["mean_student_margin_improvement"])
            for run in online
        ],
        "student_win_rate_change": [
            float(run["game_changes"]["student_win_rate_change"])
            for run in online
        ],
    }
    recomputed = {name: _mean_ci(values) for name, values in metrics.items()}
    aggregate = payload["aggregate"]["by_condition"]["rules-online"]
    aggregate_names = {
        "probe_loss_improvement": "probe_loss_improvement",
        "conditional_placement_loss_improvement": (
            "probe_conditional_placement_loss_improvement"
        ),
        "pass_brier_improvement": "probe_pass_brier_improvement",
        "student_margin_improvement": "eval_student_margin_improvement",
        "student_win_rate_change": "eval_win_rate_change",
    }
    for local_name, aggregate_name in aggregate_names.items():
        _close(aggregate[aggregate_name]["mean"], recomputed[local_name][0])

    margin_mean, _margin_sd, margin_ci = recomputed["student_margin_improvement"]
    loss_mean, _loss_sd, loss_ci = recomputed["probe_loss_improvement"]
    pass_mean, _pass_sd, pass_ci = recomputed["pass_brier_improvement"]
    assert margin_ci[0] < 0.0 < margin_ci[1]
    assert loss_ci[1] < 0.0
    assert pass_ci[1] < 0.0
    assert all(value < 0.0 for value in metrics["probe_loss_improvement"])
    assert all(value < 0.0 for value in metrics["pass_brier_improvement"])
    assert sum(value > 0.0 for value in metrics["student_margin_improvement"]) == 4
    assert sum(
        value < 0.0 for value in metrics["conditional_placement_loss_improvement"]
    ) == 4

    digest = hashlib.sha256(raw).hexdigest()
    print(f"verified {path}")
    print(f"sha256={digest}")
    print(
        f"masked margin change={margin_mean:+.3f}, Student-t 95% CI="
        f"[{margin_ci[0]:+.3f}, {margin_ci[1]:+.3f}] (inconclusive)"
    )
    print(
        f"native probe loss improvement={loss_mean:+.3f}, Student-t 95% CI="
        f"[{loss_ci[0]:+.3f}, {loss_ci[1]:+.3f}] (strict degradation)"
    )
    print(
        f"pass-Brier improvement={pass_mean:+.4f}, Student-t 95% CI="
        f"[{pass_ci[0]:+.4f}, {pass_ci[1]:+.4f}] (strict degradation)"
    )


if __name__ == "__main__":
    main()
