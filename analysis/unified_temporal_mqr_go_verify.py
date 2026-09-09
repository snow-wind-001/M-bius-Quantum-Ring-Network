#!/usr/bin/env python3
"""Fail-closed verifier for the unified Temporal Utility MQR Go result."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


METHODS = ("mqr_unistochastic", "mqr_identity", "gru", "fast_weight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result",
        nargs="?",
        type=Path,
        default=Path("analysis/results/unified_temporal_mqr_go.json"),
    )
    return parser.parse_args()


def _ratio(values: Iterable[float]) -> float:
    prepared = [float(value) for value in values]
    return max(prepared) / min(prepared)


def _finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"non-finite value at {path}: {value}")


def _at(seed: Mapping[str, Any], method: str, path: Sequence[str]) -> float:
    current: Any = seed["methods"][method]
    for key in path:
        current = current[key]
    return float(current)


def verify(payload: Dict[str, Any]) -> Dict[str, Any]:
    assert payload["schema_version"] == 1
    assert payload["experiment"] == "unified_temporal_mqr_go"
    protocol = payload["protocol"]
    assert protocol["legality_mask_during_action"] is False
    assert protocol["prediction_before_update"] is True
    assert protocol["offline_stage"] == "advantage-weighted imitation"
    assert protocol["write_target"] == "paired future negative multi-head loss"
    _finite_tree(payload)

    audits = payload["resources"]["audits"]
    assert tuple(item["method"] for item in audits) == METHODS
    parameter_ratio = _ratio(item["trainable_parameters"] for item in audits)
    state_ratio = _ratio(item["persistent_state_scalars"] for item in audits)
    mac_ratio = _ratio(item["estimated_forward_macs"] for item in audits)
    strict = payload["resources"]["strict_equal_resource_gate"]
    assert math.isclose(parameter_ratio, strict["parameter_ratio"], rel_tol=1e-12)
    assert math.isclose(state_ratio, strict["state_ratio"], rel_tol=1e-12)
    assert math.isclose(mac_ratio, strict["estimated_mac_ratio"], rel_tol=1e-12)
    parameter_limit = float(strict.get("parameter_ratio_limit", 1.05))
    state_limit = float(strict.get("state_ratio_limit", 1.05))
    mac_limit = float(strict.get("estimated_mac_ratio_limit", 1.05))
    assert math.isclose(parameter_limit, 1.05, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(state_limit, 1.05, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mac_limit, 1.05, rel_tol=0.0, abs_tol=1e-12)
    assert strict["parameters_matched"] == (parameter_ratio <= parameter_limit)
    assert strict["state_matched"] == (state_ratio <= state_limit)
    assert strict["estimated_macs_matched"] == (mac_ratio <= mac_limit)
    assert strict["all_resources_matched"] == bool(
        strict["parameters_matched"]
        and strict["state_matched"]
        and strict["estimated_macs_matched"]
    )
    board_size = int(payload["config"].get("board_size", 3))
    if board_size >= 10:
        assert strict.get("decision_role") == "effectiveness_gate"
        relaxed = payload["resources"][
            "parameter_flop_gate_with_reported_state_cap"
        ]
        assert relaxed.get("decision_role") == "diagnostic_only"

    seeds = payload["seed_results"]
    assert len(seeds) == len(payload["config"]["seeds"])
    assert len({item["stream_hash"] for item in seeds}) == len(seeds)
    for seed in seeds:
        assert tuple(seed["methods"].keys()) == METHODS
        for method in METHODS:
            invariant = seed["methods"][method]["invariants"]
            assert invariant["pending_tickets"] == 0
            assert invariant["online_state_bytes"] >= 0
            if method == "mqr_unistochastic":
                assert invariant["max_unitary_error"] < 1e-4
                assert invariant["max_stochastic_error"] < 1e-4
            offline = seed["methods"][method]["offline_awr"]
            assert offline["task_a"]["updates"] > 0
            assert offline["task_b"]["updates"] > 0
            assert 0.0 <= offline["task_a"]["critic_oracle_sign_accuracy"] <= 1.0
            assert 0.0 <= offline["task_b"]["critic_oracle_sign_accuracy"] <= 1.0

    gate = payload["effectiveness_gate"]
    final_stage = gate["final_stage"]
    expected_sources = {
        "unmasked_legal_rate": (
            f"game_returns.{final_stage}.unmasked_legal_rate"
        ),
        "useful_legal_placement_rate": (
            f"game_returns.{final_stage}.useful_legal_placement_rate"
        ),
        "game_return": f"game_returns.{final_stage}.mean_return",
        "history_focus_accuracy": (
            f"probes.{final_stage}.task_b.focus_legality_accuracy"
        ),
        "forgetting_loss": "forgetting.task_a_positive_forgetting",
    }
    if board_size >= 10:
        assert gate.get("metric_sources") == expected_sources
    metric_sources = gate.get("metric_sources")
    decisive = True
    for comparator in METHODS[1:]:
        for metric in (
            "unmasked_legal_rate",
            "useful_legal_placement_rate",
            "game_return",
            "history_focus_accuracy",
            "forgetting_loss",
        ):
            recorded = gate["comparisons"][comparator][metric]
            if metric_sources is not None:
                path = tuple(str(metric_sources[metric]).split("."))
            elif metric == "forgetting_loss":
                path = ("forgetting", "task_a_positive_forgetting")
            elif metric == "game_return":
                path = ("game_returns", final_stage, "mean_return")
            elif metric == "history_focus_accuracy":
                path = (
                    "probes",
                    final_stage,
                    "task_b",
                    "focus_legality_accuracy",
                )
            elif metric == "useful_legal_placement_rate":
                path = (
                    "probes",
                    final_stage,
                    "task_a",
                    "useful_legal_placement_rate",
                )
            else:
                path = (
                    "probes",
                    final_stage,
                    "task_a",
                    "unmasked_legal_rate",
                )
            if metric == "forgetting_loss":
                left = [
                    _at(seed, comparator, path)
                    for seed in seeds
                ]
                right = [
                    _at(seed, "mqr_unistochastic", path)
                    for seed in seeds
                ]
            else:
                left = [_at(seed, "mqr_unistochastic", path) for seed in seeds]
                right = [_at(seed, comparator, path) for seed in seeds]
            mean_difference = statistics.fmean(a - b for a, b in zip(left, right))
            assert math.isclose(
                mean_difference,
                recorded["mean_difference"],
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
            assert recorded["decisive"] == (recorded["ci95_low"] > 0.0)
            decisive = decisive and bool(recorded["decisive"])
    assert gate["all_performance_gates_decisive"] == decisive
    expected_effective = bool(strict["all_resources_matched"] and decisive)
    assert gate["mqr_effective"] == expected_effective
    if not expected_effective:
        assert gate["rejection_reasons"]
    return {
        "seeds": len(seeds),
        "parameter_ratio": parameter_ratio,
        "state_ratio": state_ratio,
        "estimated_mac_ratio": mac_ratio,
        "all_performance_gates_decisive": decisive,
        "mqr_effective": expected_effective,
    }


def main() -> int:
    args = parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    print(json.dumps(verify(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
