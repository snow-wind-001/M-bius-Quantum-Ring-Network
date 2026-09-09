#!/usr/bin/env python3
"""Fail-closed verifier for one scale or the paired 10x10/13x13 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unified_temporal_mqr_go import _paired_bootstrap
from experiments.unified_temporal_mqr_go_competitive import (
    FORMAL_SEEDS,
    METHODS,
    OGD_METHODS,
    PRIMARY_METHOD,
)


METRIC_SPECS = (
    (
        "useful_legal_placement_rate",
        ("game_returns", "after_task_b", "useful_legal_placement_rate"),
        False,
    ),
    ("game_return", ("game_returns", "after_task_b", "mean_return"), False),
    (
        "history_focus_accuracy",
        ("probes", "after_task_b", "task_b", "focus_legality_accuracy"),
        False,
    ),
    ("forgetting_loss", ("forgetting", "task_a_positive_forgetting"), True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument(
        "--require-replication",
        action="store_true",
        help="Require exactly one 10x10 and one 13x13 formal result.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"non-finite value at {path}: {value}")


def _ratio(values: Iterable[int]) -> float:
    prepared = [int(value) for value in values]
    low, high = min(prepared), max(prepared)
    if low == high == 0:
        return 1.0
    return float("inf") if low <= 0 else high / low


def _config_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _values(
    seeds: Sequence[Mapping[str, Any]],
    method: str,
    path: Sequence[str],
) -> list[float]:
    result = []
    for seed in seeds:
        current: Any = seed["methods"][method]
        for key in path:
            current = current[key]
        result.append(float(current))
    return result


def _assert_close_dict(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key, value in expected.items():
        assert key in actual, f"missing bootstrap field: {key}"
        if isinstance(value, float):
            assert math.isclose(
                float(actual[key]),
                value,
                rel_tol=1e-11,
                abs_tol=1e-12,
            ), (key, actual[key], value)
        else:
            assert actual[key] == value, (key, actual[key], value)


def verify_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    assert payload["schema_version"] == 2
    assert payload["experiment"] == "unified_temporal_mqr_go_competitive"
    assert payload["complete"] is True
    assert payload["config_sha256"] == _config_hash(payload["config"])
    assert tuple(payload["methods"]) == METHODS
    _finite_tree(payload)
    config = payload["config"]
    board_size = int(config["board_size"])
    assert board_size in (10, 13)
    assert config["encoder"].startswith("lossless_identity")
    assert int(config["input_dim"]) == 3 * board_size * board_size + 6

    audits = payload["resources"]["audits"]
    assert tuple(item["method"] for item in audits) == METHODS
    assert all(int(item["persistent_state_scalars"]) == 12 for item in audits)
    assert all(int(item["persistent_state_bytes"]) == 48 for item in audits)
    assert len({int(item["frozen_parameters"]) for item in audits}) == 1
    assert len({int(item["allocated_continual_memory_bytes"]) for item in audits}) == 1
    calculated = {
        "trainable_parameter_ratio": _ratio(
            item["trainable_parameters"] for item in audits
        ),
        "recurrent_state_byte_ratio": _ratio(
            item["persistent_state_bytes"] for item in audits
        ),
        "agent_forward_mac_ratio": _ratio(
            item["estimated_agent_forward_macs"] for item in audits
        ),
        "allocated_update_mac_ratio": _ratio(
            item["allocated_update_budget_macs"] for item in audits
        ),
        "frozen_parameter_ratio": _ratio(item["frozen_parameters"] for item in audits),
        "continual_capacity_ratio": _ratio(
            item["allocated_continual_memory_bytes"] for item in audits
        ),
    }
    resource_gate = payload["resources"]["strict_gate"]
    for name, value in calculated.items():
        assert math.isclose(
            float(resource_gate[name]), value, rel_tol=1e-12, abs_tol=1e-12
        )
    expected_resource = bool(
        calculated["trainable_parameter_ratio"] <= 1.05
        and calculated["recurrent_state_byte_ratio"] == 1.0
        and calculated["agent_forward_mac_ratio"] <= 1.05
        and calculated["allocated_update_mac_ratio"] <= 1.05
        and calculated["frozen_parameter_ratio"] == 1.0
        and calculated["continual_capacity_ratio"] == 1.0
    )
    assert resource_gate["all_resources_matched"] == expected_resource

    seeds = payload["seed_results"]
    assert [int(item["seed"]) for item in seeds] == list(config["seeds"])
    assert len({item["trajectory_hash"] for item in seeds}) == len(seeds)
    feedback_equal = True
    initialization_exact = True
    capacity_respected = True
    mechanism_occupancy = True
    routing_exact = True
    for seed in seeds:
        assert tuple(seed["methods"].keys()) == METHODS
        equivalence = seed["initial_equivalence"]
        initialization_exact = initialization_exact and bool(
            equivalence["all_base_hashes_equal"]
            and equivalence["all_head_hashes_equal"]
            and equivalence["lora_family_hashes_equal"]
            and equivalence["all_adaptive_residuals_zero"]
            and equivalence["initial_policies_exact"]
        )
        routing_exact = routing_exact and bool(
            len(
                {
                    seed["methods"][method]["online_training"]["task_a"][
                        "write_schedule_sha256"
                    ]
                    for method in METHODS
                }
            )
            == 1
            and len(
                {
                    seed["methods"][method]["online_training"]["task_b"][
                        "write_schedule_sha256"
                    ]
                    for method in METHODS
                }
            )
            == 1
        )
        for method in METHODS:
            result = seed["methods"][method]
            phase_a = result["online_training"]["task_a"]
            phase_b = result["online_training"]["task_b"]
            feedback_equal = feedback_equal and bool(
                int(phase_a["unique_external_feedback_examples"])
                == int(config["train_exposures"])
                and int(phase_b["unique_external_feedback_examples"])
                == int(config["train_exposures"])
                and int(phase_a["total_gradient_examples"])
                == int(config["train_exposures"])
                and int(phase_b["total_gradient_examples"])
                == 2 * int(config["train_exposures"])
                and phase_a["predict_before_update"] is True
                and phase_b["predict_before_update"] is True
            )
            memory = result["memory_occupancy"]
            capacity_respected = capacity_respected and bool(
                memory["capacity_respected"]
                and int(memory["actual_total_continual_bytes"])
                <= int(memory["allocated_continual_memory_bytes"])
            )
            if method in OGD_METHODS:
                mechanism_occupancy = mechanism_occupancy and (
                    int(phase_a["ogd_rank"]) == int(config["ogd_rank"])
                )
            elif method == "replay_lora":
                mechanism_occupancy = mechanism_occupancy and (
                    int(memory["replay_items"]) > 0
                    and int(memory["actual_replay_tensor_bytes"]) > 0
                    and int(memory["actual_ogd_bytes"]) == 0
                )
            else:
                mechanism_occupancy = mechanism_occupancy and (
                    int(memory["actual_ogd_bytes"]) == 0
                )
            invariant = result["invariants"]
            assert int(invariant["pending_tickets"]) == 0
            assert int(invariant["recurrent_state_bytes_per_stream"]) == 48
            if method == PRIMARY_METHOD:
                assert float(invariant["max_unitary_error"]) < 1e-5
                assert float(invariant["max_stochastic_error"]) < 1e-5
            if method == "sinkhorn_ogd":
                assert float(invariant["max_stochastic_error"]) < 1e-5

    formal_conditions = {
        "board_size_supported": board_size in (10, 13),
        "seed_count_10_to_20": 10 <= len(seeds) <= 20,
        "pre_registered_seed_list": tuple(config["seeds"]) == FORMAL_SEEDS,
        "train_exposures_at_least_128": int(config["train_exposures"]) >= 128,
        "probe_exposures_at_least_64": int(config["probe_exposures"]) >= 64,
        "eval_games_at_least_8": int(config["eval_games"]) >= 8,
        "bootstrap_at_least_5000": int(config["bootstrap_resamples"]) >= 5000,
        "all_requested_seeds_complete": len(seeds) == len(config["seeds"]),
        "feedback_and_gradient_budgets_equal": feedback_equal,
        "training_write_routes_exact": routing_exact,
        "shared_initialization_exact": initialization_exact,
        "continual_memory_cap_respected": capacity_respected,
    }
    recorded_protocol = payload["protocol_gate"]
    for name, value in formal_conditions.items():
        assert recorded_protocol[name] == value
    expected_formal = bool(config["formal"] and all(formal_conditions.values()))
    assert recorded_protocol["formal_protocol_pass"] == expected_formal
    # OGD/replay must really be instantiated even though occupancy is not a
    # publication resource-equality requirement.
    assert mechanism_occupancy

    effectiveness = payload["effectiveness_gate"]
    assert effectiveness["metric_sources"] == {
        name: ".".join(path) for name, path, _reverse in METRIC_SPECS
    }
    all_decisive = True
    resamples = int(config["bootstrap_resamples"])
    for comparator_index, comparator in enumerate(METHODS[1:]):
        for metric_index, (metric, path, reverse) in enumerate(METRIC_SPECS):
            mqr = _values(seeds, PRIMARY_METHOD, path)
            control = _values(seeds, comparator, path)
            left, right = (control, mqr) if reverse else (mqr, control)
            expected = _paired_bootstrap(
                left,
                right,
                resamples=resamples,
                seed=12000 + 101 * comparator_index + metric_index,
            )
            expected["decisive"] = bool(expected["ci95_low"] > 0.0)
            expected["positive_means_mqr_better"] = True
            recorded = effectiveness["comparisons"][comparator][metric]
            _assert_close_dict(recorded, expected)
            all_decisive = all_decisive and bool(expected["decisive"])
    assert effectiveness["all_comparator_metric_cis_above_zero"] == all_decisive

    learning = _values(
        seeds,
        PRIMARY_METHOD,
        ("online_learning", "task_a_loss_reduction"),
    )
    expected_learning = _paired_bootstrap(
        learning,
        [0.0 for _ in learning],
        resamples=resamples,
        seed=15001,
    )
    recorded_learning = effectiveness["online_learning"]["task_a_loss_reduction"]
    _assert_close_dict(recorded_learning, expected_learning)
    online_supported = expected_learning["ci95_low"] > 0.0
    assert effectiveness["online_learning"]["supported"] == online_supported
    expected_scale_effective = bool(
        expected_formal and expected_resource and online_supported and all_decisive
    )
    assert effectiveness["mqr_effective_on_this_scale"] == expected_scale_effective
    assert effectiveness["mqr_effective"] is False
    assert effectiveness["cross_scale_replication_required"] is True
    return {
        "board_size": board_size,
        "seeds": len(seeds),
        "formal_protocol_pass": expected_formal,
        "resource_gate_pass": expected_resource,
        "mechanism_occupancy_valid": mechanism_occupancy,
        "trainable_parameter_ratio": calculated["trainable_parameter_ratio"],
        "recurrent_state_byte_ratio": calculated["recurrent_state_byte_ratio"],
        "agent_forward_mac_ratio": calculated["agent_forward_mac_ratio"],
        "allocated_update_mac_ratio": calculated["allocated_update_mac_ratio"],
        "online_learning_supported": online_supported,
        "all_comparator_metric_cis_above_zero": all_decisive,
        "mqr_effective_on_this_scale": expected_scale_effective,
    }


def verify_replication(payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    assert len(payloads) == 2, "replication verification requires exactly two results"
    by_size = {int(item["config"]["board_size"]): item for item in payloads}
    assert set(by_size) == {10, 13}
    left, right = by_size[10]["config"], by_size[13]["config"]
    allowed_differences = {"board_size", "input_dim", "rank_config"}
    for key in set(left) | set(right):
        if key not in allowed_differences:
            assert left.get(key) == right.get(key), f"cross-scale protocol mismatch: {key}"
    scale_results = [verify_one(by_size[size]) for size in (10, 13)]
    joint = bool(all(item["mqr_effective_on_this_scale"] for item in scale_results))
    return {
        "scales": scale_results,
        "same_protocol_except_board_resources": True,
        "mqr_effective": joint,
        "decision": (
            "MQR independent advantage replicated on both scales"
            if joint
            else "MQR independent advantage not established"
        ),
    }


def main() -> int:
    args = parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    if args.require_replication:
        result = verify_replication(payloads)
    else:
        result = {"results": [verify_one(payload) for payload in payloads]}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
