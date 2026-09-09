#!/usr/bin/env python3
"""Fail-closed verifier for Phase-IV contextual-topology evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mqr_contextual_topology_qualification import (
    EVALUATED_METHODS,
    TRAINABLE_METHODS,
    generate_context_topologies,
    generate_matched_batch,
    marginal_audit,
    resource_audit,
)


def _mean(values: Iterable[float]) -> float:
    parsed = [float(value) for value in values]
    return sum(parsed) / len(parsed)


def _t95(count: int) -> float:
    table = {
        1: float("inf"),
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
    }
    return table.get(count, 1.96)


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    tensor = torch.tensor(tuple(float(value) for value in values), dtype=torch.double)
    count = int(tensor.numel())
    mean = float(tensor.mean().item())
    if count <= 1:
        return {
            "count": count,
            "mean": mean,
            "std": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
            "interval": "singleton",
        }
    std = float(tensor.std(unbiased=True).item())
    radius = _t95(count) * std / math.sqrt(count)
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
        "interval": "two_sided_student_t_95",
    }


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise AssertionError(f"{label}: mapping keys differ")
        for key in expected:
            _assert_close(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{label}: list shape differs")
        for index, value in enumerate(expected):
            _assert_close(actual[index], value, f"{label}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)):
            raise AssertionError(f"{label}: expected numeric value")
        if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
            if actual != expected:
                raise AssertionError(f"{label}: non-finite mismatch")
            return
        tolerance = 1e-10 * max(1.0, abs(float(expected)))
        if abs(float(actual) - float(expected)) > tolerance:
            raise AssertionError(f"{label}: {actual} != {expected}")
        return
    if actual != expected:
        raise AssertionError(f"{label}: value mismatch")


def _args_from_protocol(protocol: Mapping[str, Any]) -> SimpleNamespace:
    learning_rates = protocol["learning_rates"]
    return SimpleNamespace(
        slots=int(protocol["slots"]),
        epsilon=float(protocol["epsilon"]),
        route_base_angle=float(protocol["route_base_angle"]),
        eval_episodes=int(protocol["eval_episodes"]),
        phase_a_steps=int(protocol["phase_a_steps"]),
        phase_b_steps=int(protocol["phase_b_steps"]),
        query_cycles=int(protocol["query_cycles"]),
        givens_lr=float(learning_rates["mqr_sparse_givens"]),
        block_cayley_lr=float(learning_rates["mqr_block_cayley"]),
        sparse_permutation_lr=float(
            learning_rates["learnable_sparse_permutation"]
        ),
    )


def _recompute_aggregate(
    args: SimpleNamespace, runs: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    method_summaries: Dict[str, Any] = {}
    for method in EVALUATED_METHODS:
        method_summaries[method] = {
            metric: _summary([run["methods"][method][metric] for run in runs])
            for metric in (
                "final_mean_accuracy",
                "final_mean_repeated_query_accuracy",
                "context_a_forgetting",
            )
        }
        if method in TRAINABLE_METHODS:
            method_summaries[method]["route_recovery"] = _summary(
                [
                    run["methods"][method]["route_recovery"]["mean"]
                    for run in runs
                ]
            )
            method_summaries[method]["context_b_prequential_regret"] = _summary(
                [
                    run["training"][method]["context_b"]["prequential_regret"]
                    for run in runs
                ]
            )
    primary = "mqr_sparse_givens"
    controls = (
        "identity",
        "learnable_sparse_permutation",
        "direct_ring_buffer",
        "mqr_identity_replacement",
    )
    paired = {
        control: _summary(
            [
                run["methods"][primary]["final_mean_accuracy"]
                - run["methods"][control]["final_mean_accuracy"]
                for run in runs
            ]
        )
        for control in controls
    }
    gate_summaries = {
        metric: _summary([run["gate_metrics"][metric] for run in runs])
        for metric in (
            "auprc",
            "brier",
            "ece",
            "event_write_rate",
            "distractor_write_rate",
            "event_distractor_write_gap",
            "write_rate",
            "budget_violation",
        )
    }
    matched = all(
        bool(audit["joint_marginals_exactly_matched"])
        and int(audit["joint_histogram_l1_difference"]) == 0
        and int(audit["key_histogram_l1_difference"]) == 0
        and int(audit["positive_value_count_difference"]) == 0
        and float(audit["absolute_magnitude_mean_difference"]) == 0.0
        for run in runs
        for audit in run["marginal_audits"]
    )
    resource_parity = all(
        bool(
            run["resource_audit"][
                "same_factor_graph_parameters_and_resources_within_1_05"
            ]
        )
        for run in runs
    )
    primary_summary = method_summaries[primary]
    block_summary = method_summaries["mqr_block_cayley"]
    mechanism = {
        "formal_seed_count_at_least_10": len(runs) >= 10,
        "held_out_episodes_at_least_64": args.eval_episodes >= 64,
        "opaque_context_topologies_are_distinct": all(
            run["topologies"][0] != run["topologies"][1] for run in runs
        ),
        "topology_not_provided_to_route_learner": True,
        "predict_before_route_update": True,
        "event_distractor_joint_marginals_exactly_matched": matched,
        "write_gap_ci_positive": gate_summaries[
            "event_distractor_write_gap"
        ]["ci95_low"]
        > 0.0,
        "hard_pairwise_write_budget_never_violated": gate_summaries[
            "budget_violation"
        ]["ci95_high"]
        <= 1e-12,
        "givens_route_recovery_ci_above_chance": primary_summary[
            "route_recovery"
        ]["ci95_low"]
        > 0.5,
        "block_cayley_route_recovery_ci_above_chance": block_summary[
            "route_recovery"
        ]["ci95_low"]
        > 0.5,
        "primary_accuracy_ci_above_0_80": primary_summary[
            "final_mean_accuracy"
        ]["ci95_low"]
        > 0.8,
        "beats_identity": paired["identity"]["ci95_low"] > 0.0,
        "identity_replacement_hurts": paired[
            "mqr_identity_replacement"
        ]["ci95_low"]
        > 0.0,
        "context_a_forgetting_ci_not_above_0_01": primary_summary[
            "context_a_forgetting"
        ]["ci95_high"]
        <= 0.01,
        "equal_route_resource_gate_vs_sparse_permutation": resource_parity,
    }
    mechanism["contextual_route_mechanism_qualified"] = all(
        mechanism.values()
    )
    independent = {
        "contextual_route_mechanism_qualified": mechanism[
            "contextual_route_mechanism_qualified"
        ],
        "beats_learnable_sparse_permutation": paired[
            "learnable_sparse_permutation"
        ]["ci95_low"]
        > 0.0,
        "beats_direct_ring_buffer": paired["direct_ring_buffer"]["ci95_low"]
        > 0.0,
    }
    independent["mqr_independent_algorithm_advantage"] = all(
        independent.values()
    )
    return {
        "method_summaries": method_summaries,
        "paired_primary_mqr_minus_control_accuracy": paired,
        "gate_summaries": gate_summaries,
        "mechanism_gates": mechanism,
        "contextual_route_mechanism_qualified": bool(
            mechanism["contextual_route_mechanism_qualified"]
        ),
        "independent_advantage_gates": independent,
        "mqr_independent_algorithm_advantage": bool(
            independent["mqr_independent_algorithm_advantage"]
        ),
        "claim_boundary": (
            "Passing the independent gate would support only this matched-"
            "marginal contextual routing benchmark. Go, MiniCPM, policy RL, "
            "general intelligence, and animal-like learning remain out of scope."
        ),
    }


def verify(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise AssertionError("unsupported schema version")
    protocol = payload.get("protocol")
    runs = payload.get("runs")
    aggregate = payload.get("aggregate")
    if not isinstance(protocol, Mapping) or not isinstance(runs, list) or not runs:
        raise AssertionError("missing protocol or runs")
    if not isinstance(aggregate, Mapping):
        raise AssertionError("missing aggregate")
    required_protocol = {
        "phase": "IV unknown context-dependent topology",
        "primary_method": "mqr_sparse_givens",
        "contexts": 2,
        "context_schedule": "A train -> B online adapt -> A retained probe",
        "context_identifier_semantics": "opaque route-bank index",
        "topology_supervision_to_learner": False,
        "route_feedback": "held-out query value MSE only",
        "predict_before_update": True,
        "explicit_event_marker_in_gate": False,
        "write_critic_target": (
            "exact finite future query-loss gain of candidate write versus no-write"
        ),
        "go_minicpm_policy_rl_frozen_until_gate": True,
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise AssertionError(f"protocol mismatch: {key}")
    args = _args_from_protocol(protocol)
    if int(protocol["seeds"]) != len(runs):
        raise AssertionError("seed count mismatch")
    expected_status = (
        "formal"
        if len(runs) >= 10
        and args.eval_episodes >= 64
        and args.phase_a_steps >= 64
        and args.phase_b_steps >= 64
        else "smoke_or_pilot"
    )
    if payload.get("status") != expected_status:
        raise AssertionError("status does not match formal evidence requirements")
    expected_seeds = [
        int(protocol["seed_start"]) + index for index in range(len(runs))
    ]
    if [int(run["seed"]) for run in runs] != expected_seeds:
        raise AssertionError("seed sequence mismatch")

    expected_resource = resource_audit(args)
    for run in runs:
        seed = int(run["seed"])
        topologies, bits = generate_context_topologies(
            seed=seed, slots=args.slots
        )
        if run["topologies"] != topologies.tolist():
            raise AssertionError(f"seed {seed}: topology regeneration mismatch")
        if run["swap_bits"] != bits.to(torch.long).tolist():
            raise AssertionError(f"seed {seed}: swap-bit regeneration mismatch")
        if sorted(run["topologies"][0]) != list(range(args.slots)):
            raise AssertionError(f"seed {seed}: context A is not a permutation")
        if sorted(run["topologies"][1]) != list(range(args.slots)):
            raise AssertionError(f"seed {seed}: context B is not a permutation")
        regenerated_audits = []
        for context_id in range(2):
            batch = generate_matched_batch(
                seed=seed * 10_000_000 + 9_000_000 + context_id,
                batch_size=args.eval_episodes,
                slots=args.slots,
                query_cycles=args.query_cycles,
                context_id=context_id,
                topology=topologies[context_id],
            )
            regenerated_audits.append(marginal_audit(batch, args.slots))
        _assert_close(
            run["marginal_audits"],
            regenerated_audits,
            f"seed {seed}.marginal_audits",
        )
        _assert_close(
            run["resource_audit"],
            expected_resource,
            f"seed {seed}.resource_audit",
        )
        if set(run["methods"]) != set(EVALUATED_METHODS):
            raise AssertionError(f"seed {seed}: method set mismatch")
        if set(run["training"]) != set(TRAINABLE_METHODS):
            raise AssertionError(f"seed {seed}: training set mismatch")
        for method in TRAINABLE_METHODS:
            training = run["training"][method]
            for context_name, expected_steps in (
                ("context_a", args.phase_a_steps),
                ("context_b", args.phase_b_steps),
            ):
                phase = training[context_name]
                curve = phase["prequential_accuracy_curve"]
                if len(curve) != expected_steps:
                    raise AssertionError(
                        f"seed {seed}.{method}.{context_name}: curve length mismatch"
                    )
                if any(not 0.0 <= float(value) <= 1.0 for value in curve):
                    raise AssertionError("prequential accuracy outside [0,1]")
                expected_regret = _mean(1.0 - float(value) for value in curve)
                if abs(expected_regret - float(phase["prequential_regret"])) > 1e-12:
                    raise AssertionError("prequential regret mismatch")
            if float(training["context_a_parameter_drift_during_b_linf"]) != 0.0:
                raise AssertionError(
                    f"seed {seed}.{method}: inactive context changed during B"
                )
            recovery = run["methods"][method]["route_recovery"]
            if float(recovery["doubly_stochastic_max_error"]) > 1e-5:
                raise AssertionError("doubly stochastic route error is too large")
        gate = run["gate_metrics"]
        if abs(float(gate["write_rate"]) - 0.5) > 1e-12:
            raise AssertionError("hard pairwise write budget mismatch")
        if float(gate["budget_violation"]) != 0.0:
            raise AssertionError("write budget violation recorded")

    expected_aggregate = _recompute_aggregate(args, runs)
    _assert_close(aggregate, expected_aggregate, "aggregate")
    return {
        "verified": True,
        "status": expected_status,
        "seeds": len(runs),
        "matched_marginals": expected_aggregate["mechanism_gates"][
            "event_distractor_joint_marginals_exactly_matched"
        ],
        "contextual_route_mechanism_qualified": expected_aggregate[
            "contextual_route_mechanism_qualified"
        ],
        "mqr_independent_algorithm_advantage": expected_aggregate[
            "mqr_independent_algorithm_advantage"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.result), indent=2))


if __name__ == "__main__":
    main()
