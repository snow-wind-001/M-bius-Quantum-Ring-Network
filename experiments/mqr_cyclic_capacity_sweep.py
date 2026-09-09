#!/usr/bin/env python3
"""Equal-state capacity sweep for cyclic MQR versus its identity ablation.

The sweep reuses the marker-free delayed-memory task from
``mqr_mechanism_qualification.py`` at 48, 192, and 768 recurrent-state bytes.
Both methods receive identical trajectories, causal routing, heads, losses,
feedback, and recurrent-state capacity.  It reports peak transition-refresh
cost separately from amortized online cost; no dummy parameters, state, or
compute are added to make a resource gate pass.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mqr_mechanism_qualification import (  # noqa: E402
    LEAK_RATES,
    _episode,
    _paired_episodes,
    _train_episode,
    build_agents,
    identity_replacement_effect,
    paired_history_separation,
)
from mqr.agent_baselines import audit_agent_resources  # noqa: E402


METHODS = ("cyclic_trace", "identity_trace")
STATE_BUDGETS = (48, 192, 768)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


_T975_BY_DF = (
    12.706205, 4.302653, 3.182446, 2.776445, 2.570582,
    2.446912, 2.364624, 2.306004, 2.262157, 2.228139,
    2.200985, 2.178813, 2.160369, 2.144787, 2.131450,
    2.119905, 2.109816, 2.100922, 2.093024, 2.085963,
    2.079614, 2.073873, 2.068658, 2.063899, 2.059539,
    2.055529, 2.051831, 2.048407, 2.045230, 2.042272,
)


def _student_t_critical_95(sample_count: int) -> float:
    if sample_count <= 1:
        return 0.0
    degrees_of_freedom = sample_count - 1
    if degrees_of_freedom <= len(_T975_BY_DF):
        return _T975_BY_DF[degrees_of_freedom - 1]
    return _T975_BY_DF[-1]


def _summary(values: Sequence[float]) -> Dict[str, float | int | str]:
    mean = _mean(values)
    if len(values) <= 1:
        radius = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        radius = _student_t_critical_95(len(values)) * math.sqrt(
            variance / len(values)
        )
    return {
        "mean": mean,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
        "n": len(values),
        "ci_method": "two_sided_student_t_95",
    }


def _ratio(values: Sequence[float]) -> float:
    low, high = min(values), max(values)
    if low == high == 0:
        return 1.0
    return float("inf") if low <= 0 else high / low


def _ring_dim(state_bytes: int) -> int:
    denominator = len(LEAK_RATES) * torch.tensor([], dtype=torch.float32).element_size()
    if state_bytes <= 0 or state_bytes % denominator:
        raise ValueError("state budget must exactly fit float32 rings")
    result = state_bytes // denominator
    if result < 2 or result % 2:
        raise ValueError("capacity sweep requires a positive even ring dimension")
    return result


def _resource_record(agents: Mapping[str, torch.nn.Module]) -> Dict[str, Any]:
    audits = [audit_agent_resources(agents[method], method) for method in METHODS]
    interval = int(agents["cyclic_trace"].transition_update_interval)
    rows = []
    for audit in audits:
        base_update = 3 * int(audit.estimated_agent_forward_macs)
        amortized = base_update + float(audit.estimated_transition_refresh_macs) / interval
        rows.append({
            **audit.to_dict(),
            "transition_update_interval": interval,
            "estimated_amortized_update_macs": amortized,
        })
    ratios = {
        "trainable_parameter_ratio": _ratio(
            [float(row["trainable_parameters"]) for row in rows]
        ),
        "recurrent_state_byte_ratio": _ratio(
            [float(row["persistent_state_bytes"]) for row in rows]
        ),
        "agent_forward_mac_ratio": _ratio(
            [float(row["estimated_agent_forward_macs"]) for row in rows]
        ),
        "amortized_update_mac_ratio": _ratio(
            [float(row["estimated_amortized_update_macs"]) for row in rows]
        ),
        "peak_update_mac_ratio": _ratio(
            [float(row["estimated_task_update_macs"]) for row in rows]
        ),
    }
    average_gate = bool(
        ratios["trainable_parameter_ratio"] <= 1.05
        and ratios["recurrent_state_byte_ratio"] == 1.0
        and ratios["agent_forward_mac_ratio"] <= 1.05
        and ratios["amortized_update_mac_ratio"] <= 1.05
    )
    return {
        "audits": rows,
        "ratios": ratios,
        "average_resource_gate": average_gate,
        "peak_update_resource_gate": bool(
            average_gate and ratios["peak_update_mac_ratio"] <= 1.05
        ),
        "accounting_note": (
            "amortized cost divides a real structured-transition refresh by the "
            "declared update interval; peak cost remains separately fail-closed"
        ),
    }


def run_one(payload: Tuple[argparse.Namespace, int, int]) -> Dict[str, Any]:
    args, state_bytes, seed = payload
    torch.set_num_threads(1)
    ring_dim = _ring_dim(state_bytes)
    agents = build_agents(
        ring_dim=ring_dim,
        seed=seed,
        task_lr=args.task_lr,
        utility_lr=args.utility_lr,
        max_trace_horizon=args.sequence_length,
        transition_update_interval=args.transition_update_interval,
        methods=METHODS,
    )
    train = [
        _episode(
            seed * 1_000_000 + state_bytes * 1000 + index,
            length=args.sequence_length,
            context=index % 2,
            reversed_mapping=False,
            bit=(index // 2) % 2,
        )
        for index in range(args.train_episodes)
    ]
    pairs = [
        _paired_episodes(
            seed * 1_000_000 + state_bytes * 1000 + 700_000 + index,
            length=args.sequence_length,
            context=0,
            reversed_mapping=False,
        )
        for index in range(args.paired_episodes)
    ]
    flat_eval = [episode for pair in pairs for episode in pair]
    methods: Dict[str, Any] = {}
    for method, agent in agents.items():
        replay: List[Tuple[torch.Tensor, float]] = []
        logs = []
        started = time.perf_counter()
        for episode_index, episode in enumerate(train):
            logs.append(
                _train_episode(
                    agent,
                    method,
                    episode,
                    episode_index=episode_index,
                    oracle_warmup_episodes=args.oracle_warmup_episodes,
                    critic_calibration_episodes=args.critic_calibration_episodes,
                    remember_gradient=False,
                    critic_replay=replay,
                )
            )
        history = paired_history_separation(agent, pairs)
        identity_effect = identity_replacement_effect(agent, flat_eval)
        methods[method] = {
            "history": history,
            "identity_replacement": identity_effect,
            "mean_first_quarter_loss": _mean(
                item["loss"] for item in logs[: max(1, len(logs) // 4)]
            ),
            "mean_last_quarter_loss": _mean(
                item["loss"] for item in logs[-max(1, len(logs) // 4) :]
            ),
            "mean_future_injection_gradient": _mean(
                item["future_block_gradient_norms"]["injection"] for item in logs
            ),
            "mean_future_transition_gradient": _mean(
                item["future_block_gradient_norms"]["transition"] for item in logs
            ),
            "transition_refreshes": sum(
                int(item["transition_refresh_required"]) for item in logs
            ),
            "runtime_seconds": time.perf_counter() - started,
            "max_unitary_error": agent.core.max_unitary_error(),
            "max_stochastic_error": agent.core.max_stochastic_error(),
        }
    return {
        "state_bytes": state_bytes,
        "ring_dim": ring_dim,
        "seed": seed,
        "methods": methods,
        "resources": _resource_record(agents),
    }


def aggregate(
    args: argparse.Namespace,
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_budget: Dict[str, Any] = {}
    all_budget_gates = []
    all_advantages = []
    all_average_independent_gates = []
    all_peak_independent_gates = []
    for state_bytes in args.state_bytes:
        selected = [run for run in runs if int(run["state_bytes"]) == state_bytes]
        methods: Dict[str, Any] = {}
        for method in METHODS:
            methods[method] = {
                "history_accuracy": _summary(
                    [float(run["methods"][method]["history"]["accuracy"]) for run in selected]
                ),
                "history_probability_separation": _summary(
                    [
                        float(run["methods"][method]["history"]["probability_separation"])
                        for run in selected
                    ]
                ),
                "event_write_rate": _summary(
                    [float(run["methods"][method]["history"]["event_write_rate"]) for run in selected]
                ),
                "query_write_rate": _summary(
                    [float(run["methods"][method]["history"]["query_write_rate"]) for run in selected]
                ),
                "runtime_seconds": _summary(
                    [float(run["methods"][method]["runtime_seconds"]) for run in selected]
                ),
            }
        paired = _summary([
            float(run["methods"]["cyclic_trace"]["history"]["accuracy"])
            - float(run["methods"]["identity_trace"]["history"]["accuracy"])
            for run in selected
        ])
        resource = selected[0]["resources"]
        gates = {
            "ten_seeds": len(selected) >= 10,
            "average_resources_within_1_05": bool(resource["average_resource_gate"]),
            "peak_update_resources_within_1_05": bool(
                resource["peak_update_resource_gate"]
            ),
            "cyclic_above_chance": methods["cyclic_trace"]["history_accuracy"]["ci95_low"] > 0.5,
            "cyclic_separation_positive": methods["cyclic_trace"]["history_probability_separation"]["ci95_low"] > 0.0,
            "cyclic_beats_identity": paired["ci95_low"] > 0.0,
        }
        all_budget_gates.append(bool(gates["average_resources_within_1_05"]))
        all_advantages.append(bool(gates["cyclic_beats_identity"]))
        average_independent = bool(
            all(
                value
                for name, value in gates.items()
                if name != "peak_update_resources_within_1_05"
            )
        )
        peak_independent = bool(all(gates.values()))
        all_average_independent_gates.append(average_independent)
        all_peak_independent_gates.append(peak_independent)
        by_budget[str(state_bytes)] = {
            "methods": methods,
            "paired_cyclic_minus_identity_accuracy": paired,
            "resources": resource,
            "decision_gates": gates,
            "average_budget_advantage_at_budget": average_independent,
            "peak_qualified_advantage_at_budget": peak_independent,
        }
    return {
        "budgets": by_budget,
        "all_average_resource_gates_pass": all(all_budget_gates),
        "cyclic_beats_identity_at_every_budget": all(all_advantages),
        "mqr_capacity_advantage": bool(all(all_average_independent_gates)),
        "mqr_peak_qualified_capacity_advantage": bool(
            all(all_peak_independent_gates)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=41)
    parser.add_argument("--state-bytes", type=int, nargs="+", default=list(STATE_BUDGETS))
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--train-episodes", type=int, default=160)
    parser.add_argument("--oracle-warmup-episodes", type=int, default=100)
    parser.add_argument("--critic-calibration-episodes", type=int, default=40)
    parser.add_argument("--paired-episodes", type=int, default=40)
    parser.add_argument("--transition-update-interval", type=int, default=32)
    parser.add_argument("--task-lr", type=float, default=0.03)
    parser.add_argument("--utility-lr", type=float, default=0.08)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/results/mqr_cyclic_capacity_sweep_formal_10seed.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 0 or args.workers <= 0:
        raise ValueError("seeds and workers must be positive")
    if args.transition_update_interval <= 0 or args.sequence_length < 3:
        raise ValueError("transition interval must be positive and sequence length >=3")
    for value in args.state_bytes:
        _ring_dim(int(value))
    seeds = [args.seed_start + index for index in range(args.seeds)]
    jobs = [(args, int(state_bytes), seed) for state_bytes in args.state_bytes for seed in seeds]
    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, len(jobs))
        ) as executor:
            runs = list(executor.map(run_one, jobs))
    else:
        runs = [run_one(job) for job in jobs]
    formal = bool(
        args.seeds >= 10
        and args.sequence_length >= 10
        and args.train_episodes >= 160
        and args.paired_episodes >= 40
        and tuple(args.state_bytes) == STATE_BUDGETS
    )
    result = {
        "schema_version": 2,
        "experiment": "mqr_cyclic_capacity_sweep",
        "status": "formal_capacity_sweep" if formal else "smoke_or_pilot",
        "protocol": {
            "predict_before_update": True,
            "explicit_event_marker": False,
            "future_label_in_gate_features": False,
            "ci_method": "two_sided_student_t_95",
            "methods": list(METHODS),
            "state_bytes": list(args.state_bytes),
            "seeds": args.seeds,
            "seed_start": args.seed_start,
            "sequence_length": args.sequence_length,
            "train_episodes": args.train_episodes,
            "oracle_warmup_episodes": args.oracle_warmup_episodes,
            "critic_calibration_episodes": args.critic_calibration_episodes,
            "paired_episodes": args.paired_episodes,
            "transition_update_interval": args.transition_update_interval,
        },
        "runs": runs,
        "aggregate": aggregate(args, runs),
        "claim_scope": (
            "Capacity and topology ablation only; peak refresh cost and Go-scale "
            "competitive evidence are reported separately."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
