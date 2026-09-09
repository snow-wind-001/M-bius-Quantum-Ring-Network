#!/usr/bin/env python3
"""Mechanism-first qualification for causal Temporal MQR.

The task is deliberately smaller than Go but harder than a two-step marker
probe: one hidden bit must survive 12 observations, distractors, and an
identical query.  All predictions are committed before a single trajectory-end
update.  The experiment isolates three questions before any expensive 10x10
rerun:

1. does a non-flat base make transition gradients observable;
2. does future credit train the content written at the first observation; and
3. does the learned ring beat identity and a non-MQR recurrent control under an
   explicitly audited resource protocol?

This script never overwrites the historical Go result files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mqr.agent import GoLossWeights, TemporalUtilityMQRAgent
from mqr.agent_baselines import (
    MultiTimescaleFastWeightCore,
    MultiTimescaleGRUCore,
    MultiTimescaleLSTMCore,
    MultiTimescaleSinkhornCore,
    audit_agent_resources,
    matched_competitive_resource_gate,
)
from mqr.temporal import MultiTimescaleMQR


INPUT_DIM = 8
LATENT_DIM = 8
BOARD_SIZE = 2
LEAK_RATES = (1.0, 0.08)
QUERY_INDEX = -1
METHODS = (
    "legacy_dense_truncated",
    "dense_nonflat_truncated",
    "dense_nonflat_trace",
    "cyclic_trace",
    "cyclic_trace_ogd",
    "identity_trace",
    "gru_trace",
    "lstm_trace",
    "fast_weight_trace",
    "sinkhorn_trace",
)
COMPETITIVE_METHODS = (
    "cyclic_trace",
    "identity_trace",
    "gru_trace",
    "lstm_trace",
    "fast_weight_trace",
    "sinkhorn_trace",
)


@dataclass(frozen=True)
class DelayedEpisode:
    observations: torch.Tensor
    bit: int
    context: int
    action: int
    reversed_mapping: bool


def _query_indices(length: int) -> Tuple[int, ...]:
    """Three separated, identical queries for one persistent hidden variable."""

    return tuple(sorted({max(1, length // 3), max(1, 2 * length // 3), length - 1}))


def _episode(
    seed: int,
    *,
    length: int,
    context: int,
    reversed_mapping: bool,
    bit: int | None = None,
) -> DelayedEpisode:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    resolved_bit = int(torch.randint(0, 2, (), generator=generator).item()) if bit is None else int(bit)
    if resolved_bit not in (0, 1) or context not in (0, 1):
        raise ValueError("bit and context must be binary")
    values = 0.25 * torch.randn(length, INPUT_DIM, generator=generator)
    # Content carries the bit; there is no separate event-type or write marker.
    values[0].zero_()
    values[0, resolved_bit] = 1.0
    values[0, 2 + context] = 0.75
    # The query is identical for bit zero and bit one within a context.
    for query_index in _query_indices(length):
        values[query_index].zero_()
        values[query_index, 2 + context] = 0.75
        values[query_index, 7] = 1.0
    action = resolved_bit ^ int(bool(reversed_mapping and context == 1))
    return DelayedEpisode(
        observations=values,
        bit=resolved_bit,
        context=int(context),
        action=int(action),
        reversed_mapping=bool(reversed_mapping),
    )


def _paired_episodes(
    seed: int,
    *,
    length: int,
    context: int,
    reversed_mapping: bool,
) -> Tuple[DelayedEpisode, DelayedEpisode]:
    return (
        _episode(
            seed,
            length=length,
            context=context,
            reversed_mapping=reversed_mapping,
            bit=0,
        ),
        _episode(
            seed,
            length=length,
            context=context,
            reversed_mapping=reversed_mapping,
            bit=1,
        ),
    )


def _method_metadata(method: str) -> Dict[str, bool]:
    return {
        "uses_trace": not method.endswith("truncated"),
        "uses_ogd": method.endswith("_ogd"),
        "is_mqr": method.startswith(("legacy_dense", "dense_nonflat", "cyclic")),
    }


def _core(method: str, ring_dim: int, seed: int) -> torch.nn.Module:
    common = {
        "input_dim": INPUT_DIM,
        "output_dim": LATENT_DIM,
        "leak_rates": LEAK_RATES,
    }
    if method == "legacy_dense_truncated":
        return MultiTimescaleMQR(
            ring_dim=ring_dim,
            injection_rank=4,
            transition_mode="unistochastic",
            transition_structure="dense_cayley",
            learn_transitions=True,
            cayley_coordinate_mode="minimal",
            base_unitary_init="identity",
            readout_bias=True,
            **common,
        )
    if method.startswith("dense_nonflat"):
        return MultiTimescaleMQR(
            ring_dim=ring_dim,
            injection_rank=4,
            transition_mode="unistochastic",
            transition_structure="dense_cayley",
            learn_transitions=True,
            cayley_coordinate_mode="minimal",
            base_unitary_init="random",
            base_unitary_scale=0.25,
            base_unitary_seed=seed + 101,
            readout_bias=True,
            **common,
        )
    if method.startswith("cyclic"):
        return MultiTimescaleMQR(
            ring_dim=ring_dim,
            injection_rank=4,
            transition_mode="unistochastic",
            transition_structure="cyclic_givens",
            cyclic_givens_layers=2,
            learn_transitions=True,
            base_unitary_init="random",
            base_unitary_scale=0.25,
            base_unitary_seed=seed + 211,
            readout_bias=True,
            **common,
        )
    if method == "identity_trace":
        return MultiTimescaleMQR(
            ring_dim=ring_dim,
            injection_rank=4,
            transition_mode="identity",
            learn_transitions=False,
            readout_bias=True,
            **common,
        )
    if method == "gru_trace":
        return MultiTimescaleGRUCore(
            INPUT_DIM,
            ring_dim,
            LATENT_DIM,
            leak_rates=LEAK_RATES,
            input_rank=4,
        )
    if method == "lstm_trace":
        if ring_dim % 2:
            raise ValueError("LSTM matched state requires an even ring_dim")
        return MultiTimescaleLSTMCore(
            INPUT_DIM,
            ring_dim // 2,
            LATENT_DIM,
            leak_rates=LEAK_RATES,
            input_rank=4,
        )
    if method == "fast_weight_trace":
        memory_dim = max(1, round(math.sqrt(ring_dim)))
        return MultiTimescaleFastWeightCore(
            INPUT_DIM,
            memory_dim,
            LATENT_DIM,
            leak_rates=LEAK_RATES,
            input_rank=4,
        )
    if method == "sinkhorn_trace":
        return MultiTimescaleSinkhornCore(
            INPUT_DIM,
            ring_dim,
            LATENT_DIM,
            leak_rates=LEAK_RATES,
            injection_rank=4,
            sinkhorn_iterations=8,
        )
    raise ValueError(f"unknown method: {method}")


def _copy_common_agent_modules(
    source: TemporalUtilityMQRAgent,
    target: TemporalUtilityMQRAgent,
) -> None:
    for name in (
        "placement_head",
        "legality_head",
        "pass_head",
        "value_head",
        "utility_gate",
    ):
        getattr(target, name).load_state_dict(getattr(source, name).state_dict())
    target.utility_content_projection.copy_(source.utility_content_projection)


def build_agents(
    *,
    ring_dim: int,
    seed: int,
    task_lr: float,
    utility_lr: float,
    max_trace_horizon: int,
    transition_update_interval: int = 2,
    methods: Sequence[str] = METHODS,
) -> Dict[str, TemporalUtilityMQRAgent]:
    agents: Dict[str, TemporalUtilityMQRAgent] = {}
    source: TemporalUtilityMQRAgent | None = None
    for method in methods:
        torch.manual_seed(seed * 1009 + 17)
        core = _core(method, ring_dim, seed)
        metadata = _method_metadata(method)
        agent = TemporalUtilityMQRAgent(
            INPUT_DIM,
            board_size=BOARD_SIZE,
            latent_dim=LATENT_DIM,
            core=core,
            utility_rank=6,
            utility_content_dim=INPUT_DIM,
            utility_content_seed=seed + 307,
            utility_content_projection_mode="identity",
            initial_write_advantage=-0.001,
            write_exploration_probability=0.20,
            write_exploration_decay_observations=1200,
            memory_cost=0.0001,
            advantage_scale=0.02,
            task_lr=task_lr,
            utility_lr=utility_lr,
            ogd_max_rank=4 if metadata["uses_ogd"] else 0,
            max_update_norm=0.10,
            utility_max_update_norm=0.05,
            transition_update_interval=transition_update_interval,
            max_trace_horizon=max_trace_horizon,
            loss_weights=GoLossWeights(
                placement=1.0,
                legality=0.0,
                pass_decision=0.0,
                value=0.0,
            ),
        )
        if source is None:
            source = agent
        else:
            _copy_common_agent_modules(source, agent)
        agents[method] = agent
    return agents


@torch.no_grad()
def _twin_return(
    agent: TemporalUtilityMQRAgent,
    ticket_id: int,
    observations: torch.Tensor,
    current_index: int,
    action: int,
) -> Tuple[float, float]:
    no_state, write_state = agent.shadow_states(ticket_id)
    record = agent._pending_tickets[ticket_id]
    no_output = record["no_write_output"]
    write_output = record["write_output"]
    for index in range(current_index + 1, observations.size(0)):
        x = observations[index : index + 1]
        no_output, no_state = agent.branch_step(x, no_state, slow_write=False)
        write_output, write_state = agent.branch_step(x, write_state, slow_write=False)
    target = torch.tensor([action], dtype=torch.long)
    no_loss = F.cross_entropy(no_output.placement_logits, target)
    write_loss = F.cross_entropy(write_output.placement_logits, target)
    return -float(no_loss.item()), -float(write_loss.item())


def _train_episode(
    agent: TemporalUtilityMQRAgent,
    method: str,
    episode: DelayedEpisode,
    *,
    episode_index: int,
    oracle_warmup_episodes: int,
    critic_calibration_episodes: int,
    remember_gradient: bool,
    critic_replay: List[Tuple[torch.Tensor, float]] | None = None,
) -> Dict[str, Any]:
    stream_id = "train"
    agent.train()
    agent.reset_state(stream_id)
    tickets: List[int] = []
    writes = []
    learned_writes = []
    for index, value in enumerate(episode.observations):
        external_write = None
        if episode_index < oracle_warmup_episodes + critic_calibration_episodes:
            external_write = index == 0
        result = agent.commit_step(
            value.unsqueeze(0),
            stream_id=stream_id,
            external_write=external_write,
        )
        tickets.append(int(result["ticket_id"]))
        writes.append(float(bool(result["effective_write"])))
        learned_writes.append(float(bool(result["learned_write"])))

    selected_critic_indices = {0, *_query_indices(len(tickets))}
    train_critic = episode_index >= oracle_warmup_episodes
    critic_advantages = []
    critic_advantages_by_index: Dict[str, float] = {}
    for index, ticket_id in enumerate(tickets):
        if train_critic and index in selected_critic_indices:
            no_return, write_return = _twin_return(
                agent,
                ticket_id,
                episode.observations,
                index,
                episode.action,
            )
            advantage = write_return - no_return - agent.memory_cost
            feature = agent._pending_tickets[ticket_id]["causal_features"].detach().clone()
            if critic_replay is not None:
                critic_replay.append((feature, float(advantage)))
                del critic_replay[:-256]
            critic_advantages.append(float(advantage))
            critic_advantages_by_index[str(index)] = float(advantage)
            agent.cancel_ticket_branch(ticket_id, "utility")
        else:
            agent.cancel_ticket_branch(ticket_id, "utility")

    if train_critic and critic_advantages:
        batch = (
            critic_replay
            if critic_replay is not None
            else [
                (
                    agent._pending_tickets[tickets[index]]["causal_features"].detach().clone(),
                    critic_advantages[position],
                )
                for position, index in enumerate(sorted(selected_critic_indices))
            ]
        )
        features = torch.cat([item[0] for item in batch], dim=0)
        advantages = torch.tensor(
            [item[1] for item in batch],
            device=features.device,
            dtype=features.dtype,
        )
        agent.fit_write_critic_batch(features, advantages)

    metadata = _method_metadata(method)
    action = torch.tensor([episode.action], dtype=torch.long)
    if metadata["uses_trace"]:
        feedback = agent.apply_trajectory_feedback(
            tickets,
            [{"action": action} for _ticket in tickets],
            loss_scales=[
                1.0 if index in _query_indices(len(tickets)) else 0.0
                for index in range(len(tickets))
            ],
            remember_gradient=remember_gradient,
        )
        future_norms = feedback["future_block_gradient_norms"]
        earliest_future_gradient = feedback["earliest_input_future_gradient_norm"]
    else:
        for ticket_id in tickets[:-1]:
            agent.cancel_ticket_branch(ticket_id, "task")
        feedback = agent.apply_feedback(
            tickets[-1],
            action,
            remember_gradient=remember_gradient,
        )
        future_norms = {
            "injection": 0.0,
            "transition": 0.0,
            "core_readout": 0.0,
            "core_other": 0.0,
            "heads": 0.0,
        }
        earliest_future_gradient = 0.0
    assert agent.pending_ticket_count == 0
    return {
        "loss": float(feedback["loss"]),
        "write_rate": sum(writes) / len(writes),
        "learned_write_rate": sum(learned_writes) / len(learned_writes),
        "critic_advantage_mean": (
            sum(critic_advantages) / len(critic_advantages)
            if critic_advantages
            else 0.0
        ),
        "critic_advantages_by_index": critic_advantages_by_index,
        "future_block_gradient_norms": future_norms,
        "earliest_input_future_gradient_norm": earliest_future_gradient,
        "transition_update_due": bool(feedback["transition_update_due"]),
        "transition_refresh_required": bool(
            feedback["transition_refresh_required"]
        ),
        "update_norm": float(feedback["update_norm"]),
    }


@torch.no_grad()
def evaluate(
    agent: TemporalUtilityMQRAgent,
    episodes: Sequence[DelayedEpisode],
) -> Dict[str, float | int]:
    was_training = agent.training
    observation_count = agent.online_observations.detach().clone()
    agent.eval()
    correct = 0
    probability = 0.0
    effective_writes = 0
    learned_writes = 0
    total_steps = 0
    event_writes = 0
    distractor_writes = 0
    query_writes = 0
    event_observations = 0
    distractor_observations = 0
    query_observations = 0
    for episode_index, episode in enumerate(episodes):
        stream_id = f"eval-{episode_index}"
        agent.reset_state(stream_id)
        final_output = None
        query_indices = set(_query_indices(episode.observations.size(0)))
        for index, value in enumerate(episode.observations):
            result = agent.commit_step(
                value.unsqueeze(0),
                stream_id=stream_id,
                issue_ticket=False,
            )
            final_output = result["output"]
            write = int(bool(result["effective_write"]))
            learned = int(bool(result["learned_write"]))
            effective_writes += write
            learned_writes += learned
            total_steps += 1
            if index == 0:
                event_writes += write
                event_observations += 1
            elif index in query_indices:
                query_writes += write
                query_observations += 1
            else:
                distractor_writes += write
                distractor_observations += 1
        assert final_output is not None
        probabilities = F.softmax(final_output.placement_logits, dim=1)
        prediction = int(probabilities.argmax(dim=1).item())
        correct += int(prediction == episode.action)
        probability += float(probabilities[0, episode.action].item())
        agent.reset_state(stream_id)
    agent.online_observations.copy_(observation_count)
    agent.train(was_training)
    count = max(1, len(episodes))
    return {
        "accuracy": correct / count,
        "target_probability": probability / count,
        "effective_write_rate": effective_writes / max(1, total_steps),
        "learned_write_rate": learned_writes / max(1, total_steps),
        "event_write_rate": event_writes / max(1, event_observations),
        "distractor_write_rate": distractor_writes / max(1, distractor_observations),
        "query_write_rate": query_writes / max(1, query_observations),
        "event_observations": event_observations,
        "distractor_observations": distractor_observations,
        "query_observations": query_observations,
    }


@torch.no_grad()
def paired_history_separation(
    agent: TemporalUtilityMQRAgent,
    pairs: Sequence[Tuple[DelayedEpisode, DelayedEpisode]],
) -> Dict[str, float]:
    flat = [episode for pair in pairs for episode in pair]
    result = evaluate(agent, flat)
    was_training = agent.training
    observation_count = agent.online_observations.detach().clone()
    agent.eval()
    separations = []
    for pair_index, (zero, one) in enumerate(pairs):
        values = []
        for suffix, episode in (("zero", zero), ("one", one)):
            stream_id = f"pair-{pair_index}-{suffix}"
            agent.reset_state(stream_id)
            output = None
            for observation in episode.observations:
                output = agent.commit_step(
                    observation.unsqueeze(0),
                    stream_id=stream_id,
                    issue_ticket=False,
                )["output"]
            assert output is not None
            values.append(float(F.softmax(output.placement_logits, dim=1)[0, 1].item()))
            agent.reset_state(stream_id)
        separations.append(values[1] - values[0])
    agent.online_observations.copy_(observation_count)
    agent.train(was_training)
    result["probability_separation"] = sum(separations) / max(1, len(separations))
    return result


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
    """Return a conservative two-sided 95% Student-t critical value."""

    if sample_count <= 1:
        return 0.0
    degrees_of_freedom = sample_count - 1
    if degrees_of_freedom <= len(_T975_BY_DF):
        return _T975_BY_DF[degrees_of_freedom - 1]
    # Student-t critical values decrease with degrees of freedom. Reusing the
    # df=30 value is conservative and avoids a SciPy runtime dependency.
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


def _transition_snapshot(agent: TemporalUtilityMQRAgent) -> List[torch.Tensor]:
    if not isinstance(agent.core, MultiTimescaleMQR):
        return []
    return [
        agent.core.transition_matrix(index).detach().clone()
        for index in range(agent.core.num_timescales)
    ]


@torch.no_grad()
def identity_replacement_effect(
    agent: TemporalUtilityMQRAgent,
    episodes: Sequence[DelayedEpisode],
) -> Dict[str, float | None]:
    if not isinstance(agent.core, MultiTimescaleMQR) or agent.core.transition_mode == "identity":
        return {"native_accuracy": None, "identity_accuracy": None, "accuracy_drop": None}
    native = evaluate(agent, episodes)["accuracy"]
    original_mode = agent.core.transition_mode
    agent.core.transition_mode = "identity"
    try:
        counterfactual = evaluate(agent, episodes)["accuracy"]
    finally:
        agent.core.transition_mode = original_mode
    return {
        "native_accuracy": native,
        "identity_accuracy": counterfactual,
        "accuracy_drop": native - counterfactual,
    }


def run_seed(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    agents = build_agents(
        ring_dim=args.ring_dim,
        seed=seed,
        task_lr=args.task_lr,
        utility_lr=args.utility_lr,
        max_trace_horizon=args.sequence_length,
        transition_update_interval=args.transition_update_interval,
    )
    initial_transitions = {
        method: _transition_snapshot(agent) for method, agent in agents.items()
    }
    initial_diagnostics = {
        method: (
            agent.core.transition_diagnostics(include_jacobian=True)
            if isinstance(agent.core, MultiTimescaleMQR)
            else []
        )
        for method, agent in agents.items()
    }
    pretrain = [
        _episode(
            seed * 100000 + index,
            length=args.sequence_length,
            context=index % 2,
            reversed_mapping=False,
            bit=(index // 2) % 2,
        )
        for index in range(args.train_episodes)
    ]
    reversal = [
        _episode(
            seed * 100000 + 50000 + index,
            length=args.sequence_length,
            context=1,
            reversed_mapping=True,
            bit=index % 2,
        )
        for index in range(args.reversal_episodes)
    ]
    eval_a = [
        _episode(
            seed * 100000 + 70000 + index,
            length=args.sequence_length,
            context=0,
            reversed_mapping=False,
        )
        for index in range(args.eval_episodes)
    ]
    eval_b_normal = [
        _episode(
            seed * 100000 + 80000 + index,
            length=args.sequence_length,
            context=1,
            reversed_mapping=False,
        )
        for index in range(args.eval_episodes)
    ]
    eval_b_reversed = [
        _episode(
            seed * 100000 + 90000 + index,
            length=args.sequence_length,
            context=1,
            reversed_mapping=True,
        )
        for index in range(args.eval_episodes)
    ]
    pairs = [
        _paired_episodes(
            seed * 100000 + 95000 + index,
            length=args.sequence_length,
            context=0,
            reversed_mapping=False,
        )
        for index in range(args.paired_episodes)
    ]

    methods: Dict[str, Any] = {}
    for method, agent in agents.items():
        started = time.perf_counter()
        train_logs = []
        critic_replay: List[Tuple[torch.Tensor, float]] = []
        for episode_index, episode in enumerate(pretrain):
            remember = bool(
                _method_metadata(method)["uses_ogd"]
                and agent.task_gradient_memory.rank < agent.task_gradient_memory.max_rank
            )
            train_logs.append(
                _train_episode(
                    agent,
                    method,
                    episode,
                    episode_index=episode_index,
                    oracle_warmup_episodes=args.oracle_warmup_episodes,
                    critic_calibration_episodes=args.critic_calibration_episodes,
                    remember_gradient=remember,
                    critic_replay=critic_replay,
                )
            )
        before_a = evaluate(agent, eval_a)
        before_b = evaluate(agent, eval_b_normal)
        paired = paired_history_separation(agent, pairs)
        transition_before_reversal = _transition_snapshot(agent)
        reversal_logs = []
        for reversal_index, episode in enumerate(reversal):
            reversal_logs.append(
                _train_episode(
                    agent,
                    method,
                    episode,
                    episode_index=args.oracle_warmup_episodes + reversal_index,
                    oracle_warmup_episodes=0,
                    critic_calibration_episodes=0,
                    remember_gradient=False,
                    critic_replay=critic_replay,
                )
            )
        after_a = evaluate(agent, eval_a)
        after_b = evaluate(agent, eval_b_reversed)
        identity_effect = identity_replacement_effect(agent, eval_a)
        final_transitions = _transition_snapshot(agent)
        transition_drift = [
            float(torch.linalg.matrix_norm(final - initial, ord="fro").item())
            for initial, final in zip(initial_transitions[method], final_transitions)
        ]
        reversal_transition_drift = [
            float(torch.linalg.matrix_norm(final - initial, ord="fro").item())
            for initial, final in zip(transition_before_reversal, final_transitions)
        ]
        methods[method] = {
            "metadata": _method_metadata(method),
            "before_reversal": {"task_a": before_a, "task_b": before_b},
            "paired_history": paired,
            "after_reversal": {"task_a": after_a, "task_b_reversed": after_b},
            "task_a_forgetting": before_a["accuracy"] - after_a["accuracy"],
            "task_b_reversal_gain": after_b["accuracy"] - (1.0 - before_b["accuracy"]),
            "identity_replacement": identity_effect,
            "mean_train_loss_first_quarter": _mean(
                item["loss"] for item in train_logs[: max(1, len(train_logs) // 4)]
            ),
            "mean_train_loss_last_quarter": _mean(
                item["loss"] for item in train_logs[-max(1, len(train_logs) // 4) :]
            ),
            "mean_train_write_rate": _mean(item["write_rate"] for item in train_logs),
            "mean_future_injection_gradient": _mean(
                item["future_block_gradient_norms"]["injection"] for item in train_logs
            ),
            "mean_future_transition_gradient": _mean(
                item["future_block_gradient_norms"]["transition"] for item in train_logs
            ),
            "mean_earliest_input_future_gradient": _mean(
                item["earliest_input_future_gradient_norm"] for item in train_logs
            ),
            "transition_drift_fro": transition_drift,
            "reversal_transition_drift_fro": reversal_transition_drift,
            "ogd_rank": agent.task_gradient_memory.rank,
            "max_unitary_error": agent.core.max_unitary_error(),
            "max_stochastic_error": agent.core.max_stochastic_error(),
            "runtime_seconds": time.perf_counter() - started,
            "reversal_loss": _mean(item["loss"] for item in reversal_logs),
        }
    audits = [
        audit_agent_resources(agents[method], method) for method in METHODS
    ]
    competitive_audits = [
        audit for audit in audits if audit.method in COMPETITIVE_METHODS
    ]
    return {
        "seed": seed,
        "initial_transition_diagnostics": initial_diagnostics,
        "methods": methods,
        "resource_audits": [asdict(value) for value in audits],
        "diagnostic_resource_gate": matched_competitive_resource_gate(
            audits, ratio_limit=1.05
        ),
        "competitive_resource_gate": matched_competitive_resource_gate(
            competitive_audits, ratio_limit=1.05
        ),
    }


def aggregate(args: argparse.Namespace, runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for method in METHODS:
        values = [run["methods"][method] for run in runs]
        summaries[method] = {
            "history_accuracy": _summary(
                [value["paired_history"]["accuracy"] for value in values]
            ),
            "history_probability_separation": _summary(
                [value["paired_history"]["probability_separation"] for value in values]
            ),
            "learned_write_rate": _summary(
                [value["paired_history"]["learned_write_rate"] for value in values]
            ),
            "event_write_rate": _summary(
                [value["paired_history"]["event_write_rate"] for value in values]
            ),
            "distractor_write_rate": _summary(
                [value["paired_history"]["distractor_write_rate"] for value in values]
            ),
            "task_a_forgetting": _summary(
                [value["task_a_forgetting"] for value in values]
            ),
            "task_b_reversed_accuracy": _summary(
                [value["after_reversal"]["task_b_reversed"]["accuracy"] for value in values]
            ),
            "task_b_reversal_gain": _summary(
                [value["task_b_reversal_gain"] for value in values]
            ),
            "future_injection_gradient": _summary(
                [value["mean_future_injection_gradient"] for value in values]
            ),
            "future_transition_gradient": _summary(
                [value["mean_future_transition_gradient"] for value in values]
            ),
            "earliest_input_future_gradient": _summary(
                [value["mean_earliest_input_future_gradient"] for value in values]
            ),
            "identity_replacement_accuracy_drop": _summary(
                [
                    value["identity_replacement"]["accuracy_drop"]
                    for value in values
                    if value["identity_replacement"]["accuracy_drop"] is not None
                ]
            ),
            "runtime_seconds": _summary([value["runtime_seconds"] for value in values]),
        }

    competitive_resource_gate_all_seeds = all(
        bool(run["competitive_resource_gate"]["all_resources_matched"])
        for run in runs
    )
    mqr = summaries["cyclic_trace"]
    identity = summaries["identity_trace"]
    recurrent_controls = ("gru_trace", "lstm_trace", "fast_weight_trace", "sinkhorn_trace")
    paired_accuracy_differences: Dict[str, Any] = {}
    paired_forgetting_improvements: Dict[str, Any] = {}
    for control in ("identity_trace", *recurrent_controls):
        paired_accuracy_differences[control] = _summary(
            [
                run["methods"]["cyclic_trace"]["paired_history"]["accuracy"]
                - run["methods"][control]["paired_history"]["accuracy"]
                for run in runs
            ]
        )
        # Positive means cyclic MQR forgets less than the control.
        paired_forgetting_improvements[control] = _summary(
            [
                run["methods"][control]["task_a_forgetting"]
                - run["methods"]["cyclic_trace"]["task_a_forgetting"]
                for run in runs
            ]
        )
    beats_identity = (
        paired_accuracy_differences["identity_trace"]["ci95_low"] > 0.0
    )
    beats_all_non_mqr = all(
        paired_accuracy_differences[control]["ci95_low"] > 0.0
        for control in recurrent_controls
    )
    reduces_forgetting_vs_identity = (
        paired_forgetting_improvements["identity_trace"]["ci95_low"] > 0.0
    )
    reduces_forgetting_vs_all_non_mqr = all(
        paired_forgetting_improvements[control]["ci95_low"] > 0.0
        for control in recurrent_controls
    )
    gates = {
        "seed_count_at_least_10": len(runs) >= 10,
        "formal_protocol_satisfied": bool(
            len(runs) >= 10
            and args.sequence_length >= 10
            and args.train_episodes >= 160
            and args.reversal_episodes >= 40
            and args.eval_episodes >= 40
            and args.paired_episodes >= 40
        ),
        "state_budget_is_192_bytes": args.ring_dim * len(LEAK_RATES) * 4 == 192,
        "competitive_resources_within_1_05": competitive_resource_gate_all_seeds,
        "history_ci_above_chance": mqr["history_accuracy"]["ci95_low"] > 0.5,
        "history_separation_ci_positive": mqr["history_probability_separation"]["ci95_low"] > 0.0,
        "reversal_accuracy_ci_above_chance": (
            mqr["task_b_reversed_accuracy"]["ci95_low"] > 0.5
        ),
        "reversal_gain_ci_positive": mqr["task_b_reversal_gain"]["ci95_low"] > 0.0,
        "learned_write_not_collapsed": (
            mqr["learned_write_rate"]["mean"] > 0.05
            and mqr["learned_write_rate"]["mean"] < 0.95
        ),
        "future_credit_reaches_injection": mqr["future_injection_gradient"]["mean"] > 1e-8,
        "future_credit_reaches_transition": mqr["future_transition_gradient"]["mean"] > 1e-8,
        "identity_replacement_hurts": mqr["identity_replacement_accuracy_drop"]["ci95_low"] > 0.0,
        "beats_identity": beats_identity,
        "beats_all_non_mqr_controls": beats_all_non_mqr,
        "reduces_forgetting_vs_identity": reduces_forgetting_vs_identity,
        "reduces_forgetting_vs_all_non_mqr_controls": (
            reduces_forgetting_vs_all_non_mqr
        ),
    }
    gates["mqr_independent_advantage"] = all(gates.values())
    return {
        "method_summaries": summaries,
        "paired_cyclic_minus_control_accuracy": paired_accuracy_differences,
        "paired_control_minus_cyclic_forgetting": paired_forgetting_improvements,
        "decision_gates": gates,
        "mqr_effective": bool(gates["mqr_independent_advantage"]),
    }


def _run_seed_entry(payload: Tuple[argparse.Namespace, int]) -> Dict[str, Any]:
    args, seed = payload
    torch.set_num_threads(1)
    return run_seed(args, seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=41)
    parser.add_argument("--ring-dim", type=int, default=24)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument("--reversal-episodes", type=int, default=20)
    parser.add_argument("--oracle-warmup-episodes", type=int, default=12)
    parser.add_argument("--critic-calibration-episodes", type=int, default=12)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--paired-episodes", type=int, default=30)
    parser.add_argument("--task-lr", type=float, default=0.03)
    parser.add_argument("--utility-lr", type=float, default=0.08)
    parser.add_argument("--transition-update-interval", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/results/mqr_mechanism_qualification_smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 0 or args.sequence_length < 3 or args.ring_dim <= 0:
        raise ValueError("seeds/ring_dim must be positive and sequence_length at least three")
    if args.workers <= 0 or args.transition_update_interval <= 0:
        raise ValueError("workers and transition_update_interval must be positive")
    if args.ring_dim % 2:
        raise ValueError("ring_dim must be even for cyclic/LSTM state matching")
    random.seed(args.seed_start)
    torch.set_num_threads(1)
    seeds = [args.seed_start + index for index in range(args.seeds)]
    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, args.seeds)
        ) as executor:
            runs = list(executor.map(_run_seed_entry, ((args, seed) for seed in seeds)))
    else:
        runs = [run_seed(args, seed) for seed in seeds]
    result = {
        "schema_version": 2,
        "status": (
            "formal_mechanism_qualification"
            if (
                args.seeds >= 10
                and args.sequence_length >= 10
                and args.train_episodes >= 160
                and args.reversal_episodes >= 40
                and args.eval_episodes >= 40
                and args.paired_episodes >= 40
            )
            else "smoke_or_pilot"
        ),
        "claim_scope": (
            "A mechanism qualification only; it does not replace the canonical "
            "10x10/13x13 Go evidence."
        ),
        "protocol": {
            "predict_before_update": True,
            "future_label_in_gate_features": False,
            "query_identical_across_hidden_bits": True,
            "repeated_query_indices": list(_query_indices(args.sequence_length)),
            "ci_method": "two_sided_student_t_95",
            "independent_advantage_requires_all_non_mqr_controls": True,
            "forgetting_improvement_required": True,
            "competitive_methods": list(COMPETITIVE_METHODS),
            "diagnostic_methods_excluded_from_competitive_resource_gate": [
                method for method in METHODS if method not in COMPETITIVE_METHODS
            ],
            "explicit_event_marker": False,
            "oracle_routing_only_during_warmup": True,
            "sequence_length": args.sequence_length,
            "ring_dim": args.ring_dim,
            "state_budget_bytes_for_mqr_identity_gru_lstm_sinkhorn": (
                args.ring_dim * len(LEAK_RATES) * 4
            ),
            "seeds": args.seeds,
            "train_episodes": args.train_episodes,
            "reversal_episodes": args.reversal_episodes,
            "oracle_warmup_episodes": args.oracle_warmup_episodes,
            "critic_calibration_episodes": args.critic_calibration_episodes,
            "transition_update_interval": args.transition_update_interval,
            "eval_episodes": args.eval_episodes,
            "paired_episodes": args.paired_episodes,
        },
        "runs": runs,
        "aggregate": aggregate(args, runs),
        "preserved_historical_evidence": [
            "analysis/results/unified_temporal_mqr_go_competitive_10x10_formal_10seed.json",
            "analysis/results/unified_temporal_mqr_go_competitive_13x13_formal_10seed.json",
            "analysis/results/unified_temporal_mqr_go_competitive_joint_verify.json",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
