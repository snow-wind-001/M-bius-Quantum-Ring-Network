#!/usr/bin/env python3
"""Matched-core Go benchmark for the unified Temporal Utility MQR agent.

The default run is a small, deterministic mechanism test, not a claim of Go
strength.  It performs marker-free offline advantage-weighted imitation on an
opening stream, protects selected gradients with OGD, changes to later-game
positions, and finally runs short unmasked actor-critic games.  A simulator
twin rollout trains the write/no-write utility critic after every action.

MQR, identity-transition MQR, GRU, and fast-weight cores share the encoder,
utility route, four heads, losses, OGD rank, data order, and update count.  The
chosen dimensions match trainable parameters and analytical forward MACs to a
5% tolerance.  Raw persistent state is reported separately and is a hard
rejection gate for any MQR-effectiveness claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mqr.agent import (  # noqa: E402
    GoLossWeights,
    TemporalUtilityMQRAgent,
    generalized_advantage_estimate,
)
from mqr.agent_baselines import (  # noqa: E402
    AgentResourceAudit,
    MultiTimescaleFastWeightCore,
    MultiTimescaleGRUCore,
    audit_agent_resources,
    matched_resource_gate,
)
from mqr.go import EMPTY, GoBoard, HeuristicGoTeacher  # noqa: E402
from mqr.go_agent import (  # noqa: E402
    GoAgentExample,
    GoAgentTrajectory,
    GoVectorEncoder,
    example_targets,
    generate_go_agent_trajectories,
    generate_ko_history_pairs,
    legality_target,
    twin_write_returns,
)
from mqr.temporal import MultiTimescaleMQR  # noqa: E402


METHODS = ("mqr_unistochastic", "mqr_identity", "gru", "fast_weight")
PROJECTED_INPUT_DIM = 214
LATENT_DIM = 23
LEAK_RATES = (1.0, 0.20, 0.05)
CORE_CONFIG = {
    "mqr_unistochastic": {"ring_dim": 3, "injection_rank": 34},
    "mqr_identity": {"ring_dim": 14, "injection_rank": 27},
    "gru": {"ring_dim": 4},
    "fast_weight": {"memory_dim": 5},
}
LARGE_BOARD_CORE_CONFIG = {
    # Retuned only from analytical parameter/MAC formulas at input_dim=306;
    # no task result was consulted.  All ring widths remain >=3.
    "mqr_unistochastic": {"ring_dim": 3, "injection_rank": 25},
    "mqr_identity": {"ring_dim": 4, "injection_rank": 26},
    "gru": {"ring_dim": 3},
    "fast_weight": {"memory_dim": 4},
}

OFFLINE_WEIGHTS = GoLossWeights(
    placement=1.0,
    legality=0.5,
    pass_decision=0.5,
    value=0.25,
    awr=0.5,
    illegal_mass=0.75,
)
EVALUATION_WEIGHTS = GoLossWeights(
    placement=1.0,
    legality=0.5,
    pass_decision=0.5,
    value=0.25,
)
ACTOR_CRITIC_WEIGHTS = GoLossWeights(
    placement=0.0,
    legality=0.5,
    pass_decision=0.0,
    value=0.5,
    ppo=1.0,
    entropy=0.01,
    reference_kl=0.01,
    illegal_mass=1.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29, 43, 71])
    parser.add_argument("--board-size", type=int, default=10)
    parser.add_argument(
        "--encoder-mode",
        choices=("lossless", "projected"),
        default="lossless",
        help="Use exact board features or the historical random projection.",
    )
    parser.add_argument(
        "--projected-input-dim",
        type=int,
        default=PROJECTED_INPUT_DIM,
        help="Feature width used only when --encoder-mode projected.",
    )
    parser.add_argument("--komi", type=float, default=2.5)
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--probe-episodes", type=int, default=6)
    parser.add_argument("--recorded-moves", type=int, default=6)
    parser.add_argument("--task-b-prefix", type=int, default=3)
    parser.add_argument(
        "--task-b",
        choices=("ko-history", "later-game"),
        default="ko-history",
        help="Use the history-identical superko challenge or a generic later-game shift.",
    )
    parser.add_argument("--twin-horizon", type=int, default=3)
    parser.add_argument("--actor-episodes", type=int, default=3)
    parser.add_argument("--actor-max-decisions", type=int, default=7)
    parser.add_argument("--eval-games", type=int, default=6)
    parser.add_argument("--ogd-rank", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument(
        "--skip-actor-critic",
        action="store_true",
        help="Stop after the non-RL offline AWR and continual-learning phases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/results/unified_temporal_mqr_go.json"),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.train_episodes,
        args.probe_episodes,
        args.recorded_moves,
        args.twin_horizon,
        args.eval_games,
        args.bootstrap_resamples,
    )
    if args.board_size < 2 or any(value <= 0 for value in positive):
        raise ValueError("board size and dataset/evaluation counts must be positive")
    if args.task_b_prefix < 0 or args.actor_episodes < 0 or args.actor_max_decisions <= 0:
        raise ValueError("prefix/actor arguments are out of range")
    if args.ogd_rank < 0:
        raise ValueError("ogd_rank must be non-negative")
    if args.projected_input_dim <= 0:
        raise ValueError("projected_input_dim must be positive")
    if args.task_b == "ko-history" and args.board_size < 3:
        raise ValueError("ko-history requires board_size >= 3")


def _resolved_input_dim(args: argparse.Namespace) -> int:
    if args.encoder_mode == "lossless":
        return 3 * int(args.board_size) * int(args.board_size) + 6
    return int(args.projected_input_dim)


def _resolved_core_config(args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    if args.encoder_mode == "lossless" and args.board_size >= 10:
        return LARGE_BOARD_CORE_CONFIG
    return CORE_CONFIG


def _trajectory_hash(groups: Iterable[Sequence[GoAgentTrajectory]]) -> str:
    digest = hashlib.sha256()
    for group in groups:
        for trajectory in group:
            digest.update(trajectory.task.encode("utf-8"))
            digest.update(int(trajectory.winner).to_bytes(2, "little", signed=True))
            for example in trajectory.examples:
                digest.update(bytes(value + 1 for value in example.board.board))
                digest.update(int(example.target_action).to_bytes(4, "little", signed=False))
                digest.update(float(example.value_target).hex().encode("ascii"))
    return digest.hexdigest()


def _make_core(
    method: str,
    input_dim: int,
    core_config: Dict[str, Dict[str, int]],
):
    if method == "mqr_unistochastic":
        return MultiTimescaleMQR(
            input_dim,
            core_config[method]["ring_dim"],
            LATENT_DIM,
            leak_rates=LEAK_RATES,
            injection_rank=core_config[method]["injection_rank"],
            transition_mode="unistochastic",
            learn_transitions=True,
            readout_bias=True,
        )
    if method == "mqr_identity":
        return MultiTimescaleMQR(
            input_dim,
            core_config[method]["ring_dim"],
            LATENT_DIM,
            leak_rates=LEAK_RATES,
            injection_rank=core_config[method]["injection_rank"],
            transition_mode="identity",
            learn_transitions=False,
            readout_bias=True,
        )
    if method == "gru":
        return MultiTimescaleGRUCore(
            input_dim,
            core_config[method]["ring_dim"],
            LATENT_DIM,
            leak_rates=LEAK_RATES,
        )
    if method == "fast_weight":
        return MultiTimescaleFastWeightCore(
            input_dim,
            core_config[method]["memory_dim"],
            LATENT_DIM,
            leak_rates=LEAK_RATES,
        )
    raise ValueError(f"unknown method: {method}")


def _make_agent(
    method: str,
    board_size: int,
    ogd_rank: int,
    input_dim: int,
    core_config: Dict[str, Dict[str, int]],
) -> TemporalUtilityMQRAgent:
    return TemporalUtilityMQRAgent(
        input_dim,
        board_size=board_size,
        latent_dim=LATENT_DIM,
        core=_make_core(method, input_dim, core_config),
        utility_rank=4,
        initial_write_advantage=0.001,
        memory_cost=0.0005,
        advantage_scale=0.02,
        task_lr=0.015,
        utility_lr=0.20,
        ogd_max_rank=ogd_rank,
        # Counterfactual utility targets move as the task head learns; retaining
        # their old gradients would violate OGD's fixed-old-objective premise.
        utility_ogd_max_rank=0,
        max_update_norm=0.10,
        utility_max_update_norm=0.08,
        awr_temperature=1.0,
        awr_max_weight=10.0,
        ppo_clip=0.20,
        legality_policy_scale=1.0,
    )


def _copy_shared_modules(
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


@torch.no_grad()
def evaluate_probes(
    agent: TemporalUtilityMQRAgent,
    encoder: GoVectorEncoder,
    trajectories: Sequence[GoAgentTrajectory],
    *,
    stream_prefix: str,
) -> Dict[str, float]:
    totals: Dict[str, List[float]] = {
        "total_loss": [],
        "placement_loss": [],
        "legality_loss": [],
        "pass_loss": [],
        "value_loss": [],
    }
    teacher_matches = 0
    raw_legal = 0
    raw_passes = 0
    placement_decisions = 0
    legal_placements = 0
    legality_correct = 0
    legality_count = 0
    write_count = 0
    count = 0
    focus_count = 0
    focus_correct = 0
    focus_probabilities: Dict[str, List[float]] = {"ko": [], "fresh": []}
    first_writes = 0
    first_count = 0
    later_writes = 0
    later_count = 0
    first_predicted_advantages: List[float] = []
    later_predicted_advantages: List[float] = []
    reference = next(agent.parameters())
    for trajectory_index, trajectory in enumerate(trajectories):
        stream = f"{stream_prefix}-{trajectory_index}"
        agent.reset_state(stream)
        for example_index, example in enumerate(trajectory.examples):
            x = encoder.encode_board(example.board).to(reference)
            result = agent.commit_step(x, stream_id=stream, issue_ticket=False)
            output = result["output"]
            action, legal, value = example_targets(example, device=reference.device)
            losses = agent.compute_go_loss(
                output,
                action,
                legality_target=legal,
                value_target=value,
                weights=EVALUATION_WEIGHTS,
            )
            totals["total_loss"].append(float(losses["total"].item()))
            totals["placement_loss"].append(float(losses["placement"].item()))
            totals["legality_loss"].append(float(losses["legality"].item()))
            totals["pass_loss"].append(float(losses["pass"].item()))
            totals["value_loss"].append(float(losses["value"].item()))
            prediction = int(torch.argmax(output.policy_logits[0]).item())
            teacher_matches += int(prediction == example.target_action)
            raw_legal += int(example.board.is_legal(prediction))
            if prediction == example.board.pass_action:
                raw_passes += 1
            else:
                placement_decisions += 1
                legal_placements += int(example.board.is_legal(prediction))
            legality_prediction = output.legality_logits.sigmoid() >= 0.5
            legality_correct += int((legality_prediction == legal.bool()).sum().item())
            legality_count += legal.numel()
            write_count += int(result["effective_write"])
            if example_index == 0:
                first_writes += int(result["effective_write"])
                first_count += 1
                first_predicted_advantages.append(
                    float(result["predicted_write_advantage"])
                )
            else:
                later_writes += int(result["effective_write"])
                later_count += 1
                later_predicted_advantages.append(
                    float(result["predicted_write_advantage"])
                )
            if example.focus_point is not None:
                focus_point = int(example.focus_point)
                focus_probability = float(
                    output.legality_logits[0, focus_point].sigmoid().item()
                )
                focus_target = bool(example.board.is_legal(focus_point))
                focus_correct += int((focus_probability >= 0.5) == focus_target)
                focus_count += 1
                if example.history_condition in focus_probabilities:
                    focus_probabilities[example.history_condition].append(
                        focus_probability
                    )
            count += 1
        agent.reset_state(stream)
    if count == 0:
        raise RuntimeError("probe set is empty")
    ko_probability = (
        statistics.fmean(focus_probabilities["ko"])
        if focus_probabilities["ko"]
        else 0.0
    )
    fresh_probability = (
        statistics.fmean(focus_probabilities["fresh"])
        if focus_probabilities["fresh"]
        else 0.0
    )
    return {
        name: statistics.fmean(values) for name, values in totals.items()
    } | {
        "count": float(count),
        "teacher_agreement": teacher_matches / count,
        "unmasked_legal_rate": raw_legal / count,
        "raw_pass_rate": raw_passes / count,
        "raw_placement_rate": placement_decisions / count,
        "placement_conditional_legal_rate": legal_placements
        / max(1, placement_decisions),
        "useful_legal_placement_rate": legal_placements / count,
        "legality_accuracy": legality_correct / legality_count,
        "write_rate": write_count / count,
        "first_observation_write_rate": first_writes / max(1, first_count),
        "later_observation_write_rate": later_writes / max(1, later_count),
        "first_observation_predicted_advantage": statistics.fmean(
            first_predicted_advantages
        ),
        "later_observation_predicted_advantage": (
            statistics.fmean(later_predicted_advantages)
            if later_predicted_advantages
            else 0.0
        ),
        "focus_count": float(focus_count),
        "focus_legality_accuracy": focus_correct / max(1, focus_count),
        "focus_ko_legal_probability": ko_probability,
        "focus_fresh_legal_probability": fresh_probability,
        "history_legality_separation": fresh_probability - ko_probability,
    }


@torch.no_grad()
def evaluate_game_return(
    agent: TemporalUtilityMQRAgent,
    encoder: GoVectorEncoder,
    *,
    size: int,
    komi: float,
    games: int,
    seed: int,
) -> Dict[str, float]:
    teacher = HeuristicGoTeacher()
    rng = random.Random(int(seed))
    returns: List[float] = []
    legal_actions = 0
    decisions = 0
    agreements = 0
    raw_passes = 0
    placement_decisions = 0
    legal_placements = 0
    reference = next(agent.parameters())
    for game_index in range(int(games)):
        student_color = 1 if game_index % 2 == 0 else -1
        board = GoBoard(size, komi=komi)
        stream = f"game-return-{seed}-{game_index}"
        agent.reset_state(stream)
        illegal_count = 0
        plies = 0
        maximum_plies = 2 * size * size + 4
        while not board.game_over and plies < maximum_plies:
            if board.to_play != student_color:
                board.play(int(teacher.select_move(board.copy())))
                plies += 1
                continue
            snapshot = board.copy()
            x = encoder.encode_board(snapshot).to(reference)
            output = agent.commit_step(
                x,
                stream_id=stream,
                issue_ticket=False,
            )["output"]
            raw_action = int(torch.argmax(output.policy_logits[0]).item())
            target = int(teacher.select_move(snapshot))
            legal = snapshot.is_legal(raw_action)
            decisions += 1
            legal_actions += int(legal)
            agreements += int(raw_action == target)
            if raw_action == snapshot.pass_action:
                raw_passes += 1
            else:
                placement_decisions += 1
                legal_placements += int(legal)
            if not legal:
                illegal_count += 1
                played = snapshot.pass_action
            else:
                played = raw_action
            board.play(int(played))
            plies += 1
        outcome = float(board.winner() * student_color)
        returns.append(outcome - 0.25 * illegal_count)
        agent.reset_state(stream)
        # Consume a deterministic draw only to make extending this evaluator
        # preserve a stable seed protocol.
        rng.random()
    return {
        "games": float(games),
        "mean_return": statistics.fmean(returns),
        "unmasked_legal_rate": legal_actions / max(1, decisions),
        "raw_pass_rate": raw_passes / max(1, decisions),
        "placement_conditional_legal_rate": legal_placements
        / max(1, placement_decisions),
        "useful_legal_placement_rate": legal_placements / max(1, decisions),
        "teacher_agreement": agreements / max(1, decisions),
        "decisions": float(decisions),
    }


def train_offline_awr(
    agent: TemporalUtilityMQRAgent,
    encoder: GoVectorEncoder,
    trajectories: Sequence[GoAgentTrajectory],
    *,
    phase: str,
    twin_horizon: int,
    protect_with_ogd: bool,
) -> Dict[str, Any]:
    losses: List[float] = []
    advantages: List[float] = []
    critic_errors: List[float] = []
    oracle_matches = 0
    observed_positive = 0
    predicted_positive = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    decision_regrets: List[float] = []
    first_advantages: List[float] = []
    later_advantages: List[float] = []
    utility_replay: List[Tuple[torch.Tensor, float]] = []
    utility_batch_losses: List[float] = []
    write_count = 0
    update_count = 0
    reference = next(agent.parameters())
    started = time.perf_counter()
    for trajectory_index, trajectory in enumerate(trajectories):
        stream = f"train-{phase}-{trajectory_index}"
        agent.reset_state(stream)
        for example_index, example in enumerate(trajectory.examples):
            x = encoder.encode_board(example.board).to(reference)
            committed = agent.commit_step(x, stream_id=stream)
            ticket = int(committed["ticket_id"])
            future = trajectory.examples[example_index + 1 :]
            no_return, write_return = twin_write_returns(
                agent,
                ticket,
                future,
                encoder,
                horizon=twin_horizon,
                weights=EVALUATION_WEIGHTS,
            )
            remember = bool(
                not protect_with_ogd
                and agent.task_gradient_memory.rank < agent.task_gradient_memory.max_rank
                and update_count % 3 == 0
            )
            utility = agent.calibrate_write_critic(
                ticket,
                no_write_return=no_return,
                write_return=write_return,
                learn=False,
                remember_gradient=False,
                project_with_memory=False,
            )
            observed_advantage = float(utility["write_advantage"])
            predicted_advantage = float(utility["predicted_advantage_before_update"])
            advantages.append(observed_advantage)
            (first_advantages if example_index == 0 else later_advantages).append(
                observed_advantage
            )
            critic_errors.append(abs(predicted_advantage - observed_advantage))
            predicted_write = predicted_advantage > 0.0
            oracle_write = observed_advantage > 0.0
            oracle_matches += int(predicted_write == oracle_write)
            observed_positive += int(oracle_write)
            predicted_positive += int(predicted_write)
            true_positive += int(predicted_write and oracle_write)
            true_negative += int(not predicted_write and not oracle_write)
            false_positive += int(predicted_write and not oracle_write)
            false_negative += int(not predicted_write and oracle_write)
            selected_return = write_return if committed["effective_write"] else no_return
            decision_regrets.append(max(no_return, write_return) - selected_return)
            write_count += int(committed["effective_write"])
            utility_replay.append(
                (committed["features"].detach().clone(), observed_advantage)
            )
            if len(utility_replay) > 16:
                utility_replay.pop(0)
            if (update_count + 1) % 4 == 0:
                replay_features = torch.cat(
                    [item[0] for item in utility_replay], dim=0
                )
                replay_advantages = replay_features.new_tensor(
                    [item[1] for item in utility_replay]
                )
                for _ in range(2):
                    batch_info = agent.fit_write_critic_batch(
                        replay_features,
                        replay_advantages,
                        project_with_memory=False,
                    )
                    utility_batch_losses.append(float(batch_info["loss"]))

            action, legal, value = example_targets(example, device=reference.device)
            awr_advantage = value - committed["output"].value.to(value)
            feedback = agent.apply_feedback(
                ticket,
                action,
                legality_target=legal,
                value_target=value,
                awr_advantage=awr_advantage,
                weights=OFFLINE_WEIGHTS,
                remember_gradient=remember,
                project_with_memory=protect_with_ogd,
            )
            losses.append(float(feedback["loss"]))
            update_count += 1
        agent.reset_state(stream)
    if utility_replay and update_count % 4 != 0:
        replay_features = torch.cat([item[0] for item in utility_replay], dim=0)
        replay_advantages = replay_features.new_tensor(
            [item[1] for item in utility_replay]
        )
        for _ in range(2):
            batch_info = agent.fit_write_critic_batch(
                replay_features,
                replay_advantages,
                project_with_memory=False,
            )
            utility_batch_losses.append(float(batch_info["loss"]))
    elapsed = time.perf_counter() - started
    positive_total = true_positive + false_negative
    negative_total = true_negative + false_positive
    sensitivity = true_positive / positive_total if positive_total else 0.0
    specificity = true_negative / negative_total if negative_total else 0.0
    represented_classes = int(positive_total > 0) + int(negative_total > 0)
    balanced_accuracy = (
        (sensitivity + specificity) / represented_classes
        if represented_classes
        else 0.0
    )
    positive_rate = observed_positive / max(1, update_count)
    return {
        "updates": float(update_count),
        "mean_loss": statistics.fmean(losses),
        "mean_write_advantage": statistics.fmean(advantages),
        "critic_mae": statistics.fmean(critic_errors),
        "critic_oracle_sign_accuracy": oracle_matches / max(1, update_count),
        "critic_balanced_sign_accuracy": balanced_accuracy,
        "oracle_write_rate": positive_rate,
        "predicted_write_rate": predicted_positive / max(1, update_count),
        "majority_sign_baseline": max(positive_rate, 1.0 - positive_rate),
        "mean_write_decision_regret": statistics.fmean(decision_regrets),
        "first_observation_mean_advantage": statistics.fmean(first_advantages),
        "first_observation_mean_normalized_target": statistics.fmean(
            value / agent.advantage_scale for value in first_advantages
        ),
        "first_observation_oracle_write_rate": sum(
            value > 0.0 for value in first_advantages
        ) / max(1, len(first_advantages)),
        "later_observation_mean_advantage": (
            statistics.fmean(later_advantages) if later_advantages else 0.0
        ),
        "later_observation_mean_normalized_target": (
            statistics.fmean(
                value / agent.advantage_scale for value in later_advantages
            )
            if later_advantages
            else 0.0
        ),
        "later_observation_oracle_write_rate": sum(
            value > 0.0 for value in later_advantages
        ) / max(1, len(later_advantages)),
        "write_rate": write_count / max(1, update_count),
        "task_ogd_rank": float(agent.task_gradient_memory.rank),
        "utility_ogd_rank": float(agent.utility_gradient_memory.rank),
        "utility_replay_capacity": 16.0,
        "utility_batch_interval": 4.0,
        "utility_batch_updates": float(len(utility_batch_losses)),
        "utility_batch_loss": statistics.fmean(utility_batch_losses),
        "seconds": elapsed,
        "updates_per_second": update_count / max(elapsed, 1e-12),
    }


def calibrate_current_write_utility(
    agent: TemporalUtilityMQRAgent,
    encoder: GoVectorEncoder,
    trajectories: Sequence[GoAgentTrajectory],
    *,
    phase: str,
    twin_horizon: int,
    optimization_steps: int = 20,
) -> Dict[str, float]:
    """Re-estimate utility after freezing the phase's learned task head."""

    features: List[torch.Tensor] = []
    advantages: List[float] = []
    positions: List[int] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        stream = f"critic-calibration-{phase}-{trajectory_index}"
        agent.reset_state(stream)
        for example_index, example in enumerate(trajectory.examples):
            reference = next(agent.parameters())
            x = encoder.encode_board(example.board).to(reference)
            committed = agent.commit_step(x, stream_id=stream)
            ticket = int(committed["ticket_id"])
            no_return, write_return = twin_write_returns(
                agent,
                ticket,
                trajectory.examples[example_index + 1 :],
                encoder,
                horizon=twin_horizon,
                weights=EVALUATION_WEIGHTS,
            )
            utility = agent.calibrate_write_critic(
                ticket,
                no_write_return=no_return,
                write_return=write_return,
                learn=False,
                project_with_memory=False,
            )
            agent.cancel_ticket_branch(ticket, "task")
            features.append(committed["features"].detach().clone())
            advantages.append(float(utility["write_advantage"]))
            positions.append(example_index)
        agent.reset_state(stream)
    feature_batch = torch.cat(features, dim=0)
    advantage_batch = feature_batch.new_tensor(advantages)
    losses = []
    for _ in range(int(optimization_steps)):
        info = agent.fit_write_critic_batch(
            feature_batch,
            advantage_batch,
            project_with_memory=False,
        )
        losses.append(float(info["loss"]))
    with torch.no_grad():
        predictions = (
            agent.advantage_scale * agent.utility_gate(feature_batch).squeeze(1)
        )
    first = [index for index, position in enumerate(positions) if position == 0]
    later = [index for index, position in enumerate(positions) if position > 0]

    def mean_at(values: Sequence[float] | torch.Tensor, indices: Sequence[int]) -> float:
        if not indices:
            return 0.0
        if isinstance(values, torch.Tensor):
            return float(values[torch.tensor(indices, device=values.device)].mean().item())
        return statistics.fmean(values[index] for index in indices)

    return {
        "examples": float(len(advantages)),
        "optimization_steps": float(optimization_steps),
        "initial_batch_loss": losses[0],
        "final_batch_loss": losses[-1],
        "mean_advantage": statistics.fmean(advantages),
        "first_mean_advantage": mean_at(advantages, first),
        "later_mean_advantage": mean_at(advantages, later),
        "first_predicted_advantage": mean_at(predictions, first),
        "later_predicted_advantage": mean_at(predictions, later),
        "first_predicted_write_rate": sum(
            float(predictions[index].item()) > 0.0 for index in first
        ) / max(1, len(first)),
        "later_predicted_write_rate": sum(
            float(predictions[index].item()) > 0.0 for index in later
        ) / max(1, len(later)),
    }


def run_short_actor_critic(
    agent: TemporalUtilityMQRAgent,
    encoder: GoVectorEncoder,
    *,
    size: int,
    komi: float,
    episodes: int,
    max_decisions: int,
    twin_horizon: int,
    seed: int,
) -> Dict[str, float]:
    if episodes <= 0:
        return {
            "episodes": 0.0,
            "updates": 0.0,
            "mean_return": 0.0,
            "unmasked_legal_rate": 0.0,
            "mean_ppo_loss": 0.0,
        }
    teacher = HeuristicGoTeacher()
    generator = torch.Generator().manual_seed(int(seed))
    reference = next(agent.parameters())
    episode_returns: List[float] = []
    ppo_losses: List[float] = []
    legal_count = 0
    decision_count = 0
    raw_pass_count = 0
    placement_count = 0
    legal_placement_count = 0
    started = time.perf_counter()
    offline_lr = agent.task_lr
    agent.task_lr = 0.25 * offline_lr
    for episode_index in range(int(episodes)):
        board = GoBoard(size, komi=komi)
        stream = f"actor-{seed}-{episode_index}"
        agent.reset_state(stream)
        records: List[Dict[str, Any]] = []
        while not board.game_over and len(records) < int(max_decisions):
            snapshot = board.copy()
            target = int(teacher.select_move(snapshot))
            x = encoder.encode_board(snapshot).to(reference)
            committed = agent.commit_step(x, stream_id=stream)
            probabilities = F.softmax(committed["policy_logits"][0], dim=0)
            raw_action = int(torch.multinomial(probabilities.cpu(), 1, generator=generator).item())
            old_log_prob = float(torch.log(probabilities[raw_action].clamp_min(1e-12)).item())
            legal = snapshot.is_legal(raw_action)
            legal_count += int(legal)
            decision_count += 1
            if raw_action == snapshot.pass_action:
                raw_pass_count += 1
            else:
                placement_count += 1
                legal_placement_count += int(legal)
            reward = 0.05 + 0.20 * float(raw_action == target) if legal else -1.0
            played = raw_action if legal else snapshot.pass_action
            move_result = board.play(int(played))
            reward += 0.10 * float(move_result.captures)
            if not board.game_over:
                opponent = int(teacher.select_move(board.copy()))
                board.play(opponent)
            records.append(
                {
                    "snapshot": snapshot,
                    "target": target,
                    "ticket": int(committed["ticket_id"]),
                    "action": raw_action,
                    "old_log_prob": old_log_prob,
                    "old_log_probs": F.log_softmax(
                        committed["policy_logits"], dim=1
                    ).detach(),
                    "features": committed["features"].detach().clone(),
                    "value": float(committed["output"].value.item()),
                    "reward": reward,
                }
            )
        if not records:
            agent.reset_state(stream)
            continue
        outcome = float(board.winner())
        records[-1]["reward"] += outcome
        episode_returns.append(sum(float(record["reward"]) for record in records))
        examples = tuple(
            GoAgentExample(
                board=record["snapshot"],
                target_action=int(record["target"]),
                value_target=outcome,
                episode_index=episode_index,
                move_index=index,
            )
            for index, record in enumerate(records)
        )

        # Critic calibration happens after the actions and before policy updates.
        actor_utility_advantages: List[float] = []
        for index, record in enumerate(records):
            no_return, write_return = twin_write_returns(
                agent,
                int(record["ticket"]),
                examples[index + 1 :],
                encoder,
                horizon=twin_horizon,
                weights=EVALUATION_WEIGHTS,
            )
            utility = agent.calibrate_write_critic(
                int(record["ticket"]),
                no_write_return=no_return,
                write_return=write_return,
                learn=False,
                project_with_memory=True,
            )
            actor_utility_advantages.append(float(utility["write_advantage"]))
        actor_utility_features = torch.cat(
            [record["features"] for record in records], dim=0
        )
        actor_utility_targets = actor_utility_features.new_tensor(
            actor_utility_advantages
        )
        for _ in range(2):
            agent.fit_write_critic_batch(
                actor_utility_features,
                actor_utility_targets,
                project_with_memory=False,
            )

        rewards = torch.tensor([[record["reward"]] for record in records], dtype=torch.float32)
        values = torch.tensor([[record["value"]] for record in records], dtype=torch.float32)
        dones = torch.zeros_like(rewards)
        dones[-1] = 1.0
        advantages, returns = generalized_advantage_estimate(
            rewards,
            values,
            dones,
            gamma=0.97,
            gae_lambda=0.90,
        )
        flat_advantage = advantages[:, 0]
        if flat_advantage.numel() > 1 and float(flat_advantage.std(unbiased=False).item()) > 1e-8:
            flat_advantage = (
                flat_advantage - flat_advantage.mean()
            ) / flat_advantage.std(unbiased=False)
        for index, record in enumerate(records):
            snapshot = record["snapshot"]
            feedback = agent.apply_feedback(
                int(record["ticket"]),
                torch.tensor([record["action"]], device=reference.device),
                legality_target=legality_target(snapshot, device=reference.device),
                value_target=returns[index].to(device=reference.device),
                ppo_advantage=flat_advantage[index : index + 1].to(reference.device),
                old_log_prob=torch.tensor(
                    [record["old_log_prob"]], device=reference.device
                ),
                reference_log_probs=record["old_log_probs"].to(reference.device),
                weights=ACTOR_CRITIC_WEIGHTS,
                project_with_memory=True,
                allow_stale=index > 0,
            )
            ppo_losses.append(float(feedback["losses"]["ppo"]))
        agent.reset_state(stream)
    agent.task_lr = offline_lr
    elapsed = time.perf_counter() - started
    return {
        "episodes": float(len(episode_returns)),
        "updates": float(len(ppo_losses)),
        "mean_return": statistics.fmean(episode_returns),
        "unmasked_legal_rate": legal_count / max(1, decision_count),
        "raw_pass_rate": raw_pass_count / max(1, decision_count),
        "placement_conditional_legal_rate": legal_placement_count
        / max(1, placement_count),
        "useful_legal_placement_rate": legal_placement_count
        / max(1, decision_count),
        "mean_ppo_loss": statistics.fmean(ppo_losses),
        "seconds": elapsed,
        "actor_lr_ratio": 0.25,
    }


def _paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> Dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap inputs must have equal non-zero length")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    rng = random.Random(int(seed))
    estimates = []
    for _ in range(int(resamples)):
        estimates.append(
            statistics.fmean(rng.choice(differences) for _ in differences)
        )
    estimates.sort()
    lower_index = max(0, int(0.025 * len(estimates)))
    upper_index = min(len(estimates) - 1, int(0.975 * len(estimates)))
    return {
        "mean_difference": statistics.fmean(differences),
        "ci95_low": estimates[lower_index],
        "ci95_high": estimates[upper_index],
        "wins": float(sum(value > 0.0 for value in differences)),
        "ties": float(sum(value == 0.0 for value in differences)),
        "seeds": float(len(differences)),
    }


def _metric_by_method(
    seed_results: Sequence[Mapping[str, Any]],
    method: str,
    path: Sequence[str],
) -> List[float]:
    values = []
    for seed_result in seed_results:
        current: Any = seed_result["methods"][method]
        for key in path:
            current = current[key]
        values.append(float(current))
    return values


def summarize_and_gate(
    seed_results: Sequence[Mapping[str, Any]],
    resource_gate: Mapping[str, Any],
    *,
    actor_enabled: bool,
    bootstrap_resamples: int,
) -> Dict[str, Any]:
    final_stage = "after_actor_critic" if actor_enabled else "after_task_b"
    metric_specs = (
        (
            "unmasked_legal_rate",
            ("game_returns", final_stage, "unmasked_legal_rate"),
            False,
        ),
        (
            "useful_legal_placement_rate",
            ("game_returns", final_stage, "useful_legal_placement_rate"),
            False,
        ),
        (
            "game_return",
            ("game_returns", final_stage, "mean_return"),
            False,
        ),
        (
            "history_focus_accuracy",
            ("probes", final_stage, "task_b", "focus_legality_accuracy"),
            False,
        ),
        (
            "forgetting_loss",
            ("forgetting", "task_a_positive_forgetting"),
            True,
        ),
    )
    comparisons: Dict[str, Dict[str, Any]] = {}
    decisive = True
    for comparator_index, comparator in enumerate(METHODS[1:]):
        current: Dict[str, Any] = {}
        for metric, path, reverse in metric_specs:
            mqr_values = _metric_by_method(seed_results, "mqr_unistochastic", path)
            control_values = _metric_by_method(seed_results, comparator, path)
            left, right = (
                (control_values, mqr_values) if reverse else (mqr_values, control_values)
            )
            interval = _paired_bootstrap(
                left,
                right,
                resamples=bootstrap_resamples,
                seed=9100 + 101 * comparator_index + len(metric),
            )
            interval["positive_means_mqr_better"] = True
            interval["decisive"] = bool(interval["ci95_low"] > 0.0)
            decisive = decisive and bool(interval["decisive"])
            current[metric] = interval
        comparisons[comparator] = current

    reasons: List[str] = []
    if not bool(resource_gate["all_resources_matched"]):
        reasons.append("strict parameter/state/MAC resource gate failed")
    if not decisive:
        reasons.append(
            "not every legality/return/history/forgetting paired CI is above zero"
        )
    return {
        "final_stage": final_stage,
        "metric_sources": {
            metric: ".".join(path) for metric, path, _reverse in metric_specs
        },
        "comparisons": comparisons,
        "all_performance_gates_decisive": decisive,
        "mqr_effective": bool(resource_gate["all_resources_matched"] and decisive),
        "rejection_reasons": reasons,
    }


def run_seed(args: argparse.Namespace, seed: int) -> Tuple[Dict[str, Any], List[AgentResourceAudit]]:
    input_dim = _resolved_input_dim(args)
    core_config = _resolved_core_config(args)
    encoder = GoVectorEncoder(
        args.board_size,
        input_dim,
        seed=seed + 4000,
        projection_mode=("identity" if args.encoder_mode == "lossless" else "random"),
    )
    train_a = generate_go_agent_trajectories(
        args.train_episodes,
        size=args.board_size,
        komi=args.komi,
        seed=seed + 10,
        task="opening",
        start_random_moves=0,
        recorded_moves=args.recorded_moves,
        random_move_probability=0.35,
    )
    probe_a = generate_go_agent_trajectories(
        args.probe_episodes,
        size=args.board_size,
        komi=args.komi,
        seed=seed + 30,
        task="opening_probe",
        start_random_moves=0,
        recorded_moves=args.recorded_moves,
        random_move_probability=0.45,
    )
    if args.task_b == "ko-history":
        train_pairs = math.ceil(args.train_episodes * args.recorded_moves / 4)
        probe_pairs = math.ceil(args.probe_episodes * args.recorded_moves / 4)
        train_b = generate_ko_history_pairs(
            train_pairs,
            size=args.board_size,
            komi=args.komi,
            seed=seed + 20,
            symmetry_pool=(0, 1, 2, 3),
        )
        probe_b = generate_ko_history_pairs(
            probe_pairs,
            size=args.board_size,
            komi=args.komi,
            seed=seed + 40,
            symmetry_pool=(4, 5, 6, 7),
        )
    else:
        train_b = generate_go_agent_trajectories(
            args.train_episodes,
            size=args.board_size,
            komi=args.komi,
            seed=seed + 20,
            task="later_game",
            start_random_moves=args.task_b_prefix,
            recorded_moves=args.recorded_moves,
            random_move_probability=0.60,
        )
        probe_b = generate_go_agent_trajectories(
            args.probe_episodes,
            size=args.board_size,
            komi=args.komi,
            seed=seed + 40,
            task="later_game_probe",
            start_random_moves=args.task_b_prefix,
            recorded_moves=args.recorded_moves,
            random_move_probability=0.55,
        )
    stream_hash = _trajectory_hash((train_a, train_b, probe_a, probe_b))

    agents: Dict[str, TemporalUtilityMQRAgent] = {}
    audits: List[AgentResourceAudit] = []
    shared_source = None
    for method_index, method in enumerate(METHODS):
        torch.manual_seed(seed * 100 + method_index)
        agent = _make_agent(
            method,
            args.board_size,
            args.ogd_rank,
            input_dim,
            core_config,
        )
        if shared_source is None:
            shared_source = agent
        else:
            _copy_shared_modules(shared_source, agent)
        agents[method] = agent
        audits.append(audit_agent_resources(agent, method))

    methods: Dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        agent = agents[method]
        initial_a = evaluate_probes(
            agent,
            encoder,
            probe_a,
            stream_prefix=f"{method}-initial-a",
        )
        initial_b = evaluate_probes(
            agent,
            encoder,
            probe_b,
            stream_prefix=f"{method}-initial-b",
        )
        initial_return = evaluate_game_return(
            agent,
            encoder,
            size=args.board_size,
            komi=args.komi,
            games=args.eval_games,
            seed=seed * 1000 + method_index,
        )
        phase_a = train_offline_awr(
            agent,
            encoder,
            train_a,
            phase="task_a",
            twin_horizon=args.twin_horizon,
            protect_with_ogd=False,
        )
        phase_a["post_phase_critic_calibration"] = calibrate_current_write_utility(
            agent,
            encoder,
            train_a,
            phase="task_a",
            twin_horizon=args.twin_horizon,
        )
        after_a_a = evaluate_probes(
            agent,
            encoder,
            probe_a,
            stream_prefix=f"{method}-after-a-a",
        )
        after_a_b = evaluate_probes(
            agent,
            encoder,
            probe_b,
            stream_prefix=f"{method}-after-a-b",
        )
        after_a_return = evaluate_game_return(
            agent,
            encoder,
            size=args.board_size,
            komi=args.komi,
            games=args.eval_games,
            seed=seed * 1000 + 100 + method_index,
        )
        phase_b = train_offline_awr(
            agent,
            encoder,
            train_b,
            phase="task_b",
            twin_horizon=args.twin_horizon,
            protect_with_ogd=True,
        )
        phase_b["post_phase_critic_calibration"] = calibrate_current_write_utility(
            agent,
            encoder,
            train_b,
            phase="task_b",
            twin_horizon=args.twin_horizon,
        )
        after_b_a = evaluate_probes(
            agent,
            encoder,
            probe_a,
            stream_prefix=f"{method}-after-b-a",
        )
        after_b_b = evaluate_probes(
            agent,
            encoder,
            probe_b,
            stream_prefix=f"{method}-after-b-b",
        )
        after_b_return = evaluate_game_return(
            agent,
            encoder,
            size=args.board_size,
            komi=args.komi,
            games=args.eval_games,
            seed=seed * 1000 + 200 + method_index,
        )

        actor = run_short_actor_critic(
            agent,
            encoder,
            size=args.board_size,
            komi=args.komi,
            episodes=0 if args.skip_actor_critic else args.actor_episodes,
            max_decisions=args.actor_max_decisions,
            twin_horizon=args.twin_horizon,
            seed=seed * 100 + method_index,
        )
        if args.skip_actor_critic:
            after_actor_a = dict(after_b_a)
            after_actor_b = dict(after_b_b)
            after_actor_return = dict(after_b_return)
        else:
            after_actor_a = evaluate_probes(
                agent,
                encoder,
                probe_a,
                stream_prefix=f"{method}-after-actor-a",
            )
            after_actor_b = evaluate_probes(
                agent,
                encoder,
                probe_b,
                stream_prefix=f"{method}-after-actor-b",
            )
            after_actor_return = evaluate_game_return(
                agent,
                encoder,
                size=args.board_size,
                komi=args.komi,
                games=args.eval_games,
                seed=seed * 1000 + 300 + method_index,
            )
        methods[method] = {
            "offline_awr": {"task_a": phase_a, "task_b": phase_b},
            "actor_critic": actor,
            "probes": {
                "initial": {"task_a": initial_a, "task_b": initial_b},
                "after_task_a": {"task_a": after_a_a, "task_b": after_a_b},
                "after_task_b": {"task_a": after_b_a, "task_b": after_b_b},
                "after_actor_critic": {
                    "task_a": after_actor_a,
                    "task_b": after_actor_b,
                },
            },
            "game_returns": {
                "initial": initial_return,
                "after_task_a": after_a_return,
                "after_task_b": after_b_return,
                "after_actor_critic": after_actor_return,
            },
            "forgetting": {
                "task_a_total_loss_increase": (
                    after_b_a["total_loss"] - after_a_a["total_loss"]
                ),
                "task_a_positive_forgetting": max(
                    0.0,
                    after_b_a["total_loss"] - after_a_a["total_loss"],
                ),
                "task_a_teacher_agreement_drop": (
                    after_a_a["teacher_agreement"] - after_b_a["teacher_agreement"]
                ),
                "task_a_unmasked_legality_drop": (
                    after_a_a["unmasked_legal_rate"] - after_b_a["unmasked_legal_rate"]
                ),
            },
            "invariants": {
                "max_unitary_error": agent.core.max_unitary_error(),
                "max_stochastic_error": agent.core.max_stochastic_error(),
                "pending_tickets": agent.pending_ticket_count,
                "online_state_bytes": agent.online_state_bytes,
            },
        }
    return {"seed": seed, "stream_hash": stream_hash, "methods": methods}, audits


def main() -> int:
    args = parse_args()
    _validate_args(args)
    all_seed_results: List[Dict[str, Any]] = []
    canonical_audits: List[AgentResourceAudit] = []
    started = time.perf_counter()
    for seed_index, seed in enumerate(args.seeds):
        print(f"seed={seed} ({seed_index + 1}/{len(args.seeds)})", flush=True)
        result, audits = run_seed(args, int(seed))
        all_seed_results.append(result)
        if not canonical_audits:
            canonical_audits = audits
        elif [asdict(item) for item in audits] != [asdict(item) for item in canonical_audits]:
            raise RuntimeError("resource audit changed across seeds")

    parameter_flop_gate = matched_resource_gate(
        canonical_audits,
        parameter_ratio_limit=1.05,
        state_ratio_limit=9.0,
        mac_ratio_limit=1.05,
    )
    strict_resource_gate = matched_resource_gate(
        canonical_audits,
        parameter_ratio_limit=1.05,
        state_ratio_limit=1.05,
        mac_ratio_limit=1.05,
    )
    effectiveness = summarize_and_gate(
        all_seed_results,
        strict_resource_gate,
        actor_enabled=not args.skip_actor_critic,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    payload = {
        "schema_version": 1,
        "experiment": "unified_temporal_mqr_go",
        "claim_scope": (
            "controlled simulator mechanism test; not evidence of general Go strength "
            "or animal-like learning"
        ),
        "config": {
            **vars(args),
            "output": str(args.output),
            "input_dim": _resolved_input_dim(args),
            "encoder_information_preserving": args.encoder_mode == "lossless",
            "latent_dim": LATENT_DIM,
            "leak_rates": list(LEAK_RATES),
            "core_config": _resolved_core_config(args),
            "offline_weights": asdict(OFFLINE_WEIGHTS),
            "actor_critic_weights": asdict(ACTOR_CRITIC_WEIGHTS),
        },
        "protocol": {
            "write_target": "paired future negative multi-head loss",
            "future_branch_policy": "fast writes open, all later slow writes closed in both branches",
            "offline_stage": "advantage-weighted imitation",
            "rl_stage": "short unmasked PPO-style actor-critic",
            "lora_stage": "disabled in this core-isolation benchmark",
            "legality_mask_during_action": False,
            "prediction_before_update": True,
        },
        "resources": {
            "audits": [item.to_dict() for item in canonical_audits],
            "parameter_flop_gate_with_reported_state_cap": {
                **parameter_flop_gate,
                "decision_role": "diagnostic_only",
            },
            "strict_equal_resource_gate": {
                **strict_resource_gate,
                "decision_role": "effectiveness_gate",
            },
        },
        "seed_results": all_seed_results,
        "effectiveness_gate": effectiveness,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
        },
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "mqr_effective": effectiveness["mqr_effective"],
        "rejection_reasons": effectiveness["rejection_reasons"],
        "wall_seconds": payload["wall_seconds"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
