#!/usr/bin/env python3
"""Fail-closed, equal-resource 10x10/13x13 online Go benchmark.

This experiment is deliberately separate from the historical unified Go
pilot.  It compares a unistochastic MQR sidecar with an identity ablation,
GRU, LSTM, fast-weight, LoRA, OGD-LoRA, replay-LoRA, and an implicitly
differentiated Sinkhorn core.  Every method uses the same frozen base, four
heads, causal utility route, loss, trajectories, and predict-before-update
feedback order.  The competitive configurations expose exactly twelve
float32 recurrent scalars and are calibrated to a five-percent parameter/MAC
window on both supported board sizes.

Smoke and qualification runs are never publication evidence.  ``--formal``
enforces 10--20 unique seeds, at least 128 primary online updates per task,
64 held-out probes per task, and eight unmasked evaluation games per seed.
Even a complete formal run remains negative unless every resource, protocol,
learning, and paired performance gate passes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unified_temporal_mqr_go import (  # noqa: E402
    EVALUATION_WEIGHTS,
    OFFLINE_WEIGHTS,
    _paired_bootstrap,
    evaluate_game_return,
    evaluate_probes,
)
from mqr.agent import TemporalUtilityMQRAgent  # noqa: E402
from mqr.agent_baselines import (  # noqa: E402
    AgentResourceAudit,
    FrozenResidualCoreWrapper,
    MultiTimescaleFastWeightCore,
    MultiTimescaleGRUCore,
    MultiTimescaleLSTMCore,
    MultiTimescaleLoRAResidualCore,
    MultiTimescaleSinkhornCore,
    audit_agent_resources,
    matched_competitive_resource_gate,
)
from mqr.go_agent import (  # noqa: E402
    GoAgentExample,
    GoAgentTrajectory,
    GoVectorEncoder,
    example_targets,
    generate_go_agent_trajectories,
    generate_ko_history_pairs,
    twin_write_returns,
)
from mqr.temporal import MultiTimescaleMQR  # noqa: E402


METHODS = (
    "mqr_unistochastic_ogd",
    "mqr_identity_ogd",
    "gru_ogd",
    "lstm_ogd",
    "fast_weight_ogd",
    "lora",
    "ogd_lora",
    "replay_lora",
    "sinkhorn_ogd",
)
LORA_FAMILY = ("lora", "ogd_lora", "replay_lora")
OGD_METHODS = (
    "mqr_unistochastic_ogd",
    "mqr_identity_ogd",
    "gru_ogd",
    "lstm_ogd",
    "fast_weight_ogd",
    "ogd_lora",
    "sinkhorn_ogd",
)
PRIMARY_METHOD = "mqr_unistochastic_ogd"
FORMAL_SEEDS = (7, 17, 29, 43, 71, 89, 107, 131, 149, 173)
LATENT_DIM = 23
LEAK_RATES = (1.0, 0.20, 0.05)
RING_STATE_DIM = 4
SINKHORN_ITERATIONS = 30
SPATIAL_SKIP_CHANNELS = 8

# These ranks were selected solely from analytical parameter/MAC formulas.
# No task metric was consulted.  They minimize the largest charged forward
# MAC count while keeping every rank >= 32 and both ratios <= 1.05.
BOARD_RANKS: Dict[int, Dict[str, int]] = {
    10: {
        "mqr_unistochastic_ogd": 73,
        "mqr_identity_ogd": 77,
        "gru_ogd": 71,
        "lstm_ogd": 74,
        "fast_weight_ogd": 77,
        "lora": 75,
        "ogd_lora": 75,
        "replay_lora": 75,
        "sinkhorn_ogd": 73,
    },
    13: {
        "mqr_unistochastic_ogd": 38,
        "mqr_identity_ogd": 40,
        "gru_ogd": 38,
        "lstm_ogd": 39,
        "fast_weight_ogd": 40,
        "lora": 40,
        "ogd_lora": 40,
        "replay_lora": 40,
        "sinkhorn_ogd": 38,
    },
}


@dataclass(frozen=True)
class ReplayTensorExample:
    """Tensor-only replay item with an auditable payload byte count."""

    features: torch.Tensor
    action: torch.Tensor
    legality: torch.Tensor
    value: torch.Tensor

    @property
    def storage_bytes(self) -> int:
        return int(
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in (self.features, self.action, self.legality, self.value)
            )
        )


class ByteCappedReplay:
    """FIFO replay whose capacity is measured from tensor payloads only."""

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.capacity_bytes = int(capacity_bytes)
        self.items: List[ReplayTensorExample] = []
        self.storage_bytes = 0
        self.evictions = 0

    def append(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
        legality: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        item = ReplayTensorExample(
            features.detach().to(device="cpu", dtype=torch.float32).clone(),
            action.detach().to(device="cpu", dtype=torch.int64).clone(),
            legality.detach().to(device="cpu", dtype=torch.bool).clone(),
            value.detach().to(device="cpu", dtype=torch.float32).clone(),
        )
        if item.storage_bytes > self.capacity_bytes:
            raise RuntimeError("one replay item exceeds the allocated capacity")
        self.items.append(item)
        self.storage_bytes += item.storage_bytes
        while self.storage_bytes > self.capacity_bytes:
            removed = self.items.pop(0)
            self.storage_bytes -= removed.storage_bytes
            self.evictions += 1

    def cyclic(self, index: int) -> ReplayTensorExample:
        if not self.items:
            raise RuntimeError("cannot sample an empty replay buffer")
        return self.items[int(index) % len(self.items)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-size", type=int, choices=(10, 13), default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument("--train-exposures", type=int, default=128)
    parser.add_argument("--probe-exposures", type=int, default=64)
    parser.add_argument("--recorded-moves", type=int, default=4)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--twin-horizon", type=int, default=3)
    parser.add_argument("--komi", type=float, default=2.5)
    parser.add_argument("--task-lr", type=float, default=0.03)
    parser.add_argument("--utility-lr", type=float, default=0.10)
    parser.add_argument("--training-write-probability", type=float, default=0.50)
    parser.add_argument("--ogd-rank", type=int, default=4)
    parser.add_argument("--transition-update-interval", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/results/unified_temporal_mqr_go_competitive_10x10.json"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.train_exposures,
        args.probe_exposures,
        args.recorded_moves,
        args.eval_games,
        args.twin_horizon,
        args.ogd_rank,
        args.transition_update_interval,
        args.bootstrap_resamples,
        args.workers,
        args.torch_threads,
    )
    if any(int(value) <= 0 for value in positive):
        raise ValueError("all count, rank, and interval arguments must be positive")
    if args.task_lr <= 0.0 or args.utility_lr <= 0.0:
        raise ValueError("learning rates must be positive")
    if not 0.0 <= args.training_write_probability <= 1.0:
        raise ValueError("training_write_probability must lie in [0, 1]")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    if args.train_exposures % 4 or args.probe_exposures % 4:
        raise ValueError("train/probe exposures must be divisible by four")
    if args.train_exposures % args.recorded_moves:
        raise ValueError("train exposures must be divisible by recorded moves")
    if args.probe_exposures % args.recorded_moves:
        raise ValueError("probe exposures must be divisible by recorded moves")
    if args.formal:
        failures = []
        if not (10 <= len(args.seeds) <= 20):
            failures.append("formal runs require 10--20 seeds")
        if tuple(args.seeds) != FORMAL_SEEDS:
            failures.append("formal seed list must equal the pre-registered list")
        if args.train_exposures < 128:
            failures.append("formal runs require >=128 primary updates per task")
        if args.probe_exposures < 64:
            failures.append("formal runs require >=64 held-out probes per task")
        if args.eval_games < 8:
            failures.append("formal runs require >=8 unmasked games per seed")
        if args.bootstrap_resamples < 5000:
            failures.append("formal runs require >=5000 paired bootstrap resamples")
        if failures:
            raise ValueError("; ".join(failures))


def input_dim(board_size: int) -> int:
    return 3 * int(board_size) * int(board_size) + 6


def _make_adaptive_core(method: str, board_size: int) -> torch.nn.Module:
    width = input_dim(board_size)
    rank = BOARD_RANKS[int(board_size)][str(method)]
    if method == "mqr_unistochastic_ogd":
        return MultiTimescaleMQR(
            width,
            RING_STATE_DIM,
            LATENT_DIM,
            leak_rates=LEAK_RATES,
            injection_rank=rank,
            transition_mode="unistochastic",
            learn_transitions=True,
            cayley_coordinate_mode="minimal",
            readout_bias=True,
            zero_init_readout=True,
        )
    if method == "mqr_identity_ogd":
        return MultiTimescaleMQR(
            width,
            RING_STATE_DIM,
            LATENT_DIM,
            leak_rates=LEAK_RATES,
            injection_rank=rank,
            transition_mode="identity",
            learn_transitions=False,
            readout_bias=True,
            zero_init_readout=True,
        )
    if method == "gru_ogd":
        return MultiTimescaleGRUCore(
            width,
            RING_STATE_DIM,
            LATENT_DIM,
            leak_rates=LEAK_RATES,
            input_rank=rank,
            readout_bias=True,
            zero_init_readout=True,
        )
    if method == "lstm_ogd":
        return MultiTimescaleLSTMCore(
            width,
            RING_STATE_DIM // 2,
            LATENT_DIM,
            leak_rates=LEAK_RATES,
            input_rank=rank,
            readout_bias=True,
            zero_init_readout=True,
        )
    if method == "fast_weight_ogd":
        return MultiTimescaleFastWeightCore(
            width,
            int(math.isqrt(RING_STATE_DIM)),
            LATENT_DIM,
            leak_rates=LEAK_RATES,
            input_rank=rank,
            readout_bias=True,
            zero_init_readout=True,
        )
    if method in LORA_FAMILY:
        return MultiTimescaleLoRAResidualCore(
            width,
            rank,
            LATENT_DIM,
            num_timescales=len(LEAK_RATES),
            state_dim_per_timescale=RING_STATE_DIM,
        )
    if method == "sinkhorn_ogd":
        return MultiTimescaleSinkhornCore(
            width,
            RING_STATE_DIM,
            LATENT_DIM,
            leak_rates=LEAK_RATES,
            injection_rank=rank,
            sinkhorn_iterations=SINKHORN_ITERATIONS,
            readout_bias=True,
            zero_init_readout=True,
        )
    raise ValueError(f"unknown method: {method}")


def _copy_common_modules(
    source: TemporalUtilityMQRAgent,
    target: TemporalUtilityMQRAgent,
) -> None:
    target.core.frozen_base.load_state_dict(source.core.frozen_base.state_dict())
    for name in (
        "placement_head",
        "legality_head",
        "pass_head",
        "value_head",
        "utility_gate",
        "spatial_skip_heads",
    ):
        source_module = getattr(source, name)
        target_module = getattr(target, name)
        if source_module is not None:
            assert target_module is not None
            target_module.load_state_dict(source_module.state_dict())


def build_agents(
    board_size: int,
    *,
    seed: int,
    task_lr: float = 0.03,
    utility_lr: float = 0.10,
    ogd_rank: int = 4,
    transition_update_interval: int = 4,
) -> Dict[str, TemporalUtilityMQRAgent]:
    """Build all methods with shared base/heads and identical LoRA initials."""

    agents: Dict[str, TemporalUtilityMQRAgent] = {}
    common_source: TemporalUtilityMQRAgent | None = None
    lora_source: TemporalUtilityMQRAgent | None = None
    for method_index, method in enumerate(METHODS):
        torch.manual_seed(int(seed) * 1009 + method_index)
        core = FrozenResidualCoreWrapper(_make_adaptive_core(method, board_size))
        method_ogd_rank = int(ogd_rank) if method in OGD_METHODS else 0
        agent = TemporalUtilityMQRAgent(
            input_dim(board_size),
            board_size=board_size,
            latent_dim=LATENT_DIM,
            core=core,
            utility_rank=4,
            initial_write_advantage=0.001,
            memory_cost=0.0005,
            advantage_scale=0.02,
            task_lr=float(task_lr),
            utility_lr=float(utility_lr),
            ogd_max_rank=method_ogd_rank,
            utility_ogd_max_rank=0,
            max_update_norm=0.10,
            utility_max_update_norm=0.08,
            transition_update_interval=int(transition_update_interval),
            awr_temperature=1.0,
            awr_max_weight=10.0,
            ppo_clip=0.20,
            legality_policy_scale=1.0,
            spatial_skip_channels=SPATIAL_SKIP_CHANNELS,
        )
        if common_source is None:
            common_source = agent
        else:
            _copy_common_modules(common_source, agent)
        if method == "lora":
            lora_source = agent
        elif method in ("ogd_lora", "replay_lora"):
            assert lora_source is not None
            agent.core.adaptive.load_state_dict(lora_source.core.adaptive.state_dict())
        agents[method] = agent
    return agents


def _tensor_module_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def initial_equivalence(
    agents: Mapping[str, TemporalUtilityMQRAgent],
    board_size: int,
) -> Dict[str, Any]:
    width = input_dim(board_size)
    reference_input = torch.linspace(-1.0, 1.0, width).reshape(1, width)
    base_hashes = {
        method: _tensor_module_hash(agent.core.frozen_base)
        for method, agent in agents.items()
    }
    head_hashes = {
        method: hashlib.sha256(
            "".join(
                _tensor_module_hash(getattr(agent, name))
                for name in (
                    "placement_head",
                    "legality_head",
                    "pass_head",
                    "value_head",
                    "utility_gate",
                    "spatial_skip_heads",
                )
            ).encode("ascii")
        ).hexdigest()
        for method, agent in agents.items()
    }
    policies: Dict[str, torch.Tensor] = {}
    residual_max_abs: Dict[str, float] = {}
    for method, agent in agents.items():
        state = agent.core.zero_state(
            1,
            device=reference_input.device,
            dtype=reference_input.dtype,
        )
        adaptive, _ = agent.core.adaptive.forward_step(
            reference_input,
            state=state,
            write_gate=torch.ones(1, len(LEAK_RATES)),
        )
        residual_max_abs[method] = float(adaptive.abs().amax().item())
        policies[method] = agent.preview_step(reference_input)["policy_logits"].detach()
    reference = policies[METHODS[0]]
    policy_max_abs_difference = max(
        float((value - reference).abs().amax().item())
        for value in policies.values()
    )
    lora_hashes = {
        method: _tensor_module_hash(agents[method].core.adaptive)
        for method in LORA_FAMILY
    }
    return {
        "base_hashes": base_hashes,
        "head_hashes": head_hashes,
        "lora_family_hashes": lora_hashes,
        "all_base_hashes_equal": len(set(base_hashes.values())) == 1,
        "all_head_hashes_equal": len(set(head_hashes.values())) == 1,
        "lora_family_hashes_equal": len(set(lora_hashes.values())) == 1,
        "adaptive_residual_max_abs": residual_max_abs,
        "all_adaptive_residuals_zero": max(residual_max_abs.values()) == 0.0,
        "initial_policy_max_abs_difference": policy_max_abs_difference,
        "initial_policies_exact": policy_max_abs_difference == 0.0,
    }


def resource_audits(
    agents: Mapping[str, TemporalUtilityMQRAgent],
    *,
    ogd_rank: int,
) -> Tuple[List[AgentResourceAudit], int, Dict[str, Any]]:
    """Return audits under one common continual-memory allocation."""

    maximum_task_parameters = max(agent.task_parameter_count for agent in agents.values())
    capacity_bytes = int(maximum_task_parameters * int(ogd_rank) * 4)
    audits = [
        audit_agent_resources(
            agents[method],
            method,
            allocated_continual_memory_bytes=capacity_bytes,
        )
        for method in METHODS
    ]
    return (
        audits,
        capacity_bytes,
        matched_competitive_resource_gate(audits, ratio_limit=1.05),
    )


def _trajectory_hash(groups: Iterable[Sequence[GoAgentTrajectory]]) -> str:
    digest = hashlib.sha256()
    for group in groups:
        for trajectory in group:
            digest.update(trajectory.task.encode("utf-8"))
            for example in trajectory.examples:
                digest.update(bytes(value + 1 for value in example.board.board))
                digest.update(int(example.board.to_play).to_bytes(2, "little", signed=True))
                digest.update(int(example.target_action).to_bytes(4, "little"))
                digest.update(float(example.value_target).hex().encode("ascii"))
                digest.update(str(example.history_condition).encode("utf-8"))
    return digest.hexdigest()


def _datasets(
    args: argparse.Namespace,
    seed: int,
) -> Tuple[
    List[GoAgentTrajectory],
    List[GoAgentTrajectory],
    List[GoAgentTrajectory],
    List[GoAgentTrajectory],
]:
    train_a = generate_go_agent_trajectories(
        args.train_exposures // args.recorded_moves,
        size=args.board_size,
        komi=args.komi,
        seed=seed + 10,
        task="opening",
        start_random_moves=0,
        recorded_moves=args.recorded_moves,
        random_move_probability=0.35,
    )
    train_b = generate_ko_history_pairs(
        args.train_exposures // 4,
        size=args.board_size,
        komi=args.komi,
        seed=seed + 20,
        symmetry_pool=(0, 1, 2, 3),
    )
    probe_a = generate_go_agent_trajectories(
        args.probe_exposures // args.recorded_moves,
        size=args.board_size,
        komi=args.komi,
        seed=seed + 30,
        task="opening_probe",
        start_random_moves=0,
        recorded_moves=args.recorded_moves,
        random_move_probability=0.45,
    )
    probe_b = generate_ko_history_pairs(
        args.probe_exposures // 4,
        size=args.board_size,
        komi=args.komi,
        seed=seed + 40,
        symmetry_pool=(4, 5, 6, 7),
    )
    counts = [sum(len(item.examples) for item in group) for group in (train_a, train_b, probe_a, probe_b)]
    expected = [
        args.train_exposures,
        args.train_exposures,
        args.probe_exposures,
        args.probe_exposures,
    ]
    if counts != expected:
        raise RuntimeError(f"dataset exposure mismatch: {counts} != {expected}")
    return train_a, train_b, probe_a, probe_b


def _auxiliary_feedback(
    agent: TemporalUtilityMQRAgent,
    item: ReplayTensorExample,
    *,
    stream: str,
    project_with_memory: bool,
    external_write: bool,
) -> Dict[str, Any]:
    reference = next(agent.parameters())
    features = item.features.to(reference)
    committed = agent.commit_step(
        features,
        stream_id=stream,
        external_write=bool(external_write),
    )
    ticket = int(committed["ticket_id"])
    value = item.value.to(reference)
    feedback = agent.apply_feedback(
        ticket,
        item.action.to(device=reference.device),
        legality_target=item.legality.to(device=reference.device, dtype=torch.float32),
        value_target=value,
        awr_advantage=value - committed["output"].value.to(value),
        weights=OFFLINE_WEIGHTS,
        remember_gradient=False,
        project_with_memory=bool(project_with_memory),
    )
    agent.cancel_ticket_branch(ticket, "utility")
    agent.reset_state(stream)
    return feedback


def train_online_phase(
    agent: TemporalUtilityMQRAgent,
    encoder: GoVectorEncoder,
    trajectories: Sequence[GoAgentTrajectory],
    *,
    method: str,
    phase: str,
    twin_horizon: int,
    replay: ByteCappedReplay | None,
    write_schedule: Sequence[bool],
) -> Dict[str, Any]:
    """Prequential task learning with delayed twin-return write supervision."""

    if phase not in ("task_a", "task_b"):
        raise ValueError("phase must be task_a or task_b")
    primary_count = sum(len(item.examples) for item in trajectories)
    if len(write_schedule) != primary_count:
        raise ValueError("write_schedule length must equal the primary exposure count")
    remember_stride = max(
        1,
        math.ceil(primary_count / max(1, agent.task_gradient_memory.max_rank)),
    )
    protect = phase == "task_b" and agent.task_gradient_memory.max_rank > 0
    losses: List[float] = []
    auxiliary_losses: List[float] = []
    write_advantages: List[float] = []
    critic_errors: List[float] = []
    write_count = 0
    task_updates = 0
    utility_updates = 0
    remembered = 0
    auxiliary_updates = 0
    branch_loss_evaluations = 0
    started = time.perf_counter()
    reference = next(agent.parameters())
    update_index = 0
    for trajectory_index, trajectory in enumerate(trajectories):
        stream = f"{method}-{phase}-{trajectory_index}"
        agent.reset_state(stream)
        for example_index, example in enumerate(trajectory.examples):
            features = encoder.encode_board(example.board).to(reference)
            forced_write = bool(write_schedule[update_index])
            committed = agent.commit_step(
                features,
                stream_id=stream,
                external_write=forced_write,
            )
            ticket = int(committed["ticket_id"])
            future = trajectory.examples[example_index + 1 :]
            evaluated_horizon = min(len(future), int(twin_horizon))
            no_return, write_return = twin_write_returns(
                agent,
                ticket,
                future,
                encoder,
                horizon=twin_horizon,
                weights=EVALUATION_WEIGHTS,
            )
            branch_loss_evaluations += 2 * evaluated_horizon
            utility = agent.calibrate_write_critic(
                ticket,
                no_write_return=no_return,
                write_return=write_return,
                learn=True,
                remember_gradient=False,
                project_with_memory=False,
            )
            utility_updates += int(utility["did_update"])
            write_advantages.append(float(utility["write_advantage"]))
            critic_errors.append(
                abs(
                    float(utility["predicted_advantage_before_update"])
                    - float(utility["write_advantage"])
                )
            )
            write_count += int(committed["effective_write"])
            action, legality, value = example_targets(
                example,
                device=reference.device,
            )
            remember = bool(
                phase == "task_a"
                and agent.task_gradient_memory.max_rank > 0
                and agent.task_gradient_memory.rank
                < agent.task_gradient_memory.max_rank
                and update_index % remember_stride == 0
            )
            feedback = agent.apply_feedback(
                ticket,
                action,
                legality_target=legality,
                value_target=value,
                awr_advantage=value - committed["output"].value.to(value),
                weights=OFFLINE_WEIGHTS,
                remember_gradient=remember,
                project_with_memory=protect,
            )
            losses.append(float(feedback["loss"]))
            task_updates += int(feedback["did_update"])
            remembered += int(feedback["ogd_memory_added"])

            current_item = ReplayTensorExample(
                features.detach().cpu().float().clone(),
                action.detach().cpu().clone(),
                legality.detach().cpu().bool().clone(),
                value.detach().cpu().float().clone(),
            )
            if phase == "task_a" and replay is not None:
                replay.append(features, action, legality, value)
            if phase == "task_b":
                if method == "replay_lora":
                    if replay is None:
                        raise RuntimeError("replay_lora has no replay allocation")
                    auxiliary_item = replay.cyclic(update_index)
                else:
                    auxiliary_item = current_item
                auxiliary = _auxiliary_feedback(
                    agent,
                    auxiliary_item,
                    stream=f"aux-{method}-{phase}-{update_index}",
                    project_with_memory=protect,
                    external_write=forced_write,
                )
                auxiliary_losses.append(float(auxiliary["loss"]))
                auxiliary_updates += int(auxiliary["did_update"])
            update_index += 1
        agent.reset_state(stream)
    elapsed = time.perf_counter() - started
    return {
        "phase": phase,
        "primary_feedback_examples": int(primary_count),
        "unique_external_feedback_examples": int(primary_count),
        "auxiliary_gradient_examples": int(primary_count if phase == "task_b" else 0),
        "total_gradient_examples": int(
            primary_count + (primary_count if phase == "task_b" else 0)
        ),
        "counterfactual_branch_loss_evaluations": int(branch_loss_evaluations),
        "task_updates": int(task_updates),
        "utility_updates": int(utility_updates),
        "auxiliary_updates": int(auxiliary_updates),
        "mean_primary_loss": statistics.fmean(losses),
        "mean_auxiliary_loss": (
            statistics.fmean(auxiliary_losses) if auxiliary_losses else 0.0
        ),
        "mean_write_advantage": statistics.fmean(write_advantages),
        "critic_mae": statistics.fmean(critic_errors),
        "write_rate": write_count / max(1, primary_count),
        "training_route_source": "shared_content_independent_bernoulli",
        "write_schedule_sha256": hashlib.sha256(
            bytes(int(value) for value in write_schedule)
        ).hexdigest(),
        "ogd_rank": int(agent.task_gradient_memory.rank),
        "remembered_directions": int(remembered),
        "seconds": elapsed,
        "primary_updates_per_second": primary_count / max(elapsed, 1e-12),
        "predict_before_update": True,
    }


def _method_occupancy(
    agent: TemporalUtilityMQRAgent,
    replay: ByteCappedReplay | None,
    capacity_bytes: int,
) -> Dict[str, Any]:
    actual_ogd = int(agent.task_gradient_memory.storage_bytes)
    actual_replay = 0 if replay is None else int(replay.storage_bytes)
    actual = actual_ogd + actual_replay
    return {
        "allocated_continual_memory_bytes": int(capacity_bytes),
        "actual_ogd_bytes": actual_ogd,
        "actual_replay_tensor_bytes": actual_replay,
        "actual_total_continual_bytes": actual,
        "capacity_respected": actual <= capacity_bytes,
        "replay_items": 0 if replay is None else len(replay.items),
        "replay_evictions": 0 if replay is None else replay.evictions,
        "unused_capacity_bytes": int(capacity_bytes - actual),
    }


def _run_seed(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    train_a, train_b, probe_a, probe_b = _datasets(args, seed)
    encoder = GoVectorEncoder(
        args.board_size,
        input_dim(args.board_size),
        seed=seed + 4000,
        projection_mode="identity",
    )
    agents = build_agents(
        args.board_size,
        seed=seed,
        task_lr=args.task_lr,
        utility_lr=args.utility_lr,
        ogd_rank=args.ogd_rank,
        transition_update_interval=args.transition_update_interval,
    )
    _audits, capacity_bytes, _gate = resource_audits(
        agents,
        ogd_rank=args.ogd_rank,
    )
    equivalence = initial_equivalence(agents, args.board_size)
    route_a_rng = random.Random(seed + 50001)
    route_b_rng = random.Random(seed + 50002)
    route_a = tuple(
        route_a_rng.random() < args.training_write_probability
        for _ in range(args.train_exposures)
    )
    route_b = tuple(
        route_b_rng.random() < args.training_write_probability
        for _ in range(args.train_exposures)
    )
    methods: Dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        agent = agents[method]
        replay = ByteCappedReplay(capacity_bytes) if method == "replay_lora" else None
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
            seed=seed * 10000 + method_index,
        )
        phase_a = train_online_phase(
            agent,
            encoder,
            train_a,
            method=method,
            phase="task_a",
            twin_horizon=args.twin_horizon,
            replay=replay,
            write_schedule=route_a,
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
            seed=seed * 10000 + 1000 + method_index,
        )
        phase_b = train_online_phase(
            agent,
            encoder,
            train_b,
            method=method,
            phase="task_b",
            twin_horizon=args.twin_horizon,
            replay=replay,
            write_schedule=route_b,
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
            seed=seed * 10000 + 2000 + method_index,
        )
        agent.reset_state(None)
        methods[method] = {
            "online_training": {"task_a": phase_a, "task_b": phase_b},
            "probes": {
                "initial": {"task_a": initial_a, "task_b": initial_b},
                "after_task_a": {"task_a": after_a_a, "task_b": after_a_b},
                "after_task_b": {"task_a": after_b_a, "task_b": after_b_b},
            },
            "game_returns": {
                "initial": initial_return,
                "after_task_a": after_a_return,
                "after_task_b": after_b_return,
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
            },
            "online_learning": {
                "task_a_loss_reduction": (
                    initial_a["total_loss"] - after_a_a["total_loss"]
                ),
                "task_b_loss_reduction_from_initial": (
                    initial_b["total_loss"] - after_b_b["total_loss"]
                ),
                "dynamic_return_change": (
                    after_b_return["mean_return"] - initial_return["mean_return"]
                ),
            },
            "memory_occupancy": _method_occupancy(agent, replay, capacity_bytes),
            "invariants": {
                "pending_tickets": agent.pending_ticket_count,
                "max_unitary_error": agent.core.max_unitary_error(),
                "max_stochastic_error": agent.core.max_stochastic_error(),
                "recurrent_state_bytes_per_stream": 12 * 4,
                "online_parameter_version": int(agent.online_parameter_version.item()),
            },
        }
    return {
        "seed": int(seed),
        "trajectory_hash": _trajectory_hash((train_a, train_b, probe_a, probe_b)),
        "initial_equivalence": equivalence,
        "methods": methods,
    }


def _metric_values(
    seed_results: Sequence[Mapping[str, Any]],
    method: str,
    path: Sequence[str],
) -> List[float]:
    values: List[float] = []
    for seed in seed_results:
        current: Any = seed["methods"][method]
        for name in path:
            current = current[name]
        values.append(float(current))
    return values


def summarize_results(
    seed_results: Sequence[Mapping[str, Any]],
    resource_gate: Mapping[str, Any],
    protocol_gate: Mapping[str, Any],
    *,
    bootstrap_resamples: int,
) -> Dict[str, Any]:
    metric_specs = (
        (
            "useful_legal_placement_rate",
            ("game_returns", "after_task_b", "useful_legal_placement_rate"),
            False,
        ),
        (
            "game_return",
            ("game_returns", "after_task_b", "mean_return"),
            False,
        ),
        (
            "history_focus_accuracy",
            ("probes", "after_task_b", "task_b", "focus_legality_accuracy"),
            False,
        ),
        (
            "forgetting_loss",
            ("forgetting", "task_a_positive_forgetting"),
            True,
        ),
    )
    comparisons: Dict[str, Any] = {}
    all_decisive = True
    for comparator_index, comparator in enumerate(METHODS[1:]):
        metrics: Dict[str, Any] = {}
        for metric_index, (metric, path, reverse) in enumerate(metric_specs):
            mqr = _metric_values(seed_results, PRIMARY_METHOD, path)
            control = _metric_values(seed_results, comparator, path)
            left, right = (control, mqr) if reverse else (mqr, control)
            interval = _paired_bootstrap(
                left,
                right,
                resamples=bootstrap_resamples,
                seed=12000 + 101 * comparator_index + metric_index,
            )
            interval["decisive"] = bool(interval["ci95_low"] > 0.0)
            interval["positive_means_mqr_better"] = True
            metrics[metric] = interval
            all_decisive = all_decisive and bool(interval["decisive"])
        comparisons[comparator] = metrics

    learning_values = _metric_values(
        seed_results,
        PRIMARY_METHOD,
        ("online_learning", "task_a_loss_reduction"),
    )
    learning_interval = _paired_bootstrap(
        learning_values,
        [0.0 for _ in learning_values],
        resamples=bootstrap_resamples,
        seed=15001,
    )
    online_learning_supported = bool(learning_interval["ci95_low"] > 0.0)
    complete_formal = bool(protocol_gate["formal_protocol_pass"])
    resources_pass = bool(resource_gate["all_resources_matched"])
    effective = bool(
        complete_formal
        and resources_pass
        and online_learning_supported
        and all_decisive
    )
    reasons: List[str] = []
    if not complete_formal:
        reasons.append("formal seed/scale/exposure/evaluation protocol failed")
    if not resources_pass:
        reasons.append("strict parameter/state/frozen/capacity/MAC gate failed")
    if not online_learning_supported:
        reasons.append("MQR task-A online loss reduction CI does not clear zero")
    if not all_decisive:
        reasons.append("MQR does not decisively beat every pre-registered control")
    return {
        "metric_sources": {
            name: ".".join(path) for name, path, _reverse in metric_specs
        },
        "comparisons": comparisons,
        "all_comparator_metric_cis_above_zero": all_decisive,
        "online_learning": {
            "task_a_loss_reduction": learning_interval,
            "supported": online_learning_supported,
        },
        "mqr_effective_on_this_scale": effective,
        "mqr_effective": False,
        "cross_scale_replication_required": True,
        "rejection_reasons": reasons + [
            "13x13 replication must independently pass before global effectiveness"
        ],
    }


def _protocol_gate(
    args: argparse.Namespace,
    seed_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    expected_a = args.train_exposures
    expected_b_total = 2 * args.train_exposures
    count_checks = []
    equivalence_checks = []
    capacity_checks = []
    routing_checks = []
    for seed in seed_results:
        eq = seed["initial_equivalence"]
        equivalence_checks.append(
            bool(
                eq["all_base_hashes_equal"]
                and eq["all_head_hashes_equal"]
                and eq["lora_family_hashes_equal"]
                and eq["all_adaptive_residuals_zero"]
                and eq["initial_policies_exact"]
            )
        )
        task_a_routes = {
            seed["methods"][method]["online_training"]["task_a"][
                "write_schedule_sha256"
            ]
            for method in METHODS
        }
        task_b_routes = {
            seed["methods"][method]["online_training"]["task_b"][
                "write_schedule_sha256"
            ]
            for method in METHODS
        }
        routing_checks.append(len(task_a_routes) == 1 and len(task_b_routes) == 1)
        for method in METHODS:
            phases = seed["methods"][method]["online_training"]
            count_checks.append(
                phases["task_a"]["total_gradient_examples"] == expected_a
                and phases["task_b"]["total_gradient_examples"] == expected_b_total
                and phases["task_a"]["unique_external_feedback_examples"]
                == expected_a
                and phases["task_b"]["unique_external_feedback_examples"]
                == expected_a
            )
            capacity_checks.append(
                bool(seed["methods"][method]["memory_occupancy"]["capacity_respected"])
            )
    formal_conditions = {
        "board_size_supported": args.board_size in (10, 13),
        "seed_count_10_to_20": 10 <= len(seed_results) <= 20,
        "pre_registered_seed_list": tuple(args.seeds) == FORMAL_SEEDS,
        "train_exposures_at_least_128": args.train_exposures >= 128,
        "probe_exposures_at_least_64": args.probe_exposures >= 64,
        "eval_games_at_least_8": args.eval_games >= 8,
        "bootstrap_at_least_5000": args.bootstrap_resamples >= 5000,
        "all_requested_seeds_complete": len(seed_results) == len(args.seeds),
        "feedback_and_gradient_budgets_equal": all(count_checks),
        "training_write_routes_exact": all(routing_checks),
        "shared_initialization_exact": all(equivalence_checks),
        "continual_memory_cap_respected": all(capacity_checks),
    }
    return {
        **formal_conditions,
        "formal_flag_requested": bool(args.formal),
        "formal_protocol_pass": bool(args.formal and all(formal_conditions.values())),
    }


def _scientific_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "board_size": int(args.board_size),
        "seeds": [int(value) for value in args.seeds],
        "train_exposures": int(args.train_exposures),
        "probe_exposures": int(args.probe_exposures),
        "recorded_moves": int(args.recorded_moves),
        "eval_games": int(args.eval_games),
        "twin_horizon": int(args.twin_horizon),
        "komi": float(args.komi),
        "task_lr": float(args.task_lr),
        "utility_lr": float(args.utility_lr),
        "training_write_probability": float(args.training_write_probability),
        "ogd_rank": int(args.ogd_rank),
        "transition_update_interval": int(args.transition_update_interval),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "formal": bool(args.formal),
        "input_dim": input_dim(args.board_size),
        "encoder": "lossless_identity_3planes_plus_6_causal_scalars",
        "latent_dim": LATENT_DIM,
        "leak_rates": list(LEAK_RATES),
        "rank_config": BOARD_RANKS[args.board_size],
        "sinkhorn_iterations": SINKHORN_ITERATIONS,
        "spatial_skip_channels": SPATIAL_SKIP_CHANNELS,
    }


def _config_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _initial_payload(
    args: argparse.Namespace,
    audits: Sequence[AgentResourceAudit],
    capacity_bytes: int,
    resource_gate: Mapping[str, Any],
) -> Dict[str, Any]:
    config = _scientific_config(args)
    return {
        "schema_version": 2,
        "experiment": "unified_temporal_mqr_go_competitive",
        "complete": False,
        "claim_scope": (
            "frozen-small-model sidecar mechanism test; not evidence of general Go "
            "strength, general capability enhancement, or animal-like learning"
        ),
        "config": config,
        "config_sha256": _config_hash(config),
        "methods": list(METHODS),
        "protocol": {
            "prediction_before_update": True,
            "legality_mask_during_action": False,
            "task_order": "A opening -> B hidden-history ko/fresh",
            "write_target": "paired future negative four-head loss",
            "primary_loss": "same advantage-weighted four-head imitation loss",
            "task_b_auxiliary_rule": (
                "replay-LoRA uses a cyclic task-A tensor item; every other method "
                "repeats the current task-B item"
            ),
            "feedback_budget_rule": (
                "same unique teacher labels, primary updates, auxiliary gradient "
                "examples, and twin-branch evaluations"
            ),
            "core_controls": "same OGD rank; LoRA/no-OGD and replay are named ablations",
            "training_write_route": (
                "same seeded content-independent Bernoulli schedule for every method; "
                "learned gate is evaluated without override"
            ),
            "policy_rl": "disabled to isolate online sidecar learning",
        },
        "resources": {
            "audits": [item.to_dict() for item in audits],
            "strict_gate": dict(resource_gate),
            "recurrent_state_contract": "3 timescales x 4 float32 scalars = 48 bytes",
            "allocated_continual_memory_bytes": int(capacity_bytes),
            "actual_occupancy_policy": (
                "no fake padding; actual OGD/replay use is reported per method"
            ),
            "mac_semantics": (
                "analytical charged MACs, including core normalization/solve, frozen "
                "base, four heads, utility critic, backward factor, and common memory cap"
            ),
        },
        "seed_results": [],
        "protocol_gate": None,
        "effectiveness_gate": {
            "mqr_effective": False,
            "rejection_reasons": ["experiment incomplete"],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads_per_worker": int(args.torch_threads),
            "workers": int(args.workers),
        },
        "wall_seconds": 0.0,
    }


def _load_resume(args: argparse.Namespace, expected: Mapping[str, Any]) -> Dict[str, Any]:
    if not args.output.exists():
        return dict(expected)
    if not args.resume:
        raise FileExistsError(
            f"{args.output} already exists; pass --resume or choose a new output"
        )
    loaded = json.loads(args.output.read_text(encoding="utf-8"))
    if loaded.get("schema_version") != 2:
        raise ValueError("resume file has the wrong schema")
    if loaded.get("config_sha256") != expected.get("config_sha256"):
        raise ValueError("resume file scientific configuration does not match")
    return loaded


def main() -> int:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(int(args.torch_threads))
    calibration_agents = build_agents(
        args.board_size,
        seed=0,
        task_lr=args.task_lr,
        utility_lr=args.utility_lr,
        ogd_rank=args.ogd_rank,
        transition_update_interval=args.transition_update_interval,
    )
    audits, capacity_bytes, resource_gate = resource_audits(
        calibration_agents,
        ogd_rank=args.ogd_rank,
    )
    expected = _initial_payload(args, audits, capacity_bytes, resource_gate)
    payload = _load_resume(args, expected)
    existing: MutableMapping[int, Dict[str, Any]] = {
        int(item["seed"]): item for item in payload.get("seed_results", [])
    }
    pending = [int(seed) for seed in args.seeds if int(seed) not in existing]
    started = time.perf_counter()
    if pending and args.workers == 1:
        for index, seed in enumerate(pending, 1):
            print(f"seed={seed} ({index}/{len(pending)})", flush=True)
            existing[seed] = _run_seed(args, seed)
            payload["seed_results"] = [
                existing[value] for value in args.seeds if value in existing
            ]
            payload["wall_seconds"] = float(payload.get("wall_seconds", 0.0)) + (
                time.perf_counter() - started
            )
            _atomic_write(args.output, payload)
            started = time.perf_counter()
    elif pending:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(args.workers),
            mp_context=context,
        ) as executor:
            futures = {executor.submit(_run_seed, args, seed): seed for seed in pending}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                seed = futures[future]
                existing[seed] = future.result()
                completed += 1
                print(f"completed seed={seed} ({completed}/{len(pending)})", flush=True)
                payload["seed_results"] = [
                    existing[value] for value in args.seeds if value in existing
                ]
                payload["wall_seconds"] = float(payload.get("wall_seconds", 0.0)) + (
                    time.perf_counter() - started
                )
                _atomic_write(args.output, payload)
                started = time.perf_counter()

    seed_results = [existing[value] for value in args.seeds if value in existing]
    protocol_gate = _protocol_gate(args, seed_results)
    effectiveness = summarize_results(
        seed_results,
        resource_gate,
        protocol_gate,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    payload["complete"] = len(seed_results) == len(args.seeds)
    payload["seed_results"] = seed_results
    payload["protocol_gate"] = protocol_gate
    payload["effectiveness_gate"] = effectiveness
    payload["wall_seconds"] = float(payload.get("wall_seconds", 0.0)) + (
        time.perf_counter() - started
    )
    _atomic_write(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": payload["complete"],
                "formal_protocol_pass": protocol_gate["formal_protocol_pass"],
                "resource_gate_pass": resource_gate["all_resources_matched"],
                "mqr_effective_on_this_scale": effectiveness[
                    "mqr_effective_on_this_scale"
                ],
                "mqr_effective": effectiveness["mqr_effective"],
                "rejection_reasons": effectiveness["rejection_reasons"],
                "wall_seconds": payload["wall_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
