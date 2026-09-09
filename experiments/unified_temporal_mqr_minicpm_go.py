#!/usr/bin/env python3
"""Local MiniCPM LoRA consolidation smoke test for the unified MQR agent.

This experiment verifies the slowest stage of the architecture, after the
vector-encoder core comparison.  Fast MQR/head updates happen on every labeled
position; the frozen AWQ backbone receives an external feature gradient only
on an auditable slow schedule, and only LoRA A/B tensors may change.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mqr.agent import (  # noqa: E402
    GoLossWeights,
    SlowLoRAConsolidator,
    SlowLoRAConsolidationSchedule,
    TemporalUtilityMQRAgent,
)
from mqr.go_agent import (  # noqa: E402
    GoAgentExample,
    GoAgentTrajectory,
    MiniCPMGoBoardEncoder,
    example_targets,
    generate_go_agent_trajectories,
    twin_write_returns,
)
from mqr.minicpm import load_minicpm_awq_encoder  # noqa: E402


MODEL_PATH = Path("/home/spikebai/checkpoints/MiniCPM5-1B-AWQ-INT4")
SUPERVISED_WEIGHTS = GoLossWeights(
    placement=1.0,
    legality=0.5,
    pass_decision=0.5,
    value=0.25,
    awr=0.5,
    illegal_mass=0.75,
)
EVAL_WEIGHTS = GoLossWeights(
    placement=1.0,
    legality=0.5,
    pass_decision=0.5,
    value=0.25,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--board-size", type=int, default=3)
    parser.add_argument("--komi", type=float, default=2.5)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--probe-examples", type=int, default=8)
    parser.add_argument("--twin-horizon", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--lora-warmup", type=int, default=2)
    parser.add_argument("--lora-interval", type=int, default=3)
    parser.add_argument("--lora-ogd-rank", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/results/unified_temporal_mqr_minicpm_go_smoke.json"),
    )
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _flatten_with_future(
    trajectories: Sequence[GoAgentTrajectory],
    count: int,
) -> List[Tuple[GoAgentExample, Sequence[GoAgentExample]]]:
    result: List[Tuple[GoAgentExample, Sequence[GoAgentExample]]] = []
    for trajectory in trajectories:
        for index, example in enumerate(trajectory.examples):
            result.append((example, trajectory.examples[index + 1 :]))
            if len(result) >= count:
                return result
    if len(result) < count:
        raise RuntimeError("generated trajectory stream is shorter than --updates")
    return result


@torch.no_grad()
def evaluate(
    agent: TemporalUtilityMQRAgent,
    board_encoder: MiniCPMGoBoardEncoder,
    examples: Sequence[GoAgentExample],
    *,
    stream: str,
) -> Dict[str, float]:
    losses: List[float] = []
    legal = 0
    teacher = 0
    reference = next(agent.parameters())
    agent.reset_state(stream)
    for example in examples:
        features = board_encoder.encode_board(example.board).to(reference)
        output = agent.commit_step(
            features,
            stream_id=stream,
            issue_ticket=False,
        )["output"]
        action, legal_target, value = example_targets(example, device=reference.device)
        loss = agent.compute_go_loss(
            output,
            action,
            legality_target=legal_target,
            value_target=value,
            weights=EVAL_WEIGHTS,
        )["total"]
        prediction = int(torch.argmax(output.policy_logits[0]).item())
        losses.append(float(loss.item()))
        legal += int(example.board.is_legal(prediction))
        teacher += int(prediction == example.target_action)
    agent.reset_state(stream)
    return {
        "count": float(len(examples)),
        "loss": statistics.fmean(losses),
        "unmasked_legal_rate": legal / len(examples),
        "teacher_agreement": teacher / len(examples),
    }


def _adapter_snapshot(encoder: Any) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in encoder.named_lora_parameters()
    }


def _frozen_probe_snapshot(encoder: Any) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().reshape(-1)[:16].cpu().clone()
        for name, parameter in encoder.named_parameters()
        if not (name.endswith("lora_A") or name.endswith("lora_B"))
    }


def _adapter_drift(before: Dict[str, torch.Tensor], encoder: Any) -> Dict[str, float]:
    values = {
        name: float((parameter.detach().cpu() - before[name]).abs().max().item())
        for name, parameter in encoder.named_lora_parameters()
    }
    return {
        "max_abs": max(values.values(), default=0.0),
        "changed_tensors": float(sum(value > 0.0 for value in values.values())),
        "tensor_count": float(len(values)),
    }


def main() -> int:
    args = parse_args()
    if min(args.updates, args.probe_examples, args.twin_horizon) <= 0:
        raise ValueError("updates, probes, and twin horizon must be positive")
    torch.manual_seed(args.seed)
    encoder = load_minicpm_awq_encoder(
        args.model_path,
        device=args.device,
        dtype=_dtype(args.dtype),
        layer_index=-1,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
    )
    board_encoder = MiniCPMGoBoardEncoder(
        encoder,
        max_length=args.max_length,
        prompt_mode="rules",
    )
    agent = TemporalUtilityMQRAgent(
        encoder.hidden_size,
        board_size=args.board_size,
        ring_dim=8,
        latent_dim=24,
        leak_rates=(1.0, 0.2, 0.05),
        injection_rank=8,
        utility_rank=4,
        initial_write_advantage=-0.001,
        memory_cost=0.0005,
        advantage_scale=0.02,
        task_lr=0.01,
        utility_lr=0.05,
        ogd_max_rank=2,
        utility_ogd_max_rank=2,
        max_update_norm=0.05,
        utility_max_update_norm=0.02,
        legality_policy_scale=1.0,
    ).to(encoder.device)
    schedule = SlowLoRAConsolidationSchedule(
        warmup_updates=args.lora_warmup,
        interval=args.lora_interval,
        minimum_abs_advantage=0.0,
    )
    consolidator = SlowLoRAConsolidator(
        schedule,
        lr=args.lora_lr,
        ogd_max_rank=args.lora_ogd_rank,
        max_grad_norm=0.5,
    ).to(encoder.device)

    episode_count = max(2, math.ceil((args.updates + args.probe_examples) / 5))
    trajectories = generate_go_agent_trajectories(
        episode_count,
        size=args.board_size,
        komi=args.komi,
        seed=args.seed + 1,
        task="minicpm_lora_smoke",
        recorded_moves=5,
        random_move_probability=0.45,
    )
    stream = _flatten_with_future(trajectories, args.updates + args.probe_examples)
    train = stream[: args.updates]
    probe = [item[0] for item in stream[args.updates :]]
    before_metrics = evaluate(agent, board_encoder, probe, stream="probe-before")
    adapter_before = _adapter_snapshot(encoder)
    frozen_before = _frozen_probe_snapshot(encoder)
    update_records: List[Dict[str, Any]] = []
    reference = next(agent.parameters())
    agent.reset_state("train")
    for update_index, (example, future) in enumerate(train):
        needs_graph = schedule.should_consolidate(update_index, write_advantage=None)
        features = board_encoder.encode_board(
            example.board,
            require_encoder_grad=needs_graph,
        ).to(reference)
        committed = agent.commit_step(features.detach(), stream_id="train")
        ticket = int(committed["ticket_id"])
        no_return, write_return = twin_write_returns(
            agent,
            ticket,
            future,
            board_encoder,
            horizon=args.twin_horizon,
            weights=EVAL_WEIGHTS,
        )
        utility = agent.calibrate_write_critic(
            ticket,
            no_write_return=no_return,
            write_return=write_return,
        )
        action, legal_target, value = example_targets(example, device=reference.device)
        awr_advantage = value - committed["output"].value.to(value)
        feedback = agent.apply_feedback(
            ticket,
            action,
            legality_target=legal_target,
            value_target=value,
            awr_advantage=awr_advantage,
            weights=SUPERVISED_WEIGHTS,
            return_grad_features=needs_graph,
        )
        lora = consolidator.maybe_step(
            encoder,
            features,
            (
                feedback["grad_features"].to(features)
                if needs_graph
                else torch.zeros_like(features)
            ),
            update_index=update_index,
            write_advantage=utility["write_advantage"],
            remember_gradient=bool(
                needs_graph and consolidator.gradient_memory.rank == 0
            ),
        )
        update_records.append(
            {
                "index": update_index,
                "task_loss": feedback["loss"],
                "write_advantage": utility["write_advantage"],
                "lora": lora,
            }
        )
    agent.reset_state("train")
    after_metrics = evaluate(agent, board_encoder, probe, stream="probe-after")
    drift = _adapter_drift(adapter_before, encoder)
    frozen_after = _frozen_probe_snapshot(encoder)
    frozen_probe_drift = max(
        float((frozen_after[name] - value).abs().max().item())
        for name, value in frozen_before.items()
    )
    scheduled = sum(bool(item["lora"]["scheduled"]) for item in update_records)
    did_update = sum(bool(item["lora"]["did_update"]) for item in update_records)
    if scheduled == 0:
        raise RuntimeError("LoRA schedule produced no update opportunity")
    if did_update == 0 or drift["max_abs"] <= 0.0:
        raise RuntimeError("scheduled LoRA consolidation did not change the adapter")
    if frozen_probe_drift != 0.0:
        raise RuntimeError("a frozen MiniCPM parameter changed during LoRA consolidation")
    if agent.pending_ticket_count != 0:
        raise RuntimeError("ticket leak after MiniCPM consolidation run")

    payload = {
        "schema_version": 1,
        "experiment": "unified_temporal_mqr_minicpm_go",
        "claim_scope": "external-gradient and schedule smoke test only",
        "config": {
            **vars(args),
            "model_path": str(args.model_path.resolve()),
            "output": str(args.output),
            "loss_weights": asdict(SUPERVISED_WEIGHTS),
        },
        "metrics": {"before": before_metrics, "after": after_metrics},
        "lora": {
            "trainable_parameters": encoder.trainable_parameter_count,
            "schedule": asdict(schedule),
            "opportunities": int(consolidator.opportunities.item()),
            "scheduled": scheduled,
            "updates": did_update,
            "ogd_rank": consolidator.gradient_memory.rank,
            "adapter_drift": drift,
            "frozen_parameter_probe_max_abs_drift": frozen_probe_drift,
        },
        "updates": update_records,
        "invariants": {
            "pending_tickets": agent.pending_ticket_count,
            "max_unitary_error": agent.core.max_unitary_error(),
            "max_stochastic_error": agent.core.max_stochastic_error(),
            "prediction_before_update": True,
            "backbone_frozen": frozen_probe_drift == 0.0,
        },
        "mqr_effective": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "scheduled": scheduled,
        "lora_updates": did_update,
        "adapter_max_abs_drift": drift["max_abs"],
        "before": before_metrics,
        "after": after_metrics,
        "mqr_effective": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
