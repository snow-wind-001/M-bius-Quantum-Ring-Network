#!/usr/bin/env python3
"""MiniCPM5 + LoRA + multi-ring MQR prequential learning on 5x5 Go.

This is a mechanism experiment.  Agreement with either the deterministic
heuristic or the optional Sayuri policy/MCTS teacher measures online
adaptation to that teacher and must not be interpreted as independent Go
strength.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mqr import OnlineMultiRingClassifier, OrthogonalGradientMemory  # noqa: E402
from mqr.go import (  # noqa: E402
    GoBoard,
    GoTrainingExample,
    HeuristicGoTeacher,
    action_to_gtp,
    color_name,
    format_go_prompt,
    generate_basic_go_dataset,
)
from mqr.minicpm import load_minicpm_awq_encoder  # noqa: E402
from mqr.sayuri import (  # noqa: E402
    SayuriGTPClient,
    SayuriTeacher,
    discover_sayuri_paths,
)


LOGGER = logging.getLogger("minicpm_go_online")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/home/spikebai/checkpoints/MiniCPM5-1B-AWQ-INT4",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--komi", type=float, default=5.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-positions", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument(
        "--stream-mode",
        choices=("game", "dataset"),
        default="game",
        help="Play one evolving game or cycle through a fixed basic-position dataset.",
    )
    parser.add_argument("--dataset-size", type=int, default=8)
    parser.add_argument(
        "--teacher",
        choices=("heuristic", "sayuri-policy", "sayuri-mcts"),
        default="sayuri-policy",
    )
    parser.add_argument("--sayuri-binary", type=Path)
    parser.add_argument("--sayuri-weights", type=Path)
    parser.add_argument("--sayuri-playouts", type=int, default=16)
    parser.add_argument("--ring-hidden", type=int, default=64)
    parser.add_argument("--ring-rank", type=int, default=8)
    parser.add_argument("--relaxation-steps", type=int, default=24)
    parser.add_argument("--adjoint-steps", type=int, default=32)
    parser.add_argument("--mqr-lr", type=float, default=0.01)
    parser.add_argument("--max-mqr-update-norm", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-lr", type=float, default=3e-4)
    parser.add_argument("--max-lora-grad-norm", type=float, default=1.0)
    parser.add_argument("--ogd-rank", type=int, default=8)
    parser.add_argument("--remember-every", type=int, default=6)
    parser.add_argument(
        "--rollout-policy",
        choices=("model", "teacher"),
        default="model",
        help="Apply the model's legal-masked move or teacher move after feedback.",
    )
    parser.add_argument("--no-lora", action="store_true", help="Train MQR but freeze LoRA.")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=ROOT / "analysis/results/minicpm_go_online.json",
    )
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _masked_action(logits: torch.Tensor, board: GoBoard) -> int:
    legal = board.legal_moves()
    return max(legal, key=lambda action: (float(logits[action].item()), -action))


@torch.no_grad()
def evaluate_examples(
    encoder,
    learner: OnlineMultiRingClassifier,
    examples: Sequence[GoTrainingExample],
    *,
    max_length: int,
    batch_size: int,
) -> Dict[str, float]:
    if not examples:
        return {
            "count": 0,
            "loss": 0.0,
            "raw_teacher_agreement": 0.0,
            "masked_teacher_agreement": 0.0,
            "raw_legal_rate": 0.0,
        }
    losses: List[float] = []
    raw_agreements = 0
    masked_agreements = 0
    raw_legal = 0
    for offset in range(0, len(examples), batch_size):
        current = examples[offset : offset + batch_size]
        features = encoder.encode_prompts(
            [format_go_prompt(example.board) for example in current],
            max_length=max_length,
            require_lora_grad=False,
        )
        features = F.layer_norm(features, (features.size(-1),))
        for row, example in enumerate(current):
            context = color_name(example.board.to_play)
            logits = learner(features[row : row + 1], context_id=context)
            target = torch.tensor([example.target_action], device=logits.device)
            losses.append(float(F.cross_entropy(logits, target).item()))
            prediction = int(torch.argmax(logits[0]).item())
            raw_agreements += int(prediction == example.target_action)
            masked_agreements += int(
                _masked_action(logits[0], example.board) == example.target_action
            )
            raw_legal += int(example.board.is_legal(prediction))
    count = len(examples)
    return {
        "count": count,
        "loss": statistics.fmean(losses),
        "raw_teacher_agreement": raw_agreements / count,
        "masked_teacher_agreement": masked_agreements / count,
        "raw_legal_rate": raw_legal / count,
    }


def _load_resume(path: Path, encoder, learner, lora_memory) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "mqr-minicpm-go-v1":
        raise ValueError("Unsupported experiment checkpoint")
    encoder.load_adapter_state_dict(payload["lora_adapter"])
    learner.load_state_dict(payload["mqr_state"])
    lora_memory.load_state_dict(payload["lora_ogd_state"])
    state = payload.get("experiment_state")
    return {} if state is None else dict(state)


def _replay_board(size: int, komi: float, moves: Sequence[int]) -> GoBoard:
    board = GoBoard(size, komi=komi)
    for action in moves:
        board.play(int(action))
    return board


def _restore_stream_state(
    state: Dict[str, Any],
    args: argparse.Namespace,
    learner: OnlineMultiRingClassifier,
) -> Tuple[GoBoard, int, Dict[str, int], List[Dict[str, float]]]:
    if not state:
        # Legacy checkpoints contain a recurrent ring state without the board
        # that generated it. Keeping that state would splice two unrelated games.
        learner.reset_state()
        return GoBoard(args.board_size, komi=args.komi), 0, {}, []
    if int(state.get("version", -1)) != 1:
        raise ValueError("Unsupported experiment stream state")
    expected = {
        "stream_mode": args.stream_mode,
        "board_size": args.board_size,
        "teacher": args.teacher,
        "rollout_policy": args.rollout_policy,
        "seed": args.seed,
        "dataset_size": args.dataset_size,
        "sayuri_playouts": args.sayuri_playouts,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(
                f"Resume mismatch for {key}: checkpoint={state.get(key)!r}, current={value!r}"
            )
    if abs(float(state.get("komi", args.komi)) - args.komi) > 1e-12:
        raise ValueError("Resume mismatch for komi")
    board = _replay_board(
        args.board_size,
        args.komi,
        [int(action) for action in state.get("board_move_history", [])],
    )
    step_offset = int(state.get("steps_completed", 0))
    if step_offset < 0:
        raise ValueError("steps_completed must be non-negative")
    context_counts = {
        str(key): int(value)
        for key, value in dict(state.get("context_sample_counts", {})).items()
    }
    if any(value < 0 for value in context_counts.values()):
        raise ValueError("context sample counts must be non-negative")
    completed = [
        {str(key): float(value) for key, value in dict(score).items()}
        for score in state.get("completed_games", [])
    ]
    return board, step_offset, context_counts, completed


def _relabel_examples(
    examples: Sequence[GoTrainingExample], teacher
) -> List[GoTrainingExample]:
    return [
        GoTrainingExample(example.board, teacher.select_move(example.board))
        for example in examples
    ]


def _execution_description(args: argparse.Namespace, encoder) -> str:
    execution_dtype = str(encoder.execution_dtype).removeprefix("torch.").upper()
    adaptation = "online MQR + LoRA adapters"
    if args.no_lora:
        adaptation = "online MQR only (LoRA adapters frozen)"
    return (
        "AWQ INT4 checkpoint dequantized once to a frozen "
        f"{execution_dtype} backbone; {adaptation}"
    )


def _limitations(args: argparse.Namespace) -> List[str]:
    if args.teacher == "heuristic":
        teacher_limit = (
            "The heuristic is a deterministic mechanism-test teacher, not a measure of Go strength."
        )
    else:
        teacher_limit = (
            f"{args.teacher} is a fixed imitation target; agreement is not an independent measure "
            "of Go strength or generalization."
        )
    return [
        teacher_limit,
        (
            "The packed AWQ integers are never updated; the backbone executes after one-time "
            f"{args.dtype.upper()} dequantization."
        ),
        "A short stream validates gradient flow and prequential semantics, not animal-level learning.",
    ]


def _advance_context_memory_schedule(
    counts: Dict[str, int],
    context_id: str,
    *,
    ogd_rank: int,
    remember_every: int,
) -> Tuple[int, bool]:
    """Advance one context-local counter and decide whether to retain its gradient."""

    ordinal = counts.get(context_id, 0) + 1
    counts[context_id] = ordinal
    remember = bool(
        ogd_rank > 0
        and remember_every > 0
        and ordinal % remember_every == 0
    )
    return ordinal, remember


def main() -> None:
    args = parse_args()
    if min(args.steps, args.eval_positions, args.eval_batch_size, args.dataset_size) <= 0:
        raise ValueError("steps, dataset/evaluation sizes, and eval-batch-size must be positive")
    if args.board_size != 5:
        LOGGER.warning("The implementation supports size=%d, but reported defaults are calibrated for 5x5", args.board_size)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    load_started = time.perf_counter()
    encoder = load_minicpm_awq_encoder(
        args.model_path,
        device=device,
        dtype=_dtype(args.dtype),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    _sync(device)
    load_seconds = time.perf_counter() - load_started

    action_size = args.board_size * args.board_size + 1
    carry_state = args.stream_mode == "game"
    learner = OnlineMultiRingClassifier(
        encoder.hidden_size,
        args.ring_hidden,
        action_size,
        num_rings=2,
        lr=args.mqr_lr,
        unitary_lr_ratio=0.2,
        injection_lr_ratio=1.0,
        readout_lr_ratio=1.0,
        adjoint_steps=args.adjoint_steps,
        ogd_max_rank=args.ogd_rank,
        carry_state=carry_state,
        max_update_norm=args.max_mqr_update_norm,
        ring_kwargs={
            "alpha": 0.3,
            "relaxation_steps": args.relaxation_steps,
            "lora_rank": args.ring_rank,
            "readout_dim": args.ring_hidden,
            "h_mix_beta": 0.7,
            "state_activation": "tanh",
            "base_unitary_init": "random",
            "base_unitary_scale": 0.02,
            "base_unitary_seed": args.seed,
        },
    ).to(device)
    lora_memory = OrthogonalGradientMemory(args.ogd_rank).to(device)
    resume_stream_state: Dict[str, Any] = {}
    if args.resume is not None:
        resume_stream_state = _load_resume(args.resume, encoder, learner, lora_memory)
        LOGGER.info("Resumed adapters, rings, and OGD memory from %s", args.resume)

    sayuri_client = None
    sayuri_metadata: Dict[str, Any] = {"enabled": False}
    if args.teacher == "heuristic":
        teacher = HeuristicGoTeacher()
    else:
        discovered_binary = discovered_weights = None
        if args.sayuri_binary is None or args.sayuri_weights is None:
            discovered_binary, discovered_weights = discover_sayuri_paths(ROOT)
        sayuri_binary = args.sayuri_binary or discovered_binary
        sayuri_weights = args.sayuri_weights or discovered_weights
        sayuri_client = SayuriGTPClient(
            sayuri_binary,
            sayuri_weights,
            board_size=args.board_size,
            komi=args.komi,
            threads=1,
            playouts=args.sayuri_playouts,
            scoring_rule="area",
            friendly_pass=True,
        )
        atexit.register(sayuri_client.close, force=True)
        teacher = SayuriTeacher(
            sayuri_client,
            mode="policy" if args.teacher == "sayuri-policy" else "mcts",
        )
        sayuri_metadata = {
            "enabled": True,
            "teacher": args.teacher,
            "binary": str(Path(sayuri_binary).resolve()),
            "weights": str(Path(sayuri_weights).resolve()),
            "engine_name": sayuri_client.engine_name,
            "engine_version": sayuri_client.engine_version,
            "playouts": args.sayuri_playouts,
        }

    evaluation_set = generate_basic_go_dataset(
        args.eval_positions,
        size=args.board_size,
        komi=args.komi,
        seed=args.seed + 1000,
    )
    if sayuri_client is not None:
        evaluation_set = _relabel_examples(evaluation_set, teacher)
    before_eval = evaluate_examples(
        encoder,
        learner,
        evaluation_set,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
    )
    LOGGER.info("Before online stream: %s", before_eval)

    board, step_offset, context_sample_counts, completed_games = _restore_stream_state(
        resume_stream_state,
        args,
        learner,
    )
    if args.resume is not None and not resume_stream_state:
        LOGGER.warning(
            "Legacy checkpoint has no board-aligned stream state; recurrent ring states were reset"
        )
    training_set = generate_basic_go_dataset(
        args.dataset_size,
        size=args.board_size,
        komi=args.komi,
        seed=args.seed + 100,
    )
    if sayuri_client is not None:
        training_set = _relabel_examples(training_set, teacher)
        mismatch_count = 0
        mismatch_examples = []
        for example in evaluation_set[: min(8, len(evaluation_set))]:
            comparison = sayuri_client.compare_legal_moves(example.board)
            if comparison["local_only"] or comparison["sayuri_only"]:
                mismatch_count += 1
                mismatch_examples.append(
                    {
                        "local_only": sorted(comparison["local_only"]),
                        "sayuri_only": sorted(comparison["sayuri_only"]),
                    }
                )
        sayuri_metadata["rule_positions_checked"] = min(8, len(evaluation_set))
        sayuri_metadata["rule_mismatch_count"] = mismatch_count
        sayuri_metadata["rule_mismatch_examples"] = mismatch_examples
    visited: List[GoTrainingExample] = []
    records: List[Dict[str, Any]] = []
    for local_step in range(args.steps):
        step = step_offset + local_step
        if args.stream_mode == "game":
            if board.game_over:
                completed_games.append(board.score())
                board = GoBoard(args.board_size, komi=args.komi)
                learner.reset_state()
            snapshot = board.copy()
            teacher_action = teacher.select_move(snapshot)
        else:
            example = training_set[step % len(training_set)]
            snapshot = example.board.copy()
            teacher_action = example.target_action
        visited.append(GoTrainingExample(snapshot, teacher_action))
        prompt = format_go_prompt(snapshot)  # Deliberately contains no feedback label.
        _sync(device)
        started = time.perf_counter()
        features = encoder.encode_prompts(
            [prompt],
            max_length=args.max_length,
            require_lora_grad=not args.no_lora,
        )
        # MiniCPM's final RMSNorm controls scale but not mean.  Explicit
        # LayerNorm makes the online ring input distribution stationary while
        # preserving an autograd path from the MQR input gradient to LoRA.
        features = F.layer_norm(features, (features.size(-1),))
        target = torch.tensor([teacher_action], device=device, dtype=torch.long)
        context_id = color_name(snapshot.to_play)
        context_ordinal, remember = _advance_context_memory_schedule(
            context_sample_counts,
            context_id,
            ogd_rank=args.ogd_rank,
            remember_every=args.remember_every,
        )
        info = learner.online_step(
            features,
            target,
            context_id=context_id,
            remember_gradient=remember,
            project_with_memory=True,
            carry_state=carry_state,
            return_grad_x=not args.no_lora,
        )
        logits = info["logits"][0].detach()
        raw_action = int(torch.argmax(logits).item())
        raw_is_legal = snapshot.is_legal(raw_action)
        legal_model_action = _masked_action(logits, snapshot)

        if args.no_lora:
            lora_info = {
                "raw_grad_norm": 0.0,
                "update_norm": 0.0,
                "ogd_rank": lora_memory.rank,
                "ogd_retained_norm": 1.0,
                "ogd_memory_added": False,
            }
        else:
            lora_info = encoder.step_from_external_gradient(
                features,
                info["grad_x"],
                lr=args.lora_lr,
                orthogonal_memory=lora_memory if args.ogd_rank > 0 else None,
                remember_gradient=remember,
                project_with_memory=True,
                max_grad_norm=args.max_lora_grad_norm,
            )
        applied_action = teacher_action if args.rollout_policy == "teacher" else legal_model_action
        if args.stream_mode == "game":
            move_result = board.play(applied_action)
        else:
            move_result = snapshot.copy().play(applied_action)
        _sync(device)
        latency = time.perf_counter() - started
        record = {
            "step": step,
            "player": context_id,
            "context_ordinal": context_ordinal,
            "remember_gradient": remember,
            "teacher_action": teacher_action,
            "teacher_gtp": action_to_gtp(teacher_action, args.board_size),
            "raw_action": raw_action,
            "raw_gtp": action_to_gtp(raw_action, args.board_size),
            "raw_legal": raw_is_legal,
            "teacher_agreement": raw_action == teacher_action,
            "masked_teacher_agreement": legal_model_action == teacher_action,
            "legal_model_action": legal_model_action,
            "legal_model_gtp": action_to_gtp(legal_model_action, args.board_size),
            "applied_action": applied_action,
            "applied_gtp": action_to_gtp(applied_action, args.board_size),
            "captures": move_result.captures,
            "loss": float(info["loss"]),
            "ring_index": int(info["ring_index"]),
            "mqr_update_norm": float(info["update_norm"]),
            "mqr_unitary_error": float(info["unitary_error"]),
            "mqr_ogd_rank": int(info["ogd_rank"]),
            "mqr_ogd_retained_norm": float(info["ogd_retained_norm"]),
            "mqr_ogd_memory_added": bool(info["ogd_memory_added"]),
            "mqr_update_clip_scale": float(info["update_clip_scale"]),
            "lora_grad_norm": float(lora_info["raw_grad_norm"]),
            "lora_update_norm": float(lora_info["update_norm"]),
            "lora_ogd_rank": int(lora_info["ogd_rank"]),
            "lora_ogd_retained_norm": float(lora_info["ogd_retained_norm"]),
            "lora_ogd_memory_added": bool(lora_info["ogd_memory_added"]),
            "latency_seconds": latency,
        }
        records.append(record)
        LOGGER.info(
            "step=%d player=%s target=%s raw=%s legal=%s loss=%.4f ring=%d latency=%.3fs",
            step,
            record["player"],
            record["teacher_gtp"],
            record["raw_gtp"],
            raw_is_legal,
            record["loss"],
            record["ring_index"],
            latency,
        )

    after_eval = evaluate_examples(
        encoder,
        learner,
        evaluation_set,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
    )
    replay_eval = evaluate_examples(
        encoder,
        learner,
        visited,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
    )
    losses = [record["loss"] for record in records]
    window = max(1, len(losses) // 4)
    summary = {
        "execution_mode": _execution_description(args, encoder),
        "model_path": str(Path(args.model_path).resolve()),
        "load_seconds": load_seconds,
        "steps": args.steps,
        "step_offset": step_offset,
        "total_steps": step_offset + args.steps,
        "board_size": args.board_size,
        "stream_mode": args.stream_mode,
        "dataset_size": args.dataset_size,
        "rollout_policy": args.rollout_policy,
        "carry_state": carry_state,
        "teacher": args.teacher,
        "memory_schedule": "per-context",
        "context_sample_counts": dict(context_sample_counts),
        "prequential_raw_legal_rate": statistics.fmean(float(r["raw_legal"]) for r in records),
        "prequential_teacher_agreement": statistics.fmean(
            float(r["teacher_agreement"]) for r in records
        ),
        "prequential_masked_teacher_agreement": statistics.fmean(
            float(r["masked_teacher_agreement"]) for r in records
        ),
        "mean_loss": statistics.fmean(losses),
        "first_quarter_loss": statistics.fmean(losses[:window]),
        "last_quarter_loss": statistics.fmean(losses[-window:]),
        "mean_step_latency_seconds": statistics.fmean(r["latency_seconds"] for r in records),
        "before_heldout": before_eval,
        "after_heldout": after_eval,
        "after_stream_replay": replay_eval,
        "completed_games": completed_games,
        "lora_trainable_parameters": encoder.trainable_parameter_count,
        "lora_updated_parameters": 0 if args.no_lora else encoder.trainable_parameter_count,
        "backbone_frozen_parameters": encoder.frozen_parameter_count,
        "mqr_trainable_parameters": sum(
            parameter.numel() for parameter in learner.parameters() if parameter.requires_grad
        ),
        "lora_ogd_rank": lora_memory.rank,
        "mqr_ogd_ranks": [memory.rank for memory in learner.gradient_memories],
        "max_mqr_unitary_error": max(r["mqr_unitary_error"] for r in records),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    result = {
        "format": "mqr-minicpm-go-results-v1",
        "limitations": _limitations(args),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sayuri": sayuri_metadata,
        "summary": summary,
        "steps": records,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Final summary: %s", summary)
    LOGGER.info("Wrote metrics to %s", args.metrics_path)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "mqr-minicpm-go-v1",
                "lora_adapter": encoder.adapter_state_dict(),
                "mqr_state": learner.state_dict(),
                "lora_ogd_state": lora_memory.state_dict(),
                "result_summary": summary,
                "experiment_state": {
                    "version": 1,
                    "steps_completed": step_offset + args.steps,
                    "stream_mode": args.stream_mode,
                    "board_size": args.board_size,
                    "komi": args.komi,
                    "teacher": args.teacher,
                    "rollout_policy": args.rollout_policy,
                    "seed": args.seed,
                    "dataset_size": args.dataset_size,
                    "sayuri_playouts": args.sayuri_playouts,
                    "board_move_history": list(board.move_history),
                    "context_sample_counts": dict(context_sample_counts),
                    "completed_games": list(completed_games),
                },
            },
            args.save_checkpoint,
        )
        LOGGER.info("Saved trainable state to %s", args.save_checkpoint)

    if sayuri_client is not None:
        atexit.unregister(sayuri_client.close)
        sayuri_client.close()


if __name__ == "__main__":
    main()
