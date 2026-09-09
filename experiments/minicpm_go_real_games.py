#!/usr/bin/env python3
"""Causal multi-game validation of MiniCPM + MQR online Go learning.

The student always previews an action with the old parameters.  On student
turns that legal-masked action is played before Sayuri labels the pre-move
position; only then is the online update committed.  Sayuri turns use the same
predict-before-feedback transaction.  Fixed probes and paired pre/post games
separate parameter adaptation from externally enforced move legality.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import logging
import math
import random
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mqr import OnlineMultiRingClassifier, OrthogonalGradientMemory  # noqa: E402
from mqr.go import (  # noqa: E402
    BLACK,
    EMPTY,
    WHITE,
    GoBoard,
    GoTrainingExample,
    action_to_gtp,
    color_name,
    format_go_prompt,
)
from mqr.minicpm import load_minicpm_awq_encoder  # noqa: E402
from mqr.sayuri import SayuriGTPClient, SayuriTeacher, discover_sayuri_paths  # noqa: E402


LOGGER = logging.getLogger("minicpm_go_real_games")


@dataclass(frozen=True)
class ConditionSpec:
    prompt_mode: str
    learn: bool
    update_mqr: bool
    update_lora: bool


CONDITIONS: Dict[str, ConditionSpec] = {
    "rules-online": ConditionSpec("rules", True, True, True),
    "wrong-rules-online": ConditionSpec("wrong-rules", True, True, True),
    "rules-no-learning": ConditionSpec("rules", False, False, False),
    "board-only-online": ConditionSpec("board-only", True, True, True),
    "rules-mqr-only": ConditionSpec("rules", True, True, False),
    "rules-lora-only": ConditionSpec("rules", True, False, True),
}


class PassGatedSayuriPolicyTeacher:
    """Use Sayuri's raw policy while suppressing premature pass labels.

    The gate is an explicit curriculum intervention for small boards whose
    native Sayuri weights may prefer pass at very low occupancy.  Above the
    threshold, or when no stone move exists, selection is identical to the
    legal-masked raw policy teacher.
    """

    def __init__(self, client: SayuriGTPClient, *, min_pass_occupancy: float):
        threshold = float(min_pass_occupancy)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("min_pass_occupancy must be in [0, 1]")
        self.client = client
        self.min_pass_occupancy = threshold

    def select_move(self, board: GoBoard) -> int:
        policy = self.client.raw_policy(board)
        legal = board.legal_moves()
        legal_stones = [action for action in legal if action != board.pass_action]
        occupancy = 1.0 - board.board.count(EMPTY) / float(board.pass_action)
        if legal_stones and occupancy < self.min_pass_occupancy:
            legal = legal_stones
        return max(legal, key=lambda action: (policy[action], -action))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/home/spikebai/checkpoints/MiniCPM5-1B-AWQ-INT4",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(CONDITIONS),
        default=list(CONDITIONS),
    )
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--komi", type=float, default=5.5)
    parser.add_argument("--train-games", type=int, default=4)
    parser.add_argument("--eval-games", type=int, default=4)
    parser.add_argument("--probe-positions", type=int, default=16)
    parser.add_argument(
        "--probe-pass-fraction",
        type=float,
        default=0.25,
        help="Exact fraction of fixed native probes whose teacher target is pass.",
    )
    parser.add_argument("--train-opening-moves", type=int, default=2)
    parser.add_argument("--eval-opening-moves", type=int, default=4)
    parser.add_argument("--max-game-moves", type=int, default=60)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--sayuri-binary", type=Path)
    parser.add_argument("--sayuri-weights", type=Path)
    parser.add_argument("--sayuri-playouts", type=int, default=16)
    parser.add_argument(
        "--evaluation-teacher",
        choices=("training", "native-policy", "mcts"),
        default="training",
        help="Use the training teacher or a held-out Sayuri policy/search evaluator.",
    )
    parser.add_argument(
        "--min-pass-occupancy",
        type=float,
        default=0.0,
        help=(
            "Suppress Sayuri pass labels below this occupied-point fraction; "
            "0 keeps the native raw policy."
        ),
    )
    parser.add_argument(
        "--pass-update-scale",
        type=float,
        default=1.0,
        help=(
            "Scale MQR/LoRA updates on pass labels; 0 postpones terminal-action "
            "learning while retaining pass losses for curriculum diagnostics."
        ),
    )
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
        "--curve-every-games",
        type=int,
        default=1,
        help="Evaluate the fixed probe suite after this many training games.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=ROOT / "analysis/results/minicpm_go_real_games.json",
    )
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _masked_action(logits: torch.Tensor, board: GoBoard) -> int:
    legal = board.legal_moves()
    return max(legal, key=lambda action: (float(logits[action].item()), -action))


def _advance_memory_schedule(
    counts: Dict[str, int],
    context_id: str,
    *,
    learn: bool,
    ogd_rank: int,
    remember_every: int,
) -> Tuple[int, bool]:
    ordinal = counts.get(context_id, 0) + 1
    counts[context_id] = ordinal
    remember = bool(
        learn
        and ogd_rank > 0
        and remember_every > 0
        and ordinal % remember_every == 0
    )
    return ordinal, remember


def _replay_board(size: int, komi: float, moves: Sequence[int]) -> GoBoard:
    board = GoBoard(size, komi=komi)
    for action in moves:
        board.play(int(action))
    return board


def _random_opening_history(
    *, size: int, komi: float, moves: int, rng: random.Random
) -> List[int]:
    board = GoBoard(size, komi=komi)
    for _ in range(moves):
        legal_stones = board.legal_moves(include_pass=False)
        if not legal_stones:
            break
        board.play(rng.choice(legal_stones))
    return list(board.move_history)


def _generate_openings(
    count: int,
    *,
    size: int,
    komi: float,
    moves: int,
    seed: int,
) -> List[List[int]]:
    rng = random.Random(int(seed))
    return [
        _random_opening_history(size=size, komi=komi, moves=moves, rng=rng)
        for _ in range(count)
    ]


def _generate_probe_examples(
    count: int,
    *,
    size: int,
    komi: float,
    seed: int,
    teacher: SayuriTeacher,
    pass_fraction: float = 0.25,
) -> List[GoTrainingExample]:
    """Generate fixed probes with an exact pass/placement target split."""

    rng = random.Random(int(seed))
    result: List[GoTrainingExample] = []
    pass_quota = int(round(count * float(pass_fraction)))
    placement_quota = count - pass_quota
    pass_count = 0
    placement_count = 0
    seen = set()
    attempts = 0
    while len(result) < count:
        attempts += 1
        if attempts > 500 * count:
            raise RuntimeError("could not generate enough unique held-out Go positions")
        board = GoBoard(size, komi=komi)
        depth = rng.randint(2, max(3, min(2 * size * size // 3, 18)))
        for _ in range(depth):
            if board.game_over:
                break
            legal_stones = board.legal_moves(include_pass=False)
            if legal_stones and rng.random() < 0.35:
                action = rng.choice(legal_stones)
            else:
                action = teacher.select_move(board.copy())
            board.play(action)
        if board.game_over or board.position_key() in seen:
            continue
        seen.add(board.position_key())
        target = teacher.select_move(board.copy())
        is_pass = target == board.pass_action
        if is_pass and pass_count >= pass_quota:
            continue
        if not is_pass and placement_count >= placement_quota:
            continue
        result.append(GoTrainingExample(board.copy(), target))
        pass_count += int(is_pass)
        placement_count += int(not is_pass)
    return result


def _build_learner(args: argparse.Namespace, encoder, seed: int, device: torch.device):
    _seed_everything(seed + 17)
    learner = OnlineMultiRingClassifier(
        encoder.hidden_size,
        args.ring_hidden,
        args.board_size * args.board_size + 1,
        num_rings=2,
        lr=args.mqr_lr,
        unitary_lr_ratio=0.2,
        injection_lr_ratio=1.0,
        readout_lr_ratio=1.0,
        adjoint_steps=args.adjoint_steps,
        ogd_max_rank=args.ogd_rank,
        carry_state=True,
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
            "base_unitary_seed": seed,
        },
    ).to(device)
    # Context routing is part of the experimental design, not learned from the
    # ordering of probes or games.
    assert learner.route_context("black") == 0
    assert learner.route_context("white") == 1
    return learner


def _fresh_adapter_state(encoder, seed: int) -> Dict[str, Any]:
    """Create a deterministic identity-output LoRA initialization for one seed."""

    _seed_everything(seed + 29)
    with torch.no_grad():
        for name, parameter in encoder.named_lora_parameters():
            if name.endswith("lora_A"):
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
            elif name.endswith("lora_B"):
                parameter.zero_()
            else:
                raise RuntimeError(f"unexpected LoRA parameter: {name}")
    return encoder.adapter_state_dict()


def _snapshot_named_parameters(items: Iterator[Tuple[str, nn.Parameter]]) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in items
    }


def _drift_statistics(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    *,
    group_name,
) -> Dict[str, Any]:
    if set(before) != set(after):
        raise ValueError("parameter layouts differ while measuring drift")

    def summarize(names: Sequence[str]) -> Dict[str, Any]:
        reference_sq = 0.0
        delta_sq = 0.0
        maximum = 0.0
        count = 0
        for name in names:
            reference = before[name]
            delta = after[name] - reference
            reference_sq += float(reference.square().sum().item())
            delta_sq += float(delta.square().sum().item())
            maximum = max(maximum, float(delta.abs().max().item()))
            count += delta.numel()
        reference_l2 = math.sqrt(reference_sq)
        delta_l2 = math.sqrt(delta_sq)
        return {
            "parameter_count": count,
            "reference_l2": reference_l2,
            "delta_l2": delta_l2,
            "relative_delta_l2": (
                delta_l2 / reference_l2 if reference_l2 > 0.0 else None
            ),
            "max_abs_delta": maximum,
        }

    names = sorted(before)
    groups: Dict[str, List[str]] = {}
    for name in names:
        groups.setdefault(str(group_name(name)), []).append(name)
    return {
        "all": summarize(names),
        "groups": {name: summarize(group_names) for name, group_names in groups.items()},
    }


def _mqr_group(name: str) -> str:
    if ".unitary_param." in name:
        return "cayley"
    if ".injection." in name:
        return "injection"
    if ".readout." in name:
        return "readout"
    return "other"


def _lora_group(name: str) -> str:
    return "lora_A" if name.endswith("lora_A") else "lora_B"


@contextmanager
def _preserve_learner_state(learner: OnlineMultiRingClassifier):
    state = copy.deepcopy(learner.state_dict())
    try:
        yield
    finally:
        learner.load_state_dict(state)


def _illegal_kind(board: GoBoard, action: int) -> str:
    if board.is_legal(action):
        return "legal"
    if 0 <= action < board.pass_action and board.board[action] != EMPTY:
        return "occupied"
    return "suicide_or_superko"


@torch.no_grad()
def _evaluate_probes(
    encoder,
    learner: OnlineMultiRingClassifier,
    examples: Sequence[GoTrainingExample],
    *,
    prompt_mode: str,
    max_length: int,
    batch_size: int,
) -> Dict[str, Any]:
    losses: List[float] = []
    raw_agreement = 0
    masked_agreement = 0
    legal = 0
    illegal_kinds = {"occupied": 0, "suicide_or_superko": 0}
    placement_losses: List[float] = []
    placement_agreement = 0
    placement_targets = 0
    pass_targets = 0
    pass_probability_sum = 0.0
    pass_brier: List[float] = []
    pass_binary_nll: List[float] = []
    predicted_pass = 0
    true_positive_pass = 0
    false_pass_on_placement = 0
    with _preserve_learner_state(learner):
        for offset in range(0, len(examples), batch_size):
            current = examples[offset : offset + batch_size]
            features = encoder.encode_prompts(
                [
                    format_go_prompt(example.board, prompt_mode=prompt_mode)
                    for example in current
                ],
                max_length=max_length,
                require_lora_grad=False,
            )
            features = F.layer_norm(features, (features.size(-1),))
            for row, example in enumerate(current):
                logits = learner(
                    features[row : row + 1],
                    context_id=color_name(example.board.to_play),
                )[0]
                target = torch.tensor([example.target_action], device=logits.device)
                losses.append(float(F.cross_entropy(logits.unsqueeze(0), target).item()))
                raw_action = int(torch.argmax(logits).item())
                masked_action = _masked_action(logits, example.board)
                probabilities = F.softmax(logits, dim=0)
                pass_probability = float(
                    probabilities[example.board.pass_action].item()
                )
                target_is_pass = example.target_action == example.board.pass_action
                conditional_placement = int(
                    torch.argmax(logits[: example.board.pass_action]).item()
                )
                pass_probability_sum += pass_probability
                pass_brier.append(
                    (pass_probability - float(target_is_pass)) ** 2
                )
                binary_probability = (
                    pass_probability if target_is_pass else 1.0 - pass_probability
                )
                pass_binary_nll.append(-math.log(max(binary_probability, 1e-12)))
                predicted_pass += int(raw_action == example.board.pass_action)
                true_positive_pass += int(
                    target_is_pass and raw_action == example.board.pass_action
                )
                if target_is_pass:
                    pass_targets += 1
                else:
                    placement_targets += 1
                    placement_agreement += int(
                        conditional_placement == example.target_action
                    )
                    false_pass_on_placement += int(
                        raw_action == example.board.pass_action
                    )
                    placement_target = torch.tensor(
                        [example.target_action], device=logits.device
                    )
                    placement_losses.append(
                        float(
                            F.cross_entropy(
                                logits[: example.board.pass_action].unsqueeze(0),
                                placement_target,
                            ).item()
                        )
                    )
                kind = _illegal_kind(example.board, raw_action)
                legal += int(kind == "legal")
                if kind != "legal":
                    illegal_kinds[kind] += 1
                raw_agreement += int(raw_action == example.target_action)
                masked_agreement += int(masked_action == example.target_action)
    count = len(examples)
    return {
        "count": count,
        "loss": statistics.fmean(losses),
        "raw_teacher_agreement": raw_agreement / count,
        "masked_teacher_agreement": masked_agreement / count,
        "raw_legal_rate": legal / count,
        "raw_illegal_occupied_rate": illegal_kinds["occupied"] / count,
        "raw_illegal_suicide_or_superko_rate": (
            illegal_kinds["suicide_or_superko"] / count
        ),
        "placement_target_count": placement_targets,
        "pass_target_count": pass_targets,
        "conditional_placement_loss": (
            statistics.fmean(placement_losses) if placement_losses else None
        ),
        "conditional_placement_agreement": (
            placement_agreement / placement_targets if placement_targets else None
        ),
        "false_pass_rate_on_placement_targets": (
            false_pass_on_placement / placement_targets if placement_targets else None
        ),
        "pass_brier": statistics.fmean(pass_brier),
        "pass_binary_nll": statistics.fmean(pass_binary_nll),
        "mean_pass_probability": pass_probability_sum / count,
        "raw_pass_prediction_rate": predicted_pass / count,
        "pass_recall": (
            true_positive_pass / pass_targets if pass_targets else None
        ),
        "pass_precision": (
            true_positive_pass / predicted_pass if predicted_pass else None
        ),
    }


def _play_game(
    encoder,
    learner: OnlineMultiRingClassifier,
    teacher: SayuriTeacher,
    *,
    args: argparse.Namespace,
    condition: ConditionSpec,
    lora_memory: OrthogonalGradientMemory,
    context_counts: Dict[str, int],
    opening_history: Sequence[int],
    student_color: int,
    game_index: int,
    phase: str,
    learn: bool,
    collect_records: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    board = _replay_board(args.board_size, args.komi, opening_history)
    learner.reset_state()
    losses: List[float] = []
    raw_legal_count = 0
    raw_agreement_count = 0
    masked_agreement_count = 0
    student_masked_agreement_count = 0
    student_turns = 0
    student_raw_legal_count = 0
    student_raw_agreement_count = 0
    student_raw_pass_count = 0
    student_placement_targets = 0
    student_conditional_placement_agreement = 0
    student_false_pass_on_placement = 0
    student_pass_targets = 0
    student_pass_recall_count = 0
    records: List[Dict[str, Any]] = []

    for ply in range(args.max_game_moves):
        if board.game_over:
            break
        snapshot = board.copy()
        context_id = color_name(snapshot.to_play)
        require_lora_grad = bool(learn and condition.update_lora)
        features = encoder.encode_prompts(
            [format_go_prompt(snapshot, prompt_mode=condition.prompt_mode)],
            max_length=args.max_length,
            require_lora_grad=require_lora_grad,
        )
        features = F.layer_norm(features, (features.size(-1),))

        # This is a read-only transaction.  No feedback has been requested yet.
        preview = learner.preview_step(features, context_id=context_id, carry_state=True)
        logits = preview["logits"][0].detach()
        raw_action = int(torch.argmax(logits).item())
        conditional_placement_action = int(
            torch.argmax(logits[: snapshot.pass_action]).item()
        )
        legal_model_action = _masked_action(logits, snapshot)
        student_turn = snapshot.to_play == student_color

        # On a student turn, the old-policy move is physically committed before
        # querying Sayuri for the label of the saved pre-move position.
        if student_turn:
            applied_action = legal_model_action
            move_result = board.play(applied_action)
            teacher_action = teacher.select_move(snapshot)
        else:
            teacher_action = teacher.select_move(snapshot)
            applied_action = teacher_action
            move_result = board.play(applied_action)

        target = torch.tensor([teacher_action], device=features.device, dtype=torch.long)
        loss = float(F.cross_entropy(preview["logits"], target).item())
        context_ordinal, scheduled_remember = _advance_memory_schedule(
            context_counts,
            context_id,
            learn=learn,
            ogd_rank=args.ogd_rank,
            remember_every=args.remember_every,
        )
        update_scale = (
            args.pass_update_scale if teacher_action == snapshot.pass_action else 1.0
        )
        apply_feedback_update = bool(learn and update_scale > 0.0)
        mqr_remember = bool(
            scheduled_remember and apply_feedback_update and condition.update_mqr
        )
        lora_remember = bool(
            scheduled_remember and apply_feedback_update and condition.update_lora
        )
        if apply_feedback_update:
            original_lr = learner.lr
            original_ratios = (
                learner.unitary_lr_ratio,
                learner.injection_lr_ratio,
                learner.readout_lr_ratio,
                learner.state_target_lr_ratio,
                learner.h_mix_beta_lr_ratio,
            )
            learner.lr = original_lr * update_scale
            if not condition.update_mqr:
                learner.unitary_lr_ratio = 0.0
                learner.injection_lr_ratio = 0.0
                learner.readout_lr_ratio = 0.0
                learner.state_target_lr_ratio = 0.0
                learner.h_mix_beta_lr_ratio = 0.0
            try:
                committed = learner.online_step(
                    features,
                    target,
                    context_id=context_id,
                    learn=True,
                    remember_gradient=mqr_remember,
                    project_with_memory=True,
                    carry_state=True,
                    return_grad_x=condition.update_lora,
                )
            finally:
                learner.lr = original_lr
                (
                    learner.unitary_lr_ratio,
                    learner.injection_lr_ratio,
                    learner.readout_lr_ratio,
                    learner.state_target_lr_ratio,
                    learner.h_mix_beta_lr_ratio,
                ) = original_ratios
        else:
            committed = learner.online_step(
                features,
                None,
                context_id=context_id,
                learn=False,
                carry_state=True,
                return_grad_x=False,
            )
        torch.testing.assert_close(
            committed["logits"], preview["logits"], rtol=0.0, atol=0.0
        )
        if apply_feedback_update and abs(float(committed["loss"]) - loss) > 1e-6:
            raise RuntimeError("preview and deferred-update losses diverged")

        if apply_feedback_update and condition.update_lora:
            lora_info = encoder.step_from_external_gradient(
                features,
                committed["grad_x"],
                lr=args.lora_lr * update_scale,
                orthogonal_memory=lora_memory if args.ogd_rank > 0 else None,
                remember_gradient=lora_remember,
                project_with_memory=True,
                max_grad_norm=args.max_lora_grad_norm,
            )
        else:
            lora_info = {
                "raw_grad_norm": 0.0,
                "update_norm": 0.0,
                "ogd_rank": lora_memory.rank,
                "ogd_retained_norm": 1.0,
                "ogd_memory_added": False,
            }

        raw_legal = snapshot.is_legal(raw_action)
        raw_agreement = raw_action == teacher_action
        masked_agreement = legal_model_action == teacher_action
        losses.append(loss)
        raw_legal_count += int(raw_legal)
        raw_agreement_count += int(raw_agreement)
        masked_agreement_count += int(masked_agreement)
        if student_turn:
            student_turns += 1
            student_masked_agreement_count += int(masked_agreement)
            student_raw_legal_count += int(raw_legal)
            student_raw_agreement_count += int(raw_agreement)
            student_raw_pass_count += int(raw_action == snapshot.pass_action)
            if teacher_action == snapshot.pass_action:
                student_pass_targets += 1
                student_pass_recall_count += int(
                    raw_action == snapshot.pass_action
                )
            else:
                student_placement_targets += 1
                student_conditional_placement_agreement += int(
                    conditional_placement_action == teacher_action
                )
                student_false_pass_on_placement += int(
                    raw_action == snapshot.pass_action
                )

        if collect_records:
            records.append(
                {
                    "phase": phase,
                    "game": game_index,
                    "ply": ply,
                    "player": context_id,
                    "student_turn": student_turn,
                    "context_ordinal": context_ordinal,
                    "remember_mqr_gradient": mqr_remember,
                    "remember_lora_gradient": lora_remember,
                    "feedback_update_applied": apply_feedback_update,
                    "feedback_update_scale": update_scale if learn else 0.0,
                    "teacher_action": teacher_action,
                    "teacher_gtp": action_to_gtp(teacher_action, args.board_size),
                    "raw_action": raw_action,
                    "raw_gtp": action_to_gtp(raw_action, args.board_size),
                    "raw_legal": raw_legal,
                    "raw_teacher_agreement": raw_agreement,
                    "conditional_placement_action": conditional_placement_action,
                    "conditional_placement_gtp": action_to_gtp(
                        conditional_placement_action, args.board_size
                    ),
                    "masked_teacher_agreement": masked_agreement,
                    "applied_action": applied_action,
                    "applied_gtp": action_to_gtp(applied_action, args.board_size),
                    "legality_mask_enforced": student_turn,
                    "action_committed_before_model_update": True,
                    "loss": loss,
                    "captures": move_result.captures,
                    "ring_index": int(committed["ring_index"]),
                    "mqr_update_norm": float(committed.get("update_norm", 0.0)),
                    "mqr_unitary_error": float(committed["unitary_error"]),
                    "mqr_ogd_rank": int(committed.get("ogd_rank", 0)),
                    "mqr_ogd_retained_norm": float(
                        committed.get("ogd_retained_norm", 1.0)
                    ),
                    "lora_grad_norm": float(lora_info["raw_grad_norm"]),
                    "lora_update_norm": float(lora_info["update_norm"]),
                    "lora_ogd_rank": int(lora_info["ogd_rank"]),
                    "lora_ogd_retained_norm": float(lora_info["ogd_retained_norm"]),
                }
            )

    truncated = not board.game_over
    score = board.score()
    margin_black = float(score["margin_black"])
    student_margin = margin_black if student_color == BLACK else -margin_black
    moves_played = len(board.move_history) - len(opening_history)
    count = len(losses)
    summary = {
        "game": game_index,
        "phase": phase,
        "student_color": color_name(student_color),
        "opening_history": list(opening_history),
        "moves_played": moves_played,
        "terminal": board.game_over,
        "truncated": truncated,
        "margin_black": margin_black,
        "student_margin": student_margin,
        "student_win": student_margin > 0.0,
        "loss": statistics.fmean(losses),
        "raw_legal_rate": raw_legal_count / count,
        "raw_teacher_agreement": raw_agreement_count / count,
        "masked_teacher_agreement": masked_agreement_count / count,
        "student_masked_teacher_agreement": (
            student_masked_agreement_count / student_turns if student_turns else 0.0
        ),
        "student_turns": student_turns,
        "student_raw_legal_rate": (
            student_raw_legal_count / student_turns if student_turns else 0.0
        ),
        "student_raw_teacher_agreement": (
            student_raw_agreement_count / student_turns if student_turns else 0.0
        ),
        "student_raw_pass_rate": (
            student_raw_pass_count / student_turns if student_turns else 0.0
        ),
        "student_placement_target_count": student_placement_targets,
        "student_conditional_placement_agreement": (
            student_conditional_placement_agreement / student_placement_targets
            if student_placement_targets
            else None
        ),
        "student_false_pass_rate_on_placement_targets": (
            student_false_pass_on_placement / student_placement_targets
            if student_placement_targets
            else None
        ),
        "student_pass_target_count": student_pass_targets,
        "student_pass_recall": (
            student_pass_recall_count / student_pass_targets
            if student_pass_targets
            else None
        ),
        "final_move_history": list(board.move_history),
    }
    return summary, records


def _summarize_games(games: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    placement_targets = sum(
        int(game["student_placement_target_count"]) for game in games
    )
    pass_targets = sum(int(game["student_pass_target_count"]) for game in games)
    placement_hits = sum(
        float(game["student_conditional_placement_agreement"] or 0.0)
        * int(game["student_placement_target_count"])
        for game in games
    )
    false_passes = sum(
        float(game["student_false_pass_rate_on_placement_targets"] or 0.0)
        * int(game["student_placement_target_count"])
        for game in games
    )
    pass_hits = sum(
        float(game["student_pass_recall"] or 0.0)
        * int(game["student_pass_target_count"])
        for game in games
    )
    return {
        "count": len(games),
        "student_win_rate": statistics.fmean(float(game["student_win"]) for game in games),
        "mean_student_margin": statistics.fmean(float(game["student_margin"]) for game in games),
        "mean_loss": statistics.fmean(float(game["loss"]) for game in games),
        "raw_legal_rate": statistics.fmean(float(game["raw_legal_rate"]) for game in games),
        "masked_teacher_agreement": statistics.fmean(
            float(game["masked_teacher_agreement"]) for game in games
        ),
        "student_masked_teacher_agreement": statistics.fmean(
            float(game["student_masked_teacher_agreement"]) for game in games
        ),
        "student_raw_legal_rate": statistics.fmean(
            float(game["student_raw_legal_rate"]) for game in games
        ),
        "student_raw_teacher_agreement": statistics.fmean(
            float(game["student_raw_teacher_agreement"]) for game in games
        ),
        "student_raw_pass_rate": statistics.fmean(
            float(game["student_raw_pass_rate"]) for game in games
        ),
        "student_placement_target_count": placement_targets,
        "student_conditional_placement_agreement": (
            placement_hits / placement_targets if placement_targets else None
        ),
        "student_false_pass_rate_on_placement_targets": (
            false_passes / placement_targets if placement_targets else None
        ),
        "student_pass_target_count": pass_targets,
        "student_pass_recall": pass_hits / pass_targets if pass_targets else None,
        "truncation_rate": statistics.fmean(float(game["truncated"]) for game in games),
        "mean_moves": statistics.fmean(float(game["moves_played"]) for game in games),
        "games": list(games),
    }


def _probe_curve_trends(curve: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    directions = {
        "loss": "lower",
        "conditional_placement_loss": "lower",
        "conditional_placement_agreement": "higher",
        "false_pass_rate_on_placement_targets": "lower",
        "pass_brier": "lower",
        "raw_legal_rate": "higher",
        "raw_teacher_agreement": "higher",
    }
    result: Dict[str, Any] = {}
    for metric, favorable in directions.items():
        points = [
            (float(item["after_train_games"]), float(item[metric]))
            for item in curve
            if item.get(metric) is not None
        ]
        if len(points) < 2:
            result[metric] = None
            continue
        x = torch.tensor([point[0] for point in points], dtype=torch.float64)
        y = torch.tensor([point[1] for point in points], dtype=torch.float64)
        centered = x - x.mean()
        denominator = float(centered.square().sum().item())
        slope = (
            0.0
            if denominator == 0.0
            else float((centered * (y - y.mean())).sum().item() / denominator)
        )
        favorable_steps = sum(
            (right <= left if favorable == "lower" else right >= left)
            for left, right in zip(y[:-1].tolist(), y[1:].tolist())
        )
        result[metric] = {
            "direction": favorable,
            "slope_per_training_game": slope,
            "first": float(y[0].item()),
            "last": float(y[-1].item()),
            "favorable_change": (
                float(y[0].item() - y[-1].item())
                if favorable == "lower"
                else float(y[-1].item() - y[0].item())
            ),
            "favorable_step_fraction": favorable_steps / (len(points) - 1),
        }
    return result


def _evaluate_games(
    encoder,
    learner: OnlineMultiRingClassifier,
    teacher: SayuriTeacher,
    *,
    args: argparse.Namespace,
    condition: ConditionSpec,
    openings: Sequence[Sequence[int]],
    phase: str,
) -> Dict[str, Any]:
    games = []
    dummy_memory = OrthogonalGradientMemory(0).to(learner.ring_usage.device)
    with _preserve_learner_state(learner):
        for game_index, opening in enumerate(openings):
            student_color = BLACK if game_index % 2 == 0 else WHITE
            game, _ = _play_game(
                encoder,
                learner,
                teacher,
                args=args,
                condition=condition,
                lora_memory=dummy_memory,
                context_counts={},
                opening_history=opening,
                student_color=student_color,
                game_index=game_index,
                phase=phase,
                learn=False,
                collect_records=False,
            )
            games.append(game)
    return _summarize_games(games)


def _train_games(
    encoder,
    learner: OnlineMultiRingClassifier,
    teacher: SayuriTeacher,
    *,
    args: argparse.Namespace,
    condition: ConditionSpec,
    openings: Sequence[Sequence[int]],
    lora_memory: OrthogonalGradientMemory,
    probes: Sequence[GoTrainingExample],
    initial_probe: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    games = []
    records: List[Dict[str, Any]] = []
    context_counts: Dict[str, int] = {}
    probe_curve: List[Dict[str, Any]] = [
        {"after_train_games": 0, **copy.deepcopy(dict(initial_probe))}
    ]
    for game_index, opening in enumerate(openings):
        student_color = BLACK if game_index % 2 == 0 else WHITE
        game, current = _play_game(
            encoder,
            learner,
            teacher,
            args=args,
            condition=condition,
            lora_memory=lora_memory,
            context_counts=context_counts,
            opening_history=opening,
            student_color=student_color,
            game_index=game_index,
            phase="train",
            learn=condition.learn,
            collect_records=True,
        )
        games.append(game)
        records.extend(current)
        LOGGER.info(
            "train game=%d student=%s margin=%+.1f moves=%d truncated=%s",
            game_index,
            game["student_color"],
            game["student_margin"],
            game["moves_played"],
            game["truncated"],
        )
        completed_games = game_index + 1
        if (
            completed_games % args.curve_every_games == 0
            or completed_games == len(openings)
        ):
            checkpoint = _evaluate_probes(
                encoder,
                learner,
                probes,
                prompt_mode=condition.prompt_mode,
                max_length=args.max_length,
                batch_size=args.eval_batch_size,
            )
            probe_curve.append(
                {"after_train_games": completed_games, **checkpoint}
            )

    losses = [float(record["loss"]) for record in records]
    window = max(1, len(losses) // 4)
    game_summary = _summarize_games(games)
    game_summary.update(
        {
            "steps": len(records),
            "context_sample_counts": context_counts,
            "first_quarter_loss": statistics.fmean(losses[:window]),
            "last_quarter_loss": statistics.fmean(losses[-window:]),
            "loss_reduction_first_to_last": (
                statistics.fmean(losses[:window]) - statistics.fmean(losses[-window:])
            ),
            "prequential_raw_legal_rate": statistics.fmean(
                float(record["raw_legal"]) for record in records
            ),
            "prequential_masked_teacher_agreement": statistics.fmean(
                float(record["masked_teacher_agreement"]) for record in records
            ),
            "mean_mqr_update_norm": statistics.fmean(
                float(record["mqr_update_norm"]) for record in records
            ),
            "mean_lora_update_norm": statistics.fmean(
                float(record["lora_update_norm"]) for record in records
            ),
            "max_mqr_unitary_error": max(
                float(record["mqr_unitary_error"]) for record in records
            ),
            "probe_learning_curve": probe_curve,
            "probe_trends": _probe_curve_trends(probe_curve),
        }
    )
    return game_summary, records


def _paired_game_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    before_games = before["games"]
    after_games = after["games"]
    if len(before_games) != len(after_games):
        raise ValueError("paired game suites have different lengths")
    margins = []
    for left, right in zip(before_games, after_games):
        if (
            left["opening_history"] != right["opening_history"]
            or left["student_color"] != right["student_color"]
        ):
            raise ValueError("paired game openings or student colors differ")
        margins.append(float(right["student_margin"]) - float(left["student_margin"]))
    result = {
        "paired_student_margin_changes": margins,
        "mean_student_margin_improvement": statistics.fmean(margins),
        "student_win_rate_change": (
            float(after["student_win_rate"]) - float(before["student_win_rate"])
        ),
        "raw_legal_rate_change": float(after["raw_legal_rate"]) - float(before["raw_legal_rate"]),
        "masked_teacher_agreement_change": (
            float(after["masked_teacher_agreement"])
            - float(before["masked_teacher_agreement"])
        ),
        "student_raw_legal_rate_change": (
            float(after["student_raw_legal_rate"])
            - float(before["student_raw_legal_rate"])
        ),
        "student_raw_pass_rate_change": (
            float(after["student_raw_pass_rate"])
            - float(before["student_raw_pass_rate"])
        ),
    }
    for metric in (
        "student_conditional_placement_agreement",
        "student_false_pass_rate_on_placement_targets",
        "student_pass_recall",
    ):
        if before.get(metric) is not None and after.get(metric) is not None:
            result[f"{metric}_change"] = float(after[metric]) - float(before[metric])
    return result


def _probe_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, float]:
    result = {
        "loss_improvement": float(before["loss"]) - float(after["loss"]),
        "raw_legal_rate_change": float(after["raw_legal_rate"]) - float(before["raw_legal_rate"]),
        "raw_teacher_agreement_change": (
            float(after["raw_teacher_agreement"])
            - float(before["raw_teacher_agreement"])
        ),
        "masked_teacher_agreement_change": (
            float(after["masked_teacher_agreement"])
            - float(before["masked_teacher_agreement"])
        ),
        "pass_brier_improvement": float(before["pass_brier"])
        - float(after["pass_brier"]),
        "false_pass_rate_reduction": float(
            before["false_pass_rate_on_placement_targets"]
        )
        - float(after["false_pass_rate_on_placement_targets"]),
    }
    if (
        before.get("conditional_placement_loss") is not None
        and after.get("conditional_placement_loss") is not None
    ):
        result["conditional_placement_loss_improvement"] = float(
            before["conditional_placement_loss"]
        ) - float(after["conditional_placement_loss"])
        result["conditional_placement_agreement_change"] = float(
            after["conditional_placement_agreement"]
        ) - float(before["conditional_placement_agreement"])
    return result


def _game_suites_match_ignoring_phase(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> bool:
    left = copy.deepcopy(dict(before))
    right = copy.deepcopy(dict(after))
    for game in left.get("games", []):
        game.pop("phase", None)
    for game in right.get("games", []):
        game.pop("phase", None)
    return left == right


def _mean_sd(values: Sequence[float]) -> Dict[str, Any]:
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = (
        0.0 if len(values) <= 1 else 1.96 * sample_sd / math.sqrt(len(values))
    )
    # Two-sided 95% Student-t critical values for the small-seed regime used
    # here.  Fall back to 1.96 only beyond the table.
    t_critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }.get(len(values) - 1, 1.96)
    t_half_width = (
        0.0
        if len(values) <= 1
        else t_critical * sample_sd / math.sqrt(len(values))
    )
    return {
        "n": len(values),
        "mean": mean,
        "sample_sd": sample_sd,
        "normal_95ci": [mean - half_width, mean + half_width],
        "student_t_95ci": [mean - t_half_width, mean + t_half_width],
        "values": list(values),
    }


def _aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = {
        "probe_loss_improvement": lambda run: run["probe_changes"]["loss_improvement"],
        "probe_raw_legal_rate_change": lambda run: run["probe_changes"]["raw_legal_rate_change"],
        "probe_masked_agreement_change": lambda run: run["probe_changes"]["masked_teacher_agreement_change"],
        "probe_conditional_placement_loss_improvement": lambda run: run["probe_changes"]["conditional_placement_loss_improvement"],
        "probe_conditional_placement_agreement_change": lambda run: run["probe_changes"]["conditional_placement_agreement_change"],
        "probe_false_pass_rate_reduction": lambda run: run["probe_changes"]["false_pass_rate_reduction"],
        "probe_pass_brier_improvement": lambda run: run["probe_changes"]["pass_brier_improvement"],
        "eval_student_margin_improvement": lambda run: run["game_changes"]["mean_student_margin_improvement"],
        "eval_win_rate_change": lambda run: run["game_changes"]["student_win_rate_change"],
        "eval_student_raw_legal_rate_change": lambda run: run["game_changes"]["student_raw_legal_rate_change"],
        "eval_student_raw_pass_rate_change": lambda run: run["game_changes"]["student_raw_pass_rate_change"],
        "train_loss_reduction": lambda run: run["training"]["loss_reduction_first_to_last"],
        "probe_loss_trend_per_game": lambda run: run["training"]["probe_trends"]["loss"]["slope_per_training_game"],
        "placement_loss_trend_per_game": lambda run: run["training"]["probe_trends"]["conditional_placement_loss"]["slope_per_training_game"],
        "pass_brier_trend_per_game": lambda run: run["training"]["probe_trends"]["pass_brier"]["slope_per_training_game"],
        "mqr_parameter_delta_l2": lambda run: run["parameter_drift"]["mqr"]["all"]["delta_l2"],
        "lora_parameter_delta_l2": lambda run: run["parameter_drift"]["lora"]["all"]["delta_l2"],
    }
    aggregate: Dict[str, Any] = {}
    for condition in CONDITIONS:
        selected = [run for run in runs if run["condition"] == condition]
        if not selected:
            continue
        aggregate[condition] = {
            name: _mean_sd([float(extract(run)) for run in selected])
            for name, extract in metrics.items()
        }

    by_seed = {(int(run["seed"]), str(run["condition"])): run for run in runs}
    contrast_specs = {
        "rules_prompt_minus_board_only": ("rules-online", "board-only-online"),
        "correct_rules_minus_wrong_rules": ("rules-online", "wrong-rules-online"),
        "lora_increment_over_mqr_only": ("rules-online", "rules-mqr-only"),
        "mqr_increment_over_lora_only": ("rules-online", "rules-lora-only"),
        "mqr_only_minus_lora_only": ("rules-mqr-only", "rules-lora-only"),
        "online_minus_no_learning": ("rules-online", "rules-no-learning"),
    }
    contrasts: Dict[str, Any] = {}
    for name, (left_name, right_name) in contrast_specs.items():
        common_seeds = sorted(
            seed
            for seed, condition in by_seed
            if condition == left_name and (seed, right_name) in by_seed
        )
        if not common_seeds:
            continue
        contrasts[name] = {}
        for metric_name, extract in metrics.items():
            values = [
                float(extract(by_seed[(seed, left_name)]))
                - float(extract(by_seed[(seed, right_name)]))
                for seed in common_seeds
            ]
            contrasts[name][metric_name] = _mean_sd(values)
    return {"by_condition": aggregate, "paired_contrasts": contrasts}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_condition(
    encoder,
    training_teacher,
    evaluation_teacher,
    *,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    condition_name: str,
    adapter_initial: Mapping[str, Any],
    probes: Sequence[GoTrainingExample],
    training_openings: Sequence[Sequence[int]],
    evaluation_openings: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    condition = CONDITIONS[condition_name]
    encoder.load_adapter_state_dict(adapter_initial)
    learner = _build_learner(args, encoder, seed, device)
    lora_memory = OrthogonalGradientMemory(args.ogd_rank).to(device)
    initial_mqr = _snapshot_named_parameters(learner.named_parameters())
    initial_lora = _snapshot_named_parameters(encoder.named_lora_parameters())

    before_probe = _evaluate_probes(
        encoder,
        learner,
        probes,
        prompt_mode=condition.prompt_mode,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
    )
    before_games = _evaluate_games(
        encoder,
        learner,
        evaluation_teacher,
        args=args,
        condition=condition,
        openings=evaluation_openings,
        phase="before",
    )
    LOGGER.info(
        "seed=%d condition=%s before probe_loss=%.4f margin=%+.2f",
        seed,
        condition_name,
        before_probe["loss"],
        before_games["mean_student_margin"],
    )

    training, records = _train_games(
        encoder,
        learner,
        training_teacher,
        args=args,
        condition=condition,
        openings=training_openings,
        lora_memory=lora_memory,
        probes=probes,
        initial_probe=before_probe,
    )
    after_probe = _evaluate_probes(
        encoder,
        learner,
        probes,
        prompt_mode=condition.prompt_mode,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
    )
    after_games = _evaluate_games(
        encoder,
        learner,
        evaluation_teacher,
        args=args,
        condition=condition,
        openings=evaluation_openings,
        phase="after",
    )

    final_mqr = _snapshot_named_parameters(learner.named_parameters())
    final_lora = _snapshot_named_parameters(encoder.named_lora_parameters())
    drift = {
        "mqr": _drift_statistics(initial_mqr, final_mqr, group_name=_mqr_group),
        "lora": _drift_statistics(initial_lora, final_lora, group_name=_lora_group),
    }
    probe_changes = _probe_changes(before_probe, after_probe)
    game_changes = _paired_game_changes(before_games, after_games)
    no_learning_audit = None
    if not condition.learn:
        no_learning_audit = {
            "zero_mqr_parameter_drift": drift["mqr"]["all"]["delta_l2"] == 0.0,
            "zero_lora_parameter_drift": drift["lora"]["all"]["delta_l2"] == 0.0,
            "identical_probe_metrics": before_probe == after_probe,
            "identical_paired_games": _game_suites_match_ignoring_phase(
                before_games, after_games
            ),
        }
        if not all(no_learning_audit.values()):
            raise RuntimeError(f"no-learning control changed stateful results: {no_learning_audit}")

    LOGGER.info(
        "seed=%d condition=%s probe_loss_improvement=%+.4f margin_improvement=%+.2f "
        "mqr_drift=%.4g lora_drift=%.4g",
        seed,
        condition_name,
        probe_changes["loss_improvement"],
        game_changes["mean_student_margin_improvement"],
        drift["mqr"]["all"]["delta_l2"],
        drift["lora"]["all"]["delta_l2"],
    )
    return {
        "seed": seed,
        "condition": condition_name,
        "spec": asdict(condition),
        "probe_target_histogram": dict(
            sorted(
                collections.Counter(
                    action_to_gtp(example.target_action, args.board_size)
                    for example in probes
                ).items()
            )
        ),
        "before_probe": before_probe,
        "after_probe": after_probe,
        "probe_changes": probe_changes,
        "before_games": before_games,
        "after_games": after_games,
        "game_changes": game_changes,
        "training": training,
        "parameter_drift": drift,
        "mqr_ogd_ranks": [memory.rank for memory in learner.gradient_memories],
        "lora_ogd_rank": lora_memory.rank,
        "no_learning_audit": no_learning_audit,
        "training_steps": records,
    }


def main() -> None:
    args = parse_args()
    positive = (
        args.train_games,
        args.eval_games,
        args.probe_positions,
        args.max_game_moves,
        args.max_length,
        args.eval_batch_size,
        args.sayuri_playouts,
        args.curve_every_games,
    )
    if min(positive) <= 0:
        raise ValueError("game, probe, length, batch, move, and playout counts must be positive")
    if min(args.train_opening_moves, args.eval_opening_moves, args.ogd_rank) < 0:
        raise ValueError("opening moves and OGD rank must be non-negative")
    if not (0.0 <= args.min_pass_occupancy <= 1.0):
        raise ValueError("min-pass-occupancy must be in [0, 1]")
    if not (0.0 <= args.pass_update_scale <= 1.0):
        raise ValueError("pass-update-scale must be in [0, 1]")
    if not (0.0 <= args.probe_pass_fraction <= 1.0):
        raise ValueError("probe-pass-fraction must be in [0, 1]")
    pass_probe_count = int(round(args.probe_positions * args.probe_pass_fraction))
    if pass_probe_count <= 0 or pass_probe_count >= args.probe_positions:
        raise ValueError(
            "probe split must contain at least one pass and one placement target"
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("conditions must be unique")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device(args.device)
    _seed_everything(args.seeds[0])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    binary = args.sayuri_binary
    weights = args.sayuri_weights
    if binary is None or weights is None:
        discovered_binary, discovered_weights = discover_sayuri_paths(ROOT)
        binary = binary or discovered_binary
        weights = weights or discovered_weights
    assert binary is not None and weights is not None

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
    runs: List[Dict[str, Any]] = []
    experiment_started = time.perf_counter()

    with SayuriGTPClient(
        binary,
        weights,
        board_size=args.board_size,
        komi=args.komi,
        threads=1,
        playouts=args.sayuri_playouts,
        scoring_rule="area",
        friendly_pass=True,
    ) as client:
        native_teacher = SayuriTeacher(client, mode="policy")
        if args.min_pass_occupancy > 0.0:
            training_teacher = PassGatedSayuriPolicyTeacher(
                client,
                min_pass_occupancy=args.min_pass_occupancy,
            )
        else:
            training_teacher = native_teacher
        if args.evaluation_teacher == "training":
            evaluation_teacher = training_teacher
        elif args.evaluation_teacher == "native-policy":
            evaluation_teacher = native_teacher
        else:
            evaluation_teacher = SayuriTeacher(client, mode="mcts")
        sayuri_metadata = {
            "binary": str(Path(binary).resolve()),
            "weights": str(Path(weights).resolve()),
            "weights_sha256": _file_sha256(Path(weights)),
            "engine_name": client.engine_name,
            "engine_version": client.engine_version,
            "teacher_mode": "raw-policy",
            "min_pass_occupancy": args.min_pass_occupancy,
            "pass_gate_enabled": args.min_pass_occupancy > 0.0,
            "evaluation_teacher": args.evaluation_teacher,
            "playouts": args.sayuri_playouts,
        }
        for seed in args.seeds:
            adapter_initial = _fresh_adapter_state(encoder, seed)
            probes = _generate_probe_examples(
                args.probe_positions,
                size=args.board_size,
                komi=args.komi,
                seed=seed + 10_000,
                teacher=evaluation_teacher,
                pass_fraction=args.probe_pass_fraction,
            )
            training_openings = _generate_openings(
                args.train_games,
                size=args.board_size,
                komi=args.komi,
                moves=args.train_opening_moves,
                seed=seed + 20_000,
            )
            evaluation_openings = _generate_openings(
                args.eval_games,
                size=args.board_size,
                komi=args.komi,
                moves=args.eval_opening_moves,
                seed=seed + 30_000,
            )
            for condition_name in args.conditions:
                runs.append(
                    _run_condition(
                        encoder,
                        training_teacher,
                        evaluation_teacher,
                        args=args,
                        device=device,
                        seed=seed,
                        condition_name=condition_name,
                        adapter_initial=adapter_initial,
                        probes=probes,
                        training_openings=training_openings,
                        evaluation_openings=evaluation_openings,
                    )
                )

    _sync(device)
    result = {
        "format": "mqr-minicpm-go-real-games-v2",
        "claim_boundary": (
            "This tests online imitation on learner-influenced 5x5 Go states. Raw pass, "
            "conditional placement, and legality metrics are separated. Legality of played "
            "student moves is externally masked and is not evidence that the raw model "
            "learned Go rules; territory margin remains a masked-policy outcome."
        ),
        "causal_protocol": (
            "preview theta_t -> commit actual action -> obtain/withhold Sayuri feedback from "
            "the saved pre-action state -> update to theta_(t+1)"
        ),
        "execution_mode": (
            "AWQ INT4 checkpoint dequantized once into a frozen "
            f"{str(encoder.execution_dtype).removeprefix('torch.').upper()} backbone; "
            "only FP32 LoRA and/or MQR sidecar parameters update by condition"
        ),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": {
            "path": str(Path(args.model_path).resolve()),
            "hidden_size": encoder.hidden_size,
            "frozen_parameters": encoder.frozen_parameter_count,
            "lora_parameters": encoder.trainable_parameter_count,
            "load_seconds": load_seconds,
        },
        "sayuri": sayuri_metadata,
        "runs": runs,
        "aggregate": _aggregate_runs(runs),
        "runtime_seconds": time.perf_counter() - experiment_started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote causal multi-game metrics to %s", args.metrics_path)


if __name__ == "__main__":
    main()
