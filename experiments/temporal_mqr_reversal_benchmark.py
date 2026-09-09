#!/usr/bin/env python3
"""Resource-capped reversal and cross-task-forgetting benchmark for MQR.

The stream is marker-free.  Every episode contains two warm-up copies of a
background digit followed by five shuffled candidates: four background copies
and one different (novel) digit.  Task A asks for the novel digit, Task B asks
for the repeated background digit, and Task A then returns.  The input does not
identify the task; only prediction-after-action feedback reveals which events
were worth retaining.

All methods receive the same projected Digits examples, phase order, labels,
and one current-example feedback transaction per episode.  The comparison is
resource-capped rather than falsely described as bit/FLOP-identical: methods
use 288--320 learned scalars and at most 4096 bytes of tensor-valued online
state, shadow tickets, OGD basis, or replay payload.  Exact measured resource
use and latency are written to the result file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.temporal_mqr_delayed_digits import prepare_digits  # noqa: E402
from mqr.online import OrthogonalGradientMemory  # noqa: E402
from mqr.utility import UtilityDrivenMQR  # noqa: E402


METHODS = (
    "mqr_utility",
    "mqr_utility_ogd",
    "mqr_context_utility_ogd",
    "mqr_expert_utility",
    "mqr_expert_utility_ogd",
    "gru",
    "fast_weight",
    "lora",
    "ogd_lora",
    "replay_lora",
)
TASK_BY_PHASE = {
    "task_a": "novel",
    "task_b_reversal": "repeat",
    "task_a_return": "novel",
}
ONLINE_PARAMETER_MIN = 288
ONLINE_PARAMETER_CAP = 320
ONLINE_TENSOR_BYTE_CAP = 4096
INPUT_DIM = 16
OUTPUT_DIM = 10
WARMUP_COUNT = 2
CANDIDATE_COUNT = 5


@dataclass(frozen=True)
class Episode:
    frames: torch.Tensor
    novel_label: int
    repeat_label: int
    novel_candidate_index: int
    novel_feature: torch.Tensor
    repeat_feature: torch.Tensor

    def target(self, task: str) -> int:
        if task == "novel":
            return self.novel_label
        if task == "repeat":
            return self.repeat_label
        raise ValueError(f"unknown task: {task}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29, 43, 71])
    parser.add_argument("--calibration-samples", type=int, default=400)
    parser.add_argument("--phase-episodes", type=int, default=160)
    parser.add_argument("--probe-episodes", type=int, default=80)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--adaptation-threshold", type=float, default=0.60)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--reversal-context",
        choices=("warmup-shift", "unobservable"),
        default="warmup-shift",
        help=(
            "warmup-shift makes the latent task identifiable through a changed "
            "same-class warm-up exemplar; unobservable is a pure reward-reversal "
            "impossibility control with identical observation distributions"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/results/temporal_mqr_reversal_matched.json"),
    )
    return parser.parse_args()


def _parameter_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _named_parameter_snapshot(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def _max_parameter_drift(
    before: Mapping[str, torch.Tensor], module: nn.Module
) -> float:
    return max(
        float((parameter.detach() - before[name]).abs().max().item())
        for name, parameter in module.named_parameters()
    )


def _project_features(features: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed) + 9_981)
    matrix = torch.randn(64, INPUT_DIM, generator=generator)
    basis = torch.linalg.qr(matrix, mode="reduced").Q
    projected = features @ basis
    return F.layer_norm(projected, (INPUT_DIM,))


def _make_episode(
    features: torch.Tensor,
    labels: torch.Tensor,
    rng: np.random.Generator,
    *,
    warmup_shift: bool,
) -> Episode:
    novel_index = int(rng.integers(0, len(features)))
    novel_label = int(labels[novel_index].item())
    eligible = torch.nonzero(labels != novel_label, as_tuple=False).squeeze(1).numpy()
    repeat_index = int(eligible[int(rng.integers(0, len(eligible)))])
    repeat_label = int(labels[repeat_index].item())
    candidate_index = int(rng.integers(0, CANDIDATE_COUNT))
    repeat = features[repeat_index]
    novel = features[novel_index]
    frames = repeat.repeat(WARMUP_COUNT + CANDIDATE_COUNT, 1)
    if warmup_shift:
        same_class = torch.nonzero(
            labels == repeat_label, as_tuple=False
        ).squeeze(1).numpy()
        same_class = same_class[same_class != repeat_index]
        candidates = features[torch.as_tensor(same_class, dtype=torch.long)]
        distances = 0.5 * (
            1.0
            - F.cosine_similarity(
                candidates,
                repeat.unsqueeze(0).expand_as(candidates),
                dim=1,
            )
        )
        shifted_index = int(same_class[int(torch.argmax(distances).item())])
        shifted = features[shifted_index]
        frames[1] = shifted
        # Keep the context statistically observable throughout the candidate
        # interval without adding a task bit: repeated-class exemplars alternate.
        for candidate_offset in range(CANDIDATE_COUNT):
            if candidate_offset != candidate_index and candidate_offset % 2 == 1:
                frames[WARMUP_COUNT + candidate_offset] = shifted
    frames[WARMUP_COUNT + candidate_index] = novel
    return Episode(
        frames=frames,
        novel_label=novel_label,
        repeat_label=repeat_label,
        novel_candidate_index=candidate_index,
        novel_feature=novel,
        repeat_feature=repeat,
    )


def _calibration_episode(feature: torch.Tensor, label: int) -> Episode:
    return Episode(
        frames=feature.repeat(WARMUP_COUNT + CANDIDATE_COUNT, 1),
        novel_label=int(label),
        repeat_label=int(label),
        novel_candidate_index=0,
        novel_feature=feature,
        repeat_feature=feature,
    )


def _episode_stream(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    count: int,
    seed: int,
    warmup_shift: bool = False,
) -> List[Episode]:
    rng = np.random.default_rng(int(seed))
    return [
        _make_episode(features, labels, rng, warmup_shift=warmup_shift)
        for _ in range(count)
    ]


def _stream_hash(episodes: Sequence[Episode]) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(episode.frames.contiguous().numpy().tobytes())
        digest.update(bytes((episode.novel_label, episode.repeat_label)))
        digest.update(bytes((episode.novel_candidate_index,)))
    return digest.hexdigest()


def _episode_summary(episode: Episode) -> torch.Tensor:
    warmup = episode.frames[:WARMUP_COUNT].mean(0)
    candidates = episode.frames[WARMUP_COUNT:]
    distances = torch.linalg.vector_norm(candidates - warmup, dim=1)
    novel = candidates[int(torch.argmax(distances).item())]
    candidate_mean = candidates.mean(0)
    return torch.cat((warmup, novel, candidate_mean), dim=0)


class GradientLearner(nn.Module):
    """Short-sequence SGD learner with optional preconditioned OGD."""

    def __init__(
        self,
        *,
        lr: float,
        max_update_norm: float,
        ogd_rank: int = 0,
    ) -> None:
        super().__init__()
        self.lr = float(lr)
        self.max_update_norm = float(max_update_norm)
        self.gradient_memory = OrthogonalGradientMemory(int(ogd_rank))

    def logits_for_episode(self, episode: Episode) -> torch.Tensor:
        raise NotImplementedError

    def online_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    @property
    def adaptive_parameter_count(self) -> int:
        return sum(parameter.numel() for _, parameter in self.online_named_parameters())

    @property
    def task_phase_mutable_parameter_count(self) -> int:
        return self.adaptive_parameter_count

    @property
    def base_tensor_state_bytes(self) -> int:
        return 0

    @property
    def peak_tensor_state_bytes(self) -> int:
        return self.base_tensor_state_bytes + self.gradient_memory.storage_bytes

    def _apply_loss(
        self,
        loss: torch.Tensor,
        *,
        protect: bool,
        remember: bool,
    ) -> Dict[str, float | int | bool]:
        named = self.online_named_parameters()
        gradients = torch.autograd.grad(
            loss,
            [parameter for _name, parameter in named],
            create_graph=False,
            retain_graph=False,
        )
        entries = [
            (name, gradient.detach(), self.lr)
            for (name, _parameter), gradient in zip(named, gradients)
        ]
        if protect and self.gradient_memory.rank > 0:
            projected, projection = self.gradient_memory.project_preconditioned(entries)
        else:
            projected = {name: gradient for name, gradient, _step in entries}
            raw = math.sqrt(
                self.lr
                * sum(float(gradient.square().sum().item()) for _, gradient, _ in entries)
            )
            projection = {
                "raw_norm": raw,
                "projected_norm": raw,
                "retained_norm": 1.0,
                "max_abs_overlap": 0.0,
            }
        memory_added = self.gradient_memory.observe(entries) if remember else False
        update_sq = sum(
            self.lr**2 * float(projected[name].square().sum().item())
            for name, _gradient, _step in entries
        )
        raw_update_norm = math.sqrt(update_sq)
        clip_scale = min(1.0, self.max_update_norm / (raw_update_norm + 1e-12))
        with torch.no_grad():
            for name, parameter in named:
                parameter.add_(projected[name], alpha=-self.lr * clip_scale)
        return {
            "update_norm": raw_update_norm * clip_scale,
            "ogd_rank": self.gradient_memory.rank,
            "ogd_retained_norm": float(projection["retained_norm"]),
            "ogd_max_abs_overlap": float(projection["max_abs_overlap"]),
            "ogd_memory_added": bool(memory_added),
        }

    def step(
        self,
        episode: Episode,
        target: int,
        *,
        learn: bool,
        protect: bool,
        remember: bool,
        rng: random.Random,
    ) -> Dict[str, Any]:
        del rng
        with torch.enable_grad():
            logits = self.logits_for_episode(episode)
            target_tensor = torch.tensor([int(target)], dtype=torch.long)
            loss = F.cross_entropy(logits, target_tensor)
            update = (
                self._apply_loss(loss, protect=protect, remember=remember)
                if learn
                else {
                    "update_norm": 0.0,
                    "ogd_rank": self.gradient_memory.rank,
                    "ogd_retained_norm": 1.0,
                    "ogd_max_abs_overlap": 0.0,
                    "ogd_memory_added": False,
                }
            )
        return {
            "logits": logits.detach().clone(),
            "loss": float(loss.detach().item()),
            "prediction_before_update": True,
            **update,
        }


class GRULearner(GradientLearner):
    def __init__(self, seed: int) -> None:
        torch.manual_seed(int(seed) + 201)
        super().__init__(lr=0.035, max_update_norm=0.10)
        self.cell = nn.GRUCell(INPUT_DIM, 4)
        self.readout = nn.Linear(4, OUTPUT_DIM)

    def logits_for_episode(self, episode: Episode) -> torch.Tensor:
        hidden = episode.frames.new_zeros((1, 4))
        for frame in episode.frames:
            hidden = self.cell(frame.unsqueeze(0), hidden)
        return self.readout(hidden)

    @property
    def base_tensor_state_bytes(self) -> int:
        return 4 * 4


class FastWeightLearner(GradientLearner):
    """Outer-product fast-weight memory with the same causal novelty signals."""

    def __init__(self, seed: int) -> None:
        torch.manual_seed(int(seed) + 301)
        super().__init__(lr=0.04, max_update_norm=0.10)
        self.key = nn.Linear(INPUT_DIM, 5, bias=False)
        self.value = nn.Linear(INPUT_DIM, 5, bias=False)
        self.query = nn.Parameter(torch.randn(5) / math.sqrt(5.0))
        self.gate_down = nn.Linear(INPUT_DIM + 2, 4, bias=False)
        self.gate_out = nn.Linear(4, 1, bias=True)
        self.readout = nn.Linear(5, OUTPUT_DIM)

    def logits_for_episode(self, episode: Episode) -> torch.Tensor:
        memory = episode.frames.new_zeros((5, 5))
        running: Optional[torch.Tensor] = None
        previous: Optional[torch.Tensor] = None
        for frame in episode.frames:
            if running is None:
                novelty = frame.new_zeros(())
            else:
                novelty = 0.5 * (
                    1.0
                    - F.cosine_similarity(frame.unsqueeze(0), running.unsqueeze(0))[0]
                )
            if previous is None:
                change = frame.new_zeros(())
            else:
                change = 0.5 * (
                    1.0
                    - F.cosine_similarity(frame.unsqueeze(0), previous.unsqueeze(0))[0]
                )
            gate_input = torch.cat((frame, novelty.reshape(1), change.reshape(1)))
            gate = torch.sigmoid(self.gate_out(torch.tanh(self.gate_down(gate_input))))
            key = torch.tanh(self.key(frame))
            value = torch.tanh(self.value(frame))
            memory = 0.90 * memory + gate * torch.outer(value, key)
            running = frame if running is None else 0.9 * running + 0.1 * frame
            previous = frame
        state = memory @ torch.tanh(self.query)
        return self.readout(state.unsqueeze(0))

    @property
    def base_tensor_state_bytes(self) -> int:
        # 5x5 fast matrix plus running mean and previous input.
        return (25 + 2 * INPUT_DIM) * 4


class LoRALearner(GradientLearner):
    def __init__(self, seed: int, *, ogd_rank: int = 0, replay_capacity: int = 0) -> None:
        torch.manual_seed(int(seed) + 401)
        super().__init__(lr=0.08, max_update_norm=0.10, ogd_rank=ogd_rank)
        self.register_buffer("base_weight", torch.zeros(OUTPUT_DIM, 3 * INPUT_DIM))
        self.lora_a = nn.Parameter(torch.empty(5, 3 * INPUT_DIM))
        self.lora_b = nn.Parameter(torch.zeros(OUTPUT_DIM, 5))
        self.bias = nn.Parameter(torch.zeros(OUTPUT_DIM))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5.0))
        self.replay_capacity = int(replay_capacity)
        self.replay_batch = min(5, self.replay_capacity)
        self._replay: List[Tuple[torch.Tensor, int]] = []
        self._examples_seen = 0

    def logits_for_summary(self, summary: torch.Tensor) -> torch.Tensor:
        base = F.linear(summary, self.base_weight)
        delta = F.linear(F.linear(summary, self.lora_a), self.lora_b)
        return base + delta + self.bias

    def logits_for_episode(self, episode: Episode) -> torch.Tensor:
        return self.logits_for_summary(_episode_summary(episode).unsqueeze(0))

    @property
    def base_tensor_state_bytes(self) -> int:
        summary_bytes = 3 * INPUT_DIM * 4
        replay_bytes = self.replay_capacity * (3 * INPUT_DIM * 4 + 8)
        return summary_bytes + replay_bytes

    def clear_replay(self) -> None:
        self._replay.clear()
        self._examples_seen = 0

    def _reservoir_add(self, summary: torch.Tensor, target: int, rng: random.Random) -> None:
        if self.replay_capacity <= 0:
            return
        self._examples_seen += 1
        item = (summary.detach().clone(), int(target))
        if len(self._replay) < self.replay_capacity:
            self._replay.append(item)
            return
        replacement = rng.randrange(self._examples_seen)
        if replacement < self.replay_capacity:
            self._replay[replacement] = item

    def step(
        self,
        episode: Episode,
        target: int,
        *,
        learn: bool,
        protect: bool,
        remember: bool,
        rng: random.Random,
    ) -> Dict[str, Any]:
        summary = _episode_summary(episode).unsqueeze(0)
        with torch.enable_grad():
            current_logits = self.logits_for_summary(summary)
            current_target = torch.tensor([int(target)], dtype=torch.long)
            current_loss = F.cross_entropy(current_logits, current_target)
            replayed = 0
            if learn and self.replay_capacity > 0 and self._replay:
                chosen = rng.sample(self._replay, min(self.replay_batch, len(self._replay)))
                batch_x = torch.cat([summary] + [item[0] for item in chosen], dim=0)
                batch_y = torch.tensor(
                    [int(target)] + [item[1] for item in chosen], dtype=torch.long
                )
                update_loss = F.cross_entropy(self.logits_for_summary(batch_x), batch_y)
                replayed = len(chosen)
            else:
                update_loss = current_loss
            update = (
                self._apply_loss(update_loss, protect=protect, remember=remember)
                if learn
                else {
                    "update_norm": 0.0,
                    "ogd_rank": self.gradient_memory.rank,
                    "ogd_retained_norm": 1.0,
                    "ogd_max_abs_overlap": 0.0,
                    "ogd_memory_added": False,
                }
            )
        if learn:
            self._reservoir_add(summary, target, rng)
        return {
            "logits": current_logits.detach().clone(),
            "loss": float(current_loss.detach().item()),
            "prediction_before_update": True,
            "replay_examples": replayed,
            **update,
        }


class MQRBenchmarkLearner(nn.Module):
    def __init__(
        self,
        seed: int,
        *,
        ogd_rank: int,
        context_trace: bool = False,
        context_experts: bool = False,
    ) -> None:
        super().__init__()
        self.context_trace = bool(context_trace)
        self.context_experts = bool(context_experts)
        if self.context_experts and not self.context_trace:
            raise ValueError("context experts require the causal context trace")
        torch.manual_seed(int(seed) + 101)
        self.runtime = UtilityDrivenMQR(
            INPUT_DIM,
            12,
            OUTPUT_DIM,
            core_kwargs={
                "leak_rates": (1.0, 0.02),
                "write_scales": (1.0, 1.0),
                "injection_rank": 12,
                "injection_activation": "tanh",
                "state_activation": "tanh",
                "transition_mode": "unistochastic",
                "learn_transitions": False,
                "readout_bias": False,
            },
            gate_rank=8,
            utility_lr=0.08,
            readout_lr=0.15,
            utility_ogd_max_rank=int(ogd_rank),
            readout_ogd_max_rank=0,
            utility_max_update_norm=0.04,
            readout_max_update_norm=0.25,
            memory_cost=0.0,
            advantage_scale=1.0,
            advantage_target_clip=4.0,
            decision_temperature=0.25,
            initial_advantage=-0.10,
            feature_ema_decay=0.9,
            context_volatility_decay=0.9 if self.context_trace else None,
            utility_experts=2 if self.context_experts else 1,
            utility_router_feature=(
                "context_volatility" if self.context_experts else None
            ),
            utility_router_threshold=0.30,
            max_pending_candidates=CANDIDATE_COUNT + 2,
            promotion_max_norm=4.0,
        )

    @property
    def adaptive_parameter_count(self) -> int:
        return sum(p.numel() for p in self.runtime.core.readout.parameters()) + sum(
            p.numel() for p in self.runtime.utility_gate.parameters()
        )

    @property
    def task_phase_mutable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.runtime.utility_gate.parameters())

    @property
    def base_tensor_state_bytes(self) -> int:
        feature_count = len(self.runtime.utility_feature_names)
        recurrent_and_features = (
            2 * 12 + 2 * INPUT_DIM + int(self.context_trace)
        ) * 4
        per_ticket = (feature_count + 2 * 12 + 2 * (2 * 12)) * 4
        return recurrent_and_features + CANDIDATE_COUNT * per_ticket

    @property
    def peak_tensor_state_bytes(self) -> int:
        return self.base_tensor_state_bytes + self.runtime.utility_gradient_memory.storage_bytes

    def calibrate(self, feature: torch.Tensor, target: int) -> Dict[str, Any]:
        self.runtime.reset_state()
        self.runtime.observe(
            feature.unsqueeze(0), issue_candidate=False, external_write=True
        )
        output = self.runtime.observe(
            torch.zeros_like(feature).unsqueeze(0),
            issue_candidate=False,
            external_write=False,
        )
        learned = self.runtime.learn_current(torch.tensor([int(target)]), learn=True)
        return {"logits": output["logits"], "loss": learned["loss"]}

    def step(
        self,
        episode: Episode,
        target: int,
        *,
        learn: bool,
        protect: bool,
        remember: bool,
        rng: random.Random,
    ) -> Dict[str, Any]:
        del rng
        self.runtime.reset_state()
        novel_writes: List[int] = []
        repeat_writes: List[int] = []
        novel_experts: List[int] = []
        repeat_experts: List[int] = []
        for index, frame in enumerate(episode.frames):
            candidate = index >= WARMUP_COUNT
            output = self.runtime.observe(
                frame.unsqueeze(0),
                issue_candidate=candidate,
                external_write=False if not candidate else None,
            )
            if candidate:
                wrote = int(bool(output["effective_write"]))
                candidate_index = index - WARMUP_COUNT
                if candidate_index == episode.novel_candidate_index:
                    novel_writes.append(wrote)
                    novel_experts.append(int(output["utility_expert"]))
                else:
                    repeat_writes.append(wrote)
                    repeat_experts.append(int(output["utility_expert"]))
        query = self.runtime.observe(
            torch.zeros(INPUT_DIM).unsqueeze(0),
            issue_candidate=False,
            external_write=False,
        )
        target_tensor = torch.tensor([int(target)], dtype=torch.long)
        loss = F.cross_entropy(query["logits"], target_tensor)
        update_norms: List[float] = []
        retained: List[float] = []
        max_overlap = 0.0
        added = False
        for offset, ticket_id in enumerate(self.runtime.pending_candidate_ids()):
            result = self.runtime.resolve_utility(
                ticket_id,
                target_tensor,
                learn=learn,
                remember_gradient=bool(remember and offset == 0),
                project_with_memory=protect,
                promote_missed_positive=False,
            )
            update_norms.append(float(result["update_norm"]))
            retained.append(float(result["ogd_retained_norm"]))
            max_overlap = max(max_overlap, float(result["ogd_max_abs_overlap"]))
            added = added or bool(result["ogd_memory_added"])
        return {
            "logits": query["logits"].detach().clone(),
            "loss": float(loss.detach().item()),
            "prediction_before_update": True,
            "update_norm": sum(update_norms),
            "ogd_rank": self.runtime.utility_gradient_memory.rank,
            "ogd_retained_norm": statistics.fmean(retained) if retained else 1.0,
            "ogd_max_abs_overlap": max_overlap,
            "ogd_memory_added": added,
            "novel_write_rate": statistics.fmean(novel_writes),
            "repeat_write_rate": statistics.fmean(repeat_writes),
            "novel_expert_one_rate": statistics.fmean(novel_experts),
            "repeat_expert_one_rate": statistics.fmean(repeat_experts),
            "unitary_error": self.runtime.core.max_unitary_error(),
            "stochastic_error": self.runtime.core.max_stochastic_error(),
        }


def _train_linear_oracle(
    features: torch.Tensor, labels: torch.Tensor
) -> nn.Linear:
    torch.manual_seed(12_345)
    model = nn.Linear(INPUT_DIM, OUTPUT_DIM)
    for feature, label in zip(features, labels):
        logits = model(feature.unsqueeze(0))
        loss = F.cross_entropy(logits, label.reshape(1))
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        with torch.no_grad():
            for parameter, gradient in zip(model.parameters(), gradients):
                parameter.add_(gradient, alpha=-0.10)
    model.requires_grad_(False)
    return model


def _oracle_loss(model: nn.Linear, episode: Episode, task: str) -> float:
    feature = episode.novel_feature if task == "novel" else episode.repeat_feature
    target = torch.tensor([episode.target(task)], dtype=torch.long)
    return float(F.cross_entropy(model(feature.unsqueeze(0)), target).item())


def _remember_schedule(
    episode_index: int, episode_count: int, current_rank: int, maximum_rank: int
) -> bool:
    if maximum_rank <= 0 or current_rank >= maximum_rank:
        return False
    period = max(1, episode_count // maximum_rank)
    return episode_index % period == 0


def _rolling_adaptation_delay(
    correct: Sequence[int], *, window: int, threshold: float
) -> Dict[str, Any]:
    if len(correct) < window:
        return {"episodes": len(correct) + 1, "censored": True}
    for end in range(window, len(correct) + 1):
        if statistics.fmean(correct[end - window : end]) >= threshold:
            return {"episodes": end, "censored": False}
    return {"episodes": len(correct) + 1, "censored": True}


def _run_phase(
    learner: nn.Module,
    episodes: Sequence[Episode],
    *,
    task: str,
    phase_name: str,
    oracle: nn.Linear,
    rolling_window: int,
    threshold: float,
    rng_seed: int,
) -> Dict[str, Any]:
    correct: List[int] = []
    losses: List[float] = []
    oracle_losses: List[float] = []
    update_norms: List[float] = []
    retained: List[float] = []
    overlaps: List[float] = []
    novel_writes: List[float] = []
    repeat_writes: List[float] = []
    novel_experts: List[float] = []
    repeat_experts: List[float] = []
    replay_examples = 0
    started = time.perf_counter()
    rng = random.Random(int(rng_seed))
    if isinstance(learner, MQRBenchmarkLearner):
        memory = learner.runtime.utility_gradient_memory
    else:
        assert isinstance(learner, GradientLearner)
        memory = learner.gradient_memory
    protect = phase_name != "task_a" and memory.max_rank > 0
    for episode_index, episode in enumerate(episodes):
        remember = bool(
            phase_name == "task_a"
            and _remember_schedule(
                episode_index, len(episodes), memory.rank, memory.max_rank
            )
        )
        target = episode.target(task)
        output = learner.step(  # type: ignore[attr-defined]
            episode,
            target,
            learn=True,
            protect=protect,
            remember=remember,
            rng=rng,
        )
        prediction = int(output["logits"].argmax(1).item())
        correct.append(int(prediction == target))
        losses.append(float(output["loss"]))
        oracle_losses.append(_oracle_loss(oracle, episode, task))
        update_norms.append(float(output["update_norm"]))
        retained.append(float(output["ogd_retained_norm"]))
        overlaps.append(float(output["ogd_max_abs_overlap"]))
        replay_examples += int(output.get("replay_examples", 0))
        if "novel_write_rate" in output:
            novel_writes.append(float(output["novel_write_rate"]))
            repeat_writes.append(float(output["repeat_write_rate"]))
            novel_experts.append(float(output["novel_expert_one_rate"]))
            repeat_experts.append(float(output["repeat_expert_one_rate"]))
    elapsed = time.perf_counter() - started
    quarter = max(1, len(correct) // 4)
    blocks = np.array_split(np.asarray(correct, dtype=np.float64), 8)
    result: Dict[str, Any] = {
        "episodes": len(correct),
        "accuracy": statistics.fmean(correct),
        "first_quarter_accuracy": statistics.fmean(correct[:quarter]),
        "last_quarter_accuracy": statistics.fmean(correct[-quarter:]),
        "accuracy_gain_first_to_last": (
            statistics.fmean(correct[-quarter:]) - statistics.fmean(correct[:quarter])
        ),
        "accuracy_eighths": [float(block.mean()) for block in blocks],
        "mean_loss": statistics.fmean(losses),
        "first_quarter_loss": statistics.fmean(losses[:quarter]),
        "last_quarter_loss": statistics.fmean(losses[-quarter:]),
        "cumulative_excess_nll": sum(
            value - reference for value, reference in zip(losses, oracle_losses)
        ),
        "adaptation_delay": _rolling_adaptation_delay(
            correct, window=rolling_window, threshold=threshold
        ),
        "mean_update_norm": statistics.fmean(update_norms),
        "mean_ogd_retained_norm": statistics.fmean(retained),
        "max_ogd_overlap": max(overlaps),
        "ogd_rank_after_phase": memory.rank,
        "replay_feedback_exposures": replay_examples,
        "milliseconds_per_episode": 1000.0 * elapsed / len(correct),
    }
    if novel_writes:
        result["novel_write_rate"] = statistics.fmean(novel_writes)
        result["repeat_write_rate"] = statistics.fmean(repeat_writes)
        result["novel_expert_one_rate"] = statistics.fmean(novel_experts)
        result["repeat_expert_one_rate"] = statistics.fmean(repeat_experts)
    return result


def _evaluate(
    learner: nn.Module,
    episodes: Sequence[Episode],
    *,
    task: str,
    oracle: nn.Linear,
    rng_seed: int,
) -> Dict[str, Any]:
    runtime_snapshot = copy.deepcopy(learner.state_dict())
    parameter_snapshot = _named_parameter_snapshot(learner)
    correct: List[int] = []
    losses: List[float] = []
    oracle_losses: List[float] = []
    novel_writes: List[float] = []
    repeat_writes: List[float] = []
    novel_experts: List[float] = []
    repeat_experts: List[float] = []
    rng = random.Random(int(rng_seed))
    for episode in episodes:
        target = episode.target(task)
        output = learner.step(  # type: ignore[attr-defined]
            episode,
            target,
            learn=False,
            protect=False,
            remember=False,
            rng=rng,
        )
        correct.append(int(output["logits"].argmax(1).item() == target))
        losses.append(float(output["loss"]))
        oracle_losses.append(_oracle_loss(oracle, episode, task))
        if "novel_write_rate" in output:
            novel_writes.append(float(output["novel_write_rate"]))
            repeat_writes.append(float(output["repeat_write_rate"]))
            novel_experts.append(float(output["novel_expert_one_rate"]))
            repeat_experts.append(float(output["repeat_expert_one_rate"]))
    parameter_drift = _max_parameter_drift(parameter_snapshot, learner)
    learner.load_state_dict(runtime_snapshot)
    result: Dict[str, Any] = {
        "episodes": len(correct),
        "accuracy": statistics.fmean(correct),
        "mean_loss": statistics.fmean(losses),
        "mean_excess_nll": statistics.fmean(
            value - reference for value, reference in zip(losses, oracle_losses)
        ),
        "parameter_max_drift": parameter_drift,
    }
    if novel_writes:
        result["novel_write_rate"] = statistics.fmean(novel_writes)
        result["repeat_write_rate"] = statistics.fmean(repeat_writes)
        result["novel_expert_one_rate"] = statistics.fmean(novel_experts)
        result["repeat_expert_one_rate"] = statistics.fmean(repeat_experts)
    return result


def _calibrate_gradient_model(
    learner: GradientLearner,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
) -> Dict[str, float]:
    correct: List[int] = []
    losses: List[float] = []
    rng = random.Random(int(seed))
    for feature, label in zip(features, labels):
        episode = _calibration_episode(feature, int(label.item()))
        output = learner.step(
            episode,
            int(label.item()),
            learn=True,
            protect=False,
            remember=False,
            rng=rng,
        )
        correct.append(int(output["logits"].argmax(1).item() == int(label.item())))
        losses.append(float(output["loss"]))
    learner.gradient_memory.clear(reset_layout=True)
    if isinstance(learner, LoRALearner):
        learner.clear_replay()
    quarter = max(1, len(correct) // 4)
    return {
        "prequential_accuracy": statistics.fmean(correct),
        "last_quarter_accuracy": statistics.fmean(correct[-quarter:]),
        "mean_loss": statistics.fmean(losses),
    }


def _calibrate_mqr(
    learner: MQRBenchmarkLearner,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    correct: List[int] = []
    losses: List[float] = []
    for feature, label in zip(features, labels):
        output = learner.calibrate(feature, int(label.item()))
        correct.append(int(output["logits"].argmax(1).item() == int(label.item())))
        losses.append(float(output["loss"]))
    learner.runtime.reset_all_states()
    quarter = max(1, len(correct) // 4)
    return {
        "prequential_accuracy": statistics.fmean(correct),
        "last_quarter_accuracy": statistics.fmean(correct[-quarter:]),
        "mean_loss": statistics.fmean(losses),
    }


def _copy_named_parameters(source: nn.Module, target: nn.Module) -> None:
    source_items = dict(source.named_parameters())
    target_items = dict(target.named_parameters())
    if source_items.keys() != target_items.keys():
        raise RuntimeError("paired learners do not share a parameter layout")
    with torch.no_grad():
        for name, parameter in target_items.items():
            parameter.copy_(source_items[name])


def _resource_record(learner: nn.Module) -> Dict[str, Any]:
    adaptive = int(learner.adaptive_parameter_count)  # type: ignore[attr-defined]
    task_mutable = int(learner.task_phase_mutable_parameter_count)  # type: ignore[attr-defined]
    peak_bytes = int(learner.peak_tensor_state_bytes)  # type: ignore[attr-defined]
    return {
        "adaptive_parameters_including_calibration": adaptive,
        "task_phase_mutable_parameters": task_mutable,
        "total_sidecar_parameters": sum(p.numel() for p in learner.parameters()),
        "peak_tensor_online_state_bytes": peak_bytes,
        "parameter_cap_satisfied": ONLINE_PARAMETER_MIN <= adaptive <= ONLINE_PARAMETER_CAP,
        "tensor_byte_cap_satisfied": peak_bytes <= ONLINE_TENSOR_BYTE_CAP,
    }


def _make_learners(
    seed: int,
    calibration_x: torch.Tensor,
    calibration_y: torch.Tensor,
) -> Tuple[Dict[str, nn.Module], Dict[str, Dict[str, float]], Dict[str, bool]]:
    mqr_plain = MQRBenchmarkLearner(seed, ogd_rank=0)
    mqr_calibration = _calibrate_mqr(mqr_plain, calibration_x, calibration_y)
    mqr_ogd = MQRBenchmarkLearner(seed, ogd_rank=11)
    _copy_named_parameters(mqr_plain, mqr_ogd)
    mqr_context_ogd = MQRBenchmarkLearner(seed, ogd_rank=10, context_trace=True)
    mqr_context_ogd.runtime.core.load_state_dict(
        copy.deepcopy(mqr_plain.runtime.core.state_dict())
    )
    mqr_expert = MQRBenchmarkLearner(
        seed, ogd_rank=0, context_trace=True, context_experts=True
    )
    mqr_expert.runtime.core.load_state_dict(
        copy.deepcopy(mqr_plain.runtime.core.state_dict())
    )
    mqr_expert_ogd = MQRBenchmarkLearner(
        seed,
        ogd_rank=9,
        context_trace=True,
        context_experts=True,
    )
    _copy_named_parameters(mqr_expert, mqr_expert_ogd)

    gru = GRULearner(seed)
    fast = FastWeightLearner(seed)
    lora_base = LoRALearner(seed)
    calibration = {
        "mqr_utility": mqr_calibration,
        "mqr_utility_ogd": dict(mqr_calibration),
        "mqr_context_utility_ogd": dict(mqr_calibration),
        "mqr_expert_utility": dict(mqr_calibration),
        "mqr_expert_utility_ogd": dict(mqr_calibration),
        "gru": _calibrate_gradient_model(gru, calibration_x, calibration_y, seed=seed + 1),
        "fast_weight": _calibrate_gradient_model(
            fast, calibration_x, calibration_y, seed=seed + 2
        ),
        "lora": _calibrate_gradient_model(
            lora_base, calibration_x, calibration_y, seed=seed + 3
        ),
    }
    ogd_lora = LoRALearner(seed, ogd_rank=3)
    replay_lora = LoRALearner(seed, replay_capacity=19)
    _copy_named_parameters(lora_base, ogd_lora)
    _copy_named_parameters(lora_base, replay_lora)
    calibration["ogd_lora"] = dict(calibration["lora"])
    calibration["replay_lora"] = dict(calibration["lora"])
    learners: Dict[str, nn.Module] = {
        "mqr_utility": mqr_plain,
        "mqr_utility_ogd": mqr_ogd,
        "mqr_context_utility_ogd": mqr_context_ogd,
        "mqr_expert_utility": mqr_expert,
        "mqr_expert_utility_ogd": mqr_expert_ogd,
        "gru": gru,
        "fast_weight": fast,
        "lora": lora_base,
        "ogd_lora": ogd_lora,
        "replay_lora": replay_lora,
    }
    audits = {
        "mqr_pair_matched_after_calibration": (
            _parameter_hash(mqr_plain) == _parameter_hash(mqr_ogd)
        ),
        "mqr_expert_pair_matched_after_calibration": (
            _parameter_hash(mqr_expert) == _parameter_hash(mqr_expert_ogd)
        ),
        "lora_triplet_matched_after_calibration": len(
            {
                _parameter_hash(lora_base),
                _parameter_hash(ogd_lora),
                _parameter_hash(replay_lora),
            }
        )
        == 1,
    }
    return learners, calibration, audits


def _derived_metrics(
    boundaries: Mapping[str, Mapping[str, Any]],
    phases: Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    a_after_a = float(boundaries["after_task_a"]["novel"]["accuracy"])
    a_after_b = float(boundaries["after_task_b_reversal"]["novel"]["accuracy"])
    b_after_b = float(boundaries["after_task_b_reversal"]["repeat"]["accuracy"])
    b_after_return = float(boundaries["after_task_a_return"]["repeat"]["accuracy"])
    a_final = float(boundaries["after_task_a_return"]["novel"]["accuracy"])
    initial_delay = float(phases["task_a"]["adaptation_delay"]["episodes"])
    return_delay = float(phases["task_a_return"]["adaptation_delay"]["episodes"])
    return {
        "task_a_probe_drop_after_b": a_after_a - a_after_b,
        "task_b_probe_drop_after_a_return": b_after_b - b_after_return,
        "task_a_equal_training_forgetting": max(0.0, a_after_a - a_final),
        "task_a_return_recovery": a_final - a_after_b,
        "task_a_return_memory_savings_episodes": initial_delay - return_delay,
        "mean_post_task_accuracy": statistics.fmean((a_after_a, b_after_b, a_final)),
        "final_balanced_accuracy": statistics.fmean((a_final, b_after_return)),
    }


def _run_seed(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    data = prepare_digits(seed, train_samples=1_000, eval_samples=500)
    train_x, train_y = data["train"]
    eval_x, eval_y = data["eval"]
    train_x = _project_features(train_x, seed)
    eval_x = _project_features(eval_x, seed)
    calibration_x = train_x[: args.calibration_samples]
    calibration_y = train_y[: args.calibration_samples]
    oracle = _train_linear_oracle(calibration_x, calibration_y)

    contextual = args.reversal_context == "warmup-shift"
    phase_streams = {
        "task_a": _episode_stream(
            train_x, train_y, count=args.phase_episodes, seed=seed + 10_000
        ),
        "task_b_reversal": _episode_stream(
            train_x,
            train_y,
            count=args.phase_episodes,
            seed=seed + 20_000,
            warmup_shift=contextual,
        ),
        "task_a_return": _episode_stream(
            train_x, train_y, count=args.phase_episodes, seed=seed + 30_000
        ),
    }
    probes = {
        "novel": _episode_stream(
            eval_x, eval_y, count=args.probe_episodes, seed=seed + 40_000
        ),
        "repeat": _episode_stream(
            eval_x,
            eval_y,
            count=args.probe_episodes,
            seed=seed + 45_000,
            warmup_shift=contextual,
        ),
    }
    learners, calibration, initialization_audit = _make_learners(
        seed, calibration_x, calibration_y
    )
    runs: List[Dict[str, Any]] = []
    for method in METHODS:
        learner = learners[method]
        boundaries: Dict[str, Dict[str, Any]] = {
            "after_calibration": {
                task: _evaluate(
                    learner,
                    probes[task],
                    task=task,
                    oracle=oracle,
                    rng_seed=seed + 50_000,
                )
                for task in ("novel", "repeat")
            }
        }
        phases: Dict[str, Any] = {}
        expert_after_a: Optional[Dict[str, torch.Tensor]] = None
        expert_isolation: Optional[Dict[str, float | bool]] = None
        for phase_index, (phase_name, task) in enumerate(TASK_BY_PHASE.items()):
            phases[phase_name] = _run_phase(
                learner,
                phase_streams[phase_name],
                task=task,
                phase_name=phase_name,
                oracle=oracle,
                rolling_window=args.rolling_window,
                threshold=args.adaptation_threshold,
                rng_seed=seed + 60_000 + phase_index,
            )
            if isinstance(learner, MQRBenchmarkLearner) and learner.context_experts:
                if phase_name == "task_a":
                    expert_after_a = _named_parameter_snapshot(
                        learner.runtime.utility_gate
                    )
                elif phase_name == "task_b_reversal":
                    assert expert_after_a is not None
                    expert_zero_drift = max(
                        float((parameter.detach() - expert_after_a[name]).abs().max().item())
                        for name, parameter in learner.runtime.utility_gate.named_parameters()
                        if name.startswith("experts.0.")
                    )
                    expert_one_drift = max(
                        float((parameter.detach() - expert_after_a[name]).abs().max().item())
                        for name, parameter in learner.runtime.utility_gate.named_parameters()
                        if name.startswith("experts.1.")
                    )
                    expert_isolation = {
                        "expert_zero_max_abs_drift_during_task_b": expert_zero_drift,
                        "expert_one_max_abs_drift_during_task_b": expert_one_drift,
                        "exact_old_expert_parameter_isolation": expert_zero_drift == 0.0,
                        "new_expert_did_learn": expert_one_drift > 0.0,
                    }
            boundaries[f"after_{phase_name}"] = {
                probe_task: _evaluate(
                    learner,
                    probes[probe_task],
                    task=probe_task,
                    oracle=oracle,
                    rng_seed=seed + 70_000 + phase_index,
                )
                for probe_task in ("novel", "repeat")
            }
        resource = _resource_record(learner)
        if not resource["parameter_cap_satisfied"]:
            raise RuntimeError(f"{method} violates adaptive parameter cap: {resource}")
        if not resource["tensor_byte_cap_satisfied"]:
            raise RuntimeError(f"{method} violates online tensor byte cap: {resource}")
        runs.append(
            {
                "seed": int(seed),
                "method": method,
                "calibration": calibration[method],
                "resource": resource,
                "phases": phases,
                "boundaries": boundaries,
                "derived": _derived_metrics(boundaries, phases),
                "expert_isolation": expert_isolation,
                "final_parameter_sha256": _parameter_hash(learner),
                "max_unitary_error": (
                    learner.runtime.core.max_unitary_error()
                    if isinstance(learner, MQRBenchmarkLearner)
                    else None
                ),
                "max_stochastic_error": (
                    learner.runtime.core.max_stochastic_error()
                    if isinstance(learner, MQRBenchmarkLearner)
                    else None
                ),
            }
        )
        print(
            f"seed={seed:3d} {method:17s} "
            f"A={boundaries['after_task_a']['novel']['accuracy']:.3f} "
            f"A|B={boundaries['after_task_b_reversal']['novel']['accuracy']:.3f} "
            f"B={boundaries['after_task_b_reversal']['repeat']['accuracy']:.3f} "
            f"A2={boundaries['after_task_a_return']['novel']['accuracy']:.3f}"
        )
    return {
        "seed": int(seed),
        "stream_sha256": {
            name: _stream_hash(stream) for name, stream in phase_streams.items()
        }
        | {
            "probe_novel": _stream_hash(probes["novel"]),
            "probe_repeat": _stream_hash(probes["repeat"]),
        },
        "initialization_audit": initialization_audit,
        "runs": runs,
    }


def _mean_sd_ci(values: Sequence[float]) -> Dict[str, Any]:
    samples = [float(value) for value in values]
    mean = statistics.fmean(samples)
    sd = statistics.stdev(samples) if len(samples) > 1 else 0.0
    half = 0.0 if len(samples) <= 1 else 1.96 * sd / math.sqrt(len(samples))
    return {
        "n": len(samples),
        "mean": mean,
        "sample_sd": sd,
        "normal_95ci": [mean - half, mean + half],
        "values": samples,
    }


def _bootstrap_ci(
    values: Sequence[float], *, resamples: int, seed: int
) -> List[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        return [float(array[0]), float(array[0])]
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(resamples), array.size))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _aggregate(
    runs: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int
) -> Dict[str, Any]:
    by_method: Dict[str, Any] = {}
    for method in METHODS:
        selected = [run for run in runs if run["method"] == method]
        metrics: Dict[str, Sequence[float]] = {
            "task_a_prequential_last_quarter": [
                run["phases"]["task_a"]["last_quarter_accuracy"] for run in selected
            ],
            "task_b_reversal_first_quarter": [
                run["phases"]["task_b_reversal"]["first_quarter_accuracy"]
                for run in selected
            ],
            "task_b_reversal_last_quarter": [
                run["phases"]["task_b_reversal"]["last_quarter_accuracy"]
                for run in selected
            ],
            "task_a_return_first_quarter": [
                run["phases"]["task_a_return"]["first_quarter_accuracy"]
                for run in selected
            ],
            "task_a_return_last_quarter": [
                run["phases"]["task_a_return"]["last_quarter_accuracy"]
                for run in selected
            ],
            "task_a_probe_drop_after_b": [
                run["derived"]["task_a_probe_drop_after_b"] for run in selected
            ],
            "task_b_probe_drop_after_a_return": [
                run["derived"]["task_b_probe_drop_after_a_return"] for run in selected
            ],
            "task_a_equal_training_forgetting": [
                run["derived"]["task_a_equal_training_forgetting"] for run in selected
            ],
            "task_a_return_recovery": [
                run["derived"]["task_a_return_recovery"] for run in selected
            ],
            "task_a_return_memory_savings_episodes": [
                run["derived"]["task_a_return_memory_savings_episodes"]
                for run in selected
            ],
            "mean_post_task_accuracy": [
                run["derived"]["mean_post_task_accuracy"] for run in selected
            ],
            "final_balanced_accuracy": [
                run["derived"]["final_balanced_accuracy"] for run in selected
            ],
            "milliseconds_per_episode": [
                statistics.fmean(
                    run["phases"][phase]["milliseconds_per_episode"]
                    for phase in TASK_BY_PHASE
                )
                for run in selected
            ],
        }
        by_method[method] = {
            "resource": selected[0]["resource"],
            **{name: _mean_sd_ci(values) for name, values in metrics.items()},
        }

    indexed = {
        (int(run["seed"]), str(run["method"])): run
        for run in runs
    }
    seeds = sorted({int(run["seed"]) for run in runs})
    contrasts: Dict[str, Any] = {}
    for baseline in METHODS:
        if baseline == "mqr_expert_utility_ogd":
            continue
        accuracy_delta = [
            float(indexed[(seed, "mqr_expert_utility_ogd")]["derived"]["mean_post_task_accuracy"])
            - float(indexed[(seed, baseline)]["derived"]["mean_post_task_accuracy"])
            for seed in seeds
        ]
        probe_drop_reduction = [
            float(indexed[(seed, baseline)]["derived"]["task_a_probe_drop_after_b"])
            - float(indexed[(seed, "mqr_expert_utility_ogd")]["derived"]["task_a_probe_drop_after_b"])
            for seed in seeds
        ]
        equal_training_forgetting_reduction = [
            float(
                indexed[(seed, baseline)]["derived"][
                    "task_a_equal_training_forgetting"
                ]
            )
            - float(
                indexed[(seed, "mqr_expert_utility_ogd")]["derived"][
                    "task_a_equal_training_forgetting"
                ]
            )
            for seed in seeds
        ]
        contrasts[f"mqr_expert_utility_ogd_vs_{baseline}"] = {
            "mean_post_task_accuracy_delta": {
                **_mean_sd_ci(accuracy_delta),
                "paired_bootstrap_95ci": _bootstrap_ci(
                    accuracy_delta,
                    resamples=bootstrap_resamples,
                    seed=91_000 + len(contrasts),
                ),
                "positive_favors": "mqr_expert_utility_ogd",
            },
            "task_a_probe_drop_reduction": {
                **_mean_sd_ci(probe_drop_reduction),
                "paired_bootstrap_95ci": _bootstrap_ci(
                    probe_drop_reduction,
                    resamples=bootstrap_resamples,
                    seed=92_000 + len(contrasts),
                ),
                "positive_favors": "mqr_expert_utility_ogd",
            },
            "task_a_equal_training_forgetting_reduction": {
                **_mean_sd_ci(equal_training_forgetting_reduction),
                "paired_bootstrap_95ci": _bootstrap_ci(
                    equal_training_forgetting_reduction,
                    resamples=bootstrap_resamples,
                    seed=93_000 + len(contrasts),
                ),
                "positive_favors": "mqr_expert_utility_ogd",
            },
        }
    return {"by_method": by_method, "paired_contrasts": contrasts}


def _claim_gates(
    aggregate: Mapping[str, Any], *, cross_task_forgetting_interpretable: bool
) -> Dict[str, Any]:
    methods = aggregate["by_method"]
    mqr = methods["mqr_expert_utility_ogd"]
    baseline_names = ("gru", "fast_weight", "lora", "ogd_lora", "replay_lora")
    best = max(
        baseline_names,
        key=lambda name: float(methods[name]["mean_post_task_accuracy"]["mean"]),
    )
    contrast = aggregate["paired_contrasts"][
        f"mqr_expert_utility_ogd_vs_{best}"
    ]
    accuracy_ci = contrast["mean_post_task_accuracy_delta"]["paired_bootstrap_95ci"]
    forgetting_metric = (
        "task_a_probe_drop_reduction"
        if cross_task_forgetting_interpretable
        else "task_a_equal_training_forgetting_reduction"
    )
    forgetting_ci = contrast[forgetting_metric]["paired_bootstrap_95ci"]
    reversal_gain = (
        float(mqr["task_b_reversal_last_quarter"]["mean"])
        - float(mqr["task_b_reversal_first_quarter"]["mean"])
    )
    return {
        "best_baseline_by_mean_post_task_accuracy": best,
        "cross_task_forgetting_interpretable": cross_task_forgetting_interpretable,
        "forgetting_decision_metric": forgetting_metric,
        "mqr_reversal_learning_positive_mean": reversal_gain > 0.0,
        "mqr_noninferior_to_best_baseline_at_5pp_margin": float(accuracy_ci[0]) > -0.05,
        "mqr_strict_accuracy_superiority_to_best_baseline": float(accuracy_ci[0]) > 0.0,
        "mqr_strict_forgetting_reduction_vs_best_baseline": float(forgetting_ci[0]) > 0.0,
        "independent_competitive_advantage_established": bool(
            float(accuracy_ci[0]) > 0.0 and float(forgetting_ci[0]) > 0.0
        ),
        "decision_rule": (
            "Independent advantage requires paired-bootstrap lower bounds above zero "
            "for both mean post-task accuracy and the declared Task-A forgetting "
            "reduction metric versus "
            "the strongest baseline."
        ),
    }


def main() -> None:
    args = parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be non-empty and unique")
    if min(args.calibration_samples, args.phase_episodes, args.probe_episodes) <= 0:
        raise ValueError("sample counts must be positive")
    if args.rolling_window <= 0 or args.rolling_window > args.phase_episodes:
        raise ValueError("rolling-window must be in [1, phase-episodes]")
    if not (0.0 < args.adaptation_threshold <= 1.0):
        raise ValueError("adaptation-threshold must be in (0, 1]")
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap-resamples must be positive")
    torch.set_num_threads(1)
    started = time.perf_counter()
    seed_records = [_run_seed(args, seed) for seed in args.seeds]
    runs = [run for seed_record in seed_records for run in seed_record["runs"]]
    aggregate = _aggregate(runs, bootstrap_resamples=args.bootstrap_resamples)
    result = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {
            "dataset": "sklearn.datasets.load_digits (offline bundled dataset)",
            "stream": (
                "A(novel digit) -> B(repeated background; reward reversal) -> "
                "A return, with no task marker in the input"
            ),
            "feedback_order": "predict with theta_t, reveal target, then update theta_(t+1)",
            "shared_sample_order": True,
            "fixed_heldout_probes": True,
            "reversal_context": args.reversal_context,
            "cross_task_forgetting_interpretable": (
                args.reversal_context == "warmup-shift"
            ),
            "context_identifiability": (
                "Task B replaces the second warm-up image with another exemplar of the "
                "same repeated digit; this is a causal distributional cue, not a task ID."
                if args.reversal_context == "warmup-shift"
                else "A and B have identical observation distributions, so an A probe "
                "immediately after B measures interference, not identifiable task recall."
            ),
            "resource_matching": {
                "adaptive_parameter_range": [ONLINE_PARAMETER_MIN, ONLINE_PARAMETER_CAP],
                "tensor_online_state_cap_bytes": ONLINE_TENSOR_BYTE_CAP,
                "includes": [
                    "recurrent state and causal feature history",
                    "MQR shadow tickets and dense OGD basis",
                    "LoRA OGD basis or replay summaries and labels",
                ],
                "excludes": [
                    "Python/container metadata",
                    "shared input episode and frozen Digits data",
                    "temporary autograd activations",
                ],
                "compute_boundary": (
                    "same current feedback count and one update transaction per episode; "
                    "operations are not padded to equality and measured latency is reported"
                ),
            },
            "ogd_staging": (
                "collect old-task directions during A without projection; protect during "
                "B and A-return"
            ),
            "replay": (
                "19-summary reservoir fits the same 4096-byte cap; at most five old "
                "examples join each one-transaction update"
            ),
            "claim_boundary": (
                "A controlled hidden-task reversal test can falsify MQR competitiveness; "
                "it is not evidence of language-model or animal-level learning."
            ),
        },
        "configuration": {
            **vars(args),
            "output": str(args.output),
            "methods": list(METHODS),
            "input_dim": INPUT_DIM,
            "candidate_count": CANDIDATE_COUNT,
            "warmup_count": WARMUP_COUNT,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "seed_audits": [
            {
                "seed": record["seed"],
                "stream_sha256": record["stream_sha256"],
                "initialization_audit": record["initialization_audit"],
            }
            for record in seed_records
        ],
        "aggregate": aggregate,
        "claim_gates": _claim_gates(
            aggregate,
            cross_task_forgetting_interpretable=(
                args.reversal_context == "warmup-shift"
            ),
        ),
        "runs": runs,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
