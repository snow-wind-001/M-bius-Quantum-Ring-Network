#!/usr/bin/env python3
"""Causal learned-write-gate validation for interference-rich Digits streams.

Each sequence is ``cue -> real digit distractors -> zero query``.  A non-label
event marker is one only on the cue.  The controller must choose the current
write before its auxiliary event target is submitted; the digit label arrives
only after the zero-query prediction.  All held-out evaluation freezes both
the readout and controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import sklearn
import torch
import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.temporal_mqr_delayed_digits import prepare_digits  # noqa: E402
from experiments.temporal_mqr_interference_digits import (  # noqa: E402
    balanced_count_order,
)
from mqr import OnlineTemporalMQRClassifier  # noqa: E402


VARIANTS = (
    "multiscale_ungated",
    "multiscale_oracle_gate",
    "multiscale_learned_gate",
    "multiscale_learned_no_gate_updates",
    "multiscale_oracle_false_open_10",
    "multiscale_oracle_false_open_25",
    "multiscale_oracle_false_close_10",
    "single_slow_learned_gate",
)

MULTISCALE_GATE_LEARNING_VARIANTS = (
    "multiscale_ungated",
    "multiscale_oracle_gate",
    "multiscale_learned_gate",
    "multiscale_oracle_false_open_10",
    "multiscale_oracle_false_open_25",
    "multiscale_oracle_false_close_10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument("--eval-samples", type=int, default=240)
    parser.add_argument(
        "--distractor-counts",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
    )
    parser.add_argument("--total-state-dim", type=int, default=48)
    parser.add_argument("--injection-rank", type=int, default=24)
    parser.add_argument("--gate-rank", type=int, default=8)
    parser.add_argument("--readout-lr", type=float, default=0.15)
    parser.add_argument("--gate-lr", type=float, default=0.05)
    parser.add_argument("--gate-positive-weight", type=float, default=8.25)
    parser.add_argument("--max-update-norm", type=float, default=0.25)
    parser.add_argument("--gate-max-update-norm", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/results/temporal_mqr_learned_gate_digits.json"
        ),
    )
    return parser.parse_args()


def make_marked_sequence(
    cue: torch.Tensor,
    distractors: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return frames ``[count+2, 65]`` and causal event targets ``[count+2]``."""

    if cue.shape != (64,):
        raise ValueError("cue must have shape [64]")
    if distractors.dim() != 2 or distractors.size(1) != 64:
        raise ValueError("distractors must have shape [count, 64]")
    frames = torch.zeros(
        distractors.size(0) + 2,
        65,
        dtype=cue.dtype,
        device=cue.device,
    )
    frames[0, :64] = cue
    frames[0, 64] = 1.0
    if distractors.numel() > 0:
        frames[1:-1, :64] = distractors.to(
            device=cue.device,
            dtype=cue.dtype,
        )
    event_targets = torch.zeros(frames.size(0), dtype=cue.dtype, device=cue.device)
    event_targets[0] = 1.0
    return frames, event_targets


def variant_policy(variant: str) -> Tuple[str, bool]:
    if variant == "multiscale_ungated":
        return "ungated", True
    if variant == "multiscale_oracle_gate":
        return "oracle", True
    if variant in ("multiscale_learned_gate", "single_slow_learned_gate"):
        return "learned", True
    if variant == "multiscale_learned_no_gate_updates":
        return "learned", False
    if variant == "multiscale_oracle_false_open_10":
        return "false_open_10", True
    if variant == "multiscale_oracle_false_open_25":
        return "false_open_25", True
    if variant == "multiscale_oracle_false_close_10":
        return "false_close_10", True
    raise ValueError(f"unknown variant: {variant}")


def external_gate(
    policy: str,
    event_target: float,
    corruption_uniform: float,
) -> Optional[torch.Tensor]:
    if policy == "learned":
        return None
    if policy == "ungated":
        value = 1.0
    elif policy == "oracle":
        value = event_target
    elif policy == "false_open_10":
        value = float(event_target >= 0.5 or corruption_uniform < 0.10)
    elif policy == "false_open_25":
        value = float(event_target >= 0.5 or corruption_uniform < 0.25)
    elif policy == "false_close_10":
        value = float(event_target >= 0.5 and corruption_uniform >= 0.10)
    else:
        raise RuntimeError(f"unknown gate policy: {policy}")
    return torch.tensor([value], dtype=torch.float32)


def named_parameter_sha256(
    module: nn.Module,
    *,
    prefix: Optional[str] = None,
) -> str:
    digest = hashlib.sha256()
    selected = 0
    for name, parameter in module.named_parameters():
        if prefix is not None and not name.startswith(prefix):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        selected += 1
    if selected == 0:
        raise ValueError(f"no parameters selected for prefix {prefix!r}")
    return digest.hexdigest()


def parameter_snapshot(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def max_parameter_drift(
    before: Mapping[str, torch.Tensor],
    module: nn.Module,
    *,
    prefix: Optional[str] = None,
) -> float:
    values = []
    for name, parameter in module.named_parameters():
        if prefix is not None and not name.startswith(prefix):
            continue
        values.append(float((parameter.detach() - before[name]).abs().max().item()))
    return max(values, default=0.0)


def make_learner(
    args: argparse.Namespace,
    variant: str,
    seed: int,
) -> OnlineTemporalMQRClassifier:
    torch.manual_seed(seed + 101)
    if variant == "single_slow_learned_gate":
        leaks = (0.02,)
        ring_dim = int(args.total_state_dim)
    else:
        leaks = (0.5, 0.1, 0.02, 0.005)
        if args.total_state_dim % len(leaks) != 0:
            raise ValueError("total-state-dim must be divisible by four")
        ring_dim = int(args.total_state_dim) // len(leaks)
    return OnlineTemporalMQRClassifier(
        65,
        ring_dim,
        10,
        core_kwargs={
            "leak_rates": leaks,
            "write_scales": tuple(1.0 for _ in leaks),
            "injection_rank": args.injection_rank,
            "injection_activation": "tanh",
            "state_activation": "tanh",
            "transition_mode": "unistochastic",
            "learn_transitions": False,
        },
        lr=args.readout_lr,
        transition_lr_ratio=0.0,
        injection_lr_ratio=0.0,
        readout_lr_ratio=1.0,
        ogd_max_rank=0,
        # The controller receives only the non-label event channel.  This
        # isolates online event learning from digit recognition and avoids
        # claiming that importance was discovered without a usable cue.
        gate_input_dim=1,
        gate_kwargs={
            "rank": args.gate_rank,
            "activation": "tanh",
            "temperature": 1.0,
            "minimum_gate": 0.02,
            "initial_open_probability": 0.5,
        },
        gate_lr=args.gate_lr,
        gate_ogd_max_rank=0,
        gate_max_update_norm=args.gate_max_update_norm,
        gate_positive_weight=args.gate_positive_weight,
        carry_state=True,
        max_update_norm=args.max_update_norm,
    )


def run_sequences(
    learner: OnlineTemporalMQRClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    counts: Sequence[int],
    *,
    policy: str,
    train_readout: bool,
    train_gate: bool,
    random_seed: int,
) -> Dict[str, object]:
    correct: List[int] = []
    query_losses: List[float] = []
    query_update_norms: List[float] = []
    gate_losses: List[float] = []
    gate_update_norms: List[float] = []
    query_state_norms: List[float] = []
    by_count: Dict[int, List[int]] = {int(v): [] for v in sorted(set(counts))}
    controller_correct = 0
    controller_false_open = 0
    controller_false_close = 0
    effective_correct = 0
    effective_false_open = 0
    effective_false_close = 0
    positive_count = 0
    negative_count = 0
    controller_cue_sum = 0.0
    controller_distractor_sum = 0.0
    effective_cue_sum = 0.0
    effective_distractor_sum = 0.0
    per_scale_controller_sum = torch.zeros(learner.core.num_timescales)
    per_scale_effective_sum = torch.zeros(learner.core.num_timescales)
    total_gate_elements = 0
    rng = np.random.default_rng(random_seed)
    started = time.perf_counter()

    for cue, label, distractor_count in zip(features, labels, counts):
        count = int(distractor_count)
        distractor_indices = rng.integers(0, features.size(0), size=count)
        distractors = features[torch.from_numpy(distractor_indices).long()]
        frames, event_targets = make_marked_sequence(cue, distractors)
        # Draw for every policy, including learned/oracle, so future distractor
        # sequences remain exactly paired across ablations.
        corruption_uniforms = rng.random(frames.size(0))
        learner.reset_state()
        result: Dict[str, object] = {}

        for frame_index, (frame, event_target) in enumerate(
            zip(frames, event_targets)
        ):
            is_query = frame_index == frames.size(0) - 1
            write_gate = external_gate(
                policy,
                float(event_target.item()),
                float(corruption_uniforms[frame_index]),
            )
            task_target = label.reshape(1) if is_query else None
            result = learner.online_step(
                frame.unsqueeze(0),
                task_target,
                learn=bool(train_readout and is_query),
                write_gate=write_gate,
                gate_input=frame[64:].unsqueeze(0),
                gate_target=event_target.reshape(1),
                learn_gate=train_gate,
            )
            gate_losses.append(float(result["gate_loss"]))
            gate_update_norms.append(float(result["gate_update_norm"]))

            controller = result["learned_write_gate"]
            probability = result["gate_base_probability"]
            effective = result["effective_write_gate"]
            assert isinstance(controller, torch.Tensor)
            assert isinstance(probability, torch.Tensor)
            assert isinstance(effective, torch.Tensor)
            target_open = bool(float(event_target.item()) >= 0.5)
            controller_open = probability >= 0.5
            effective_open = effective >= 0.5
            controller_correct += int(
                (controller_open == target_open).sum().item()
            )
            effective_correct += int(
                (effective_open == target_open).sum().item()
            )
            if target_open:
                positive_count += controller.numel()
                controller_false_close += int((~controller_open).sum().item())
                effective_false_close += int((~effective_open).sum().item())
                controller_cue_sum += float(controller.sum().item())
                effective_cue_sum += float(effective.sum().item())
            else:
                negative_count += controller.numel()
                controller_false_open += int(controller_open.sum().item())
                effective_false_open += int(effective_open.sum().item())
                controller_distractor_sum += float(controller.sum().item())
                effective_distractor_sum += float(effective.sum().item())
            per_scale_controller_sum += controller.squeeze(0).cpu()
            per_scale_effective_sum += effective.squeeze(0).cpu()
            total_gate_elements += controller.numel()

        predicted = int(result["logits"].argmax(dim=1).item())
        hit = int(predicted == int(label.item()))
        correct.append(hit)
        by_count[count].append(hit)
        query_losses.append(float(result["loss"]))
        query_update_norms.append(float(result["update_norm"]))
        state_vector = torch.cat(result["state"].rings, dim=1)
        query_state_norms.append(
            float(torch.linalg.vector_norm(state_vector).item())
        )

    elapsed = time.perf_counter() - started
    quarter = max(1, len(correct) // 4)
    frame_count = total_gate_elements // learner.core.num_timescales
    return {
        "count": len(correct),
        "frame_count": frame_count,
        "accuracy": statistics.fmean(correct),
        "first_quarter_accuracy": statistics.fmean(correct[:quarter]),
        "last_quarter_accuracy": statistics.fmean(correct[-quarter:]),
        "mean_query_loss": statistics.fmean(query_losses),
        "mean_query_update_norm": statistics.fmean(query_update_norms),
        "mean_gate_loss": statistics.fmean(gate_losses),
        "mean_gate_update_norm": statistics.fmean(gate_update_norms),
        "mean_query_state_l2": statistics.fmean(query_state_norms),
        "controller_gate_accuracy": controller_correct / total_gate_elements,
        "controller_false_open_rate": controller_false_open / negative_count,
        "controller_false_close_rate": controller_false_close / positive_count,
        "controller_cue_mean": controller_cue_sum / positive_count,
        "controller_distractor_mean": (
            controller_distractor_sum / negative_count
        ),
        "effective_gate_accuracy": effective_correct / total_gate_elements,
        "effective_false_open_rate": effective_false_open / negative_count,
        "effective_false_close_rate": effective_false_close / positive_count,
        "effective_cue_mean": effective_cue_sum / positive_count,
        "effective_distractor_mean": effective_distractor_sum / negative_count,
        "controller_mean_by_timescale": (
            per_scale_controller_sum / frame_count
        ).tolist(),
        "effective_mean_by_timescale": (
            per_scale_effective_sum / frame_count
        ).tolist(),
        "accuracy_by_distractor_count": {
            str(value): statistics.fmean(samples)
            for value, samples in by_count.items()
        },
        "milliseconds_per_sequence": 1000.0 * elapsed / max(1, len(correct)),
        "milliseconds_per_frame": 1000.0 * elapsed / max(1, frame_count),
    }


def run_once(
    args: argparse.Namespace,
    variant: str,
    seed: int,
) -> Dict[str, object]:
    data = prepare_digits(
        seed,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
    )
    learner = make_learner(args, variant, seed)
    policy, gate_learning_enabled = variant_policy(variant)
    train_counts = balanced_count_order(
        args.train_samples,
        args.distractor_counts,
        seed + 60_000,
    )
    eval_counts = balanced_count_order(
        args.eval_samples,
        args.distractor_counts,
        seed + 70_000,
    )
    initial_hash = named_parameter_sha256(learner)
    initial_controller_hash = named_parameter_sha256(
        learner,
        prefix="write_gate_controller.",
    )
    before = parameter_snapshot(learner)
    training = run_sequences(
        learner,
        *data["train"],
        train_counts,
        policy=policy,
        train_readout=True,
        train_gate=gate_learning_enabled,
        random_seed=seed + 80_000,
    )
    training_drift = max_parameter_drift(before, learner)
    training_gate_drift = max_parameter_drift(
        before,
        learner,
        prefix="write_gate_controller.",
    )
    final_controller_hash = named_parameter_sha256(
        learner,
        prefix="write_gate_controller.",
    )
    before_evaluation = parameter_snapshot(learner)
    evaluation = run_sequences(
        learner,
        *data["eval"],
        eval_counts,
        policy=policy,
        train_readout=False,
        train_gate=False,
        random_seed=seed + 90_000,
    )
    evaluation_drift = max_parameter_drift(before_evaluation, learner)
    controller = learner.write_gate_controller
    assert controller is not None
    readout_parameters = sum(p.numel() for p in learner.core.readout.parameters())
    gate_parameters = sum(p.numel() for p in controller.parameters())
    return {
        "variant": variant,
        "seed": seed,
        "gate_policy": policy,
        "gate_learning_enabled": gate_learning_enabled,
        "initial_parameter_sha256": initial_hash,
        "initial_controller_sha256": initial_controller_hash,
        "final_controller_sha256": final_controller_hash,
        "total_parameters": sum(p.numel() for p in learner.parameters()),
        "readout_parameters": readout_parameters,
        "gate_parameters": gate_parameters,
        "online_parameter_budget": (
            readout_parameters + (gate_parameters if gate_learning_enabled else 0)
        ),
        "total_state_dim": learner.core.total_state_dim,
        "leak_rates": learner.core.leak_rates.tolist(),
        "half_lives": learner.core.memory_half_lives().tolist(),
        "training": training,
        "evaluation": evaluation,
        "training_parameter_max_drift": training_drift,
        "training_gate_max_drift": training_gate_drift,
        "evaluation_parameter_max_drift": evaluation_drift,
        "online_observations": int(learner.online_observations.item()),
        "online_updates": int(learner.online_updates.item()),
        "online_gate_updates": int(learner.online_gate_updates.item()),
        "online_gate_labels": int(learner.online_gate_labels.item()),
        "max_unitary_error": learner.core.max_unitary_error(),
        "max_stochastic_error": learner.core.max_stochastic_error(),
    }


def summarize(values: Iterable[float]) -> Dict[str, float]:
    samples = list(values)
    return {
        "mean": statistics.fmean(samples),
        "sample_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def aggregate(runs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    metrics = (
        "accuracy",
        "last_quarter_accuracy",
        "mean_gate_loss",
        "controller_gate_accuracy",
        "controller_false_open_rate",
        "controller_false_close_rate",
        "controller_cue_mean",
        "controller_distractor_mean",
        "effective_gate_accuracy",
        "effective_false_open_rate",
        "effective_false_close_rate",
        "effective_cue_mean",
        "effective_distractor_mean",
        "milliseconds_per_sequence",
    )
    result: Dict[str, object] = {}
    for variant in VARIANTS:
        selected = [run for run in runs if run["variant"] == variant]
        summary: Dict[str, object] = {
            "total_parameters": int(selected[0]["total_parameters"]),
            "readout_parameters": int(selected[0]["readout_parameters"]),
            "gate_parameters": int(selected[0]["gate_parameters"]),
            "online_parameter_budget": int(selected[0]["online_parameter_budget"]),
        }
        for split in ("training", "evaluation"):
            for metric in metrics:
                summary[f"{split}_{metric}"] = summarize(
                    float(run[split][metric])  # type: ignore[index]
                    for run in selected
                )
        summary["training_gate_max_drift"] = summarize(
            float(run["training_gate_max_drift"]) for run in selected
        )
        summary["evaluation_parameter_max_drift"] = summarize(
            float(run["evaluation_parameter_max_drift"]) for run in selected
        )
        result[variant] = summary
    return result


def paired_contrasts(runs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    indexed = {
        (int(run["seed"]), str(run["variant"])): float(
            run["evaluation"]["accuracy"]  # type: ignore[index]
        )
        for run in runs
    }
    seeds = sorted({seed for seed, _variant in indexed})
    definitions = {
        "learned_vs_ungated": (
            "multiscale_learned_gate",
            "multiscale_ungated",
        ),
        "learned_gate_update_effect": (
            "multiscale_learned_gate",
            "multiscale_learned_no_gate_updates",
        ),
        "oracle_vs_learned": (
            "multiscale_oracle_gate",
            "multiscale_learned_gate",
        ),
        "oracle_false_open_10_cost": (
            "multiscale_oracle_false_open_10",
            "multiscale_oracle_gate",
        ),
        "oracle_false_open_25_cost": (
            "multiscale_oracle_false_open_25",
            "multiscale_oracle_gate",
        ),
        "oracle_false_close_10_cost": (
            "multiscale_oracle_false_close_10",
            "multiscale_oracle_gate",
        ),
        "multiscale_vs_single_slow_learned": (
            "multiscale_learned_gate",
            "single_slow_learned_gate",
        ),
    }
    result: Dict[str, object] = {}
    for name, (left, right) in definitions.items():
        samples = [indexed[(seed, left)] - indexed[(seed, right)] for seed in seeds]
        result[name] = {
            "left": left,
            "right": right,
            "paired_differences": samples,
            **summarize(samples),
        }
    return result


def initialization_audit(runs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    seeds = sorted({int(run["seed"]) for run in runs})
    indexed = {
        (int(run["seed"]), str(run["variant"])): run
        for run in runs
    }
    multiscale = tuple(v for v in VARIANTS if v.startswith("multiscale_"))
    return {
        "multiscale_initial_parameters_matched_by_seed": {
            str(seed): len(
                {
                    str(indexed[(seed, variant)]["initial_parameter_sha256"])
                    for variant in multiscale
                }
            )
            == 1
            for seed in seeds
        },
        "multiscale_learned_controller_final_matched_by_seed": {
            str(seed): len(
                {
                    str(indexed[(seed, variant)]["final_controller_sha256"])
                    for variant in MULTISCALE_GATE_LEARNING_VARIANTS
                }
            )
            == 1
            for seed in seeds
        },
    }


def main() -> None:
    args = parse_args()
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("sample counts must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be non-empty and unique")
    if not args.distractor_counts or any(v < 0 for v in args.distractor_counts):
        raise ValueError("distractor counts must be non-empty and non-negative")
    if len(set(args.distractor_counts)) != len(args.distractor_counts):
        raise ValueError("distractor counts must be unique")
    for name in (
        "total_state_dim",
        "injection_rank",
        "gate_rank",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in (
        "readout_lr",
        "gate_lr",
        "gate_positive_weight",
        "max_update_norm",
        "gate_max_update_norm",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name} must be positive")

    torch.set_num_threads(1)
    runs: List[Dict[str, object]] = []
    for seed in args.seeds:
        for variant in VARIANTS:
            run = run_once(args, variant, seed)
            runs.append(run)
            print(
                f"{variant:40s} seed={seed:3d} "
                f"preq={run['training']['accuracy']:.3f} "
                f"heldout={run['evaluation']['accuracy']:.3f} "
                f"gate_acc={run['evaluation']['controller_gate_accuracy']:.3f} "
                f"FO={run['evaluation']['controller_false_open_rate']:.3f} "
                f"FC={run['evaluation']['controller_false_close_rate']:.3f}"
            )

    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {
            "dataset": "sklearn.datasets.load_digits (offline bundled dataset)",
            "split": "70% train / 30% held out, stratified per seed",
            "sequence": (
                "digit cue with non-label event marker -> independent real-digit "
                "distractors -> exactly zero query"
            ),
            "causal_order": (
                "use theta_t gate and predict first; submit current event target "
                "afterward; submit digit class only after zero-query prediction"
            ),
            "trainable_scope": (
                "frozen reservoir/injection; online readout plus an independent "
                "auxiliary-event gate update"
            ),
            "evaluation": "held-out streams with readout and gate updates disabled",
            "matching": (
                "seed/data/order/distractors/corruption uniforms and multiscale "
                "initial parameters are paired"
            ),
            "gate_semantics": (
                "external gates override state writes while the learned controller "
                "may train in the background; learned conditions use its continuous "
                "gate and the controller sees only the one-dimensional event marker"
            ),
            "limitations": [
                "the event label is immediate auxiliary supervision, not delayed reward",
                "the cue marker explicitly exposes event type but never digit class",
                "three seeds are a mechanism screen rather than confirmatory evidence",
                "Digits fading memory is not language learning, Go strength, or animal learning",
            ],
        },
        "configuration": {
            "seeds": args.seeds,
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
            "distractor_counts": args.distractor_counts,
            "total_state_dim": args.total_state_dim,
            "injection_rank": args.injection_rank,
            "gate_rank": args.gate_rank,
            "readout_lr": args.readout_lr,
            "gate_lr": args.gate_lr,
            "gate_positive_weight": args.gate_positive_weight,
            "max_update_norm": args.max_update_norm,
            "gate_max_update_norm": args.gate_max_update_norm,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "initialization_audit": initialization_audit(runs),
        "summary": aggregate(runs),
        "paired_contrasts": paired_contrasts(runs),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
