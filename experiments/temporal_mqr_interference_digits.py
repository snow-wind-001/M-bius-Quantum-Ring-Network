#!/usr/bin/env python3
"""Interference-rich Digits validation for externally gated Temporal MQR.

Each causal sequence contains one target digit, independently sampled digit
distractors, and a zero query.  Only the query receives feedback.  Conditions
are seed/data/order matched and train only the readout, so differences between
the multiscale gate conditions isolate state-write policy rather than parameter
count, initialization, or label leakage.
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

from experiments.temporal_mqr_delayed_digits import (  # noqa: E402
    OnlineLinear,
    learner_counts,
    max_parameter_drift,
    parameter_snapshot,
    prepare_digits,
)
from mqr import OnlineTemporalMQRClassifier  # noqa: E402


VARIANTS = (
    "memoryless_query_linear",
    "single_slow_ungated",
    "single_slow_oracle_gate",
    "multiscale_ungated",
    "multiscale_random_gate",
    "multiscale_oracle_gate",
    "multiscale_oracle_no_learning",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--train-samples", type=int, default=800)
    parser.add_argument("--eval-samples", type=int, default=320)
    parser.add_argument(
        "--distractor-counts",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
    )
    parser.add_argument("--total-state-dim", type=int, default=48)
    parser.add_argument("--injection-rank", type=int, default=24)
    parser.add_argument("--readout-lr", type=float, default=0.15)
    parser.add_argument("--max-update-norm", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/results/temporal_mqr_interference_digits.json"),
    )
    return parser.parse_args()


def make_interference_sequence(
    cue: torch.Tensor,
    distractors: torch.Tensor,
) -> torch.Tensor:
    """Return ``[cue, distractors..., zero_query]`` without a label-bearing query."""

    if cue.shape != (64,):
        raise ValueError("cue must have shape [64]")
    if distractors.dim() != 2 or distractors.size(1) != 64:
        raise ValueError("distractors must have shape [count, 64]")
    frames = torch.zeros(
        distractors.size(0) + 2,
        64,
        dtype=cue.dtype,
        device=cue.device,
    )
    frames[0] = cue
    if distractors.numel() > 0:
        frames[1:-1] = distractors.to(device=cue.device, dtype=cue.dtype)
    return frames


def balanced_count_order(length: int, counts: Sequence[int], seed: int) -> List[int]:
    values = [int(counts[index % len(counts)]) for index in range(length)]
    generator = np.random.default_rng(seed)
    generator.shuffle(values)
    return values


def variant_gate_mode(variant: str) -> str:
    if variant.endswith("ungated"):
        return "ungated"
    if variant == "multiscale_random_gate":
        return "random_one_write"
    if "oracle" in variant:
        return "oracle_cue_only"
    return "not_applicable"


def parameter_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_learner(args: argparse.Namespace, variant: str, seed: int):
    torch.manual_seed(seed + 101)
    if variant == "memoryless_query_linear":
        # Learning is enabled deliberately.  Because the query is the exact
        # zero vector and the readout has no bias, its gradient is also exactly
        # zero; this is a stronger leakage audit than disabling updates.
        return OnlineLinear(64, 10, lr=args.readout_lr), True

    total = int(args.total_state_dim)
    if variant.startswith("single_slow"):
        leaks = (0.02,)
        ring_dim = total
    else:
        leaks = (0.5, 0.1, 0.02, 0.005)
        if total % len(leaks) != 0:
            raise ValueError("total-state-dim must be divisible by four")
        ring_dim = total // len(leaks)
    learn_enabled = variant != "multiscale_oracle_no_learning"
    learner = OnlineTemporalMQRClassifier(
        64,
        ring_dim,
        10,
        core_kwargs={
            "leak_rates": leaks,
            "write_scales": tuple(1.0 for _ in leaks),
            "injection_rank": args.injection_rank,
            "injection_activation": "tanh",
            # Real distractors can accumulate without bound under a linear
            # state; tanh keeps the engineering control uniformly bounded.
            "state_activation": "tanh",
            "transition_mode": "unistochastic",
            "learn_transitions": False,
        },
        lr=args.readout_lr,
        transition_lr_ratio=0.0,
        injection_lr_ratio=0.0,
        readout_lr_ratio=1.0 if learn_enabled else 0.0,
        ogd_max_rank=0,
        carry_state=True,
        max_update_norm=args.max_update_norm,
    )
    return learner, learn_enabled


def gate_for_frame(
    gate_mode: str,
    frame_index: int,
    random_open_index: int,
) -> Optional[torch.Tensor]:
    if gate_mode == "not_applicable":
        return None
    if gate_mode == "ungated":
        value = 1.0
    elif gate_mode == "oracle_cue_only":
        value = float(frame_index == 0)
    elif gate_mode == "random_one_write":
        value = float(frame_index == random_open_index)
    else:
        raise RuntimeError(f"unknown gate mode: {gate_mode}")
    return torch.tensor([value], dtype=torch.float32)


def run_sequences(
    learner,
    features: torch.Tensor,
    labels: torch.Tensor,
    counts: Sequence[int],
    *,
    gate_mode: str,
    learn: bool,
    random_seed: int,
) -> Dict[str, object]:
    correct: List[int] = []
    losses: List[float] = []
    updates: List[float] = []
    state_norms: List[float] = []
    by_count: Dict[int, List[int]] = {int(count): [] for count in sorted(set(counts))}
    rng = np.random.default_rng(random_seed)
    started = time.perf_counter()
    total_writes = 0.0
    total_frames = 0

    for cue, label, distractor_count in zip(features, labels, counts):
        count = int(distractor_count)
        distractor_indices = rng.integers(0, features.size(0), size=count)
        distractors = features[torch.from_numpy(distractor_indices).long()]
        frames = make_interference_sequence(cue, distractors)
        # The random control writes exactly one content frame, with no access to
        # the cue position or target label.  The query is never selected.
        random_open = int(rng.integers(0, count + 1))
        learner.reset_state()
        result: Dict[str, object] = {}
        for frame_index, frame in enumerate(frames):
            gate = gate_for_frame(gate_mode, frame_index, random_open)
            total_frames += 1
            if gate is not None:
                total_writes += float(gate.item())
            target = label.reshape(1) if frame_index == frames.size(0) - 1 else None
            if isinstance(learner, OnlineLinear):
                result = learner.online_step(
                    frame.unsqueeze(0),
                    target,
                    learn=bool(learn and target is not None),
                )
            else:
                result = learner.online_step(
                    frame.unsqueeze(0),
                    target,
                    learn=bool(learn and target is not None),
                    write_gate=gate,
                )
        predicted = int(result["logits"].argmax(dim=1).item())
        hit = int(predicted == int(label.item()))
        correct.append(hit)
        by_count[count].append(hit)
        losses.append(float(result["loss"]))
        updates.append(float(result["update_norm"]))
        if not isinstance(learner, OnlineLinear):
            state = result["state"]
            state_vector = torch.cat(state.rings, dim=1)
            state_norms.append(float(torch.linalg.vector_norm(state_vector).item()))

    elapsed = time.perf_counter() - started
    quarter = max(1, len(correct) // 4)
    return {
        "count": len(correct),
        "accuracy": statistics.fmean(correct),
        "first_quarter_accuracy": statistics.fmean(correct[:quarter]),
        "last_quarter_accuracy": statistics.fmean(correct[-quarter:]),
        "mean_loss": statistics.fmean(losses),
        "mean_update_norm": statistics.fmean(updates),
        "mean_query_state_l2": (
            statistics.fmean(state_norms) if state_norms else 0.0
        ),
        "gate_open_fraction": (
            total_writes / total_frames if gate_mode != "not_applicable" else None
        ),
        "accuracy_by_distractor_count": {
            str(count): statistics.fmean(values)
            for count, values in by_count.items()
        },
        "milliseconds_per_sequence": 1000.0 * elapsed / max(1, len(correct)),
    }


def run_once(args: argparse.Namespace, variant: str, seed: int) -> Dict[str, object]:
    data = prepare_digits(
        seed,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
    )
    learner, learn_enabled = make_learner(args, variant, seed)
    gate_mode = variant_gate_mode(variant)
    initial_hash = parameter_sha256(learner)
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

    before = parameter_snapshot(learner)
    training = run_sequences(
        learner,
        *data["train"],
        train_counts,
        gate_mode=gate_mode,
        learn=learn_enabled,
        random_seed=seed + 80_000,
    )
    training_drift = max_parameter_drift(before, learner)
    before_evaluation = parameter_snapshot(learner)
    evaluation = run_sequences(
        learner,
        *data["eval"],
        eval_counts,
        gate_mode=gate_mode,
        learn=False,
        random_seed=seed + 90_000,
    )
    evaluation_drift = max_parameter_drift(before_evaluation, learner)
    total_parameters, online_parameters = learner_counts(learner)
    result: Dict[str, object] = {
        "variant": variant,
        "seed": seed,
        "gate_mode": gate_mode,
        "learn_enabled": learn_enabled,
        "initial_parameter_sha256": initial_hash,
        "total_parameters": total_parameters,
        "online_parameters": online_parameters,
        "training": training,
        "evaluation": evaluation,
        "training_parameter_max_drift": training_drift,
        "evaluation_parameter_max_drift": evaluation_drift,
    }
    if isinstance(learner, OnlineTemporalMQRClassifier):
        result.update(
            {
                "total_state_dim": learner.core.total_state_dim,
                "leak_rates": learner.core.leak_rates.tolist(),
                "half_lives": learner.core.memory_half_lives().tolist(),
                "transition_mode": learner.core.transition_mode,
                "state_activation": learner.core.state_activation,
                "max_unitary_error": learner.core.max_unitary_error(),
                "max_stochastic_error": learner.core.max_stochastic_error(),
            }
        )
    return result


def summarize(values: Iterable[float]) -> Dict[str, float]:
    samples = list(values)
    return {
        "mean": statistics.fmean(samples),
        "sample_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def aggregate(
    runs: Sequence[Mapping[str, object]],
    distractor_counts: Sequence[int],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for variant in VARIANTS:
        selected = [run for run in runs if run["variant"] == variant]
        result[variant] = {
            "total_parameters": int(selected[0]["total_parameters"]),
            "online_parameters": int(selected[0]["online_parameters"]),
            "prequential_accuracy": summarize(
                float(run["training"]["accuracy"]) for run in selected  # type: ignore[index]
            ),
            "prequential_last_quarter_accuracy": summarize(
                float(run["training"]["last_quarter_accuracy"])  # type: ignore[index]
                for run in selected
            ),
            "heldout_accuracy": summarize(
                float(run["evaluation"]["accuracy"]) for run in selected  # type: ignore[index]
            ),
            "mean_query_state_l2": summarize(
                float(run["evaluation"]["mean_query_state_l2"])  # type: ignore[index]
                for run in selected
            ),
            "training_parameter_max_drift": summarize(
                float(run["training_parameter_max_drift"]) for run in selected
            ),
            "evaluation_parameter_max_drift": summarize(
                float(run["evaluation_parameter_max_drift"]) for run in selected
            ),
            "milliseconds_per_training_sequence": summarize(
                float(run["training"]["milliseconds_per_sequence"])  # type: ignore[index]
                for run in selected
            ),
            "heldout_accuracy_by_distractor_count": {
                str(count): summarize(
                    float(
                        run["evaluation"]["accuracy_by_distractor_count"][  # type: ignore[index]
                            str(count)
                        ]
                    )
                    for run in selected
                )
                for count in distractor_counts
            },
        }
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
        "multiscale_oracle_gate_effect": (
            "multiscale_oracle_gate",
            "multiscale_ungated",
        ),
        "single_slow_oracle_gate_effect": (
            "single_slow_oracle_gate",
            "single_slow_ungated",
        ),
        "oracle_vs_random_gate": (
            "multiscale_oracle_gate",
            "multiscale_random_gate",
        ),
        "oracle_gate_learning_effect": (
            "multiscale_oracle_gate",
            "multiscale_oracle_no_learning",
        ),
        "multiscale_vs_single_slow_oracle": (
            "multiscale_oracle_gate",
            "single_slow_oracle_gate",
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
    groups = {
        "single_gate_pair": ("single_slow_ungated", "single_slow_oracle_gate"),
        "multiscale_gate_group": (
            "multiscale_ungated",
            "multiscale_random_gate",
            "multiscale_oracle_gate",
            "multiscale_oracle_no_learning",
        ),
    }
    by_key = {
        (int(run["seed"]), str(run["variant"])): str(
            run["initial_parameter_sha256"]
        )
        for run in runs
    }
    result: Dict[str, object] = {}
    seeds = sorted({int(run["seed"]) for run in runs})
    for name, variants in groups.items():
        matched = {
            str(seed): len({by_key[(seed, variant)] for variant in variants}) == 1
            for seed in seeds
        }
        result[name] = {"variants": variants, "matched_by_seed": matched}
    return result


def main() -> None:
    args = parse_args()
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("sample counts must be positive")
    if not args.distractor_counts or any(value < 0 for value in args.distractor_counts):
        raise ValueError("distractor counts must be non-empty and non-negative")
    if len(set(args.distractor_counts)) != len(args.distractor_counts):
        raise ValueError("distractor counts must be unique")
    if args.total_state_dim <= 0 or args.injection_rank <= 0:
        raise ValueError("state dimension and injection rank must be positive")
    if args.readout_lr <= 0 or args.max_update_norm <= 0:
        raise ValueError("learning rate and update norm must be positive")

    torch.set_num_threads(1)
    runs: List[Dict[str, object]] = []
    for seed in args.seeds:
        for variant in VARIANTS:
            run = run_once(args, variant, seed)
            runs.append(run)
            print(
                f"{variant:36s} seed={seed:3d} "
                f"preq={run['training']['accuracy']:.3f} "
                f"heldout={run['evaluation']['accuracy']:.3f} "
                f"drift={run['training_parameter_max_drift']:.3e}"
            )

    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {
            "dataset": "sklearn.datasets.load_digits (offline bundled dataset)",
            "split": "70% train / 30% held out, stratified independently per seed",
            "sequence": "target digit once -> independent real-digit distractors -> zero query",
            "feedback_order": (
                "predict zero query with theta_t, score, then optionally update readout"
            ),
            "trainable_scope": "readout only; injection and transitions are frozen",
            "gate_controls": {
                "ungated": "write every target/distractor frame",
                "oracle_cue_only": "write only the known cue event",
                "random_one_write": (
                    "write one uniformly sampled content frame without cue/label access"
                ),
            },
            "state_reset": "reset at every independent target sequence",
            "evaluation": (
                "held-out targets/distractors with learning disabled and zero-drift audit"
            ),
            "statistical_unit": "independently initialized seed",
            "limitations": [
                "three seeds are a mechanism screen, not confirmatory evidence",
                "the oracle gate is supplied by the protocol and is not learned",
                "the task tests protected fading memory, not delayed credit into the gate",
                "only readout weights learn; injection and transition learning are not tested",
                "the random reservoir is not a language model or a Go policy",
                "this experiment cannot support animal-level learning claims",
            ],
        },
        "configuration": {
            "seeds": args.seeds,
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
            "distractor_counts": args.distractor_counts,
            "total_state_dim": args.total_state_dim,
            "injection_rank": args.injection_rank,
            "readout_lr": args.readout_lr,
            "max_update_norm": args.max_update_norm,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": "cpu",
        },
        "initialization_audit": initialization_audit(runs),
        "summary": aggregate(runs, args.distractor_counts),
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
