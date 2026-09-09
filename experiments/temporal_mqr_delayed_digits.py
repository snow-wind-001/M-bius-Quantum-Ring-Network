#!/usr/bin/env python3
"""Causal delayed-Digits validation for temporal MQR state and online readout.

Each sequence presents one 8x8 digit once, waits for a variable number of blank
frames, and requests the class only at the final query frame.  The prediction is
scored before its label updates the readout.  The experiment separates:

* an instant linear oracle that sees the digit at the query;
* a memoryless query baseline;
* the effective decay of the equilibrium MQR (alpha=0.3, K=24);
* one-step fast, slow, and multi-timescale MQR reservoirs;
* identity versus frozen Cayley-unistochastic transitions;
* an exact no-learning control.

Only readout weights learn in the reservoir conditions.  Injection and
transition parameters remain fixed, so delayed performance is attributable to
activation memory plus a causal online readout rather than BPTT or slow feature
learning.  ``sklearn.datasets.load_digits`` is bundled with scikit-learn and
requires no network access.
"""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mqr import OnlineTemporalMQRClassifier


VARIANTS = (
    "instant_linear_oracle",
    "memoryless_query_linear",
    "equilibrium_k24_control",
    "single_fast_mqr",
    "single_slow_mqr",
    "multiscale_identity",
    "multiscale_coupled_write",
    "multiscale_unistochastic",
    "multiscale_unistochastic_no_learning",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--train-samples", type=int, default=800)
    parser.add_argument("--eval-samples", type=int, default=320)
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--total-state-dim", type=int, default=48)
    parser.add_argument("--injection-rank", type=int, default=24)
    parser.add_argument("--readout-lr", type=float, default=0.15)
    parser.add_argument("--max-update-norm", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/results/temporal_mqr_delayed_digits.json"),
    )
    return parser.parse_args()


class OnlineLinear(nn.Module):
    """Minimal predict-before-update linear control."""

    def __init__(self, input_dim: int, output_dim: int, *, lr: float, learn: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.lr = float(lr)
        self.learn_enabled = bool(learn)

    def reset_state(self) -> None:
        return None

    def online_step(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        *,
        learn: bool,
    ) -> Dict[str, object]:
        with torch.enable_grad():
            logits = self.linear(x)
            loss = None if target is None else F.cross_entropy(logits, target)
            gradient = None
            if learn and self.learn_enabled and loss is not None:
                gradient = torch.autograd.grad(loss, self.linear.weight)[0]
        before = logits.detach().clone()
        if gradient is not None:
            with torch.no_grad():
                self.linear.weight.add_(gradient, alpha=-self.lr)
        return {
            "logits": before,
            "loss": None if loss is None else float(loss.detach().item()),
            "did_update": gradient is not None,
            "update_norm": (
                0.0
                if gradient is None
                else self.lr * float(torch.linalg.vector_norm(gradient).item())
            ),
        }

    @property
    def online_parameter_count(self) -> int:
        return self.linear.weight.numel() if self.learn_enabled else 0


def prepare_digits(
    seed: int,
    *,
    train_samples: int,
    eval_samples: int,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    digits = load_digits()
    features = digits.data.astype(np.float32) / 16.0
    labels = digits.target.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    mean = x_train.mean(axis=0, keepdims=True)
    scale = x_train.std(axis=0, keepdims=True)
    scale[scale < 0.05] = 1.0
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale

    if train_samples > x_train.shape[0] or eval_samples > x_test.shape[0]:
        raise ValueError(
            "requested samples exceed the 70/30 Digits split: "
            f"train <= {x_train.shape[0]}, eval <= {x_test.shape[0]}"
        )
    rng = np.random.default_rng(seed + 30_000)
    train_order = rng.permutation(x_train.shape[0])[:train_samples]
    eval_order = rng.permutation(x_test.shape[0])[:eval_samples]
    return {
        "train": (
            torch.from_numpy(x_train[train_order].copy()),
            torch.from_numpy(y_train[train_order].copy()),
        ),
        "eval": (
            torch.from_numpy(x_test[eval_order].copy()),
            torch.from_numpy(y_test[eval_order].copy()),
        ),
    }


def make_sequence(feature: torch.Tensor, delay: int, *, instant: bool) -> torch.Tensor:
    """Return ``[cue, blank*delay, query]`` with two explicit control flags."""

    if feature.shape != (64,):
        raise ValueError("feature must have shape [64]")
    if delay < 0:
        raise ValueError("delay must be non-negative")
    frames = torch.zeros(delay + 2, 66, dtype=feature.dtype)
    frames[0, :64] = feature
    frames[0, 64] = 1.0  # cue flag
    frames[-1, 65] = 1.0  # query flag
    if instant:
        frames[-1, :64] = feature
    return frames


def make_learner(args: argparse.Namespace, variant: str, seed: int):
    torch.manual_seed(seed + 101)
    if variant in ("instant_linear_oracle", "memoryless_query_linear"):
        return OnlineLinear(66, 10, lr=args.readout_lr), variant == "instant_linear_oracle"

    total = int(args.total_state_dim)
    if variant == "equilibrium_k24_control":
        # For identity H and no state nonlinearity, K equilibrium iterations are
        # exactly one temporal step with carry q^K and leak 1-q^K.
        equilibrium_leak = 1.0 - (1.0 - 0.3) ** 24
        leaks = (equilibrium_leak,)
        ring_dim = total
        transition_mode = "identity"
        write_scales = (equilibrium_leak,)
    elif variant == "single_fast_mqr":
        leaks = (0.3,)
        ring_dim = total
        transition_mode = "unistochastic"
        write_scales = (1.0,)
    elif variant == "single_slow_mqr":
        leaks = (0.02,)
        ring_dim = total
        transition_mode = "unistochastic"
        write_scales = (1.0,)
    else:
        leaks = (0.5, 0.1, 0.02, 0.005)
        if total % len(leaks) != 0:
            raise ValueError("total-state-dim must be divisible by four for multiscale variants")
        ring_dim = total // len(leaks)
        transition_mode = "identity" if variant == "multiscale_identity" else "unistochastic"
        write_scales = (
            leaks
            if variant == "multiscale_coupled_write"
            else tuple(1.0 for _ in leaks)
        )

    learn_enabled = variant != "multiscale_unistochastic_no_learning"
    learner = OnlineTemporalMQRClassifier(
        66,
        ring_dim,
        10,
        core_kwargs={
            "leak_rates": leaks,
            # Strong one-shot write and slow decay are deliberately independent.
            "write_scales": write_scales,
            "injection_rank": args.injection_rank,
            "injection_activation": "tanh",
            "state_activation": "none",
            "transition_mode": transition_mode,
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
    return learner, False


def learner_counts(learner) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in learner.parameters())
    if isinstance(learner, OnlineLinear):
        online = learner.online_parameter_count
    else:
        online = sum(parameter.numel() for _name, parameter, _step in learner._active_parameters())
    return total, online


def parameter_snapshot(learner) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in learner.named_parameters()
    }


def max_parameter_drift(
    before: Mapping[str, torch.Tensor], learner: nn.Module
) -> float:
    current = dict(learner.named_parameters())
    return max(
        (float((current[name].detach() - value).abs().max().item()) for name, value in before.items()),
        default=0.0,
    )


def run_sequences(
    learner,
    features: torch.Tensor,
    labels: torch.Tensor,
    delays: Sequence[int],
    *,
    instant: bool,
    learn: bool,
) -> Dict[str, object]:
    correct: List[int] = []
    losses: List[float] = []
    updates: List[float] = []
    by_delay: Dict[int, List[int]] = {int(delay): [] for delay in delays}
    started = time.perf_counter()
    for index, (feature, label) in enumerate(zip(features, labels)):
        delay = int(delays[index % len(delays)])
        frames = make_sequence(feature, delay, instant=instant)
        learner.reset_state()
        for frame in frames[:-1]:
            learner.online_step(frame.unsqueeze(0), None, learn=False)
        target = label.reshape(1)
        result = learner.online_step(frames[-1:].clone(), target, learn=learn)
        predicted = int(result["logits"].argmax(dim=1).item())
        hit = int(predicted == int(label.item()))
        correct.append(hit)
        by_delay[delay].append(hit)
        losses.append(float(result["loss"]))
        updates.append(float(result["update_norm"]))
    elapsed = time.perf_counter() - started
    quarter = max(1, len(correct) // 4)
    return {
        "count": len(correct),
        "accuracy": statistics.fmean(correct),
        "first_quarter_accuracy": statistics.fmean(correct[:quarter]),
        "last_quarter_accuracy": statistics.fmean(correct[-quarter:]),
        "mean_loss": statistics.fmean(losses),
        "mean_update_norm": statistics.fmean(updates),
        "accuracy_by_delay": {
            str(delay): statistics.fmean(values)
            for delay, values in by_delay.items()
        },
        "milliseconds_per_sequence": 1000.0 * elapsed / max(1, len(correct)),
    }


def balanced_delay_order(length: int, delays: Sequence[int], seed: int) -> List[int]:
    values = [int(delays[index % len(delays)]) for index in range(length)]
    generator = np.random.default_rng(seed)
    generator.shuffle(values)
    return values


def run_once(args: argparse.Namespace, variant: str, seed: int) -> Dict[str, object]:
    data = prepare_digits(
        seed,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
    )
    learner, instant = make_learner(args, variant, seed)
    train_delays = balanced_delay_order(
        args.train_samples,
        args.delays,
        seed + 40_000,
    )
    eval_delays = balanced_delay_order(
        args.eval_samples,
        args.delays,
        seed + 50_000,
    )
    learn_enabled = variant != "multiscale_unistochastic_no_learning"
    before = parameter_snapshot(learner)
    training = run_sequences(
        learner,
        *data["train"],
        train_delays,
        instant=instant,
        learn=learn_enabled,
    )
    training_drift = max_parameter_drift(before, learner)
    before_evaluation = parameter_snapshot(learner)
    evaluation = run_sequences(
        learner,
        *data["eval"],
        eval_delays,
        instant=instant,
        learn=False,
    )
    evaluation_drift = max_parameter_drift(before_evaluation, learner)
    total_parameters, online_parameters = learner_counts(learner)
    result: Dict[str, object] = {
        "variant": variant,
        "seed": seed,
        "learn_enabled": learn_enabled,
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
                "write_scales": learner.core.write_scales.tolist(),
                "half_lives": learner.core.memory_half_lives().tolist(),
                "transition_mode": learner.core.transition_mode,
                "max_unitary_error": learner.core.max_unitary_error(),
                "max_stochastic_error": learner.core.max_stochastic_error(),
            }
        )
    return result


def aggregate(runs: Sequence[Mapping[str, object]], delays: Sequence[int]) -> Dict[str, object]:
    def summarize(values: Iterable[float]) -> Dict[str, float]:
        samples = list(values)
        return {
            "mean": statistics.fmean(samples),
            "sample_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        }

    result: Dict[str, object] = {}
    for variant in VARIANTS:
        selected = [run for run in runs if run["variant"] == variant]
        variant_summary: Dict[str, object] = {
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
            "heldout_accuracy_by_delay": {
                str(delay): summarize(
                    float(run["evaluation"]["accuracy_by_delay"][str(delay)])  # type: ignore[index]
                    for run in selected
                )
                for delay in delays
            },
        }
        result[variant] = variant_summary
    return result


def paired_contrasts(runs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Compute seed-paired held-out effects; games/examples are not replicates."""

    indexed = {
        (int(run["seed"]), str(run["variant"])): float(run["evaluation"]["accuracy"])  # type: ignore[index]
        for run in runs
    }
    seeds = sorted({seed for seed, _variant in indexed})
    definitions = {
        "multiscale_learning_effect": (
            "multiscale_unistochastic",
            "multiscale_unistochastic_no_learning",
        ),
        "temporal_vs_equilibrium": (
            "multiscale_unistochastic",
            "equilibrium_k24_control",
        ),
        "slow_vs_fast": ("single_slow_mqr", "single_fast_mqr"),
        "multiscale_vs_single_slow": (
            "multiscale_unistochastic",
            "single_slow_mqr",
        ),
        "decoupled_write_effect": (
            "multiscale_unistochastic",
            "multiscale_coupled_write",
        ),
        "unistochastic_vs_identity": (
            "multiscale_unistochastic",
            "multiscale_identity",
        ),
    }
    result: Dict[str, object] = {}
    for name, (left, right) in definitions.items():
        samples = [indexed[(seed, left)] - indexed[(seed, right)] for seed in seeds]
        result[name] = {
            "left": left,
            "right": right,
            "paired_differences": samples,
            "mean": statistics.fmean(samples),
            "sample_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        }
    return result


def main() -> None:
    args = parse_args()
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("sample counts must be positive")
    if not args.delays or any(delay < 0 for delay in args.delays):
        raise ValueError("delays must be non-empty and non-negative")
    if len(set(args.delays)) != len(args.delays):
        raise ValueError("delays must be unique")
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
                f"{variant:38s} seed={seed:3d} "
                f"preq={run['training']['accuracy']:.3f} "
                f"heldout={run['evaluation']['accuracy']:.3f} "
                f"drift={run['training_parameter_max_drift']:.3e}"
            )

    equilibrium_carry = (1.0 - 0.3) ** 24
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {
            "dataset": "sklearn.datasets.load_digits (offline bundled dataset)",
            "split": "70% train / 30% held out, stratified independently per seed",
            "sequence": "digit cue once -> variable blank delay -> label-free query",
            "feedback_order": "predict query with theta_t, score, then optionally update readout",
            "trainable_scope": "readout only; temporal injection and transitions frozen",
            "state_reset": "reset at every independent digit sequence",
            "evaluation": "held-out digits with learning disabled and exact zero-drift audit",
            "statistical_unit": "independently initialized seed",
            "limitations": [
                "three seeds are a mechanism screen, not confirmatory evidence",
                "the blank-delay task tests fading activation memory, not interference-rich recall",
                "only readout weights learn; delayed credit assignment into injection is not tested",
                "identity and unistochastic reservoirs are not compute-matched to all linear controls",
                "this does not test an LM token head, Go strength, or animal-level learning",
            ],
        },
        "configuration": {
            "seeds": args.seeds,
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
            "delays": args.delays,
            "total_state_dim": args.total_state_dim,
            "injection_rank": args.injection_rank,
            "readout_lr": args.readout_lr,
            "max_update_norm": args.max_update_norm,
        },
        "theory": {
            "equilibrium_alpha": 0.3,
            "equilibrium_internal_steps": 24,
            "equilibrium_carry_per_observation": equilibrium_carry,
            "equilibrium_two_observation_carry": equilibrium_carry**2,
            "temporal_carry_at_delay": {
                str(leak): {
                    str(delay): (1.0 - leak) ** (delay + 1)
                    for delay in args.delays
                }
                for leak in (0.5, 0.1, 0.02, 0.005)
            },
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": "cpu",
        },
        "summary": aggregate(runs, args.delays),
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
