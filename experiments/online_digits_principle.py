#!/usr/bin/env python3
"""Single-pass Digits principle test for MQR online continual learning.

The experiment is intentionally small and download-free.  It compares:

1. one shared ring updated on contexts A then B;
2. one shared ring with a low-rank OGD memory consolidated near the end of A;
3. two explicitly routed rings, one per context.

Predictions are scored before their minibatch update (prequential protocol).
This is a mechanism check, not a competitive continual-learning benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mqr import OnlineMultiRingClassifier


VARIANTS = ("shared_ring", "shared_ring_ogd", "isolated_two_rings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--ogd-rank", type=int, default=64)
    parser.add_argument("--consolidation-batches", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/results/online_digits_principle.json"),
    )
    return parser.parse_args()


def prepare_digits(seed: int) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
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

    # A signed permutation is an orthogonal context transform: it preserves
    # Euclidean norms but changes which input coordinates carry each feature.
    rng = np.random.default_rng(seed + 10_000)
    permutation = rng.permutation(x_train.shape[1])
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=x_train.shape[1])
    train_b = x_train[:, permutation] * signs
    test_b = x_test[:, permutation] * signs

    return {
        "train_a": (torch.from_numpy(x_train), torch.from_numpy(y_train)),
        "test_a": (torch.from_numpy(x_test), torch.from_numpy(y_test)),
        "train_b": (torch.from_numpy(train_b.copy()), torch.from_numpy(y_train)),
        "test_b": (torch.from_numpy(test_b.copy()), torch.from_numpy(y_test)),
    }


def make_model(args: argparse.Namespace, variant: str, seed: int) -> OnlineMultiRingClassifier:
    torch.manual_seed(seed)
    num_rings = 2 if variant == "isolated_two_rings" else 1
    ogd_rank = args.ogd_rank if variant == "shared_ring_ogd" else 0
    return OnlineMultiRingClassifier(
        input_dim=64,
        hidden_dim=args.hidden_dim,
        output_dim=10,
        num_rings=num_rings,
        lr=args.lr,
        unitary_lr_ratio=0.1,
        injection_lr_ratio=1.0,
        readout_lr_ratio=1.0,
        adjoint_steps=20,
        ogd_max_rank=ogd_rank,
        carry_state=False,
        ring_kwargs={
            "alpha": 0.35,
            "relaxation_steps": 12,
            "lora_rank": args.lora_rank,
            "inj_activation": "tanh",
            "state_activation": "none",
            "readout_dim": args.hidden_dim,
            "h_mix_beta": 0.7,
        },
    )


def batches(
    x: torch.Tensor, y: torch.Tensor, batch_size: int, *, seed: int
) -> Iterable[Tuple[int, torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(x.size(0), generator=generator)
    total = math.ceil(x.size(0) / batch_size)
    for batch_index, start in enumerate(range(0, x.size(0), batch_size)):
        indices = order[start : start + batch_size]
        yield total - batch_index, x.index_select(0, indices), y.index_select(0, indices)


@torch.no_grad()
def evaluate(
    model: OnlineMultiRingClassifier,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    context_id: str,
    batch_size: int = 128,
) -> Tuple[float, torch.Tensor]:
    ring_index = model.route_context(context_id)
    ring = model.rings[ring_index]
    logits_parts = []
    correct = 0
    for start in range(0, x.size(0), batch_size):
        logits = ring(x[start : start + batch_size])
        logits_parts.append(logits.cpu())
        correct += int((logits.argmax(dim=1) == y[start : start + batch_size]).sum().item())
    return correct / x.size(0), torch.cat(logits_parts, dim=0)


def stream_context(
    model: OnlineMultiRingClassifier,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    context_id: str,
    batch_size: int,
    shuffle_seed: int,
    remember_last: int = 0,
    project_with_memory: bool = True,
) -> Dict[str, float]:
    correct = 0
    seen = 0
    weighted_loss = 0.0
    elapsed = 0.0
    retained: List[float] = []
    overlaps: List[float] = []
    for remaining, xb, yb in batches(x, y, batch_size, seed=shuffle_seed):
        remember = remember_last > 0 and remaining <= remember_last
        start = time.perf_counter()
        info = model.online_step(
            xb,
            yb,
            context_id=context_id,
            remember_gradient=remember,
            project_with_memory=project_with_memory,
            carry_state=False,
        )
        elapsed += time.perf_counter() - start
        correct += int((info["logits"].argmax(dim=1) == yb).sum().item())
        seen += yb.numel()
        weighted_loss += float(info["loss"]) * yb.numel()
        retained.append(float(info["ogd_retained_norm"]))
        overlaps.append(float(info.get("ogd_max_abs_overlap", 0.0)))

    return {
        "prequential_accuracy": correct / seen,
        "prequential_loss": weighted_loss / seen,
        "milliseconds_per_minibatch": 1000.0 * elapsed / max(1, len(retained)),
        "mean_retained_gradient_norm": statistics.fmean(retained),
        "max_projected_overlap": max(overlaps, default=0.0),
    }


def parameter_drift(before: Dict[str, torch.Tensor], ring: torch.nn.Module) -> float:
    after = dict(ring.named_parameters())
    return max(
        float((after[name].detach() - value).abs().max().item())
        for name, value in before.items()
    )


def run_once(args: argparse.Namespace, variant: str, seed: int) -> Dict[str, object]:
    data = prepare_digits(seed)
    model = make_model(args, variant, seed)
    context_a = "context-a" if variant == "isolated_two_rings" else "shared"
    context_b = "context-b" if variant == "isolated_two_rings" else "shared"

    remember_a = args.consolidation_batches if variant == "shared_ring_ogd" else 0
    stream_a = stream_context(
        model,
        *data["train_a"],
        context_id=context_a,
        batch_size=args.batch_size,
        shuffle_seed=seed + 1,
        remember_last=remember_a,
        # Stage protected A directions without suppressing learning inside A.
        project_with_memory=variant != "shared_ring_ogd",
    )
    accuracy_a_after_a, logits_a_after_a = evaluate(
        model, *data["test_a"], context_id=context_a
    )
    accuracy_b_before_b, _ = evaluate(model, *data["test_b"], context_id=context_b)

    ring_a_index = model.route_context(context_a, allocate=False)
    ring_a_before_b = {
        name: value.detach().clone() for name, value in model.rings[ring_a_index].named_parameters()
    }
    stream_b = stream_context(
        model,
        *data["train_b"],
        context_id=context_b,
        batch_size=args.batch_size,
        shuffle_seed=seed + 2,
    )
    accuracy_a_after_b, logits_a_after_b = evaluate(
        model, *data["test_a"], context_id=context_a
    )
    accuracy_b_after_b, _ = evaluate(model, *data["test_b"], context_id=context_b)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    unitary_errors = [float(ring.unitary_param.unitary_error_fro().item()) for ring in model.rings]
    return {
        "variant": variant,
        "seed": seed,
        "num_rings": model.num_rings,
        "total_parameters": total_parameters,
        "stream_a": stream_a,
        "stream_b": stream_b,
        "accuracy_matrix": {
            "after_a": {"context_a": accuracy_a_after_a, "context_b": accuracy_b_before_b},
            "after_b": {"context_a": accuracy_a_after_b, "context_b": accuracy_b_after_b},
        },
        "context_a_forgetting": accuracy_a_after_a - accuracy_a_after_b,
        "context_a_parameter_max_drift_during_b": parameter_drift(
            ring_a_before_b, model.rings[ring_a_index]
        ),
        "context_a_logit_max_drift_during_b": float(
            (logits_a_after_b - logits_a_after_a).abs().max().item()
        ),
        "ogd_rank_after_stream": model.gradient_memories[ring_a_index].rank,
        "ogd_memory_bytes": model.gradient_memories[ring_a_index].storage_bytes,
        "ogd_basis_orthogonality_error": model.gradient_memories[
            ring_a_index
        ].orthogonality_error(),
        "max_unitary_error": max(unitary_errors),
    }


def aggregate(runs: List[Dict[str, object]]) -> Dict[str, object]:
    def metric(run: Dict[str, object], path: Tuple[str, ...]) -> float:
        value: object = run
        for key in path:
            value = value[key]  # type: ignore[index]
        return float(value)

    paths = {
        "prequential_accuracy_a": ("stream_a", "prequential_accuracy"),
        "prequential_accuracy_b": ("stream_b", "prequential_accuracy"),
        "accuracy_a_after_a": ("accuracy_matrix", "after_a", "context_a"),
        "accuracy_a_after_b": ("accuracy_matrix", "after_b", "context_a"),
        "accuracy_b_after_b": ("accuracy_matrix", "after_b", "context_b"),
        "context_a_forgetting": ("context_a_forgetting",),
        "context_a_parameter_drift": ("context_a_parameter_max_drift_during_b",),
        "milliseconds_per_minibatch_b": ("stream_b", "milliseconds_per_minibatch"),
        "mean_ogd_retained_norm_b": ("stream_b", "mean_retained_gradient_norm"),
        "max_projected_overlap_b": ("stream_b", "max_projected_overlap"),
        "max_unitary_error": ("max_unitary_error",),
        "ogd_memory_bytes": ("ogd_memory_bytes",),
    }
    summary: Dict[str, object] = {}
    for variant in VARIANTS:
        selected = [run for run in runs if run["variant"] == variant]
        values = {}
        for name, path in paths.items():
            samples = [metric(run, path) for run in selected]
            values[name] = {
                "mean": statistics.fmean(samples),
                "population_std": statistics.pstdev(samples),
            }
        values["total_parameters"] = int(selected[0]["total_parameters"])
        summary[variant] = values
    return summary


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.consolidation_batches < 0:
        raise ValueError("batch size must be positive and consolidation batches non-negative")
    torch.set_num_threads(1)
    runs = []
    for seed in args.seeds:
        for variant in VARIANTS:
            run = run_once(args, variant, seed)
            runs.append(run)
            matrix = run["accuracy_matrix"]
            print(
                f"{variant:20s} seed={seed:3d} "
                f"A: {matrix['after_a']['context_a']:.3f}->{matrix['after_b']['context_a']:.3f} "
                f"B: {matrix['after_b']['context_b']:.3f} "
                f"forget={run['context_a_forgetting']:+.3f}"
            )

    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {
            "dataset": "sklearn.datasets.load_digits (1797 samples, no download)",
            "split": "70% train / 30% test, stratified per seed",
            "stream": "single pass A then B; shuffled prequential minibatches",
            "context_a": "standardized 64-pixel features",
            "context_b": "fixed orthogonal signed permutation of context A features",
            "feedback_order": "predict with theta_t, score, then update to theta_(t+1)",
            "state_policy": "disabled for independent Digits minibatch lanes",
            "ogd_consolidation": (
                f"remember final {args.consolidation_batches} A minibatch gradients; "
                "do not project within A; project all B updates"
            ),
            "limitations": [
                "microbatch-online rather than one-sample updates",
                "explicit task IDs for exact two-ring isolation",
                "two-ring model has more parameters than the shared-ring controls",
                "OGD protects a low-rank local gradient subspace, not all old examples",
                "this is not evidence about language modeling or animal-level learning",
            ],
        },
        "configuration": {
            "seeds": args.seeds,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lora_rank": args.lora_rank,
            "learning_rate": args.lr,
            "ogd_max_rank": args.ogd_rank,
            "consolidation_batches": args.consolidation_batches,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": "cpu",
        },
        "summary": aggregate(runs),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
