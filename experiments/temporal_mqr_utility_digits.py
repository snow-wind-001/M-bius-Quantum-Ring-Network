#!/usr/bin/env python3
"""Marker-free future-utility validation on the offline Digits dataset.

Each episode begins with two repetitions of an arbitrary background digit,
then presents five candidates in random order: four repeats of that background
and one different target digit.  A zero query asks for the unique target's
class.  There is no cue bit, query bit, event label, or fixed target position.
The target is statistically discoverable from causal novelty, so this protocol
tests delayed utility learning without asking a causal controller to perform an
impossible clairvoyant decision.

A common causal calibration phase first trains only the readout on isolated
digits.  The readout is then frozen for all policy variants.  Candidate write
advantages are exact paired shadow-rollout loss differences under the frozen
readout and shared future actions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.temporal_mqr_delayed_digits import prepare_digits  # noqa: E402
from mqr import UtilityDrivenMQR  # noqa: E402


VARIANTS = (
    "learned_utility",
    "learned_utility_ogd",
    "learned_no_updates",
    "causal_novelty",
    "clairvoyant_oracle",
    "ungated",
    "random_sparse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--calibration-samples", type=int, default=500)
    parser.add_argument("--train-episodes", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--warmup-count", type=int, default=2)
    parser.add_argument("--ring-dim", type=int, default=24)
    parser.add_argument("--injection-rank", type=int, default=24)
    parser.add_argument("--gate-rank", type=int, default=8)
    parser.add_argument("--readout-lr", type=float, default=0.15)
    parser.add_argument("--utility-lr", type=float, default=0.08)
    parser.add_argument("--utility-max-update-norm", type=float, default=0.04)
    parser.add_argument("--readout-max-update-norm", type=float, default=0.25)
    parser.add_argument("--novelty-threshold", type=float, default=0.08)
    parser.add_argument("--ogd-remember-period", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/results/temporal_mqr_utility_digits.json"),
    )
    return parser.parse_args()


def parameter_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def snapshot(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def max_drift(before: Mapping[str, torch.Tensor], module: torch.nn.Module) -> float:
    return max(
        float((parameter.detach() - before[name]).abs().max().item())
        for name, parameter in module.named_parameters()
    )


def make_model(args: argparse.Namespace, seed: int, *, ogd_rank: int) -> UtilityDrivenMQR:
    torch.manual_seed(seed + 101)
    return UtilityDrivenMQR(
        64,
        args.ring_dim,
        10,
        core_kwargs={
            # The fast candidate ring is flushed by the zero query.  The slow
            # ring retains selectively written candidates.
            "leak_rates": (1.0, 0.02),
            "write_scales": (1.0, 1.0),
            "injection_rank": args.injection_rank,
            "injection_activation": "tanh",
            "state_activation": "tanh",
            "transition_mode": "unistochastic",
            "learn_transitions": False,
            "readout_bias": False,
        },
        gate_rank=args.gate_rank,
        utility_lr=args.utility_lr,
        readout_lr=args.readout_lr,
        utility_ogd_max_rank=ogd_rank,
        readout_ogd_max_rank=0,
        utility_max_update_norm=args.utility_max_update_norm,
        readout_max_update_norm=args.readout_max_update_norm,
        memory_cost=0.0,
        advantage_scale=1.0,
        advantage_target_clip=4.0,
        decision_temperature=0.25,
        initial_advantage=-0.10,
        feature_ema_decay=0.9,
        max_pending_candidates=args.warmup_count + args.candidate_count + 2,
        promotion_max_norm=4.0,
    )


def calibrate_readout(
    model: UtilityDrivenMQR,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """Causally train the common readout on isolated one-write episodes."""

    correct: List[int] = []
    losses: List[float] = []
    for feature, label in zip(features, labels):
        model.reset_state()
        model.observe(
            feature.unsqueeze(0),
            issue_candidate=False,
            external_write=True,
        )
        query = model.observe(
            torch.zeros_like(feature).unsqueeze(0),
            issue_candidate=False,
            external_write=False,
        )
        correct.append(int(query["logits"].argmax(1).item() == int(label.item())))
        result = model.learn_current(label.reshape(1), learn=True)
        losses.append(float(result["loss"]))
    model.reset_state()
    return {
        "count": len(correct),
        "prequential_accuracy": statistics.fmean(correct),
        "mean_loss": statistics.fmean(losses),
        "last_quarter_accuracy": statistics.fmean(correct[-max(1, len(correct) // 4) :]),
    }


def choose_background(
    labels: torch.Tensor,
    target_label: int,
    rng: np.random.Generator,
) -> int:
    eligible = torch.nonzero(labels != target_label, as_tuple=False).squeeze(1).numpy()
    return int(eligible[int(rng.integers(0, len(eligible)))])


def marker_free_episode(
    target: torch.Tensor,
    target_label: int,
    pool_features: torch.Tensor,
    pool_labels: torch.Tensor,
    *,
    candidate_count: int,
    warmup_count: int,
    rng: np.random.Generator,
) -> Tuple[List[torch.Tensor], List[bool], int]:
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    background_index = choose_background(pool_labels, target_label, rng)
    background = pool_features[background_index]
    target_position = int(rng.integers(0, candidate_count))
    frames = [background.clone() for _ in range(warmup_count + candidate_count)]
    usefulness = [False for _ in frames]
    absolute_target = warmup_count + target_position
    frames[absolute_target] = target.clone()
    usefulness[absolute_target] = True
    return frames, usefulness, target_position


def cosine_novelty(current: torch.Tensor, running_mean: torch.Tensor | None) -> float:
    if running_mean is None:
        return 0.0
    denominator = float(
        torch.linalg.vector_norm(current).item()
        * torch.linalg.vector_norm(running_mean).item()
    )
    similarity = 0.0 if denominator <= 1e-12 else float(
        torch.dot(current, running_mean).item() / denominator
    )
    return max(0.0, min(1.0, 0.5 * (1.0 - similarity)))


def external_policy(
    variant: str,
    *,
    useful: bool,
    novelty: float,
    novelty_threshold: float,
    random_value: float,
    candidate_count: int,
) -> bool | None:
    if variant in ("learned_utility", "learned_utility_ogd", "learned_no_updates"):
        return None
    if variant == "causal_novelty":
        return novelty > novelty_threshold
    if variant == "clairvoyant_oracle":
        return useful
    if variant == "ungated":
        return True
    if variant == "random_sparse":
        return random_value < 1.0 / candidate_count
    raise ValueError(f"unknown variant: {variant}")


def run_episodes(
    model: UtilityDrivenMQR,
    targets: torch.Tensor,
    target_labels: torch.Tensor,
    pool_features: torch.Tensor,
    pool_labels: torch.Tensor,
    *,
    variant: str,
    candidate_count: int,
    warmup_count: int,
    novelty_threshold: float,
    random_seed: int,
    learn_gate: bool,
    ogd_remember_period: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(random_seed)
    correct: List[int] = []
    predicted_advantages: List[float] = []
    realized_advantages: List[float] = []
    useful_labels: List[int] = []
    effective_writes: List[int] = []
    target_writes = 0
    target_count = 0
    distractor_writes = 0
    distractor_count = 0
    positive_utility = 0
    positive_utility_written = 0
    utility_losses: List[float] = []
    update_norms: List[float] = []
    target_positions: List[int] = []
    started = time.perf_counter()

    for episode_index, (target, label) in enumerate(zip(targets, target_labels)):
        model.reset_state()
        frames, usefulness, target_position = marker_free_episode(
            target,
            int(label.item()),
            pool_features,
            pool_labels,
            candidate_count=candidate_count,
            warmup_count=warmup_count,
            rng=rng,
        )
        target_positions.append(target_position)
        running_mean: torch.Tensor | None = None
        ticket_usefulness: Dict[int, bool] = {}
        for frame_index, (frame, useful) in enumerate(zip(frames, usefulness)):
            novelty = cosine_novelty(frame, running_mean)
            if frame_index < warmup_count:
                # Warm-up establishes strictly past context.  It is always
                # fast-only and is not a candidate action, so the oracle and
                # random controls both make one expected slow write per five
                # actual candidates.
                model.observe(
                    frame.unsqueeze(0),
                    issue_candidate=False,
                    external_write=False,
                )
                if running_mean is None:
                    running_mean = frame.detach().clone()
                else:
                    running_mean = 0.9 * running_mean + 0.1 * frame
                continue
            random_value = float(rng.random())
            override = external_policy(
                variant,
                useful=useful,
                novelty=novelty,
                novelty_threshold=novelty_threshold,
                random_value=random_value,
                candidate_count=candidate_count,
            )
            result = model.observe(
                frame.unsqueeze(0),
                issue_candidate=True,
                external_write=override,
            )
            ticket_id = int(result["candidate_id"])
            ticket_usefulness[ticket_id] = useful
            predicted_advantages.append(float(result["predicted_advantage"]))
            useful_labels.append(int(useful))
            wrote = int(bool(result["effective_write"]))
            effective_writes.append(wrote)
            if useful:
                target_count += 1
                target_writes += wrote
            else:
                distractor_count += 1
                distractor_writes += wrote
            if running_mean is None:
                running_mean = frame.detach().clone()
            else:
                running_mean = 0.9 * running_mean + 0.1 * frame

        query = model.observe(
            torch.zeros_like(target).unsqueeze(0),
            issue_candidate=False,
            external_write=False,
        )
        correct.append(int(query["logits"].argmax(1).item() == int(label.item())))
        candidate_ids = model.pending_candidate_ids()
        for offset, ticket_id in enumerate(candidate_ids):
            remember = bool(
                variant == "learned_utility_ogd"
                and learn_gate
                and model.utility_gradient_memory.rank
                < model.utility_gradient_memory.max_rank
                and episode_index % ogd_remember_period == 0
                and offset == 0
            )
            feedback = model.resolve_utility(
                ticket_id,
                label.reshape(1),
                learn=learn_gate,
                remember_gradient=remember,
                promote_missed_positive=False,
            )
            advantage = float(feedback["write_advantage"])
            realized_advantages.append(advantage)
            utility_losses.append(float(feedback["utility_loss"]))
            update_norms.append(float(feedback["update_norm"]))
            if advantage > 0.0:
                positive_utility += 1
                positive_utility_written += int(bool(feedback["action_taken"]))
            # The semantic label is retained only for evaluation; the gate was
            # trained exclusively on the numeric shadow-rollout advantage.
            assert ticket_usefulness[ticket_id] in (True, False)

    elapsed = time.perf_counter() - started
    quarter = max(1, len(correct) // 4)
    utility_positive_labels = [int(value > 0.0) for value in realized_advantages]
    if len(set(utility_positive_labels)) > 1:
        utility_auprc = float(
            average_precision_score(utility_positive_labels, predicted_advantages)
        )
    else:
        utility_auprc = float("nan")
    if len(set(useful_labels)) > 1:
        semantic_auprc = float(
            average_precision_score(useful_labels, predicted_advantages)
        )
    else:
        semantic_auprc = float("nan")
    target_predictions = [
        prediction
        for prediction, useful in zip(predicted_advantages, useful_labels)
        if useful
    ]
    distractor_predictions = [
        prediction
        for prediction, useful in zip(predicted_advantages, useful_labels)
        if not useful
    ]
    target_advantages = [
        advantage
        for advantage, useful in zip(realized_advantages, useful_labels)
        if useful
    ]
    distractor_advantages = [
        advantage
        for advantage, useful in zip(realized_advantages, useful_labels)
        if not useful
    ]
    return {
        "episodes": len(correct),
        "candidate_frames": len(useful_labels),
        "accuracy": statistics.fmean(correct),
        "first_quarter_accuracy": statistics.fmean(correct[:quarter]),
        "last_quarter_accuracy": statistics.fmean(correct[-quarter:]),
        "target_write_rate": target_writes / max(1, target_count),
        "distractor_write_rate": distractor_writes / max(1, distractor_count),
        "positive_utility_write_rate": positive_utility_written
        / max(1, positive_utility),
        "utility_auprc": utility_auprc,
        "semantic_target_auprc_diagnostic": semantic_auprc,
        "mean_realized_advantage": statistics.fmean(realized_advantages),
        "mean_target_advantage": statistics.fmean(target_advantages),
        "mean_distractor_advantage": statistics.fmean(distractor_advantages),
        "mean_predicted_target_advantage": statistics.fmean(target_predictions),
        "mean_predicted_distractor_advantage": statistics.fmean(
            distractor_predictions
        ),
        "mean_utility_loss": statistics.fmean(utility_losses),
        "mean_update_norm": statistics.fmean(update_norms),
        "milliseconds_per_episode": 1000.0 * elapsed / max(1, len(correct)),
        "target_position_counts": {
            str(position): target_positions.count(position)
            for position in range(candidate_count)
        },
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
        "first_quarter_accuracy",
        "last_quarter_accuracy",
        "target_write_rate",
        "distractor_write_rate",
        "positive_utility_write_rate",
        "utility_auprc",
        "semantic_target_auprc_diagnostic",
        "mean_realized_advantage",
        "mean_target_advantage",
        "mean_distractor_advantage",
        "mean_predicted_target_advantage",
        "mean_predicted_distractor_advantage",
        "mean_utility_loss",
        "milliseconds_per_episode",
    )
    result: Dict[str, object] = {}
    for variant in VARIANTS:
        selected = [run for run in runs if run["variant"] == variant]
        summary: Dict[str, object] = {
            "total_parameters": int(selected[0]["total_parameters"]),
            "utility_gate_parameters": int(selected[0]["utility_gate_parameters"]),
            "readout_parameters": int(selected[0]["readout_parameters"]),
        }
        for split in ("training", "evaluation"):
            for metric in metrics:
                summary[f"{split}_{metric}"] = summarize(
                    float(run[split][metric])  # type: ignore[index]
                    for run in selected
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
    definitions = {
        "learned_vs_no_updates": ("learned_utility", "learned_no_updates"),
        "learned_vs_ungated": ("learned_utility", "ungated"),
        "ogd_vs_plain": ("learned_utility_ogd", "learned_utility"),
        "ogd_vs_no_updates": ("learned_utility_ogd", "learned_no_updates"),
        "ogd_vs_ungated": ("learned_utility_ogd", "ungated"),
        "ogd_vs_random_sparse": ("learned_utility_ogd", "random_sparse"),
        "causal_heuristic_vs_learned": ("causal_novelty", "learned_utility"),
        "causal_heuristic_vs_ogd": ("causal_novelty", "learned_utility_ogd"),
        "clairvoyant_gap": ("clairvoyant_oracle", "learned_utility"),
        "clairvoyant_vs_ogd": ("clairvoyant_oracle", "learned_utility_ogd"),
        "random_sparse_vs_ungated": ("random_sparse", "ungated"),
    }
    result: Dict[str, object] = {}
    seeds = sorted({seed for seed, _variant in indexed})
    for name, (left, right) in definitions.items():
        differences = [indexed[(seed, left)] - indexed[(seed, right)] for seed in seeds]
        result[name] = {
            "left": left,
            "right": right,
            "paired_differences": differences,
            **summarize(differences),
        }
    return result


def main() -> None:
    args = parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be non-empty and unique")
    if args.calibration_samples <= 0 or args.train_episodes <= 0 or args.eval_episodes <= 0:
        raise ValueError("sample counts must be positive")
    if args.candidate_count < 2 or args.warmup_count <= 0:
        raise ValueError("candidate_count >= 2 and warmup_count > 0 are required")
    if args.ogd_remember_period <= 0:
        raise ValueError("ogd_remember_period must be positive")
    torch.set_num_threads(1)

    runs: List[Dict[str, object]] = []
    calibration_records: Dict[str, Dict[str, float]] = {}
    initialization_audit: Dict[str, Dict[str, bool]] = {}
    required_train = max(args.calibration_samples, args.train_episodes)
    required_eval = args.eval_episodes
    for seed in args.seeds:
        data = prepare_digits(
            seed,
            train_samples=required_train,
            eval_samples=required_eval,
        )
        train_features, train_labels = data["train"]
        eval_features, eval_labels = data["eval"]
        base = make_model(args, seed, ogd_rank=0)
        calibration = calibrate_readout(
            base,
            train_features[: args.calibration_samples],
            train_labels[: args.calibration_samples],
        )
        calibration_records[str(seed)] = calibration
        base.reset_all_states()
        models: Dict[str, UtilityDrivenMQR] = {}
        for variant in VARIANTS:
            ogd_rank = 8 if variant == "learned_utility_ogd" else 0
            model = make_model(args, seed, ogd_rank=ogd_rank)
            model.core.load_state_dict(copy.deepcopy(base.core.state_dict()))
            models[variant] = model
        initial_hashes = {variant: parameter_sha256(model) for variant, model in models.items()}
        # OGD memory capacity is a buffer/configuration difference, not a model
        # parameter difference; all named parameters must remain paired.
        initialization_audit[str(seed)] = {
            "all_variant_parameters_matched": len(set(initial_hashes.values())) == 1
        }

        for variant in VARIANTS:
            model = models[variant]
            train_gate = variant in ("learned_utility", "learned_utility_ogd")
            training = run_episodes(
                model,
                train_features[: args.train_episodes],
                train_labels[: args.train_episodes],
                train_features,
                train_labels,
                variant=variant,
                candidate_count=args.candidate_count,
                warmup_count=args.warmup_count,
                novelty_threshold=args.novelty_threshold,
                random_seed=seed + 50_000,
                learn_gate=train_gate,
                ogd_remember_period=args.ogd_remember_period,
            )
            before_evaluation = snapshot(model)
            evaluation = run_episodes(
                model,
                eval_features[: args.eval_episodes],
                eval_labels[: args.eval_episodes],
                train_features,
                train_labels,
                variant=variant,
                candidate_count=args.candidate_count,
                warmup_count=args.warmup_count,
                novelty_threshold=args.novelty_threshold,
                random_seed=seed + 60_000,
                learn_gate=False,
                ogd_remember_period=args.ogd_remember_period,
            )
            evaluation_drift = max_drift(before_evaluation, model)
            run = {
                "variant": variant,
                "seed": seed,
                "initial_parameter_sha256": initial_hashes[variant],
                "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "utility_gate_parameters": sum(
                    parameter.numel() for parameter in model.utility_gate.parameters()
                ),
                "readout_parameters": sum(
                    parameter.numel() for parameter in model.core.readout.parameters()
                ),
                "total_state_dim": model.core.total_state_dim,
                "shadow_state_elements_per_candidate": 2 * model.core.total_state_dim,
                "utility_ogd_rank": model.utility_gradient_memory.rank,
                "utility_ogd_bytes": model.utility_gradient_memory.storage_bytes,
                "training": training,
                "evaluation": evaluation,
                "evaluation_parameter_max_drift": evaluation_drift,
                "online_utility_updates": int(model.online_utility_updates.item()),
                "online_utility_feedback": int(model.online_utility_feedback.item()),
                "max_unitary_error": model.core.max_unitary_error(),
                "max_stochastic_error": model.core.max_stochastic_error(),
            }
            runs.append(run)
            print(
                f"{variant:24s} seed={seed:3d} "
                f"train={training['accuracy']:.3f} "
                f"heldout={evaluation['accuracy']:.3f} "
                f"target_write={evaluation['target_write_rate']:.3f} "
                f"false_open={evaluation['distractor_write_rate']:.3f}"
            )

    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {
            "dataset": "sklearn.datasets.load_digits (offline bundled dataset)",
            "sequence": (
                "two repeated background frames -> five randomly ordered candidates "
                "containing one different target -> exactly zero query"
            ),
            "marker_free": True,
            "target_position_randomized": True,
            "causal_features": [
                "novelty",
                "current readout uncertainty",
                "strictly previous feedback surprise",
                "slow-ring saturation",
                "context change",
            ],
            "utility_target": (
                "CE(no-write shadow at query) - CE(write shadow at query) - memory cost"
            ),
            "causal_order": (
                "feature/action/state first; future class label then evaluates paired "
                "shadow trajectories and trains the gate by feature replay"
            ),
            "readout_calibration": (
                "common predict-before-update isolated-digit phase; frozen thereafter"
            ),
            "limitations": [
                "the unique target is statistically exposed through novelty, not arbitrary future relevance",
                "the clairvoyant oracle is an upper bound and not a causal comparator",
                "shadow rollout cost is linear in pending tickets and is not a production LM implementation",
                "this mechanism screen does not establish superiority to GRU, fast-weight, LoRA, or replay",
            ],
        },
        "configuration": {
            "seeds": args.seeds,
            "calibration_samples": args.calibration_samples,
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "candidate_count": args.candidate_count,
            "warmup_count": args.warmup_count,
            "ring_dim": args.ring_dim,
            "injection_rank": args.injection_rank,
            "gate_rank": args.gate_rank,
            "readout_lr": args.readout_lr,
            "utility_lr": args.utility_lr,
            "utility_max_update_norm": args.utility_max_update_norm,
            "readout_max_update_norm": args.readout_max_update_norm,
            "novelty_threshold": args.novelty_threshold,
            "ogd_remember_period": args.ogd_remember_period,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "calibration": calibration_records,
        "initialization_audit": initialization_audit,
        "summary": aggregate(runs),
        "paired_contrasts": paired_contrasts(runs),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
