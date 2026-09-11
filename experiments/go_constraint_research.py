"""Explicit behavior-constraint transport: mechanism, history and online Go.

Backends differentiate the same finite-sequence function. History is a recall
diagnostic with declared replay epochs; online Go uses causal student actions.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.go_memory_research import (
    base_pretrain, build, evaluate_go, materialize, parameter_counts, play_games, recall_metrics,
)
from mqr import GoBoard
from mqr.constraint_transport import terminal_gradients
from mqr.go_agent import generate_go_agent_trajectories
from mqr.go_constraints import ConstraintGoBehaviorMemory, ConstraintGoSession
from mqr.go_history_tasks import generate_reachable_history_pairs
from mqr.go_memory import GoObservationMemory


ONLINE_METHODS = {
    "autograd8": ("autograd", 8, False, False),
    "autograd1": ("autograd", 1, False, False),
    "transport1": ("transport", 1, False, False),
    "transport1_guard": ("transport", 1, True, False),
    "norm1": ("autograd", 1, False, True),
}
HISTORY_METHODS = {
    "autograd_full": ("autograd", None),
    "transport_full": ("transport", None),
    "autograd_short": ("autograd", 8),
}


def sources():
    paths = list((ROOT / "mqr").glob("*.py")) + [
        ROOT / "experiments/go_memory_research.py", Path(__file__),
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def session_for(args, agent, encoder, method, memory=None):
    backend, refresh, guard, norm = ONLINE_METHODS[method]
    if memory is None:
        memory = ConstraintGoBehaviorMemory(args.anchors, backend=backend)
    return ConstraintGoSession(
        agent, encoder, behavior_memory=memory, refresh_every=refresh,
        max_anchor_kl=args.anchor_kl if guard else None,
        norm_matched_sgd=norm, project_with_memory=not norm,
        update_every=args.window, credit_horizon=args.credit,
    )


def run_online(args, method, seed, base):
    # All methods have exactly the same architecture, initial policy and rank.
    agent, encoder = build(args, "conditional_guard", seed)
    agent.spatial_skip_heads.load_state_dict(base)
    session = session_for(args, agent, encoder, method)
    memory = session.behavior_memory
    datasets = {
        phase: generate_go_agent_trajectories(
            args.test_games, seed=seed + offset, size=args.size, komi=2.5,
            recorded_moves=24, start_random_moves=prefix, random_move_probability=0.5,
        )
        for phase, offset, prefix in (("a", 23000, 0), ("b", 24000, 10))
    }
    evaluate = lambda: {name: evaluate_go(args, agent, encoder, data) for name, data in datasets.items()}
    result = {"method": method, "seed": seed, **parameter_counts(agent), "before": evaluate()}
    started = time.perf_counter()
    desired_projection, desired_norm = session.project_with_memory, session.norm_matched_sgd
    # One numerical update path for phase A, including the norm control.
    # The bank is empty/unfrozen, so no historical directions are constrained.
    session.project_with_memory, session.norm_matched_sgd = True, False
    result["train_a"] = play_games(args, session, seed + 25000, learn=True, collect_anchors=True)
    result["after_a"] = evaluate()
    session.project_with_memory, session.norm_matched_sgd = desired_projection, desired_norm
    memory.freeze(agent)
    result["consolidation"] = memory.refresh(agent)
    result["train_b"] = play_games(args, session, seed + 26000, learn=True, phase="b")
    result["after_b"] = evaluate()
    result["anchor_drift"] = memory.drift(agent)
    result["matches"] = play_games(args, session, seed + 27000, learn=False,
                                   opening_moves=4, paired_openings=True)
    result.update(
        seconds=time.perf_counter() - started, anchor_bytes=memory.storage_bytes,
        ogd_bytes=agent.task_gradient_memory.storage_bytes,
        refresh_count=memory.refresh_count, refresh_seconds=memory.refresh_seconds,
        peak_constraint_state_trace_bytes=memory.peak_state_trace_bytes,
        max_adjoint_norm_error=memory.max_adjoint_norm_error,
    )
    if args.checkpoint_dir:
        directory = ROOT / args.checkpoint_dir
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"config": vars(args), "method": method, "seed": seed,
                    "session": session.state_dict()}, directory / f"{method}-{seed}.pt")
    return result


def run_history(args, method, seed):
    agent, encoder = build(args, "conditional", seed, history_task=True)
    pairs = [
        generate_reachable_history_pairs(count, seed=seed + offset, size=args.size, moves=args.history_moves)
        for count, offset in ((args.history_train_pairs, 11000), (args.history_test_pairs, 12000))
    ]
    hashes = [{hashlib.sha256(repr(pair.moves).encode()).hexdigest() for pair in group} for group in pairs]
    if hashes[0] & hashes[1]:
        raise RuntimeError("history train/test overlap")
    (train_x, train_y), (test_x, test_y) = [materialize(group, encoder) for group in pairs]
    before = recall_metrics(agent, test_x, test_y)
    active = [p for _, p in agent._named_task_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(active, lr=0.006)
    generator = torch.Generator().manual_seed(seed + 13000)
    backend, credit = HISTORY_METHODS[method]
    prequential, max_trace, max_error = [], 0, 0.0
    started = time.perf_counter()
    for _ in range(args.history_epochs):
        for indices in torch.randperm(len(train_y), generator=generator).split(16):
            targets = train_y[indices]
            # The final output is formed before this objective uses its label.
            result = terminal_gradients(
                agent, train_x[indices].transpose(0, 1),
                lambda output: nn.functional.cross_entropy(output.placement_logits, targets),
                backend=backend, credit_horizon=credit,
            )
            prequential.append(float(result.values[0]))
            max_trace = max(max_trace, result.diagnostics["state_trace_bytes"])
            max_error = max(max_error, result.diagnostics["max_adjoint_norm_error"])
            optimizer.zero_grad(set_to_none=True)
            cursor = 0
            for p in active:
                p.grad = result.jacobian[0, cursor:cursor + p.numel()].reshape_as(p).clone()
                cursor += p.numel()
            nn.utils.clip_grad_norm_(active, 1.0)
            optimizer.step()
    result = {
        "method": method, "seed": seed, "before": before, "after": recall_metrics(agent, test_x, test_y),
        "updates": len(prequential), "backend": backend, "credit_horizon": credit,
        "actual_credit_steps": train_x.size(1) if credit is None else (train_x.size(1) - 1) % credit + 1,
        "training_replay_epochs": args.history_epochs, "prequential_point_nll": float(np.mean(prequential)),
        "peak_state_trace_bytes": max_trace, "max_adjoint_norm_error": max_error,
        "seconds": time.perf_counter() - started, **parameter_counts(agent),
        "train_trajectory_hash": hashlib.sha256("".join(sorted(hashes[0])).encode()).hexdigest(),
        "test_trajectory_hash": hashlib.sha256("".join(sorted(hashes[1])).encode()).hexdigest(),
        "task": "legal antipode of first black move; diagnostic recall with replay, not Go strength",
    }
    if args.checkpoint_dir:
        directory = ROOT / args.checkpoint_dir
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"config": vars(args), "method": method, "seed": seed,
                    "model_state": agent.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "generator_state": generator.get_state()}, directory / f"history-{method}-{seed}.pt")
    return result


class SavedTensorBytes:
    """Peak unique storage referenced by tensors saved for backward, not RSS.

    Includes input/parameter storage when autograd saves it. Detached prefix
    states retained outside autograd are reported separately.
    """

    def __init__(self):
        self.refs, self.live, self.peak = {}, 0, 0

    def pack(self, tensor):
        owner = self
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        size = storage.nbytes()
        if key not in self.refs:
            self.refs[key] = [0, size]
            self.live += size
            self.peak = max(self.peak, self.live)
        self.refs[key][0] += 1

        class Saved:
            def __init__(self, value):
                self.value = value

            def __del__(self):
                owner.refs[key][0] -= 1
                if owner.refs[key][0] == 0:
                    owner.live -= owner.refs.pop(key)[1]

        # Keeping grad_fn here can form a saved-output/grad_fn reference cycle.
        return Saved(tensor.detach())

    @staticmethod
    def unpack(saved):
        return saved.value


def legal_prefix(agent, encoder, length, seed):
    rng = random.Random(seed)
    for _ in range(128):
        board, memory = GoBoard(encoder.board_size, komi=2.5), GoObservationMemory(encoder)
        values, moves = [], []
        for index in range(length):
            values.append(encoder.with_history(board, memory.slots))
            memory.write(board)
            if index == length - 1:
                return torch.cat(values).to(next(agent.parameters())), moves
            candidates = [a for a in board.legal_moves() if a != board.pass_action]
            action = rng.choice(candidates) if candidates else board.pass_action
            moves.append(action)
            board.play(action)
            if board.game_over:
                break
    raise RuntimeError("could not construct a legal prefix of the requested length")


def mathematical_checks(seed):
    generator = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    R = torch.linalg.qr(rand(9, 9)).Q
    Q = torch.linalg.qr(rand(9, 3), mode="reduced").Q
    P = torch.eye(9, dtype=R.dtype) - Q @ Q.T
    next_q = R @ Q
    next_p = torch.eye(9, dtype=R.dtype) - next_q @ next_q.T
    h, u = rand(9), rand(9)
    blocks, gradients = [rand(3, p) for p in (5, 7, 4)], [rand(p) for p in (5, 7, 4)]
    A, g = torch.cat(blocks, dim=1), torch.cat(gradients)
    multiplier = torch.linalg.pinv(sum(a @ a.T for a in blocks)) @ sum(a @ v for a, v in zip(blocks, gradients))
    step = torch.cat([-(v - a.T @ multiplier) for a, v in zip(blocks, gradients)])
    centralized = -(g - A.T @ (torch.linalg.pinv(A @ A.T) @ (A @ g)))
    return {
        "projector_transport_error": float((next_p @ R - R @ P).abs().max()),
        "protected_coordinate_error": float((next_q.T @ (R @ h + next_p @ u) - Q.T @ h).abs().max()),
        "block_projection_error": float((step - centralized).abs().max()),
        "global_constraint_error": float((A @ step).abs().max()),
    }


def run_mechanisms(args, method, seed):
    case = copy.copy(args)
    case.history_moves = max(args.lengths) - 1
    model_method = "conditional_external_guard" if method == "conditional_external" else (
        "conditional_guard" if method == "conditional" else "orthogonal_guard"
    )
    agent, encoder = build(case, model_method, seed)
    with torch.no_grad():
        agent.point_correction.weight.normal_(std=0.12)
        agent.ring_channel_gain.weight.normal_(std=0.12)
        for p in agent.core.angle_controllers.parameters():
            p.normal_(std=0.04)
    rows = []
    objective = lambda output: torch.stack((output.policy_logits[0, 0], output.policy_logits[0, -1]))
    for length in args.lengths:
        x, moves = legal_prefix(agent, encoder, length, seed + 31000)
        results, costs = {}, {}
        for backend in ("autograd", "transport"):
            # One warmup is excluded from timings and memory measurements.
            terminal_gradients(agent, x, objective, backend=backend)
            times = []
            for _ in range(args.repeats):
                gc.collect()
                started = time.perf_counter()
                result = terminal_gradients(agent, x, objective, backend=backend)
                times.append(time.perf_counter() - started)
            tracker = SavedTensorBytes()
            with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
                measured = terminal_gradients(agent, x, objective, backend=backend)
            torch.testing.assert_close(measured.jacobian, result.jacobian, rtol=0, atol=0)
            if tracker.live:
                raise RuntimeError("backward graph storage remains after gradient construction")
            results[backend] = result
            costs[backend] = {"median_seconds": float(np.median(times)),
                              "peak_saved_tensor_storage_bytes": tracker.peak, **result.diagnostics}
        a, b = results["autograd"], results["transport"]
        block_errors, cursor = {}, 0
        for name, parameter in agent._named_task_parameters():
            if not parameter.requires_grad:
                continue
            ga = a.jacobian[:, cursor:cursor + parameter.numel()]
            gb = b.jacobian[:, cursor:cursor + parameter.numel()]
            block_errors[name] = {"reference_norm": float(ga.norm()),
                                  "max_absolute_error": float((ga - gb).abs().max()),
                                  "relative_error": float((ga - gb).norm() / ga.norm().clamp_min(1e-12))}
            cursor += parameter.numel()
        rows.append({
            "length": length, "moves": moves,
            "features_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
            "max_objective_error": float((a.values - b.values).abs().max()),
            "max_gradient_error": float((a.jacobian - b.jacobian).abs().max()),
            "relative_gradient_error": float((a.jacobian - b.jacobian).norm() / a.jacobian.norm().clamp_min(1e-12)),
            "block_gradient_errors": block_errors, "costs": costs,
        })
    return {"method": method, "seed": seed, "rows": rows, "proof": mathematical_checks(seed),
            **parameter_counts(agent), "seconds": sum(sum(c["median_seconds"] for c in r["costs"].values()) for r in rows)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", choices=("history", "online", "mechanisms"), required=True)
    p.add_argument("--methods", default="")
    p.add_argument("--seeds", default="401,409,419,431,443")
    p.add_argument("--size", type=int, default=5)
    p.add_argument("--ring-dim", type=int, default=16)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--channels", type=int, default=12)
    p.add_argument("--slots", type=int, default=16)
    p.add_argument("--history-moves", type=int, default=12)
    p.add_argument("--history-train-pairs", type=int, default=96)
    p.add_argument("--history-test-pairs", type=int, default=64)
    p.add_argument("--history-epochs", type=int, default=64)
    p.add_argument("--warmup-games", type=int, default=24)
    p.add_argument("--pretrain-epochs", type=int, default=24)
    p.add_argument("--train-games", type=int, default=8)
    p.add_argument("--test-games", type=int, default=8)
    p.add_argument("--match-games", type=int, default=8)
    p.add_argument("--max-game-moves", type=int, default=100)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--credit", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--anchors", type=int, default=8)
    p.add_argument("--anchor-kl", type=float, default=0.01)
    p.add_argument("--lengths", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--checkpoint-dir", default="")
    p.add_argument("--output", required=True)
    # Used only by the legacy frozen evaluation helper, which never refreshes.
    p.set_defaults(refresh=8)
    return p


def main():
    args = parser().parse_args()
    torch.set_num_threads(1)
    available = ONLINE_METHODS if args.study == "online" else (
        HISTORY_METHODS if args.study == "history" else ("orthogonal", "conditional", "conditional_external")
    )
    methods = args.methods.split(",") if args.methods else list(available)
    if any(method not in available for method in methods):
        raise ValueError("unknown method for this study")
    source = sources()
    result = {"version": 1, "config": vars(args), "source_sha256": source,
              "base_git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "runs": []}
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    for seed in map(int, args.seeds.split(",")):
        base = base_pretrain(args, seed) if args.study == "online" else None
        for method in methods:
            if args.study == "online":
                row = run_online(args, method, seed, base)
            elif args.study == "history":
                row = run_history(args, method, seed)
            else:
                row = run_mechanisms(args, method, seed)
            if sources() != source:
                raise RuntimeError("experiment source changed during execution")
            result["runs"].append(row)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(path)
            print(json.dumps({"study": args.study, "seed": seed, "method": method,
                              "seconds": round(row["seconds"], 2),
                              "metric": row.get("after_b", row.get("after", {}))}), flush=True)


if __name__ == "__main__":
    main()
