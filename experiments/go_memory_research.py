"""Versioned Go history-readability and student-induced online learning studies.

History recall is a separate diagnostic task, not Go strength. All Go win
counts require two-pass termination. No model is trained on evaluation games.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mqr import GoBoard, GoLossWeights, GoMultiHeadOutput, GoSpatialSkipHeads, HeuristicGoTeacher
from mqr.agent_baselines import MultiTimescaleGRUCore
from mqr.go_online import SpatialRingGoAgent
from mqr.go_agent import generate_go_agent_trajectories, legality_target
from mqr.go_history_tasks import generate_reachable_history_pairs
from mqr.go_memory import (GoHistoryEncoder, GoObservationMemory, GoResearchCore,
                           PositionQueryGoAgent, fixed_color_features)
from mqr.go_protection import GoBehaviorMemory, ProtectedGoSession


class FixedShift(nn.Module):
    """Parameter-free cyclic/twisted shift control, using exactly the ring budget."""

    def __init__(self, dim, twist=False):
        super().__init__()
        self.dim, self.twist = dim, twist
        self.register_buffer("angles", torch.empty(0))
        matrix = torch.eye(dim).roll(1, dims=0)
        if twist:
            matrix[0].neg_()
        self.register_buffer("matrix", matrix)

    def apply_orthogonal(self, state, *, transpose=False, angle_offsets=None):
        if angle_offsets is not None:
            raise ValueError("fixed shift does not accept angle conditioning")
        if transpose:
            state = torch.cat((-state[..., :1], state[..., 1:]), dim=-1) if self.twist else state
            return state.roll(-1, dims=-1)
        result = state.roll(1, dims=-1)
        return torch.cat((-result[..., :1], result[..., 1:]), dim=-1) if self.twist else result

    def unitary(self):
        return torch.complex(self.matrix, torch.zeros_like(self.matrix))


class GoGRU(MultiTimescaleGRUCore):
    def __init__(self, *args, board_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.board_size = board_size

    def forward_step(self, x, **kwargs):
        return super().forward_step(fixed_color_features(x, self.board_size), **kwargs)


def build(args, method, seed, *, history_task=False):
    external = "external" in method
    slots = args.slots if external else 0
    encoder = GoHistoryEncoder(args.size, history_slots=slots)
    rates = (1.0, 0.15, 0.03)
    mode = "identity" if method.startswith(("identity", "stateless", "frozen", "spatial")) else "orthogonal"
    if method.startswith(("stateless", "frozen", "spatial")):
        rates = (1.0, 1.0, 1.0)
    torch.manual_seed(seed)
    core = GoResearchCore(
        encoder.output_dim, args.ring_dim, args.latent_dim, board_size=args.size,
        fixed_colors="relative" not in method,
        conditional_scale=0.3 if "conditional" in method else 0.0,
        transition_mode=mode, transition_structure="cyclic_givens", injection_rank=8,
        base_unitary_init="random", base_unitary_seed=seed + 31,
        learn_transitions=not method.startswith(("fixed", "shift", "twisted")),
        leak_rates=rates, write_scales=tuple(math.sqrt(1 - (1 - rate) ** 2) for rate in rates),
        state_activation="none", readout_bias=True,
    )
    if method.startswith(("shift", "twisted")):
        for index in (1, 2):
            core.unitary_params[index] = FixedShift(args.ring_dim, method.startswith("twisted"))
        core._fixed_transitions = torch.stack([core._transition_from_parameter(i) for i in range(3)])
    if method == "gru":
        torch.manual_seed(seed)
        core = GoGRU(encoder.output_dim, args.ring_dim, args.latent_dim, board_size=args.size,
                     leak_rates=rates, input_rank=8)
    torch.manual_seed(seed + 100)
    rank = args.rank if any(x in method for x in ("ogd", "guard", "norm")) else 0
    agent = PositionQueryGoAgent(
        encoder.output_dim, board_size=args.size, core=core, ring_dim=args.ring_dim,
        latent_dim=args.latent_dim, channels=args.channels, query_dim=8, history_slots=slots,
        task_lr=args.lr, max_update_norm=0.15, max_trace_horizon=max(32, args.history_moves + 1),
        ogd_max_rank=rank, legality_policy_scale=0.0 if history_task else 1.0,
        loss_weights=GoLossWeights(legality=0.0 if history_task else 1.0, pass_decision=0.0 if history_task else 1.0,
                                  value=0.0, illegal_mass=0.0 if history_task else 0.2),
    )
    if method.startswith("film"):
        if slots or history_task:
            raise ValueError("FiLM comparison is a Go control without external memory")
        torch.manual_seed(seed + 100)
        agent = SpatialRingGoAgent(
            encoder.output_dim, board_size=args.size, core=core, ring_dim=args.ring_dim,
            latent_dim=args.latent_dim, spatial_skip_channels=args.channels, task_lr=args.lr,
            max_update_norm=0.15, max_trace_horizon=32, ogd_max_rank=rank,
            legality_policy_scale=1.0,
            loss_weights=GoLossWeights(value=0.0, illegal_mass=0.2),
        )
        agent.spatial_skip_heads = GoSpatialSkipHeads(encoder.base_dim, args.size, args.channels,
                                                     geometry=True, depth=2)
    for parameter in agent.utility_gate.parameters():
        parameter.requires_grad_(False)
    if not history_task:
        for parameter in agent.spatial_skip_heads.parameters():
            parameter.requires_grad_(method.startswith("spatial"))
        if method.startswith(("frozen", "spatial")):
            for name, parameter in agent.named_parameters():
                parameter.requires_grad_(method.startswith("spatial") and name.startswith("spatial_skip_heads."))
    return agent, encoder


def parameter_counts(agent):
    return {"trainable_parameters": sum(p.numel() for p in agent.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in agent.parameters()),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in agent.parameters()),
            "ring_state_bytes": agent.core.num_timescales * agent.core.ring_dim * 4}


def materialize(pairs, encoder):
    result, targets = [], []
    for pair in pairs:
        for history, target in zip(pair.histories, pair.targets):
            memory = GoObservationMemory(encoder)
            sequence = []
            for board in history:
                sequence.append(encoder.with_history(board, memory.slots))
                memory.write(board)
            result.append(torch.cat(sequence))
            targets.append(target)
    return torch.stack(result), torch.tensor(targets)


def recall_forward(agent, features, *, credit=32, reset=False, swap=False, return_state=False):
    state = agent.core.zero_state(features.size(0), device=features.device, dtype=features.dtype)
    for index in range(features.size(1)):
        if index and index % credit == 0:
            state = state.detached()
        x = features[:, index]
        if index == features.size(1) - 1:
            if reset:
                state = agent.core.zero_state(features.size(0), device=features.device, dtype=features.dtype)
                x = x.clone(); x[:, agent.board_dim:] = 0
            if swap:
                indices = torch.arange(features.size(0)) ^ 1
                state = type(state)(tuple(ring[indices] for ring in state.rings))
                x = x.clone(); x[:, agent.board_dim:] = x[indices, agent.board_dim:]
        latent, state = agent.core.forward_step(x, state=state)
    logits = agent._query_output(latent, state, x).placement_logits
    return (logits, state) if return_state else logits


@torch.no_grad()
def recall_metrics(agent, features, targets):
    metrics = {}
    for name, kwargs in (("history", {}), ("reset", {"reset": True}), ("swap", {"swap": True})):
        logits, state = recall_forward(agent, features, return_state=True, **kwargs)
        choices = targets.reshape(-1, 2).repeat_interleave(2, dim=0)
        binary = logits.gather(1, choices)
        chosen = choices.gather(1, binary.argmax(dim=1, keepdim=True)).squeeze(1)
        metrics[name] = {"pair_choice_accuracy": float((chosen == targets).float().mean()),
                         "full_action_accuracy": float((logits.argmax(dim=1) == targets).float().mean()),
                         "point_nll": float(nn.functional.cross_entropy(logits, targets)),
                         "paired_state_l2_distance": float((torch.cat(state.rings, dim=1)[::2] - torch.cat(state.rings, dim=1)[1::2]).norm(dim=1).mean()),
                         "paired_point_logit_l2_distance": float((logits[::2] - logits[1::2]).norm(dim=1).mean())}
    return metrics


def run_history(args, method, seed):
    agent, encoder = build(args, method, seed, history_task=True)
    train_pairs = generate_reachable_history_pairs(args.history_train_pairs, seed=seed + 11000,
                                                  size=args.size, moves=args.history_moves)
    test_pairs = generate_reachable_history_pairs(args.history_test_pairs, seed=seed + 12000,
                                                 size=args.size, moves=args.history_moves)
    train_hashes = {hashlib.sha256(repr(pair.moves).encode()).hexdigest() for pair in train_pairs}
    test_hashes = {hashlib.sha256(repr(pair.moves).encode()).hexdigest() for pair in test_pairs}
    if train_hashes & test_hashes:
        raise RuntimeError("history train/test trajectory overlap")
    train_x, train_y = materialize(train_pairs, encoder)
    test_x, test_y = materialize(test_pairs, encoder)
    before = recall_metrics(agent, test_x, test_y)
    optimizer = torch.optim.Adam([p for p in agent.parameters() if p.requires_grad], lr=0.006)
    generator = torch.Generator().manual_seed(seed + 13000)
    prequential = []
    started = time.perf_counter()
    credit = 8 if "short" in method else 32
    for _ in range(args.history_epochs):
        for indices in torch.randperm(len(train_y), generator=generator).split(16):
            # Commit predictions under the current parameters; labels are used
            # only by the following loss and update. Each group is an online
            # mini-batch of independent game streams, with explicit replay epochs.
            logits = recall_forward(agent, train_x[indices], credit=credit)
            prequential.append(float(nn.functional.cross_entropy(logits.detach(), train_y[indices])))
            loss = nn.functional.cross_entropy(logits, train_y[indices])
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_([p for p in agent.parameters() if p.requires_grad], 1.0)
            optimizer.step()
    result = {"method": method, "seed": seed, "before": before,
              "after": recall_metrics(agent, test_x, test_y),
              "updates": len(prequential), "credit_horizon": credit,
              "training_replay_epochs": args.history_epochs,
              "prequential_point_nll": float(np.mean(prequential)),
              "seconds": time.perf_counter() - started, **parameter_counts(agent),
              "observation_memory_bytes": encoder.history_slots * encoder.slot_dim * 4,
              "optimizer_tensor_bytes": sum(v.numel() * v.element_size() for state in optimizer.state.values()
                                            for v in state.values() if isinstance(v, torch.Tensor)),
              "materialized_training_dataset_bytes": train_x.numel() * train_x.element_size() + train_y.numel() * train_y.element_size(),
              "train_trajectory_hash": hashlib.sha256(''.join(sorted(train_hashes)).encode()).hexdigest(),
              "test_trajectory_hash": hashlib.sha256(''.join(sorted(test_hashes)).encode()).hexdigest(),
              "task": "legal_antipode_of_first_black_move; diagnostic, not optimal Go"}
    if args.checkpoint_dir:
        path = Path(args.checkpoint_dir); path.mkdir(parents=True, exist_ok=True)
        torch.save({"config": vars(args), "method": method, "seed": seed,
                    "model_state": agent.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "generator_state": generator.get_state()}, path / f"history-{method}-{seed}.pt")
    return result


def base_pretrain(args, seed):
    encoder = GoHistoryEncoder(args.size)
    data = generate_go_agent_trajectories(args.warmup_games, seed=seed + 21000, size=args.size,
                                         komi=2.5, recorded_moves=24, random_move_probability=0.5)
    examples = [example for trace in data for example in trace.examples]
    x = torch.cat([encoder.encode_board(example.board) for example in examples])
    y = torch.tensor([example.target_action for example in examples])
    legal = torch.cat([legality_target(example.board) for example in examples])
    agent, _ = build(args, "spatial", seed)
    base = agent.spatial_skip_heads
    nn.init.constant_(base.pass_decision.bias, -math.log(args.size ** 2))
    optimizer = torch.optim.Adam(base.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(seed + 22000)
    for _ in range(args.pretrain_epochs):
        for indices in torch.randperm(len(y), generator=generator).split(64):
            p, l, passing, value = base(x[indices])
            output = GoMultiHeadOutput(p, l, passing, value.tanh(), x.new_zeros(len(indices), 1))
            loss = agent.compute_go_loss(output, y[indices], legality_target=legal[indices])["total"]
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return copy.deepcopy(base.state_dict())


def session_for(args, agent, encoder, method, *, memory=None):
    return ProtectedGoSession(agent, encoder, update_every=args.window, credit_horizon=args.credit,
                              project_with_memory=("ogd" in method or "guard" in method),
                              norm_matched_sgd="norm" in method,
                              behavior_memory=memory, refresh_every=args.refresh,
                              max_anchor_kl=args.anchor_kl if "guard" in method else None)


@torch.no_grad()
def evaluate_go(args, agent, encoder, data):
    # Snapshot all live counters/state as well as parameters, preserving the
    # training stream exactly. Evaluation gets its own empty observation memory.
    snapshot = copy.deepcopy(agent.state_dict())
    session = session_for(args, agent, encoder, "evaluation")
    nlls, points, passes, legal = [], [], [], []
    try:
        for trace in data:
            session.reset_game()
            for example in trace.examples:
                result = session.observe(example.board, learn=False)
                target = example.target_action
                output = result["output"]
                nlls.append(float(-result["policy_logits"][0, target]))
                pass_nll = nn.functional.binary_cross_entropy_with_logits(output.pass_logit, output.pass_logit.new_tensor([target == agent.points]))
                passes.append(float(pass_nll))
                if target < agent.points:
                    conditional = result["policy_logits"][0, :-1]
                    points.append(float(-conditional[target] + conditional.logsumexp(0)))
                legal.append(result["raw_legal"])
    finally:
        agent.load_state_dict(snapshot)
    return {"joint_nll": float(np.mean(nlls)), "nonpass_point_nll": float(np.mean(points)),
            "pass_nll": float(np.mean(passes)), "raw_legality": float(np.mean(legal)), "positions": len(nlls)}


def play_games(args, session, seed, *, learn, collect_anchors=False, phase="a", start_game=0,
               opening_moves=0, paired_openings=False):
    teacher = HeuristicGoTeacher()
    games, updates, losses, timing = [], [], [], []
    peak_bytes = session.online_tensor_bytes
    count = args.train_games if learn else args.match_games
    for game in range(start_game, start_game + count):
        rng = random.Random(seed + 1009 * (game // 2 if paired_openings else game))
        session.reset_game()
        board = GoBoard(args.size, komi=2.5)
        student = 1 if game % 2 == 0 else -1
        inputs, moves = [], []
        for ply in range(args.max_game_moves):
            if board.game_over:
                break
            started = time.perf_counter()
            # Record the same raw context actually presented, before any label.
            x = session._encode_observation(board).clone()
            result = session.step(board, teacher, learn=learn)
            inputs.append(x)
            target = result["teacher_action"]
            losses.append(float(-result["policy_logits"][0, target]))
            if result["update"] is not None:
                updates.append(result["update"])
            if collect_anchors and ply < session.agent.max_trace_horizon and (ply + 1) % 4 == 0:
                session.behavior_memory.add(inputs, target, stratum=game % 2 + 2 * int(ply >= 8))
            if ply < (6 if phase == "b" else opening_moves):
                candidates = [a for a in board.legal_moves() if a != board.pass_action]
                action = rng.choice(candidates) if candidates else board.pass_action
            else:
                action = result["action"] if board.to_play == student else target
            moves.append(action)
            board.play(action)
            peak_bytes = max(peak_bytes, session.online_tensor_bytes)
            timing.append((time.perf_counter() - started) * 1000)
        update = session.flush()
        if update is not None:
            updates.append(update)
        games.append({"student_color": student, "moves": moves, "terminated": board.game_over,
                      "winner": board.winner() if board.game_over else None,
                      "student_win": bool(board.game_over and board.winner() == student)})
    session.reset_game()
    return {"games": games, "wins": sum(g["student_win"] for g in games),
            "random_opening_moves": 6 if phase == "b" else opening_moves,
            "paired_color_openings": paired_openings,
            "terminated_games": sum(g["terminated"] for g in games),
            "prequential_joint_nll": float(np.mean(losses)), "feedback_positions": len(losses) if learn else 0,
            "updates": len(updates), "accepted_updates": sum(u["did_update"] for u in updates),
            "mean_update_norm": float(np.mean([u["update_norm"] for u in updates])) if updates else 0,
            "mean_ogd_retained_norm": float(np.mean([u["ogd_retained_norm"] for u in updates])) if updates else None,
            "max_anchor_kl": max((u.get("anchor_drift", {}).get("max_policy_kl", 0) for u in updates), default=0),
            "backtracked_updates": sum(u.get("anchor_backtracks", 0) > 0 for u in updates),
            "peak_persistent_online_tensor_bytes": peak_bytes,
            "mean_observe_feedback_ms": float(np.mean(timing))}


def run_online(args, method, seed, base):
    agent, encoder = build(args, method, seed)
    agent.spatial_skip_heads.load_state_dict(base)
    protected = agent.task_gradient_memory.max_rank > 0
    memory = GoBehaviorMemory(args.anchors) if protected else None
    session = session_for(args, agent, encoder, method, memory=memory)
    datasets = {phase: generate_go_agent_trajectories(args.test_games, seed=seed + offset, size=args.size,
                                                    komi=2.5, recorded_moves=24, start_random_moves=prefix,
                                                    random_move_probability=0.5)
                for phase, offset, prefix in (("a", 23000, 0), ("b", 24000, 10))}
    result = {"method": method, "seed": seed, **parameter_counts(agent),
              "before": {name: evaluate_go(args, agent, encoder, data) for name, data in datasets.items()}}
    started = time.perf_counter()
    frozen = method == "frozen"
    result["train_a"] = play_games(args, session, seed + 25000, learn=not frozen, collect_anchors=protected)
    result["after_a"] = {name: evaluate_go(args, agent, encoder, data) for name, data in datasets.items()}
    if protected:
        memory.freeze(agent)
        result["consolidation"] = memory.refresh(agent)
    result["train_b"] = play_games(args, session, seed + 26000, learn=not frozen, phase="b")
    result["after_b"] = {name: evaluate_go(args, agent, encoder, data) for name, data in datasets.items()}
    if protected:
        result["anchor_drift"] = memory.drift(agent)
        result["anchor_bytes"] = memory.storage_bytes
        result["ogd_bytes"] = agent.task_gradient_memory.storage_bytes
    result["matches"] = play_games(args, session, seed + 27000, learn=False,
                                   opening_moves=4, paired_openings=True)
    result["seconds"] = time.perf_counter() - started
    if args.checkpoint_dir:
        path = Path(args.checkpoint_dir); path.mkdir(parents=True, exist_ok=True)
        torch.save({"config": vars(args), "method": method, "seed": seed, "session": session.state_dict(),
                    "continuation_games": 0}, path / f"{method}-{seed}.pt")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("history", "online"), default="history")
    parser.add_argument("--methods", default="stateless,identity,shift,twisted,fixed,gru,orthogonal_short,orthogonal,conditional,stateless_external,conditional_external")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--ring-dim", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--history-moves", type=int, default=12)
    parser.add_argument("--history-train-pairs", type=int, default=96)
    parser.add_argument("--history-test-pairs", type=int, default=32)
    parser.add_argument("--history-epochs", type=int, default=8)
    parser.add_argument("--warmup-games", type=int, default=24)
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--train-games", type=int, default=8)
    parser.add_argument("--test-games", type=int, default=8)
    parser.add_argument("--match-games", type=int, default=8)
    parser.add_argument("--max-game-moves", type=int, default=100)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--credit", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--refresh", type=int, default=8)
    parser.add_argument("--anchor-kl", type=float, default=0.01)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--output", default="analysis/results/go_memory_history_development.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources = ["mqr/agent.py", "mqr/go.py", "mqr/go_agent.py", "mqr/go_online.py", "mqr/temporal.py",
               "mqr/unitary.py", "mqr/online.py", "mqr/go_memory.py", "mqr/go_protection.py",
               "mqr/go_history_tasks.py", "experiments/go_memory_research.py"]
    result = {"version": 2, "config": vars(args), "runs": [],
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}}
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True)
    for seed in map(int, args.seeds.split(',')):
        base = base_pretrain(args, seed) if args.study == "online" else None
        for method in args.methods.split(','):
            run = run_history(args, method, seed) if args.study == "history" else run_online(args, method, seed, base)
            result["runs"].append(run)
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({"seed": seed, "method": method, "seconds": round(run["seconds"], 2),
                              "metric": run["after"] if args.study == "history" else run["after_b"]}), flush=True)


if __name__ == "__main__":
    main()
