"""Controlled Go query/spatial/outcome ablations, with separately budgeted search.

All online labels follow committed predictions. Value labels come only from
actual terminal games. Fixed-position policy probes never use their generator's
prefix-end value labels. Development and confirmation outputs remain separate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.go_memory_research import base_pretrain, build as old_build, parameter_counts
from mqr import GoBoard, GoLossWeights, GoMultiHeadOutput, HeuristicGoTeacher
from mqr.go_agent import generate_go_agent_trajectories
from mqr.go_outcome import OutcomeGoSession, PolicyBehaviorMemory
from mqr.go_policy import SignedQueryGoAgent


METHODS = ("legacy", "query", "spatial", "replay", "outcome", "reliable")
CONFIRMATION_SEEDS = (503, 509, 521, 523, 541)


def sources():
    paths = sorted((ROOT / "mqr").glob("*.py")) + [
        ROOT / "experiments/go_memory_research.py", Path(__file__),
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def build(args, method, seed):
    if method not in METHODS:
        raise ValueError("unknown policy ablation")
    agent, encoder = old_build(args, "conditional_guard", seed)
    if method != "legacy":
        core = agent.core
        torch.manual_seed(seed + 100)
        agent = SignedQueryGoAgent(
            encoder.output_dim, board_size=args.size, core=core, ring_dim=args.ring_dim,
            latent_dim=args.latent_dim, channels=args.channels, query_dim=8,
            adapt_spatial=method != "query", task_lr=args.lr, max_update_norm=0.15,
            max_trace_horizon=32, ogd_max_rank=args.rank, legality_policy_scale=1.0,
            loss_weights=GoLossWeights(value=0.0, illegal_mass=0.2),
        )
        agent.utility_gate.requires_grad_(False)
    return agent, encoder


def session_for(args, agent, encoder, method):
    mode = "policy" if method == "replay" else "outcome" if method in ("outcome", "reliable") else "none"
    return OutcomeGoSession(
        agent, encoder, replay_mode=mode, episode_capacity=args.max_game_moves,
        update_every=args.window, credit_horizon=args.credit,
        behavior_memory=PolicyBehaviorMemory(args.anchors, reliable_only=method == "reliable"),
        refresh_every=args.refresh, max_anchor_kl=args.anchor_kl, project_with_memory=True,
    )


def mean(values):
    return float(np.mean(values)) if values else None


def zero_state(agent):
    p = next(agent.parameters())
    return agent.core.zero_state(1, device=p.device, dtype=p.dtype)


@torch.no_grad()
def no_query_output(agent, full, features):
    base = agent.spatial_skip_heads
    hidden = base.spatial_features(features[:, :agent.board_dim])
    hidden *= 1 + agent.ring_channel_gain(full.latent)[:, :, None, None]
    return GoMultiHeadOutput(
        base.placement(hidden).flatten(1), base.legality(hidden).flatten(1),
        full.pass_logit, full.value, full.latent, legality_policy_scale=agent.legality_policy_scale,
    )


@torch.no_grad()
def probes(agent, encoder, dataset):
    records = {name: [] for name in ("full", "no_query", "reset_history")}
    query_rms, gain_rms = [], []
    for trajectory in dataset:
        state = zero_state(agent)
        for example in trajectory.examples:
            features = encoder.encode_board(example.board).to(next(agent.parameters()))
            full, state = agent._transition(features, state, slow_write=True)
            without = no_query_output(agent, full, features)
            reset, _ = agent._transition(features, zero_state(agent), slow_write=True)
            legal = torch.tensor(example.board.legal_moves(), device=features.device)
            chosen = int(legal[full.policy_logits[0, legal].argmax()])
            for name, output in (("full", full), ("no_query", without), ("reset_history", reset)):
                log_probs = output.policy_logits[0]
                action = int(legal[log_probs[legal].argmax()])
                records[name].append({
                    "joint_nll": float(-log_probs[example.target_action]),
                    "teacher_agreement": float(action == example.target_action),
                    "raw_legal": float(example.board.is_legal(int(log_probs.argmax()))),
                    "action_differs_from_full": float(action != chosen),
                })
            direct = full.placement_logits - without.placement_logits
            base_placement = agent.spatial_skip_heads(features[:, :agent.board_dim])[0]
            gain = without.placement_logits - base_placement
            query_rms.append(float((direct - direct.mean(-1, keepdim=True)).square().mean().sqrt()))
            gain_rms.append(float((gain - gain.mean(-1, keepdim=True)).square().mean().sqrt()))
    return {
        **{name: {key: mean([record[key] for record in rows]) for key in rows[0]}
           for name, rows in records.items()},
        "positions": len(query_rms), "query_centered_logit_rms": mean(query_rms),
        "gain_centered_logit_rms": mean(gain_rms),
    }


def tactical(board, action):
    if action == board.pass_action:
        return {"placement": 0, "noncapturing_self_atari": 0, "capture": 0}
    stones, captured, _ = board._simulate_stone(action)
    _, liberties = board._group_and_liberties(stones, action)
    return {"placement": 1, "noncapturing_self_atari": int(captured == 0 and len(liberties) == 1),
            "capture": int(captured > 0)}


def compact_update(update, source):
    keys = ("did_update", "update_norm", "ogd_retained_norm", "ogd_rank", "ogd_max_abs_overlap",
            "anchor_drift", "anchor_step_scale", "anchor_backtracks", "parameter_version",
            "max_unitary_error", "prediction_before_update", "all_predictions_committed_before_update",
            "parameter_staleness")
    return {"source": source, **{key: update[key] for key in keys if key in update}}


def play_games(args, session, seed, *, learn, phase="eval", collect=False, simulations=0, start_game=0):
    phase_started = time.perf_counter()
    count = args.train_games if learn else args.match_games
    teacher = HeuristicGoTeacher()
    games, updates, nlls, latencies = [], [], [], []
    peak_bytes = session.online_tensor_bytes
    opening = 0 if phase == "a" else 6 if phase == "b" else 4
    replay_positions, value_labels, search_evaluations, terminal_seconds = 0, 0, 0, 0.0
    student_tactics = {"placement": 0, "noncapturing_self_atari": 0, "capture": 0}
    teacher_tactics = dict(student_tactics)
    query_changed, decisions, query_rms = 0, 0, []
    for game in range(start_game, start_game + count):
        rng = random.Random(seed + 1009 * (game if learn else game // 2))
        session.reset_game()
        board = GoBoard(args.size, komi=2.5)
        student = 1 if game % 2 == 0 else -1
        inputs, moves, predictions = [], [], []
        for ply in range(args.max_game_moves):
            if board.game_over:
                break
            features = session._encode_observation(board).clone()
            started = time.perf_counter()
            # Observe BEFORE both search and teacher. The action remains fixed
            # when feedback closes a window and changes model parameters.
            result = session.observe(board, learn=learn)
            is_decision = ply >= opening and board.to_play == student
            if is_decision:
                with torch.no_grad():
                    without = no_query_output(session.agent, result["output"], features)
                    legal = torch.tensor(board.legal_moves())
                    alternative = int(legal[without.policy_logits[0, legal].argmax()])
                    query_changed += int(alternative != result["action"])
                    direct = result["output"].placement_logits - without.placement_logits
                    query_rms.append(float((direct - direct.mean(-1, keepdim=True)).square().mean().sqrt()))
                decisions += 1
                if simulations:
                    from mqr.go_search import policy_value_search
                    search = policy_value_search(session.agent, session.encoder, board,
                                                 result["state"], result["policy_logits"],
                                                 simulations=simulations, max_depth=args.search_depth)
                    result["action"] = search["action"]
                    search_evaluations += search["network_evaluations"]
            target = int(teacher.select_move(board.copy()))
            nlls.append(float(-result["policy_logits"][0, target]))
            predictions.append((board.to_play, float(result["output"].value[0]), is_decision))
            if learn:
                session.feedback(int(result["ticket_id"]), target)
                if session.pending_count == session.update_every:
                    updates.append(compact_update(session.flush(), "online"))
            inputs.append(features)
            if collect and ply < session.agent.max_trace_horizon and (ply + 1) % 4 == 0:
                session.behavior_memory.add(inputs, target, stratum=game % 2 + 2 * int(ply >= 8))
            if ply < opening:
                candidates = board.legal_moves(include_pass=False)
                action = rng.choice(candidates) if candidates else board.pass_action
            else:
                action = result["action"] if board.to_play == student else target
            if is_decision:
                for key, value in tactical(board, action).items():
                    student_tactics[key] += value
                for key, value in tactical(board, target).items():
                    teacher_tactics[key] += value
            moves.append(action)
            board.play(action)
            peak_bytes = max(peak_bytes, session.online_tensor_bytes)
            latencies.append(1000 * (time.perf_counter() - started))
        terminal = None
        if learn:
            terminal_started = time.perf_counter()
            terminal = session.finish_game(board) if board.game_over else session.abort_game()
            terminal_seconds += time.perf_counter() - terminal_started
            if terminal["online_update"] is not None:
                updates.append(compact_update(terminal["online_update"], "online"))
            updates.extend(compact_update(update, "terminal_replay") for update in terminal["replay_updates"])
            replay_positions += terminal["replayed_positions"]
            value_labels += len(terminal["value_targets"])
        winner = board.winner() if board.game_over else None
        game_value_mse = mean([(value - winner * color) ** 2 for color, value, _ in predictions]) if board.game_over else None
        student_value_mse = mean([(value - winner * color) ** 2 for color, value, decision in predictions
                                  if decision]) if board.game_over else None
        games.append({
            "student_color": student, "moves": moves, "terminated": board.game_over,
            "winner": winner, "student_win": bool(board.game_over and winner == student),
            "student_margin": board.score()["margin_black"] * student if board.game_over else None,
            "prequential_value_mse": game_value_mse, "student_value_mse": student_value_mse,
            "value_positions": len(predictions) if board.game_over else 0,
        })
    session.reset_game()
    phase_seconds = time.perf_counter() - phase_started
    return {
        "games": games, "wins": sum(game["student_win"] for game in games),
        "terminated_games": sum(game["terminated"] for game in games),
        "mean_terminal_student_margin": mean([g["student_margin"] for g in games if g["terminated"]]),
        "game_mean_value_mse": mean([g["prequential_value_mse"] for g in games if g["terminated"]]),
        "student_game_mean_value_mse": mean([g["student_value_mse"] for g in games if g["student_value_mse"] is not None]),
        "prequential_joint_nll": mean(nlls), "observations": len(nlls),
        "feedback_positions": len(nlls) if learn else 0, "replayed_positions": replay_positions,
        "terminal_value_labels": value_labels, "student_decisions": decisions,
        "query_action_changes": query_changed, "query_centered_logit_rms": mean(query_rms),
        "student_tactics": student_tactics, "teacher_on_same_positions_tactics": teacher_tactics,
        "updates": updates, "accepted_updates": sum(u["did_update"] for u in updates),
        "max_anchor_kl": max((u.get("anchor_drift", {}).get("max_policy_kl", 0) for u in updates), default=0),
        "peak_persistent_online_tensor_bytes": peak_bytes,
        "mean_observe_feedback_ms": mean(latencies), "search_simulations": simulations,
        "terminal_feedback_seconds": terminal_seconds, "phase_seconds": phase_seconds,
        "amortized_ms_per_observation": 1000 * phase_seconds / len(nlls),
        "search_network_evaluations": search_evaluations, "opening_moves": opening,
        "paired_color_openings": not learn,
    }


def run(args, method, seed, base, datasets):
    started = time.perf_counter()
    agent, encoder = build(args, method, seed)
    agent.spatial_skip_heads.load_state_dict(base)
    session = session_for(args, agent, encoder, method)
    memory = session.behavior_memory
    evaluate = lambda: {key: probes(agent, encoder, data) for key, data in datasets.items()}
    result = {"method": method, "seed": seed, **parameter_counts(agent), "before": evaluate()}
    result["train_a"] = play_games(args, session, seed + 25000, learn=True, phase="a", collect=True)
    result["after_a"] = evaluate()
    memory.freeze(agent)
    result["anchor_selection"] = {
        "candidates": memory.candidate_count, "teacher_agreements": memory.agreement_count,
        "protected": len(memory.anchors), "reliable_only": memory.reliable_only,
    }
    result["consolidation"] = memory.refresh(agent)
    result["train_b"] = play_games(args, session, seed + 26000, learn=True, phase="b")
    result["after_b"] = evaluate()
    result["anchor_drift"] = memory.drift(agent)
    snapshot = copy.deepcopy(session.state_dict())
    result["matches"] = play_games(args, session, seed + 27000, learn=False)
    session.load_state_dict(snapshot)
    if method in args.search_methods.split(",") and args.simulations:
        result["search_matches"] = play_games(args, session, seed + 27000, learn=False,
                                             simulations=args.simulations)
        session.load_state_dict(snapshot)
    result.update(
        seconds=time.perf_counter() - started, anchor_bytes=memory.storage_bytes,
        ogd_bytes=agent.task_gradient_memory.storage_bytes, refresh_count=memory.refresh_count,
        refresh_seconds=memory.refresh_seconds, max_adjoint_norm_error=memory.max_adjoint_norm_error,
        peak_constraint_state_trace_bytes=memory.peak_state_trace_bytes,
    )
    if args.checkpoint_dir:
        directory = ROOT / args.checkpoint_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{method}-{seed}.pt"
        torch.save({"config": vars(args), "method": method, "seed": seed,
                    "source_sha256": sources(), "session": session.state_dict()}, path)
        result["checkpoint"] = str(path.relative_to(ROOT))
        result["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="1801")
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--ring-dim", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--anchor-kl", type=float, default=0.01)
    parser.add_argument("--refresh", type=int, default=1)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--credit", type=int, default=8)
    parser.add_argument("--warmup-games", type=int, default=24)
    parser.add_argument("--pretrain-epochs", type=int, default=24)
    parser.add_argument("--train-games", type=int, default=8)
    parser.add_argument("--test-games", type=int, default=8)
    parser.add_argument("--match-games", type=int, default=16)
    parser.add_argument("--max-game-moves", type=int, default=100)
    parser.add_argument("--search-methods", default="legacy,reliable")
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--search-depth", type=int, default=8)
    parser.add_argument("--checkpoint-dir", default="checkpoints/go_policy_v1")
    parser.add_argument("--output", default="analysis/results/go_policy_development.json")
    args = parser.parse_args()
    # Compatibility arguments used solely by the shared spatial pretraining.
    args.slots, args.history_moves = 0, 12
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if args.confirmation and any(seed not in CONFIRMATION_SEEDS for seed in seeds):
        raise ValueError("confirmation runs use only the preregistered seeds")
    if not args.confirmation and set(seeds) & set(CONFIRMATION_SEEDS):
        raise ValueError("reserve confirmation seeds; use 1801 for development")
    if any(method not in METHODS for method in args.methods.split(",")):
        raise ValueError("unknown method")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(1)
    evidence = {
        "version": 1, "completed": False, "config": vars(args),
        "source_sha256": sources(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runs": [],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing result: {args.output}")

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    save()
    for seed in [int(value) for value in args.seeds.split(",")]:
        base = base_pretrain(args, seed)
        datasets = {
            phase: generate_go_agent_trajectories(
                args.test_games, seed=seed + offset, size=args.size, komi=2.5,
                recorded_moves=24, start_random_moves=opening, random_move_probability=0.5,
            )
            for phase, offset, opening in (("a", 23000, 0), ("b", 24000, 10))
        }
        for method in args.methods.split(","):
            result = run(args, method, seed, base, datasets)
            evidence["runs"].append(result)
            save()
            print(json.dumps({
                "seed": seed, "method": method, "wins": result["matches"]["wins"],
                "games": len(result["matches"]["games"]),
                "b_nll": result["after_b"]["b"]["full"]["joint_nll"],
                "query_changed": result["matches"]["query_action_changes"],
                "seconds": result["seconds"],
            }), flush=True)
    if sources() != evidence["source_sha256"]:
        raise RuntimeError("source changed during the experiment; results remain incomplete")
    evidence["completed"] = True
    save()


if __name__ == "__main__":
    main()
