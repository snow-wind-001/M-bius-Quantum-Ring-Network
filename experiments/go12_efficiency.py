"""Matched-budget 10x10 terminal ownership / credit study; see go12_plan.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import go11_continual as prior
from mqr import GoBoard, GoLossWeights, HeuristicGoTeacher
from mqr.go import DefensiveGoTeacher
from mqr.go_cone import TaskMarginMemory
from mqr.go_memory import GoHistoryEncoder, GoResearchCore
from mqr.go_outcome import _board_from_record, _board_record
from mqr.go_policy import SignedQueryGoAgent
from mqr.go_selfplay import SharedPolicyGoSession

SEEDS = (811, 821, 823, 827, 829)
METHODS = ("frozen", "baseline", "ownership", "credit", "combined", "unprotected")


def source_hashes():
    paths = list((ROOT / "mqr").glob("*.py")) + [Path(__file__), Path(prior.__file__)]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def parameter_hash(agent):
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in agent.parameters())).hexdigest()


def build(args, seed):
    torch.manual_seed(seed)
    encoder = GoHistoryEncoder(args.size)
    rates = (1.0, 0.5, 0.15, 0.07, 0.03, 0.01)
    core = GoResearchCore(
        encoder.output_dim, 16, 16, board_size=args.size, fixed_colors=True,
        conditional_scale=0.3, transition_mode="orthogonal",
        transition_structure="cyclic_givens", injection_rank=8,
        base_unitary_init="random", base_unitary_seed=seed + 31,
        leak_rates=rates, write_scales=tuple(math.sqrt(1 - (1 - rate) ** 2) for rate in rates),
        state_activation="none", readout_bias=True,
    )
    torch.manual_seed(seed + 100)
    agent = SignedQueryGoAgent(
        encoder.output_dim, board_size=args.size, core=core, ring_dim=16, latent_dim=16,
        channels=12, query_dim=8, task_lr=0.02, max_update_norm=0.08,
        max_trace_horizon=64, ogd_max_rank=4, legality_policy_scale=1.0,
        adapt_spatial=True, spatial_head_only=True, ownership_head=True,
        loss_weights=GoLossWeights(placement=0, pass_decision=0, value=0,
                                  legality=0.25, illegal_mass=0.1, policy_distillation=1.0),
    )
    agent.utility_gate.requires_grad_(False)
    return agent, encoder


def make_session(args, agent, encoder):
    method = args.method
    memory = None if method in ("frozen", "unprotected") else TaskMarginMemory(
        4, strata=4, reliable_only=True, margin_fraction=0.5)
    return SharedPolicyGoSession(
        agent, encoder, update_every=16, credit_horizon=16,
        replay_window=64, replay_credit_horizon=64 if method in ("credit", "combined", "unprotected") else 16,
        project_with_memory=False, behavior_memory=memory, refresh_every=1,
        max_anchor_kl=None, replay_mode="none" if method == "frozen" else "outcome",
        outcome_weight=0.5, ownership_weight=0.5 if method in ("ownership", "combined", "unprotected") else 0,
        episode_capacity=4096,
    )


def schedule(args):
    if args.method == "frozen":
        return [("final", args.eval_games, False)]
    stages = []
    previous = 0
    for total in (4, 8, args.train_games):
        if total > args.train_games or total <= previous:
            continue
        stages.append((f"train_{total}", total - previous, True))
        previous = total
        if total in (4, 8) and total < args.train_games and args.method in ("baseline", "combined"):
            stages.append((f"curve_{total}", args.curve_games, False))
    stages.append(("final", args.eval_games, False))
    return stages


def begin_game(args, session, phase, index, trained):
    session.reset_game()
    training = phase.startswith("train_")
    game_index = trained if training else index
    moves = ([] if trained < 8 else prior.opening(args.seed + 51000 + trained - 8, args.size, 12)) if training else (
        prior.opening(args.seed + 61000 + index // 2, args.size, (4, 24, 60)[(index // 2) % 3]))
    board = GoBoard(args.size, komi=5.5)
    for action in moves:
        session.observe_context(board)
        board.play(action)
    return {
        "phase": phase, "index": game_index, "opening": moves, "board": _board_record(board),
        "student_color": 1 if index % 2 == 0 else -1, "decisions": [], "predictions": [],
        "updates": [], "seconds": 0.0, "peak_online_tensor_bytes": 0,
        "ownership_predictions": [],
    }


@torch.no_grad()
def mechanism_probe(agent, data):
    """Full legal teacher prefixes, never fitted; paired readout interventions."""
    encoder = GoHistoryEncoder(agent.board_size)
    selected = {(r["game_index"], r["ply"]): r for r in data["probes"]}
    rows = []
    for index, game in enumerate(data["probe_games"]):
        board = GoBoard(agent.board_size, komi=5.5)
        terminal = GoBoard(agent.board_size, komi=5.5)
        for action in game["moves"]:
            terminal.play(action)
        ownership = torch.tensor(terminal.area_ownership()).float()
        state = agent.core.zero_state(1, device="cpu", dtype=torch.float32)
        for ply, action in enumerate(game["moves"]):
            x = encoder.encode_board(board)
            output, state = agent._transition(x, state, slow_write=True)
            record = selected.get((index, ply))
            if record is not None:
                latent = agent.core.readout_state(state)
                no_query = agent._query_output(latent, state, x, include_query=False)
                reset, _ = agent._transition(x, agent.core.zero_state(1, device="cpu", dtype=x.dtype), slow_write=True)
                hidden, correction, _ = agent.query_components(latent, state, x)
                legal = torch.tensor(board.legal_moves())
                def greedy(value):
                    return int(legal[value.policy_logits[0, legal].argmax()])
                rows.append({
                    "game_index": index, "ply": ply, "action": greedy(output),
                    "no_query_action": greedy(no_query), "reset_action": greedy(reset),
                    "nll": float(-output.policy_logits[0, record["action"]]),
                    "agreement": int(greedy(output) == record["action"]),
                    "ownership_mse": float((output.ownership[0] - ownership * board.to_play).square().mean()),
                    "ownership_zero_mse": float(ownership.square().mean()),
                    "query_rms": float(correction[..., 0].square().mean().sqrt()),
                    "spatial_rms": float(agent.spatial_skip_heads.placement(hidden).square().mean().sqrt()),
                })
            board.play(action)
    return rows


def run(args):
    started = time.perf_counter()
    base_path = prior.prepare(args)
    data = torch.load(base_path, weights_only=True)
    agent, encoder = build(args, args.seed)
    agent.spatial_skip_heads.load_state_dict(data["base"])
    session = make_session(args, agent, encoder)
    path = Path(args.directory) / f"work-{args.method}-{args.seed}.pt"
    output = Path(args.output or ROOT / "analysis/results" / f"go12_{args.method}_seed{args.seed}.json")
    config = {k: v for k, v in vars(args).items() if k not in ("max_plies", "prepare_only")}
    if path.exists():
        work = torch.load(path, weights_only=True)
        if work["config"] != config or work["source"] != source_hashes():
            raise RuntimeError("source/configuration differs from saved run; choose a new directory")
        session.load_state_dict(work["session"])
    else:
        if session.behavior_memory is not None:
            prior.initialize_memory(session, data)
            if args.seed in SEEDS and len(session.behavior_memory.anchors) != 4:
                raise RuntimeError("formal protocol requires four reliable training anchors")
        work = {
            "version": 1, "config": config, "source": source_hashes(), "stage": 0,
            "game_index": 0, "trained": 0, "games": [], "active_game": None, "session": None,
            "completed": False, "evaluation_parameter_checks": {}, "seconds": 0.,
            "initial_parameters": parameter_hash(agent),
            "mechanisms": {"before": mechanism_probe(agent, data)},
            "total_parameters": sum(p.numel() for p in agent.parameters()),
            "trainable_parameters": 0 if args.method == "frozen" else sum(p.numel() for p in agent.parameters() if p.requires_grad),
            "base_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        }

    def checkpoint():
        work["session"] = session.state_dict()
        prior.save_torch(work, path)

    consumed = 0
    stages = schedule(args)
    while work["stage"] < len(stages):
        phase, count, training = stages[work["stage"]]
        if work["game_index"] == count:
            if not training:
                check = work["evaluation_parameter_checks"][phase]
                check["parameters_after"] = parameter_hash(agent)
                check["version_after"] = int(agent.online_parameter_version)
                if (check["parameters_before"] != check["parameters_after"]
                        or check["version_before"] != check["version_after"]):
                    raise RuntimeError("frozen evaluation mutated parameters")
            work["stage"] += 1
            work["game_index"] = 0
            checkpoint()
            continue
        if work["active_game"] is None:
            if not training and work["game_index"] == 0:
                work["evaluation_parameter_checks"][phase] = {
                    "parameters_before": parameter_hash(agent), "version_before": int(agent.online_parameter_version)}
            work["active_game"] = begin_game(args, session, phase, work["game_index"], work["trained"])
        game = work["active_game"]
        board = _board_from_record(game["board"])
        teacher = DefensiveGoTeacher() if game["index"] // 2 % 2 else HeuristicGoTeacher()
        while not board.game_over:
            step_start = time.perf_counter()
            result = session.observe(board, learn=training)
            student = training or board.to_play == game["student_color"]
            search = None
            if student:
                search = prior.search_decision(
                    args, agent, encoder, board, result, phase="a" if game["index"] < 8 else "b",
                    index=game["index"], simulations=args.simulations, training=training)
            actual = search["action"] if student else teacher.select_move(board)
            if student:
                captures, atari = 0, False
                if actual != board.pass_action:
                    _, captures, liberties = board._simulate_stone(actual)
                    atari = liberties == 1 and captures == 0
                game["decisions"].append({
                    "ply": len(board.move_history), "action": actual,
                    "raw_action": result["raw_action"], "raw_legal": result["raw_legal"],
                    "captures": captures, "self_atari": atari,
                    "search_evaluations": search["network_evaluations"],
                })
            game["predictions"].append((board.to_play, float(result["output"].value[0])))
            game["ownership_predictions"].append((board.to_play, result["output"].ownership[0].tolist()))
            if training:
                session.feedback(result["ticket_id"], actual, policy_target=session.search_target(search, board.action_size))
                if session.pending_count == session.update_every:
                    game["updates"].append(prior.compact_update(session.flush()))
                game["peak_online_tensor_bytes"] = max(game["peak_online_tensor_bytes"], session.online_tensor_bytes)
            board.play(actual)
            game["board"] = _board_record(board)
            game["seconds"] += time.perf_counter() - step_start
            consumed += 1
            if consumed % 64 == 0:
                checkpoint()
            if args.max_plies and consumed >= args.max_plies and not board.game_over:
                checkpoint()
                print(json.dumps({"paused": True, "phase": phase, "ply": len(board.move_history)}), flush=True)
                return
        step_start = time.perf_counter()
        terminal = session.finish_game(board) if training else {}
        if training:
            for update in [terminal["online_update"], *terminal["replay_updates"]]:
                if update is not None:
                    game["updates"].append(prior.compact_update(update))
            work["trained"] += 1
        game["seconds"] += time.perf_counter() - step_start
        record = prior.finish_record(game, board, terminal)
        ownership = np.array(board.area_ownership())
        record.update(
            ownership_positions=terminal.get("ownership_positions", 0),
            ownership_mse=float(np.mean([(np.array(v) - c * ownership) ** 2 for c, v in game["ownership_predictions"]])),
            ownership_zero_mse=float(np.mean(ownership ** 2)),
        )
        work["games"].append(record)
        work["active_game"] = None
        work["game_index"] += 1
        checkpoint()
        print(json.dumps({"method": args.method, "seed": args.seed, "phase": phase,
                          "game": work["game_index"], "plies": len(board.move_history),
                          "win": record["win"], "seconds": round(game["seconds"], 2)}), flush=True)
        if args.max_plies and consumed >= args.max_plies:
            return
    work["completed"] = True
    work["seconds"] += time.perf_counter() - started
    work["mechanisms"]["after"] = mechanism_probe(agent, data)
    checkpoint()
    result = {k: v for k, v in work.items() if k not in ("session", "active_game")}
    result["checkpoint"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    result["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["final_anchor_drift"] = None if session.behavior_memory is None else session.behavior_memory.drift(agent)
    result["parameter_version"] = int(agent.online_parameter_version)
    result["anchor_count"] = 0 if session.behavior_memory is None else len(session.behavior_memory.anchors)
    prior.save_json(result, output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3901)
    parser.add_argument("--method", choices=METHODS, default="combined")
    parser.add_argument("--directory", default=str(ROOT / "checkpoints/go12_v1"))
    parser.add_argument("--output")
    parser.add_argument("--warmup-games", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--train-games", type=int, default=16)
    parser.add_argument("--eval-games", type=int, default=48)
    parser.add_argument("--curve-games", type=int, default=24)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--search-depth", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.size, args.window = 10, 16
    if min(args.warmup_games, args.epochs, args.train_games, args.simulations, args.search_depth) < 1:
        parser.error("budgets must be positive")
    if args.eval_games < 2 or args.eval_games % 2 or args.curve_games < 2 or args.curve_games % 2:
        parser.error("evaluation counts must be positive color pairs")
    return args


if __name__ == "__main__":
    torch.set_num_threads(1)
    args = parse_args()
    prior.prepare(args) if args.prepare_only else run(args)
