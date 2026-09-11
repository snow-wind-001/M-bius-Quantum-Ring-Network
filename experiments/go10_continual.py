"""Complete 10x10 games, online search distillation, and resumable evidence.

No move cap creates a result. A compute interruption saves the unfinished
board/session and resumes exactly that game. See analysis/go10_plan.md.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
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

from mqr import GoBoard, GoLossWeights, GoMultiHeadOutput, HeuristicGoTeacher
from mqr.go import DefensiveGoTeacher
from mqr.go_agent import legality_target
from mqr.go_memory import GoHistoryEncoder, GoResearchCore, PositionQueryGoAgent
from mqr.go_outcome import (OutcomeGoSession, PolicyBehaviorMemory, SearchOutcomeGoSession,
                            _board_from_record, _board_record)
from mqr.go_policy import SignedQueryGoAgent
from mqr.go_search import policy_value_search

SEEDS = (601, 607, 613, 617, 619)
METHODS = ("frozen", "legacy", "optimized", "identity", "unprotected")


def source_hashes():
    paths = list((ROOT / "mqr").glob("*.py")) + [Path(__file__)]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def save_torch(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build(args, method, seed):
    torch.manual_seed(seed)
    encoder = GoHistoryEncoder(args.size)
    rates = (1.0, 0.15, 0.03)
    core = GoResearchCore(
        encoder.output_dim, 16, 16, board_size=args.size, fixed_colors=True,
        conditional_scale=0.0 if method == "identity" else 0.3,
        transition_mode="identity" if method == "identity" else "orthogonal",
        transition_structure="cyclic_givens", injection_rank=8,
        base_unitary_init="random", base_unitary_seed=seed + 31,
        leak_rates=rates, write_scales=tuple(math.sqrt(1 - (1 - rate) ** 2) for rate in rates),
        state_activation="none", readout_bias=True,
    )
    torch.manual_seed(seed + 100)
    shared = dict(board_size=args.size, core=core, ring_dim=16, latent_dim=16,
                  channels=12, query_dim=8, task_lr=0.08 if method == "legacy" else 0.02,
                  max_update_norm=0.15 if method == "legacy" else 0.08,
                  max_trace_horizon=64, ogd_max_rank=4, legality_policy_scale=1.0)
    if method == "legacy":
        agent = PositionQueryGoAgent(encoder.output_dim, **shared,
                                    loss_weights=GoLossWeights(value=0, illegal_mass=0.2))
        agent.spatial_skip_heads.requires_grad_(False)
    else:
        agent = SignedQueryGoAgent(
            encoder.output_dim, **shared, adapt_spatial=True, spatial_head_only=True,
            loss_weights=GoLossWeights(placement=0, pass_decision=0, value=0,
                                      legality=0.25, illegal_mass=0.1, policy_distillation=1.0),
        )
    agent.utility_gate.requires_grad_(False)
    return agent, encoder


def make_session(args, agent, encoder, method):
    protected = method not in ("frozen", "unprotected")
    memory = PolicyBehaviorMemory(4, strata=4, reliable_only=method != "legacy",
                                  margin_fraction=0.5 if method != "legacy" else None) if protected else None
    cls = OutcomeGoSession if method == "legacy" else SearchOutcomeGoSession
    return cls(agent, encoder, update_every=args.window, credit_horizon=args.window,
               project_with_memory=protected, behavior_memory=memory, refresh_every=1,
               max_anchor_kl=0.01 if method == "legacy" else 0.05,
               replay_mode="none" if method in ("legacy", "frozen") else "outcome",
               outcome_weight=0.5, episode_capacity=4096)


def opening(seed, size, plies):
    rng, board, teacher = random.Random(seed), GoBoard(size, komi=5.5), HeuristicGoTeacher()
    for _ in range(plies):
        legal = board.legal_moves(include_pass=False)
        if not legal:
            raise RuntimeError("opening unexpectedly ran out of placements")
        action = teacher.select_move(board) if rng.random() < 0.5 else rng.choice(legal)
        board.play(rng.choice(legal) if action == board.pass_action else action)
    return list(board.move_history)


def complete_teacher_game(size, seed, *, defensive=False):
    rng, board = random.Random(seed), GoBoard(size, komi=5.5)
    teacher = DefensiveGoTeacher() if defensive else HeuristicGoTeacher()
    encoder = GoHistoryEncoder(size)
    records = []
    while not board.game_over:
        target = teacher.select_move(board)
        records.append({"x": encoder.encode_board(board), "action": target,
                        "legal": legality_target(board), "color": board.to_play})
        action = target
        if len(board.move_history) < int(0.7 * size * size) and rng.random() < 0.25:
            legal = board.legal_moves(include_pass=False)
            if legal:
                action = rng.choice(legal)
        board.play(action)
    for record in records:
        record["value"] = float(board.winner() * record["color"])
    return records, {"moves": board.move_history, "winner": board.winner(), "score": board.score(),
                     "terminated": board.game_over, "seed": seed, "defensive": defensive}


def prepare(args):
    path = Path(args.directory) / f"base-{args.seed}.pt"
    if path.exists():
        data = torch.load(path, weights_only=True)
        if (data["source"] != source_hashes() or data["size"] != args.size
                or len(data["training_games"]) != args.warmup_games or data["epochs"] != args.epochs):
            raise RuntimeError("existing base has different source or board size; use a new directory")
        return path
    started = time.perf_counter()
    traces, games = [], []
    for index in range(args.warmup_games):
        records, game = complete_teacher_game(args.size, args.seed + 31000 + index)
        traces.append(records)
        games.append(game)
    records = [item for trace in traces for item in trace]
    x, legal = torch.cat([r["x"] for r in records]), torch.cat([r["legal"] for r in records])
    y, values = torch.tensor([r["action"] for r in records]), torch.tensor([r["value"] for r in records])
    agent, _ = build(args, "optimized", args.seed)
    base = agent.spatial_skip_heads
    base.requires_grad_(True)
    nn.init.constant_(base.pass_decision.bias, -math.log(args.size ** 2))
    optimizer = torch.optim.Adam(base.parameters(), lr=0.006)
    generator = torch.Generator().manual_seed(args.seed + 32000)
    weights = GoLossWeights(value=0.25, illegal_mass=0.2)
    for epoch in range(args.epochs):
        for indices in torch.randperm(len(records), generator=generator).split(64):
            p, l, passing, v = base(x[indices])
            output = GoMultiHeadOutput(p, l, passing, v.tanh(), x.new_zeros(len(indices), 1),
                                      legality_policy_scale=1.0)
            loss = agent.compute_go_loss(output, y[indices], legality_target=legal[indices],
                                         value_target=values[indices], weights=weights)["total"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    anchors = []
    for trace in traces:
        for fraction in (0.15, 0.4, 0.65, 0.9):
            index = min(len(trace) - 1, int(len(trace) * fraction))
            anchors.append({"features": torch.cat([r["x"] for r in trace[max(0, index - 7):index + 1]]),
                            "action": trace[index]["action"], "stratum": int(fraction * 4)})
    probes, probe_games = [], []
    for index in range(4):
        trace, game = complete_teacher_game(args.size, args.seed + 41000 + index, defensive=index >= 2)
        probe_games.append(game)
        for fraction in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
            end = min(len(trace) - 1, int(len(trace) * fraction))
            probes.append({**trace[end], "features": torch.cat([r["x"] for r in trace[max(0, end - 15):end + 1]]),
                           "domain": "b" if index >= 2 else "a", "game_index": index, "ply": end})
    prior = {color: float(values[torch.tensor([r["color"] == color for r in records])].mean()) for color in (-1, 1)}
    data = {"version": 1, "source": source_hashes(), "size": args.size, "seed": args.seed,
            "base": copy.deepcopy(base.state_dict()), "training_games": games, "probe_games": probe_games,
            "anchors": anchors, "probes": probes, "color_prior": prior,
            "training_positions": len(records), "epochs": args.epochs, "seconds": time.perf_counter() - started}
    save_torch(data, path)
    print(json.dumps({"prepared": str(path), "positions": len(records), "complete_games": len(games),
                      "seconds": data["seconds"]}), flush=True)
    return path


@torch.no_grad()
def probe(agent, data, *, reset_history=False):
    rows = []
    for record in data["probes"]:
        state = agent.core.zero_state(1, device="cpu", dtype=torch.float32)
        for x in record["features"].split(1):
            if reset_history:
                state = agent.core.zero_state(1, device="cpu", dtype=torch.float32)
            output, state = agent._transition(x, state, slow_write=True)
        logits = output.policy_logits[0]
        mask = torch.cat((record["legal"][0].bool(), torch.ones(1, dtype=torch.bool)))
        action = int(logits.masked_fill(~mask, -torch.inf).argmax())
        rows.append({"domain": record["domain"], "nll": float(-logits[record["action"]]),
                     "agreement": int(action == record["action"]), "value": float(output.value[0]),
                     "target": record["value"], "color": record["color"],
                     "value_mse": (float(output.value[0]) - record["value"]) ** 2,
                     "prior_mse": (data["color_prior"][record["color"]] - record["value"]) ** 2})
    return {domain: {key: float(np.mean([r[key] for r in rows if r["domain"] == domain]))
                     for key in ("nll", "agreement", "value_mse", "prior_mse")}
            for domain in ("a", "b")} | {"rows": rows}


def initialize_memory(session, data):
    memory = session.behavior_memory
    if memory is None:
        return
    selected = set()
    with torch.no_grad():
        for record in data["anchors"]:
            stratum = record["stratum"]
            if stratum in selected:
                continue
            scores = memory._output(session.agent, record)
            if int(scores.argmax()) == record["action"]:
                memory.add(record["features"].split(1), record["action"], stratum=stratum)
                selected.add(stratum)
    if not memory.anchors:
        record = data["anchors"][0]
        memory.add(record["features"].split(1), record["action"], stratum=record["stratum"])
    memory.freeze(session.agent)
    session.refresh_protection()


def config_record(args):
    return {key: value for key, value in vars(args).items() if key not in ("max_plies", "prepare_only")}


def stages(args, method):
    result = [(name, args.train_games, True, args.simulations) for name in ("a", "b", "return_a")]
    result.append(("search_eval", args.eval_games, False, args.simulations))
    if method in ("legacy", "optimized"):
        result.append(("greedy_eval", args.eval_games, False, 0))
    return result


def stage_opening(args, phase, index):
    if phase in ("a", "return_a"):
        return []
    if phase == "b":
        return opening(args.seed + 51000 + index, args.size, 12)
    pair = index // 2
    return opening(args.seed + 61000 + pair, args.size, (4, 24, 60)[pair % 3])


def compact_update(result):
    if result is None:
        return None
    return {key: result[key] for key in ("did_update", "update_norm", "parameter_version", "anchor_drift",
                                         "anchor_step_scale", "anchor_backtracks", "prediction_before_update")
            if key in result}


def begin_game(args, session, phase, index):
    session.reset_game()
    moves = stage_opening(args, phase, index)
    board = GoBoard(args.size, komi=5.5)
    for action in moves:
        session.observe(board, learn=False)
        board.play(action)
    return {"phase": phase, "index": index, "opening": moves, "board": _board_record(board),
            "student_color": 1 if index % 2 == 0 else -1, "decisions": [], "predictions": [],
            "updates": [], "features": [], "seconds": 0.0, "peak_online_tensor_bytes": 0}


def finish_record(game, board, terminal):
    winner, color = board.winner(), game["student_color"]
    predictions = game["predictions"]
    decisions = game["decisions"]
    return {key: game[key] for key in ("phase", "index", "opening", "student_color", "decisions", "updates",
                                       "seconds", "peak_online_tensor_bytes")} | {
        "moves": list(board.move_history), "terminated": board.game_over,
        "consecutive_passes": board.consecutive_passes, "winner": winner, "score": board.score(),
        "win": int(winner == color), "margin": board.score()["margin_black"] * color,
        "observations": len(predictions), "value_mse": float(np.mean([(v - c * winner) ** 2 for c, v in predictions])),
        "search_evaluations": sum(d["search_evaluations"] for d in decisions),
        "terminal_labels": len(terminal.get("value_targets", [])),
        "replayed_positions": terminal.get("replayed_positions", 0),
    }


def run(args):
    base_path = prepare(args)
    data = torch.load(base_path, weights_only=True)
    directory, method = Path(args.directory), args.method
    work_path = directory / f"work-{method}-{args.seed}.pt"
    result_path = Path(args.output or ROOT / "analysis/results" / f"go10_{method}_seed{args.seed}.json")
    agent, encoder = build(args, method, args.seed)
    agent.spatial_skip_heads.load_state_dict(data["base"])
    session = make_session(args, agent, encoder, method)
    if work_path.exists():
        work = torch.load(work_path, weights_only=True)
        if work["config"] != config_record(args) or work["source"] != source_hashes():
            raise RuntimeError("resume configuration or source changed; use a separate directory")
        session.load_state_dict(work["session"])
    else:
        if method not in ("legacy", "frozen", "unprotected"):
            initialize_memory(session, data)
        work = {"version": 1, "config": config_record(args), "source": source_hashes(),
                "stage": 0, "game_index": 0, "active_game": None, "games": [],
                "probes": {"before": probe(agent, data)}, "session": None, "completed": False,
                "evaluation_parameter_checks": {},
                "total_parameters": sum(p.numel() for p in agent.parameters()),
                "trainable_parameters": sum(p.numel() for p in agent.parameters() if p.requires_grad)}

    def checkpoint():
        work["session"] = session.state_dict()
        save_torch(work, work_path)

    consumed = 0
    schedule = stages(args, method)
    while work["stage"] < len(schedule):
        phase, count, training, simulations = schedule[work["stage"]]
        if work["game_index"] == count:
            if phase == "a" and method == "legacy":
                if not session.behavior_memory.anchors:
                    raise RuntimeError("legacy failed to collect training anchors")
                session.behavior_memory.freeze(agent)
            work["probes"]["after_" + phase] = probe(agent, data)
            if not training:
                check = work["evaluation_parameter_checks"][phase]
                current = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in agent.parameters())).hexdigest()
                if current != check["parameters_before"] or int(agent.online_parameter_version) != check["version_before"]:
                    raise RuntimeError("evaluation changed model parameters or update version")
                check["parameters_after"] = current
                check["version_after"] = int(agent.online_parameter_version)
            work["stage"] += 1
            work["game_index"] = 0
            checkpoint()
            continue
        if work["active_game"] is None:
            if not training and work["game_index"] == 0:
                work["evaluation_parameter_checks"][phase] = {
                    "parameters_before": hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in agent.parameters())).hexdigest(),
                    "version_before": int(agent.online_parameter_version)}
            work["active_game"] = begin_game(args, session, phase, work["game_index"])
        game = work["active_game"]
        board = _board_from_record(game["board"])
        teacher = DefensiveGoTeacher() if phase == "b" or (not training and game["index"] // 2 % 2) else HeuristicGoTeacher()
        learn = training and method != "frozen"
        while not board.game_over:
            started = time.perf_counter()
            result = session.observe(board, learn=learn)
            student = board.to_play == game["student_color"]
            search = None
            if student and simulations:
                search = policy_value_search(agent, encoder, board, result["state"], result["policy_logits"],
                                             simulations=simulations, max_depth=args.search_depth)
            selected = search["action"] if search else result["action"]
            # The diagnostic/teacher query follows the student's fixed decision.
            teacher_action = teacher.select_move(board)
            actual = selected if student else teacher_action
            if student:
                captures, atari = 0, False
                if actual != board.pass_action:
                    _, captures, liberties = board._simulate_stone(actual)
                    atari = liberties == 1 and captures == 0
                game["decisions"].append({"ply": len(board.move_history), "action": actual,
                    "raw_action": result["raw_action"], "raw_legal": result["raw_legal"],
                    "teacher_action": teacher_action, "nll": float(-result["policy_logits"][0, teacher_action]),
                    "captures": captures, "self_atari": bool(atari),
                    "search_evaluations": 0 if search is None else search["network_evaluations"]})
            game["predictions"].append((board.to_play, float(result["output"].value[0])))
            if learn:
                if method == "legacy":
                    session.feedback(result["ticket_id"], teacher_action)
                else:
                    target = session.search_target(search, board.action_size) if student and search else None
                    session.feedback(result["ticket_id"], actual, policy_target=target)
                if phase == "a" and method == "legacy":
                    game["features"].append(encoder.encode_board(board))
                    game["features"] = game["features"][-8:]
                    if len(board.move_history) % 16 == 0:
                        session.behavior_memory.add(game["features"], teacher_action,
                                                    stratum=(len(board.move_history) // 16) % 4)
                if session.pending_count == session.update_every:
                    game["updates"].append(compact_update(session.flush()))
                game["peak_online_tensor_bytes"] = max(game["peak_online_tensor_bytes"], session.online_tensor_bytes)
            board.play(actual)
            game["board"] = _board_record(board)
            game["seconds"] += time.perf_counter() - started
            consumed += 1
            if consumed % 64 == 0 or (args.max_plies and consumed >= args.max_plies):
                checkpoint()
            if args.max_plies and consumed >= args.max_plies and not board.game_over:
                print(json.dumps({"paused": True, "phase": phase, "game": game["index"],
                                  "ply": len(board.move_history), "checkpoint": str(work_path)}), flush=True)
                return
        started = time.perf_counter()
        terminal = session.finish_game(board) if learn else {}
        if learn:
            for update in [terminal["online_update"], *terminal["replay_updates"]]:
                if update is not None:
                    game["updates"].append(compact_update(update))
        game["seconds"] += time.perf_counter() - started
        work["games"].append(finish_record(game, board, terminal))
        work["active_game"] = None
        work["game_index"] += 1
        checkpoint()
        print(json.dumps({"method": method, "seed": args.seed, "phase": phase,
                          "game": work["game_index"], "plies": len(board.move_history),
                          "win": work["games"][-1]["win"], "seconds": game["seconds"]}), flush=True)
        if args.max_plies and consumed >= args.max_plies:
            return
    work["completed"] = True
    work["probes"]["reset_history"] = probe(agent, data, reset_history=True)
    checkpoint()
    result = {key: value for key, value in work.items() if key not in ("session", "active_game")}
    result["checkpoint"] = str(work_path.relative_to(ROOT)) if work_path.is_relative_to(ROOT) else str(work_path)
    result["checkpoint_sha256"] = hashlib.sha256(work_path.read_bytes()).hexdigest()
    result["base_sha256"] = hashlib.sha256(base_path.read_bytes()).hexdigest()
    result["anchor_count"] = 0 if session.behavior_memory is None else len(session.behavior_memory.anchors)
    result["parameter_version"] = int(agent.online_parameter_version)
    result["final_anchor_drift"] = None if session.behavior_memory is None else session.behavior_memory.drift(agent)
    save_json(result, result_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2601)
    parser.add_argument("--method", choices=METHODS, default="optimized")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--directory", default=str(ROOT / "checkpoints/go10_v1"))
    parser.add_argument("--output")
    parser.add_argument("--warmup-games", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--train-games", type=int, default=4, help="complete games in each of A/B/A")
    parser.add_argument("--eval-games", type=int, default=24)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--search-depth", type=int, default=12)
    parser.add_argument("--max-plies", type=int, default=0, help="pause this invocation and resume; never adjudicate")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    if (args.size != 10 or args.warmup_games < 1 or args.epochs < 1 or args.train_games < 1
            or args.eval_games < 2 or args.eval_games % 2 or not 1 <= args.window <= 64
            or args.simulations < 1 or args.search_depth < 1 or args.max_plies < 0):
        parser.error("invalid 10x10 protocol or compute budget")
    if args.confirmation and (args.seed not in SEEDS or args.warmup_games != 24 or args.epochs != 24
                             or args.train_games != 4 or args.eval_games != 24 or args.window != 16
                             or args.simulations != 16 or args.search_depth != 12):
        parser.error("confirmation requires a registered seed and the complete preregistered configuration")
    return args


if __name__ == "__main__":
    torch.set_num_threads(1)
    args = parse_args()
    if args.prepare_only:
        prepare(args)
    else:
        run(args)
