"""Go13: crossed geometry/constraint study of causal online rule learning."""
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
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import go11_continual as prior
from mqr import GoBoard, GoLossWeights, HeuristicGoTeacher
from mqr.go import DefensiveGoTeacher
from mqr.go_history_tasks import generate_ko_history_pairs, generate_reachable_history_pairs
from mqr.go_memory import GoHistoryEncoder, GoResearchCore
from mqr.go_online import GoOnlineSession
from mqr.go_policy import SignedQueryGoAgent
from mqr.go_research import (SimilarityGoCore, RuleBatch, RuleConstraint, OnlineRuleLearner,
                             history_output, queried_logits, parameter_vector, task_parameters)

SEEDS = (911, 919, 929, 937, 941)
METHODS = ("orth_free", "orth_current", "similar_free", "similar_current", "orth_stale", "orth_norm", "orth_frozen")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = list((ROOT / "mqr").glob("*.py")) + [Path(__file__), Path(prior.__file__)]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def parameter_hash(agent):
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in agent.parameters())).hexdigest()


def build(seed, geometry="orth"):
    torch.manual_seed(seed)
    encoder = GoHistoryEncoder(10)
    rates = (1., .5, .15, .07, .03, .01)
    cls = GoResearchCore if geometry == "orth" else SimilarityGoCore
    core = cls(encoder.output_dim, 16, 16, board_size=10, fixed_colors=True,
        conditional_scale=.3, transition_mode="orthogonal", transition_structure="cyclic_givens",
        injection_rank=8, base_unitary_init="random", base_unitary_seed=seed + 31,
        leak_rates=rates, write_scales=tuple(math.sqrt(1 - (1 - r) ** 2) for r in rates),
        state_activation="none", readout_bias=True)
    torch.manual_seed(seed + 100)
    agent = SignedQueryGoAgent(encoder.output_dim, board_size=10, core=core, ring_dim=16, latent_dim=16,
        channels=12, query_dim=8, max_trace_horizon=64, ogd_max_rank=4, legality_policy_scale=1.,
        adapt_spatial=True, spatial_head_only=False, loss_weights=GoLossWeights())
    agent.utility_gate.requires_grad_(False)
    # Rule feedback updates shared spatial features, rings, queries and legality.
    # No fabricated policy/value target is used. Unused head weights stay fixed.
    for module in (agent.pass_head, agent.value_head, agent.spatial_skip_heads.placement,
                   agent.spatial_skip_heads.pass_decision, agent.spatial_skip_heads.value):
        module.requires_grad_(False)
    return agent, encoder


def pack_history(history, query, label, task, pair):
    first, last = history[0], history[-1]
    return {"initial": first.board, "to_play": first.to_play, "moves": list(last.move_history),
            "query": int(query), "label": int(label), "task": task, "pair": pair}


def boards(record):
    stones = record["initial"]
    b = GoBoard.from_stones(10, black=[i for i, c in enumerate(stones) if c == 1],
                           white=[i for i, c in enumerate(stones) if c == -1], to_play=record["to_play"], komi=5.5)
    history = [b.copy()]
    for move in record["moves"]:
        b.play(move)
        history.append(b.copy())
    return history


def make_suicide_pairs(count, seed):
    rng = random.Random(seed)
    records = []
    for pair in range(count):
        row, col = rng.randrange(1, 9), rng.randrange(1, 9)
        query = row * 10 + col
        neighbors = [query - 10, query + 10, query - 1, query + 1]
        missing = rng.choice(neighbors)
        color = rng.choice((-1, 1))
        occupied = set(neighbors) | {query}
        extra = {}
        candidates = list(range(100)); rng.shuffle(candidates)
        shell = GoBoard(10)
        for point in candidates:
            if len(extra) == 12:
                break
            if point in occupied or any(n in occupied for n in shell._neighbors(point)):
                continue
            extra[point] = rng.choice((-1, 1)); occupied.add(point)
        move = next(p for p in candidates if p not in occupied and not any(n in occupied for n in shell._neighbors(p)))
        for legal in (False, True):
            stones = {p: -color for p in neighbors if not legal or p != missing} | extra
            b = GoBoard.from_stones(10, black=[p for p, c in stones.items() if c == 1],
                                   white=[p for p, c in stones.items() if c == -1], to_play=-color, komi=5.5)
            first = b.copy(); b.play(move)
            if b.is_legal(query) != legal:
                raise RuntimeError("suicide example failed simulator verification")
            records.append(pack_history((first, b), query, legal, "suicide", pair))
    return records


def materialize(records):
    encoder = GoHistoryEncoder(10)
    histories = [boards(r) for r in records]
    for record, history in zip(records, histories):
        if history[-1].is_legal(record["query"]) != bool(record["label"]):
            raise RuntimeError("feedback differs from simulator legality")
    return RuleBatch(torch.stack([torch.cat([encoder.encode_board(b) for b in h]) for h in histories], 1),
                     torch.tensor([r["query"] for r in records]), torch.tensor([r["label"] for r in records]).float())


def make_data(args):
    a = []
    for index, pair in enumerate(generate_ko_history_pairs(args.a_pairs + args.test_pairs, seed=args.seed + 81000, size=10)):
        for h, legal in zip(pair.histories, pair.legal):
            a.append(pack_history(h, pair.recapture, legal, "ko", index))
    b = make_suicide_pairs(args.b_pairs + args.test_pairs, args.seed + 82000)
    data = {"a_train": a[:2 * args.a_pairs], "a_test": a[2 * args.a_pairs:],
            "b_train": b[:2 * args.b_pairs], "b_test": b[2 * args.b_pairs:]}
    # Do not let any final observed board/query cross train/held-out boundaries.
    def signatures(rows):
        return {repr((boards(r)[-1].board, boards(r)[-1].to_play, r["query"])) for r in rows}
    if (signatures(data["a_train"]) | signatures(data["b_train"])) & (signatures(data["a_test"]) | signatures(data["b_test"])):
        raise RuntimeError("train/test observation-query overlap")
    return data


@torch.no_grad()
def evaluate(agent, batch, *, intervention="full"):
    features = batch.features
    if intervention == "swap":
        features = features.clone()
        order = torch.arange(features.size(1)).reshape(-1, 2).flip(1).flatten()
        features[:-1] = features[:-1, order]
    logits = []
    for indices in torch.arange(len(batch.labels)).split(64):
        out = history_output(agent, features[:, indices], reset=intervention == "reset")
        if intervention == "no_query":
            state = agent.core.zero_state(len(indices), device=features.device, dtype=features.dtype)
            for x in features[:, indices]:
                latent, state = agent.core.forward_step(x, state=state)
            out = agent._query_output(latent, state, features[-1, indices], include_query=False)
        logits.extend(queried_logits(out, batch.queries[indices]).tolist())
    tensor = torch.tensor(logits); correct = (tensor >= 0) == batch.labels.bool()
    return {"accuracy": float(correct.float().mean()), "loss": float(F.binary_cross_entropy_with_logits(tensor, batch.labels)),
            "both_correct": int(correct.reshape(-1, 2).all(1).sum()), "pairs": len(logits) // 2, "logits": logits}


def config(args):
    return {k: v for k, v in vars(args).items() if k not in ("stage", "method", "output")}


def prepare(args):
    directory = Path(args.directory); directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"prepared-{args.seed}.pt"
    if target.exists():
        saved = torch.load(target, weights_only=True)
        if saved["source"] != source_hashes() or saved["config"] != config(args):
            raise RuntimeError("prepared research source/configuration changed")
        return target
    base_path = prior.prepare(args)
    base = torch.load(base_path, weights_only=True)
    data = make_data(args)
    batches = {k: materialize(v) for k, v in data.items()}
    save = {"source": source_hashes(), "config": config(args), "data": data,
            "base_sha256": sha(base_path), "base": base["base"], "models": {}, "optimizers": {}, "curves": {}, "a_updates": {}}
    agents = {}
    for geometry in ("orth", "similar"):
        agent, _ = build(args.seed, geometry); agent.spatial_skip_heads.load_state_dict(base["base"])
        learner = OnlineRuleLearner(agent, lr=args.lr, max_step=args.max_step)
        initial = parameter_hash(agent)
        curves, updates = [], []
        for start in range(0, len(batches["a_train"].labels), 16):
            if start in (0, 32, 128, 512, 1024):
                curves.append({"feedback": start, "a": evaluate(agent, batches["a_test"])})
            updates.append(learner.step(batches["a_train"].subset(slice(start, start + 16))))
        curves.append({"feedback": len(batches["a_train"].labels), "a": evaluate(agent, batches["a_test"])})
        save["models"][geometry] = copy.deepcopy(agent.state_dict())
        save["optimizers"][geometry] = copy.deepcopy(learner.optimizer.state_dict())
        save["curves"][geometry] = curves
        save["a_updates"][geometry] = updates
        save.setdefault("initial_hashes", {})[geometry] = initial
        agents[geometry] = agent
        print(json.dumps({"prepared_geometry": geometry, "seed": args.seed, "feedback": len(batches["a_train"].labels),
                          "a_accuracy": curves[-1]["a"]["accuracy"]}), flush=True)
    # Common train-only anchor histories, correct for both geometries.
    margins = []
    for geometry in ("orth", "similar"):
        logits = torch.tensor(evaluate(agents[geometry], batches["a_train"])["logits"])
        margins.append((2 * batches["a_train"].labels - 1) * logits)
    pair_minimum = torch.stack(margins).amin(0).reshape(-1, 2).amin(1)
    choices = [int(i) for i in pair_minimum.argsort(descending=True) if pair_minimum[i] > .02][:8]
    indices = [2 * i + branch for i in choices for branch in (0, 1)]
    save["anchor_indices"] = indices
    save["anchor_floors"] = {g: .5 * m[indices] for g, m in zip(("orth", "similar"), margins)}
    save["same_initial_parameters"] = save["initial_hashes"]["orth"] == save["initial_hashes"]["similar"]
    if not save["same_initial_parameters"]:
        raise RuntimeError("geometry control changed trainable parameter initialization")
    prior.save_torch(save, target)
    prior.save_json({k: v for k, v in save.items() if k not in ("models", "optimizers", "base", "anchor_floors")},
                    ROOT / "analysis/results" / f"go13_prepared_seed{args.seed}.json")
    return target


def restore(args, method):
    prepared = torch.load(Path(args.directory) / f"prepared-{args.seed}.pt", weights_only=True)
    if prepared["source"] != source_hashes() or prepared["config"] != config(args):
        raise RuntimeError("prepared source/configuration changed")
    geometry, mode = method.split("_")
    agent, encoder = build(args.seed, geometry)
    agent.load_state_dict(prepared["models"][geometry])
    batches = {k: materialize(v) for k, v in prepared["data"].items()}
    anchors = batches["a_train"].subset(prepared["anchor_indices"])
    constraint = None
    if mode in ("current", "stale", "norm"):
        constraint = RuleConstraint(anchors, prepared["anchor_floors"][geometry],
            backend="transport" if geometry == "orth" else "autograd", refresh_every=8 if mode == "stale" else 1)
    learner = OnlineRuleLearner(agent, mode=mode, lr=args.lr, max_step=args.max_step, constraint=constraint)
    learner.optimizer.load_state_dict(copy.deepcopy(prepared["optimizers"][geometry]))
    return prepared, agent, encoder, batches, learner


def run_learning(args):
    started = time.perf_counter()
    prepared, agent, encoder, batches, learner = restore(args, args.method)
    target = Path(args.directory) / f"learned-{args.method}-{args.seed}.pt"
    if target.exists():
        saved = torch.load(target, weights_only=True)
        if saved["source"] == source_hashes() and saved["config"] == config(args):
            return
        raise RuntimeError("learned checkpoint differs")
    before = {task: evaluate(agent, batches[task + "_test"]) for task in ("a", "b")}
    updates, curves = [], []
    for start in range(0, len(batches["b_train"].labels), 16):
        updates.append(learner.step(batches["b_train"].subset(slice(start, start + 16))))
        if start + 16 in (32, 128, 512, len(batches["b_train"].labels)):
            curves.append({"feedback": start + 16, **{task: evaluate(agent, batches[task + "_test"]) for task in ("a", "b")}})
    after = {task: evaluate(agent, batches[task + "_test"]) for task in ("a", "b")}
    interventions = {kind: evaluate(agent, batches["a_test"], intervention=kind) for kind in ("reset", "swap", "no_query")}
    geometry = args.method.split("_")[0]
    anchors = batches["a_train"].subset(prepared["anchor_indices"])
    with torch.no_grad():
        anchor_margins = ((2 * anchors.labels - 1) * queried_logits(history_output(agent, anchors.features), anchors.queries)
                          if len(anchors.labels) else torch.empty(0))
    result = {"method": args.method, "seed": args.seed, "source": source_hashes(), "config": config(args),
        "prepared_sha256": sha(Path(args.directory) / f"prepared-{args.seed}.pt"),
        "before": before, "after": after, "curves": curves, "updates": updates, "interventions": interventions,
        "anchor_indices": prepared["anchor_indices"], "anchor_floors": prepared["anchor_floors"][geometry].tolist(),
        "anchor_margins_after": anchor_margins.tolist(), "anchor_violations": int((anchor_margins < prepared["anchor_floors"][geometry] - 1e-5).sum()),
        "constraint_refreshes": 0 if learner.constraint is None else learner.constraint.refreshes,
        "total_parameters": sum(p.numel() for p in agent.parameters()),
        "trainable_parameters": sum(p.numel() for p in task_parameters(agent)),
        "initial_a_hash": hashlib.sha256(b"".join(v.numpy().tobytes() for v in prepared["models"][geometry].values() if isinstance(v, torch.Tensor))).hexdigest(),
        "final_parameter_hash": parameter_hash(agent), "seconds": time.perf_counter() - started,
        "completed": True}
    prior.save_torch({"source": source_hashes(), "config": config(args), "learner": learner.state_dict(), "result": result}, target)
    result["checkpoint_sha256"] = sha(target)
    prior.save_json(result, ROOT / "analysis/results" / f"go13_learning_{args.method}_seed{args.seed}.json")
    print(json.dumps({"method": args.method, "seed": args.seed, "a": after["a"]["accuracy"], "b": after["b"]["accuracy"],
                      "anchors": len(anchors.labels), "active": sum(u["active"] > 0 for u in updates),
                      "accepted": sum(u["accepted"] for u in updates), "seconds": result["seconds"]}), flush=True)


def run_games(args):
    _, agent, encoder, _, learner = restore(args, args.method)
    path = Path(args.directory) / f"learned-{args.method}-{args.seed}.pt"
    saved = torch.load(path, weights_only=True)
    if saved["source"] != source_hashes() or saved["config"] != config(args):
        raise RuntimeError("game checkpoint source/configuration differs")
    learner.load_state_dict(saved["learner"])
    session = GoOnlineSession(agent, encoder, update_every=16, project_with_memory=False)
    output = ROOT / "analysis/results" / f"go13_games_{args.method}_seed{args.seed}.json"
    checkpoint = Path(args.directory) / f"games-{args.method}-{args.seed}.pt"
    result = {"source": source_hashes(), "config": config(args), "method": args.method, "seed": args.seed,
              "checkpoint_sha256": sha(path), "parameter_hash": parameter_hash(agent), "games": [], "completed": False}
    if checkpoint.exists():
        result = torch.load(checkpoint, weights_only=True)
        if result["source"] != source_hashes() or result["config"] != config(args):
            raise RuntimeError("game resume source/configuration differs")
    for index in range(len(result["games"]), args.eval_games):
        session.reset_game()
        moves = prior.opening(args.seed + 91000 + index // 2, 10, (4, 24, 60)[(index // 2) % 3])
        board = GoBoard(10, komi=5.5)
        for move in moves:
            session.observe(board, learn=False); board.play(move)
        color = 1 if index % 2 == 0 else -1
        teacher = DefensiveGoTeacher() if index // 2 % 2 else HeuristicGoTeacher()
        decisions = []
        started = time.perf_counter()
        while not board.game_over:
            observation = session.observe(board, learn=False)
            if board.to_play == color:
                search = prior.search_decision(args, agent, encoder, board, observation,
                    phase="a", index=index, simulations=args.simulations, training=False)
                move = search["action"]
                decisions.append({"ply": len(board.move_history), "action": move,
                    "raw_legal": observation["raw_legal"], "search_evaluations": search["network_evaluations"]})
            else:
                move = teacher.select_move(board)
            board.play(move)
        result["games"].append({"index": index, "opening": moves, "student_color": color,
            "moves": board.move_history, "decisions": decisions, "terminated": board.game_over,
            "consecutive_passes": board.consecutive_passes, "score": board.score(), "winner": board.winner(),
            "win": int(board.winner() == color), "margin": board.score()["margin_black"] * color,
            "seconds": time.perf_counter() - started})
        if parameter_hash(agent) != result["parameter_hash"]:
            raise RuntimeError("frozen game evaluation changed parameters")
        prior.save_torch(result, checkpoint)
        print(json.dumps({"method": args.method, "seed": args.seed, "game": index + 1, "plies": len(board.move_history)}), flush=True)
    result["completed"] = True
    prior.save_torch(result, checkpoint)
    prior.save_json(result, output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "learn", "games"), default="prepare")
    parser.add_argument("--seed", type=int, default=3907)
    parser.add_argument("--method", choices=METHODS, default="orth_current")
    parser.add_argument("--directory", default=str(ROOT / "checkpoints/go13_v1"))
    parser.add_argument("--warmup-games", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--a-pairs", type=int, default=1024)
    parser.add_argument("--b-pairs", type=int, default=512)
    parser.add_argument("--test-pairs", type=int, default=128)
    parser.add_argument("--eval-games", type=int, default=24)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--search-depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--max-step", type=float, default=.5)
    args = parser.parse_args(); args.size = 10
    if min(args.warmup_games, args.epochs, args.a_pairs, args.b_pairs, args.test_pairs, args.simulations, args.search_depth) < 1:
        parser.error("budgets must be positive")
    if args.a_pairs % 8 or args.b_pairs % 8 or args.eval_games < 2 or args.eval_games % 2:
        parser.error("feedback budgets must fill 16-example windows; games require color pairs")
    return args


if __name__ == "__main__":
    torch.set_num_threads(1)
    args = parse_args()
    {"prepare": prepare, "learn": run_learning, "games": run_games}[args.stage](args)
