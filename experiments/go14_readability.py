"""Go14: controlled write/query layouts and finite-step online protection."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import go11_continual as prior
from experiments import go13_algorithm as previous
from mqr import GoBoard, GoLossWeights, HeuristicGoTeacher
from mqr.go import DefensiveGoTeacher
from mqr.go_feasible import FeasibleRuleLearner, ReadableBehaviorConstraint
from mqr.go_memory import GoHistoryEncoder
from mqr.go_readable import ReadableGoAgent, ReadableGoCore
from mqr.go_research import OnlineRuleLearner, history_output, queried_logits, task_parameters
from mqr.go_search import policy_value_search


SEEDS = (1009, 1013, 1019, 1021, 1031)
LAYOUTS = ("shared", "grouped", "spatial_pool", "spatial", "previous", "similar_spatial",
           "kernel_pool", "kernel", "similar_kernel")
METHODS = {
    "shared_free": ("shared", "free"),
    "grouped_free": ("grouped", "free"),
    "spatial_pool_free": ("spatial_pool", "free"),
    "previous_free": ("previous", "free"),
    "kernel_pool_free": ("kernel_pool", "free"),
    **{"spatial_" + variant: ("spatial", variant)
       for variant in ("free", "current", "reserve", "adaptive", "rollback", "frozen")},
    "spatial_aligned": ("spatial", "adaptive"),
    "spatial_rule24": ("spatial", "adaptive"),
    **{"similar_spatial_" + variant: ("similar_spatial", variant)
       for variant in ("free", "current", "adaptive")},
    **{"kernel_" + variant: ("kernel", variant)
       for variant in ("free", "current", "adaptive")},
    **{"similar_kernel_" + variant: ("similar_kernel", variant)
       for variant in ("free", "current", "adaptive")},
}
GAME_METHODS = ("shared_free", "spatial_free", "spatial_current",
                "spatial_adaptive", "spatial_aligned", "spatial_frozen")
DIRECTORY = ROOT / "checkpoints/go14_v2"


def source_hashes() -> dict:
    paths = list((ROOT / "mqr").glob("*.py")) + [
        Path(__file__), Path(previous.__file__), Path(prior.__file__)]
    return {str(path.relative_to(ROOT)): previous.sha(path) for path in sorted(paths)}


def build(seed: int, layout: str = "shared"):
    if layout not in LAYOUTS:
        raise ValueError("unknown study layout")
    torch.manual_seed(seed)
    encoder = GoHistoryEncoder(10)
    rates = (1., .5, .15, .07, .03, .01)
    write = "spatial" if layout in ("spatial", "spatial_pool", "similar_spatial") else layout
    if layout in ("kernel", "kernel_pool", "similar_kernel"):
        write = "convolution"
    core = ReadableGoCore(encoder.output_dim, 16, 16, board_size=10, fixed_colors=True,
        write_layout=write, geometry="similar" if layout.startswith("similar_") else "orth",
        conditional_scale=.3, transition_mode="orthogonal", transition_structure="cyclic_givens",
        injection_rank=8, base_unitary_init="random", base_unitary_seed=seed + 31,
        leak_rates=rates, write_scales=tuple(math.sqrt(1 - (1 - r) ** 2) for r in rates),
        state_activation="none", readout_bias=True)
    torch.manual_seed(seed + 100)
    agent = ReadableGoAgent(encoder.output_dim, board_size=10, core=core, ring_dim=16, latent_dim=16,
        channels=12, query_dim=8, max_trace_horizon=64, ogd_max_rank=4, legality_policy_scale=1.,
        adapt_spatial=True, spatial_head_only=False, spatial_read=layout not in ("spatial_pool", "kernel_pool"),
        loss_weights=GoLossWeights())
    agent.utility_gate.requires_grad_(False)
    for module in (agent.pass_head, agent.value_head, agent.spatial_skip_heads.placement,
                   agent.spatial_skip_heads.pass_decision, agent.spatial_skip_heads.value):
        module.requires_grad_(False)
    return agent, encoder


def resources(agent) -> dict:
    return {"total_parameters": sum(p.numel() for p in agent.parameters()),
            "trainable_parameters": sum(p.numel() for p in task_parameters(agent)),
            "state_coordinates": sum(agent.core.state_widths),
            "state_bytes_float32": 4 * sum(agent.core.state_widths)}


def make_data(seed: int, pairs: int = 1024, test_pairs: int = 128) -> dict:
    args = SimpleNamespace(seed=seed + 200000, a_pairs=pairs, b_pairs=512, test_pairs=test_pairs)
    return previous.make_data(args)


def develop(seed: int, layout: str, output: Path, prior_path: Optional[Path] = None) -> None:
    if seed in SEEDS:
        raise ValueError("development must not use a formal training seed")
    started = time.perf_counter()
    data = make_data(seed)
    batches = {key: previous.materialize(rows) for key, rows in data.items()}
    agent, _ = build(seed, layout)
    prior_path = prior_path or ROOT / "checkpoints/go14_development" / f"base-{seed}.pt"
    base = torch.load(prior_path, weights_only=True)
    agent.spatial_skip_heads.load_state_dict(base["base"])
    learner = OnlineRuleLearner(agent)
    result = {"development_only": True, "seed": seed, "layout": layout,
              "resources": resources(agent), "initial_hash": previous.parameter_hash(agent),
              "prior": {"seed": base["seed"], "epochs": base["epochs"],
                        "sha256": previous.sha(prior_path),
                        "games": len(base["training_games"])}, "curves": [], "updates": []}
    for start in range(0, len(batches["a_train"].labels), 16):
        if start in (0, 32, 128, 512, 1024):
            result["curves"].append({"feedback": start, "a": previous.evaluate(agent, batches["a_test"])})
        result["updates"].append(learner.step(batches["a_train"].subset(slice(start, start + 16))))
    result["curves"].append({"feedback": len(batches["a_train"].labels),
                             "a": previous.evaluate(agent, batches["a_test"])})
    result["interventions"] = {name: previous.evaluate(agent, batches["a_test"], intervention=name)
                               for name in ("reset", "swap", "no_query")}
    result["train"] = previous.evaluate(agent, batches["a_train"])
    result["seconds"] = time.perf_counter() - started
    prior.save_json(result, output)
    prior.save_torch({"agent": agent.state_dict(), "optimizer": learner.optimizer.state_dict(),
                      "seed": seed, "layout": layout},
                     ROOT / "checkpoints/go14_development" / (output.stem + ".pt"))
    print(json.dumps({"layout": layout, "seed": seed, "train": result["train"]["accuracy"],
                      "test": result["curves"][-1]["a"]["accuracy"],
                      "seconds": result["seconds"], **resources(agent)}), flush=True)


def check_lock() -> dict:
    lock = json.loads((ROOT / "analysis/results/go14_source.json").read_text())
    if lock["source"] != source_hashes() or lock["plan_sha256"] != previous.sha(ROOT / "analysis/go14_plan.md"):
        raise RuntimeError("formal Go14 source or protocol changed")
    return lock


def choose_rule_anchors(data: dict, agents: dict, batch) -> list[int]:
    margins = []
    for layout in ("spatial", "similar_spatial", "kernel", "similar_kernel"):
        logits = torch.tensor(previous.evaluate(agents[layout], batch)["logits"])
        margins.append((2 * batch.labels - 1) * logits)
    minimum = torch.stack(margins).amin(0).reshape(-1, 2).amin(1)
    available = [int(i) for i in minimum.argsort(descending=True) if minimum[i] > .02]
    chosen, used = [], set()
    for pair in available:
        record = data["a_train"][2 * pair]
        row, col = divmod(record["query"], 10)
        stratum = (row // 5, col // 5, record["to_play"])
        if stratum not in used:
            chosen.append(pair); used.add(stratum)
    chosen = (chosen + [pair for pair in available if pair not in chosen])[:12]
    return [2 * pair + branch for pair in chosen for branch in (0, 1)]


def policy_records(base: dict, *, training: bool) -> list[dict]:
    result = []
    fractions = (.15, .4, .65, .9)
    encoder = GoHistoryEncoder(10)
    source = base["anchors"] if training else base["probes"]
    for i, record in enumerate(source):
        if training:
            game = base["training_games"][i // 4]
            ply = min(len(game["moves"]) - 1, int(len(game["moves"]) * fractions[i % 4]))
            stratum = i % 4
        else:
            game = base["probe_games"][record["game_index"]]
            ply, stratum = record["ply"], record["game_index"]
        board = GoBoard(10, komi=5.5)
        for move in game["moves"][:ply]:
            board.play(move)
        if not torch.equal(encoder.encode_board(board)[0], record["features"][-1]):
            raise RuntimeError("policy context and recorded game disagree")
        legal = torch.zeros(101, dtype=torch.bool); legal[board.legal_moves()] = True
        result.append({"features": record["features"][-2:].clone(), "action": record["action"],
                       "legal": legal, "stratum": stratum, "game": i // 4 if training else record["game_index"],
                       "ply": ply})
    return result


@torch.no_grad()
def policy_probe(agent, records: list[dict]) -> dict:
    features = torch.stack([record["features"] for record in records], dim=1)
    output = history_output(agent, features)
    scores = output.policy_logits
    legal = torch.stack([record["legal"] for record in records])
    masked = scores.masked_fill(~legal, -torch.inf)
    actions = masked.argmax(1)
    reference = torch.tensor([record["action"] for record in records])
    return {"actions": actions.tolist(), "probabilities": masked.softmax(1).tolist(),
            "teacher_agreement": float((actions == reference).float().mean()),
            "raw_pass_mean": float(scores.softmax(1)[:, -1].mean())}


def select_policy(agent, records: list[dict]) -> dict:
    with torch.no_grad():
        output = history_output(agent, torch.stack([r["features"] for r in records], dim=1))
        scores = output.policy_logits
        actions = torch.tensor([r["action"] for r in records])
        legal = torch.stack([r["legal"] for r in records])
        alternatives = legal.clone().scatter_(1, actions[:, None], False)
        margins = scores.gather(1, actions[:, None]).squeeze(1) - scores.masked_fill(~alternatives, -torch.inf).max(1).values
    chosen = []
    for stratum in range(4):
        eligible = [i for i, r in enumerate(records) if r["stratum"] == stratum and margins[i] > .02]
        chosen.extend(sorted(eligible, key=lambda i: -float(margins[i]))[:2])
    if not chosen:
        return {}
    return {"features": torch.stack([records[i]["features"] for i in chosen], dim=1),
            "actions": actions[chosen], "legal": legal[chosen], "floors": .5 * margins[chosen],
            "indices": torch.tensor(chosen), "strata": torch.tensor([records[i]["stratum"] for i in chosen])}


def prepare(seed: int) -> None:
    check_lock()
    target = DIRECTORY / f"prepared-{seed}.pt"
    if target.exists():
        saved = torch.load(target, weights_only=True)
        if saved["source"] != source_hashes():
            raise RuntimeError("prepared checkpoint differs from the formal implementation")
        return
    args = SimpleNamespace(directory=str(DIRECTORY), seed=seed, size=10, warmup_games=24, epochs=24)
    base_path = prior.prepare(args)
    base = torch.load(base_path, weights_only=True)
    data = make_data(seed)
    batches = {key: previous.materialize(value) for key, value in data.items()}
    result = {"seed": seed, "source": source_hashes(), "base_sha256": previous.sha(base_path),
              "models": {}, "optimizers": {}, "curves": {}, "updates": {}, "resources": {},
              "initial_hashes": {}, "data": data, "seconds": {}}
    agents = {}
    for layout in LAYOUTS:
        started = time.perf_counter()
        agent, _ = build(seed, layout)
        agent.spatial_skip_heads.load_state_dict(base["base"])
        result["resources"][layout] = resources(agent)
        result["initial_hashes"][layout] = previous.parameter_hash(agent)
        learner = OnlineRuleLearner(agent)
        curve, updates = [], []
        for start in range(0, len(batches["a_train"].labels), 16):
            if start in (0, 32, 128, 512, 1024):
                curve.append({"feedback": start, "a": previous.evaluate(agent, batches["a_test"])})
            updates.append(learner.step(batches["a_train"].subset(slice(start, start + 16))))
        curve.append({"feedback": 2048, "a": previous.evaluate(agent, batches["a_test"])})
        result["models"][layout] = copy.deepcopy(agent.state_dict())
        result["optimizers"][layout] = copy.deepcopy(learner.optimizer.state_dict())
        result["curves"][layout], result["updates"][layout] = curve, updates
        result.setdefault("interventions", {})[layout] = {
            name: previous.evaluate(agent, batches["a_test"], intervention=name)
            for name in ("reset", "swap", "no_query")}
        result["seconds"][layout] = time.perf_counter() - started
        agents[layout] = agent
        print(json.dumps({"seed": seed, "layout": layout, "a_accuracy": curve[-1]["a"]["accuracy"]}), flush=True)
    if len(set(result["initial_hashes"].values())) != 1:
        raise RuntimeError("layouts do not share identical learned initialization")
    result["anchor_indices"] = choose_rule_anchors(data, agents, batches["a_train"])
    result["policy_train"] = policy_records(base, training=True)
    result["policy_test"] = policy_records(base, training=False)
    result["policy"] = select_policy(agents["spatial"], result["policy_train"])
    prior.save_torch(result, target)
    serial = {key: value for key, value in result.items()
              if key not in ("models", "optimizers", "policy_train", "policy_test", "policy")}
    serial["policy_anchor_indices"] = result["policy"].get("indices", torch.empty(0)).tolist()
    serial["policy_anchor_strata"] = result["policy"].get("strata", torch.empty(0)).tolist()
    prior.save_json(serial, ROOT / "analysis/results" / f"go14_prepared_seed{seed}.json")


def restore(seed: int, method: str):
    saved = torch.load(DIRECTORY / f"prepared-{seed}.pt", weights_only=True)
    if saved["source"] != source_hashes():
        raise RuntimeError("research source changed since preparation")
    layout, variant = METHODS[method]
    agent, encoder = build(seed, layout)
    agent.load_state_dict(saved["models"][layout])
    batches = {key: previous.materialize(value) for key, value in saved["data"].items()}
    constraint = None
    protected = variant not in ("free", "frozen")
    if layout in ("spatial", "similar_spatial", "kernel", "similar_kernel"):
        requested = 24 if method == "spatial_rule24" else 16
        indices = saved["anchor_indices"][:requested]
        if protected and len(indices) < requested:
            raise RuntimeError("insufficient mastered historical anchors; protection precondition failed")
        if indices:
            anchors = batches["a_train"].subset(indices)
            with torch.no_grad():
                margins = (2 * anchors.labels - 1) * queried_logits(history_output(agent, anchors.features), anchors.queries)
            policy = saved["policy"] if method == "spatial_aligned" and saved["policy"] else None
            constraint = ReadableBehaviorConstraint(anchors, .5 * margins, policy=policy)
    learner = FeasibleRuleLearner(agent, variant=variant, constraint=constraint if protected else None)
    learner.optimizer.load_state_dict(saved["optimizers"][layout])
    learner.updates = 128
    return saved, agent, encoder, batches, learner, constraint


def learn(seed: int, method: str) -> None:
    check_lock()
    saved, agent, _, batches, learner, guard = restore(seed, method)
    target = DIRECTORY / f"learned-{method}-{seed}.pt"
    if target.exists():
        stored = torch.load(target, weights_only=True)
        if stored["source"] != source_hashes():
            raise RuntimeError("learned checkpoint source differs")
        return
    started = time.perf_counter()
    result = {"seed": seed, "method": method, "source": source_hashes(), "resources": resources(agent),
              "before": {name: previous.evaluate(agent, batches[name + "_test"]) for name in ("a", "b")},
              "policy_before": policy_probe(agent, saved["policy_test"]), "updates": [], "curves": [],
              "rule_anchors": 0 if guard is None else guard.rule_count,
              "policy_anchors": 0 if guard is None or guard.policy is None else len(guard.policy["actions"])}
    for start in range(0, len(batches["b_train"].labels), 16):
        result["updates"].append(learner.step(batches["b_train"].subset(slice(start, start + 16))))
        if start + 16 in (32, 128, 512, 1024):
            result["curves"].append({"feedback": start + 16,
                **{name: previous.evaluate(agent, batches[name + "_test"]) for name in ("a", "b")}})
    result["after"] = {name: previous.evaluate(agent, batches[name + "_test"]) for name in ("a", "b")}
    result["interventions"] = {name: previous.evaluate(agent, batches["a_test"], intervention=name)
                               for name in ("reset", "swap", "no_query")}
    result["policy_after"] = policy_probe(agent, saved["policy_test"])
    if guard is not None:
        with torch.no_grad():
            margins = guard.margins(agent)
        result["anchor_margins"], result["anchor_floors"] = margins.tolist(), guard.floors.tolist()
        result["anchor_violations"] = int((margins < guard.floors - 1e-5).sum())
    else:
        result["anchor_violations"] = None
    result["parameter_hash"] = previous.parameter_hash(agent)
    result["seconds"] = time.perf_counter() - started
    prior.save_torch({"source": source_hashes(), "learner": learner.state_dict(), "result": result}, target)
    prior.save_json(result, ROOT / "analysis/results" / f"go14_learning_{method}_seed{seed}.json")
    print(json.dumps({"seed": seed, "method": method, "old": result["after"]["a"]["accuracy"],
        "new": result["after"]["b"]["accuracy"], "accepted": sum(u["accepted"] for u in result["updates"]),
        "seconds": result["seconds"]}), flush=True)


def games(seed: int, method: str) -> None:
    check_lock()
    if method not in GAME_METHODS:
        raise ValueError("this method is not in the fixed complete-game matrix")
    _, agent, encoder, _, learner, _ = restore(seed, method)
    checkpoint = DIRECTORY / f"learned-{method}-{seed}.pt"
    trained = torch.load(checkpoint, weights_only=True)
    learner.load_state_dict(trained["learner"])
    expected_hash = previous.parameter_hash(agent)
    target = DIRECTORY / f"games-{method}-{seed}.pt"
    result = {"seed": seed, "method": method, "source": source_hashes(), "games": [],
              "parameter_hash": expected_hash, "checkpoint_sha256": previous.sha(checkpoint), "completed": False}
    if target.exists():
        result = torch.load(target, weights_only=True)
        if result["source"] != source_hashes() or result["parameter_hash"] != expected_hash:
            raise RuntimeError("game source or frozen model changed")
    for index in range(len(result["games"]), 24):
        opening = prior.opening(seed + 291000 + index // 2, 10, (4, 24, 60)[(index // 2) % 3])
        color = 1 if index % 2 == 0 else -1
        teacher = DefensiveGoTeacher() if index // 2 % 2 else HeuristicGoTeacher()
        board = GoBoard(10, komi=5.5)
        state = agent.core.zero_state(1, device=torch.device("cpu"), dtype=torch.float32)
        started = time.perf_counter()
        decisions = []
        with torch.no_grad():
            for action in opening:
                _, state = agent._transition(encoder.encode_board(board), state, slow_write=True)
                board.play(action)
            while not board.game_over:
                output, state = agent._transition(encoder.encode_board(board), state, slow_write=True)
                if board.to_play == color:
                    search = policy_value_search(agent, encoder, board, root_state=state,
                        root_log_probs=output.policy_logits, simulations=16, max_depth=8, value_scale=0.)
                    move = search["action"]
                    raw = int(output.policy_logits.argmax(1))
                    decisions.append({"ply": len(board.move_history), "action": move,
                                      "raw_legal": board.is_legal(raw), "network_evaluations": search["network_evaluations"]})
                else:
                    move = teacher.select_move(board)
                board.play(move)
        result["games"].append({"index": index, "opening": opening, "student_color": color,
            "moves": board.move_history, "decisions": decisions, "terminated": board.game_over,
            "consecutive_passes": board.consecutive_passes, "score": board.score(), "winner": board.winner(),
            "win": int(board.winner() == color), "margin": board.score()["margin_black"] * color,
            "seconds": time.perf_counter() - started})
        if previous.parameter_hash(agent) != expected_hash:
            raise RuntimeError("frozen game evaluation mutated the model")
        prior.save_torch(result, target)
        print(json.dumps({"seed": seed, "method": method, "game": index + 1,
                          "plies": len(board.move_history)}), flush=True)
    result["completed"] = True
    prior.save_torch(result, target)
    prior.save_json(result, ROOT / "analysis/results" / f"go14_games_{method}_seed{seed}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("develop", "prepare", "learn", "games"), default="develop")
    parser.add_argument("--seed", type=int, default=4903)
    parser.add_argument("--layout", choices=LAYOUTS, default="spatial")
    parser.add_argument("--method", choices=tuple(METHODS), default="spatial_adaptive")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development-prior", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output or ROOT / "analysis/results" / f"go14_development_{args.layout}.json"
    if args.stage == "develop":
        develop(args.seed, args.layout, output, args.development_prior)
    elif args.stage == "prepare":
        prepare(args.seed)
    elif args.stage == "learn":
        learn(args.seed, args.method)
    else:
        games(args.seed, args.method)
