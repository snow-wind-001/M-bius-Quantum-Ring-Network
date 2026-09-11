"""Independently replay complete games, verify the full matrix and summarize seeds."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go10_continual import (METHODS, SEEDS, build, make_session, probe, save_json,
                                        source_hashes, stages)
from mqr import GoBoard
from mqr.go import DefensiveGoTeacher, HeuristicGoTeacher
from mqr.go_search import policy_value_search


def mean(values):
    return float(np.mean(values)) if values else None


def replay_run(path):
    torch.set_num_threads(1)
    data = json.loads(Path(path).read_text())
    config = data["config"]
    assert data["completed"] and config["confirmation"]
    assert config["size"] == 10 and config["eval_games"] == 24 and config["train_games"] == 4
    assert data["source"] == source_hashes()
    assert not data.get("active_game")
    method, seed = config["method"], config["seed"]
    counts = {}
    for game in data["games"]:
        phase = game["phase"]
        counts[phase] = counts.get(phase, 0) + 1
        board = GoBoard(10, komi=5.5)
        decisions = {d["ply"]: d for d in game["decisions"]}
        assert len(decisions) == len(game["decisions"])
        assert game["moves"][:len(game["opening"])] == game["opening"]
        for ply, action in enumerate(game["moves"]):
            if ply >= len(game["opening"]) and board.to_play == game["student_color"]:
                decision = decisions[ply]
                assert action == decision["action"]
                assert decision["raw_legal"] == board.is_legal(decision["raw_action"])
                if action != board.pass_action:
                    _, captures, liberties = board._simulate_stone(action)
                    assert captures == decision["captures"]
                    assert (liberties == 1 and captures == 0) == decision["self_atari"]
                assert 0 <= decision["search_evaluations"] <= config["simulations"]
            assert not board.game_over, "stored moves continue after termination"
            board.play(action)
        assert board.game_over and board.consecutive_passes == 2 and game["terminated"]
        assert board.move_history[-2:] == [100, 100]
        assert board.score() == game["score"] and board.winner() == game["winner"]
        assert game["win"] == (board.winner() == game["student_color"])
        assert game["observations"] == len(game["moves"]) - len(game["opening"])
        expected_replay = game["observations"] if phase in ("a", "b", "return_a") and method not in ("legacy", "frozen") else 0
        assert game["terminal_labels"] == expected_replay == game["replayed_positions"]
        for update in game["updates"]:
            if "anchor_drift" in update:
                budget = 0.01 if method == "legacy" else 0.05
                assert update["anchor_drift"]["max_policy_kl"] <= budget + 1.1e-7
                assert update["anchor_drift"].get("max_margin_violation", 0) <= 1.1e-7
    expected_counts = {name: count for name, count, _, _ in stages(argparse.Namespace(**config), method)}
    assert counts == expected_counts
    for phase, count in counts.items():
        assert sorted(g["index"] for g in data["games"] if g["phase"] == phase) == list(range(count))
    for check in data["evaluation_parameter_checks"].values():
        assert check["parameters_before"] == check["parameters_after"]
        assert check["version_before"] == check["version_after"]
    if method == "frozen":
        assert data["parameter_version"] == 0
        assert data["probes"]["before"] == data["probes"]["after_return_a"]

    checkpoint_path = Path(data["checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == data["checkpoint_sha256"]
    work = torch.load(checkpoint_path, weights_only=True)
    args = argparse.Namespace(**config)
    agent, encoder = build(args, method, seed)
    session = make_session(args, agent, encoder, method)
    session.load_state_dict(work["session"])
    base_path = Path(config["directory"]) / f"base-{seed}.pt"
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == data["base_sha256"]
    base = torch.load(base_path, weights_only=True)
    assert probe(agent, base) == data["probes"]["after_search_eval"]
    if session.behavior_memory is not None:
        assert session.behavior_memory.drift(agent) == data["final_anchor_drift"]
    verified_decisions, verified_matches = 0, 0
    # Both colors from the first paired opening, using the restored final model.
    for game in [g for g in data["games"] if g["phase"] == "search_eval"][:2]:
        session.reset_game()
        board = GoBoard(10, komi=5.5)
        predictions = []
        for ply, action in enumerate(game["moves"]):
            result = session.observe(board, learn=False)
            if ply >= len(game["opening"]):
                predictions.append((board.to_play, float(result["output"].value[0])))
                if board.to_play == game["student_color"]:
                    search = policy_value_search(agent, encoder, board, result["state"], result["policy_logits"],
                                                 simulations=args.simulations, max_depth=args.search_depth)
                    assert search["action"] == action, (method, seed, game["index"], ply)
                    verified_decisions += 1
            board.play(action)
        mse = mean([(value - color * board.winner()) ** 2 for color, value in predictions])
        assert abs(mse - game["value_mse"]) < 1e-12
        verified_matches += 1
    return {"method": method, "seed": seed, "games": len(data["games"]),
            "matches_reproduced": verified_matches, "decisions_reproduced": verified_decisions}


def paired_interval(values):
    values = np.asarray(values)
    rng = np.random.default_rng(20260912)
    sampled = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "interval95": np.quantile(sampled, [0.025, 0.975]).tolist(),
            "per_seed": values.tolist(), "unit": "fraction", "method": "descriptive paired seed bootstrap"}


def summarize(runs):
    table = {}
    for method in METHODS:
        selected = [r for r in runs if r["config"]["method"] == method]
        groups = {}
        for phase in ("a", "b", "return_a", "search_eval", "greedy_eval"):
            games = [g for r in selected for g in r["games"] if g["phase"] == phase]
            if not games:
                continue
            per_seed = [[g for g in r["games"] if g["phase"] == phase] for r in selected]
            updates = [u for g in games for u in g["updates"]]
            guarded = [u for u in updates if "anchor_step_scale" in u]
            decisions = [d for g in games for d in g["decisions"]]
            placements = [d for d in decisions if d["action"] != 100]
            groups[phase] = {
                "games": len(games), "wins": sum(g["win"] for g in games),
                "win_rate": mean([mean([g["win"] for g in rows]) for rows in per_seed]),
                "wins_per_seed": [sum(g["win"] for g in rows) for rows in per_seed],
                "margin": mean([mean([g["margin"] for g in rows]) for rows in per_seed]),
                "value_mse": mean([mean([g["value_mse"] for g in rows]) for rows in per_seed]),
                "plies_min_mean_max": [min(len(g["moves"]) for g in games), mean([len(g["moves"]) for g in games]), max(len(g["moves"]) for g in games)],
                "observations": sum(g["observations"] for g in games),
                "raw_legal": mean([int(d["raw_legal"]) for d in decisions]),
                "pass_fraction": mean([int(d["action"] == 100) for d in decisions]),
                "noncapture_self_atari_fraction": mean([int(d["self_atari"]) for d in placements]),
                "captures": sum(d["captures"] for d in decisions),
                "nll": mean([d["nll"] for d in decisions]),
                "accepted_updates": sum(bool(u["did_update"]) for u in updates), "update_attempts": len(updates),
                "guard_backtrack_fraction": mean([int(u["anchor_backtracks"] > 0) for u in guarded]),
                "guard_step_scale": mean([u["anchor_step_scale"] for u in guarded]),
                "seconds": sum(g["seconds"] for g in games),
                "ms_per_observation": 1000 * sum(g["seconds"] for g in games) / sum(g["observations"] for g in games),
                "observation_peak_tensor_bytes": max(g["peak_online_tensor_bytes"] for g in games),
                "terminal_labels": sum(g["terminal_labels"] for g in games),
                "replayed_positions": sum(g["replayed_positions"] for g in games),
                "by_opening_length": {str(length): {"games": sum(len(g["opening"]) == length for g in games),
                    "wins": sum(g["win"] for g in games if len(g["opening"]) == length)}
                    for length in sorted({len(g["opening"]) for g in games})},
            }
            if phase in ("search_eval", "greedy_eval"):
                conflicts, pairs = 0, 0
                for seed_games in per_seed:
                    ordered = sorted(seed_games, key=lambda g: g["index"])
                    for left, right in zip(ordered[::2], ordered[1::2]):
                        assert left["opening"] == right["opening"]
                        assert left["student_color"] == -right["student_color"]
                        conflicts += int(left["winner"] != right["winner"])
                        pairs += 1
                groups[phase]["role_alias_diagnostic"] = {
                    "paired_openings": pairs, "opposite_winner_pairs": conflicts,
                    "empirical_root_value_mse_lower_bound_without_student_role": conflicts / pairs,
                    "scope": "post-hoc deterministic paired-root behavior-return labels; not a bound on minimax Go value"}
        stages_table = {}
        for stage in ("before", "after_a", "after_b", "after_return_a", "reset_history"):
            stages_table[stage] = {domain: {metric: mean([r["probes"][stage][domain][metric] for r in selected])
                                           for metric in ("nll", "agreement", "value_mse", "prior_mse")}
                                  for domain in ("a", "b")}
        table[method] = {"phases": groups, "probes": stages_table,
                         "total_parameters": [r["total_parameters"] for r in selected],
                         "trainable_parameters": [0 if method == "frozen" else r["trainable_parameters"] for r in selected],
                         "anchor_counts": [r["anchor_count"] for r in selected]}
    contrasts = {}
    for baseline in ("frozen", "legacy", "identity", "unprotected"):
        a = table["optimized"]["phases"]["search_eval"]["wins_per_seed"]
        b = table[baseline]["phases"]["search_eval"]["wins_per_seed"]
        contrasts["optimized_minus_" + baseline] = paired_interval([(x-y)/24 for x, y in zip(a,b)])
    for method in ("legacy", "optimized"):
        a = table[method]["phases"]["search_eval"]["wins_per_seed"]
        b = table[method]["phases"]["greedy_eval"]["wins_per_seed"]
        contrasts[method + "_search_minus_greedy"] = paired_interval([(x-y)/24 for x,y in zip(a,b)])
    return table, contrasts


def main(args):
    paths = [ROOT / f"analysis/results/go10_{method}_seed{seed}.json" for method in METHODS for seed in SEEDS]
    assert all(path.exists() for path in paths), "the complete five-seed matrix is required"
    runs = [json.loads(path.read_text()) for path in paths]
    base_games = 0
    for seed in SEEDS:
        selected = [r for r in runs if r["config"]["seed"] == seed]
        reference = next(r for r in selected if r["config"]["method"] == "frozen")
        optimized = next(r for r in selected if r["config"]["method"] == "optimized")
        base = torch.load(Path(reference["config"]["directory"]) / f"base-{seed}.pt", weights_only=True)
        for game in base["training_games"] + base["probe_games"]:
            board = GoBoard(10, komi=5.5)
            for action in game["moves"]:
                board.play(action)
            assert board.game_over and board.score() == game["score"] and board.winner() == game["winner"]
            base_games += 1
        assert reference["probes"]["before"] == optimized["probes"]["before"]
        reference_openings = [(g["index"], g["opening"], g["student_color"]) for g in reference["games"] if g["phase"] == "search_eval"]
        for run in selected:
            assert [(g["index"], g["opening"], g["student_color"]) for g in run["games"] if g["phase"] == "search_eval"] == reference_openings
    for path, digest in source_hashes().items():
        blob = subprocess.check_output(["git", "show", f"e0504d6:{path}"], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == digest, path
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        audits = list(pool.map(replay_run, map(str, paths)))
    tables, contrasts = summarize(runs)
    assert all(tables[method]["phases"]["search_eval"]["games"] > 100 for method in METHODS)
    save_json({"verified": True, "implementation_commit": "e0504d6", "seeds": list(SEEDS),
               "tables": tables, "contrasts": contrasts, "audits": audits,
               "total_replayed_games": sum(a["games"] for a in audits),
               "additional_base_games_replayed": base_games,
               "restored_checkpoints": len(audits),
               "reproduced_matches": sum(a["matches_reproduced"] for a in audits),
               "reproduced_decisions": sum(a["decisions_reproduced"] for a in audits),
               "source": source_hashes(), "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, args.output)
    print(json.dumps({"verified": True, "games": sum(a["games"] for a in audits),
                      "search_wins": {m: tables[m]["phases"]["search_eval"]["wins"] for m in METHODS}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", default=str(ROOT / "analysis/results/go10_summary.json"))
    args = parser.parse_args()
    main(args)
