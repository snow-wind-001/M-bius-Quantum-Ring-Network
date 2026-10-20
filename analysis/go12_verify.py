"""Replay Go12 evidence and summarize paired training seeds, without training."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go12_efficiency import (
    METHODS, SEEDS, build, make_session, mechanism_probe, parameter_hash, schedule, source_hashes, prior,
)
from mqr import GoBoard, HeuristicGoTeacher
from mqr.go import DefensiveGoTeacher
from mqr.go_search import policy_value_search


def reference_score(board):
    """Independent flood fill, preserving the pre-Go12 scoring contract."""
    remaining = {i for i, color in enumerate(board.board) if color == 0}
    black, white = board.board.count(1), board.board.count(-1)
    while remaining:
        queue = [remaining.pop()]
        count, boundary = 0, set()
        while queue:
            point = queue.pop()
            count += 1
            row, col = divmod(point, board.size)
            for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= r < board.size and 0 <= c < board.size:
                    neighbor = r * board.size + c
                    if board.board[neighbor]:
                        boundary.add(board.board[neighbor])
                    elif neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
        if boundary == {1}:
            black += count
        elif boundary == {-1}:
            white += count
    return black - white - board.komi


def restore(run):
    args = argparse.Namespace(**run["config"])
    path = ROOT / run["checkpoint"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == run["checkpoint_sha256"]
    work = torch.load(path, weights_only=True)
    assert work["completed"] and work["source"] == run["source"]
    agent, encoder = build(args, args.seed)
    session = make_session(args, agent, encoder)
    session.load_state_dict(work["session"])
    return args, session


def verify_run(path):
    torch.set_num_threads(1)
    run = json.loads(Path(path).read_text())
    args = argparse.Namespace(**run["config"])
    assert run["completed"] and run["source"] == source_hashes()
    assert (args.size, args.train_games, args.eval_games, args.curve_games,
            args.simulations, args.search_depth, args.warmup_games, args.epochs) == (10, 16, 48, 24, 16, 8, 24, 24)
    phases = {}
    training_indices = []
    for game in run["games"]:
        phase = game["phase"]
        training = phase.startswith("train_")
        phases[phase] = phases.get(phase, 0) + 1
        if training:
            training_indices.append(game["index"])
        decisions = {d["ply"]: d for d in game["decisions"]}
        assert len(decisions) == len(game["decisions"])
        teacher = DefensiveGoTeacher() if game["index"] // 2 % 2 else HeuristicGoTeacher()
        board = GoBoard(10, komi=5.5)
        assert game["moves"][:len(game["opening"])] == game["opening"]
        for ply, action in enumerate(game["moves"]):
            assert not board.game_over
            if ply >= len(game["opening"]):
                if training or board.to_play == game["student_color"]:
                    decision = decisions[ply]
                    assert decision["action"] == action
                    assert decision["raw_legal"] == board.is_legal(decision["raw_action"])
                    assert 0 <= decision["search_evaluations"] <= args.simulations
                    if action != board.pass_action:
                        _, captures, liberties = board._simulate_stone(action)
                        assert captures == decision["captures"]
                        assert (liberties == 1 and captures == 0) == decision["self_atari"]
                else:
                    assert action == teacher.select_move(board)
            board.play(action)
        assert board.game_over and board.consecutive_passes == 2 and game["terminated"]
        assert board.move_history[-2:] == [100, 100]
        assert board.score() == game["score"] and board.winner() == game["winner"]
        assert reference_score(board) == game["score"]["margin_black"]
        assert sum(board.area_ownership()) - board.komi == reference_score(board)
        assert game["win"] == (board.winner() == game["student_color"])
        assert game["margin"] == reference_score(board) * game["student_color"]
        assert game["observations"] == len(game["moves"]) - len(game["opening"])
        expected = game["observations"] if training else 0
        assert game["terminal_labels"] == game["replayed_positions"] == expected
        assert game["ownership_positions"] == (expected if args.method in ("ownership", "combined", "unprotected") else 0)
        if training:
            assert len(decisions) == expected
            assert game["value_semantics"] == "shared_policy"
            # One feedback update per online 16 and terminal replay 64 positions.
            assert len(game["updates"]) == (expected + 15) // 16 + (expected + 63) // 64
        else:
            assert not game["updates"]
        for update in game["updates"]:
            if "anchor_drift" in update:
                assert update["anchor_drift"].get("max_margin_violation", 0) <= 1.1e-7
            if "constraint_projection" in update:
                projection = update["constraint_projection"]
                assert projection["norm_retention"] <= 1.00001
                if projection["converged"]:
                    assert projection.get("kkt_error", 0) <= 1e-8
                    assert projection.get("primal_error", 0) <= 1e-8
                else:
                    assert not update["did_update"]
    assert phases == {name: count for name, count, _ in schedule(args)}
    assert sorted(training_indices) == ([] if args.method == "frozen" else list(range(16)))
    for check in run["evaluation_parameter_checks"].values():
        assert check["parameters_before"] == check["parameters_after"]
        assert check["version_before"] == check["version_after"]
    _, session = restore(run)
    agent = session.agent
    before = parameter_hash(agent)
    base_path = Path(args.directory) / f"base-{args.seed}.pt"
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == run["base_sha256"]
    data = torch.load(base_path, weights_only=True)
    for name, parameter in agent.spatial_skip_heads.named_parameters():
        if name.startswith(("local.", "local_second.")):
            torch.testing.assert_close(parameter, data["base"][name], rtol=0, atol=0)
    assert not bool(data["base"]["value.weight"].count_nonzero())
    assert not bool(data["base"]["value.bias"].count_nonzero())
    assert mechanism_probe(agent, data) == run["mechanisms"]["after"]
    if args.method == "frozen":
        assert before == run["initial_parameters"] and run["parameter_version"] == 0
    if session.behavior_memory is not None:
        assert len(session.behavior_memory.anchors) == 4
        assert session.behavior_memory.drift(agent) == run["final_anchor_drift"]
    decisions_reproduced = 0
    for game in [g for g in run["games"] if g["phase"] == "final"][:2]:
        session.reset_game()
        board = GoBoard(10, komi=5.5)
        predictions, ownership_predictions = [], []
        for ply, action in enumerate(game["moves"]):
            result = session.observe(board, learn=False)
            if ply >= len(game["opening"]):
                predictions.append((board.to_play, float(result["output"].value[0])))
                ownership_predictions.append((board.to_play, result["output"].ownership[0].numpy()))
                if board.to_play == game["student_color"]:
                    search = policy_value_search(agent, session.encoder, board, result["state"],
                        result["policy_logits"], simulations=args.simulations, max_depth=args.search_depth)
                    assert search["action"] == action
                    decisions_reproduced += 1
            board.play(action)
        mse = float(np.mean([(v - c * board.winner()) ** 2 for c, v in predictions]))
        assert abs(mse - game["value_mse"]) < 1e-12
        owned = np.array(board.area_ownership())
        mse = float(np.mean([(v.astype(float) - c * owned) ** 2 for c, v in ownership_predictions]))
        assert abs(mse - game["ownership_mse"]) < 1e-12
    assert parameter_hash(agent) == before
    return {"method": args.method, "seed": args.seed, "games_replayed": len(run["games"]),
            "matches_reproduced": 2, "decisions_reproduced": decisions_reproduced}


def verify_cached(path):
    """An audit is reusable only for identical result bytes and verifier source."""
    path = Path(path)
    receipt = path.with_name(path.stem.replace("go12_", "go12_audit_", 1) + ".json")
    result_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    verifier_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if receipt.exists():
        cached = json.loads(receipt.read_text())
        if cached["result_sha256"] == result_hash and cached["verifier_sha256"] == verifier_hash:
            return cached["audit"]
    audit = verify_run(path)
    prior.save_json({"verified": True, "result_sha256": result_hash,
                     "verifier_sha256": verifier_hash, "audit": audit}, receipt)
    return audit


def verify_base_and_references(seed):
    torch.set_num_threads(1)
    runs = {m: json.loads((ROOT / f"analysis/results/go12_{m}_seed{seed}.json").read_text()) for m in METHODS}
    assert len({r["initial_parameters"] for r in runs.values()}) == 1
    assert len({r["base_sha256"] for r in runs.values()}) == 1
    args = argparse.Namespace(**runs["combined"]["config"])
    data = torch.load(Path(args.directory) / f"base-{seed}.pt", weights_only=True)
    for game in data["training_games"] + data["probe_games"]:
        board = GoBoard(10, komi=5.5)
        for action in game["moves"]:
            assert not board.game_over
            board.play(action)
        assert board.game_over and board.winner() == game["winner"] and board.score() == game["score"]
        assert reference_score(board) == game["score"]["margin_black"]
    agent, encoder = build(args, seed)
    agent.spatial_skip_heads.load_state_dict(data["base"])
    session = make_session(args, agent, encoder)
    prior.initialize_memory(session, data)
    memory = session.behavior_memory
    results = {}
    for method, run in runs.items():
        _, target = restore(run)
        rows = []
        with torch.no_grad():
            for anchor in memory.anchors:
                scores = memory._output(target.agent, anchor)
                alternatives = scores.clone()
                alternatives[anchor["action"]] = -torch.inf
                margin = float(scores[anchor["action"]] - alternatives.max())
                rows.append({"margin": margin, "margin_floor": anchor["margin_floor"],
                             "violation": max(0, anchor["margin_floor"] - margin),
                             "greedy_changed": int(scores.argmax()) != anchor["action"]})
        results[method] = {"rows": rows, "drift": memory.drift(target.agent)}
    return {"seed": seed, "base_games_replayed": 28, "references": results}


def mean(values):
    return float(np.mean(values)) if len(values) else None


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    samples = np.asarray(list(itertools.product(range(len(values)), repeat=len(values))))
    interval = np.quantile(values[samples].mean(1), [0.025, 0.975])
    return {"mean": float(values.mean()), "interval95": interval.tolist(), "per_seed": values.tolist(),
            "method": "descriptive paired training-seed percentile bootstrap, all 3125 resamples"}


def summarize(runs, references):
    indexed = {(r["config"]["method"], r["config"]["seed"]): r for r in runs}
    table, curves, seed_rates = {}, {}, {}
    for method in METHODS:
        selected = [indexed[method, seed] for seed in SEEDS]
        games = [g for r in selected for g in r["games"] if g["phase"] == "final"]
        training = [g for r in selected for g in r["games"] if g["phase"].startswith("train_")]
        updates = [u for g in training for u in g["updates"]]
        decisions = [d for g in games for d in g["decisions"]]
        probes = [p for r in selected for p in r["mechanisms"]["after"]]
        anchor_rows = [a for ref in references for a in ref["references"][method]["rows"]]
        seed_rates[method] = [mean([g["win"] for g in r["games"] if g["phase"] == "final"]) for r in selected]
        table[method] = {
            "wins": sum(g["win"] for g in games), "games": len(games),
            "per_seed_wins": [sum(g["win"] for g in r["games"] if g["phase"] == "final") for r in selected],
            "win_rate": mean([g["win"] for g in games]), "mean_margin": mean([g["margin"] for g in games]),
            "black_wins": sum(g["win"] for g in games if g["student_color"] == 1),
            "white_wins": sum(g["win"] for g in games if g["student_color"] == -1),
            "losses_by_at_least_50": sum(g["margin"] <= -50 for g in games),
            "raw_legal_rate": mean([d["raw_legal"] for d in decisions]),
            "self_atari_rate": mean([d["self_atari"] for d in decisions]),
            "training_games": len(training), "training_positions": sum(g["observations"] for g in training),
            "training_search_evaluations": sum(g["search_evaluations"] for g in training),
            "training_seconds": sum(g["seconds"] for g in training),
            "final_eval_seconds": sum(g["seconds"] for g in games),
            "attempted_updates": len(updates), "accepted_updates": sum(u["did_update"] for u in updates),
            "backtracked_updates": sum(u.get("anchor_backtracks", 0) > 0 for u in updates),
            "prequential_training_value_mse": mean([g["value_mse"] for g in training]),
            "prequential_training_ownership_mse": mean([g["ownership_mse"] for g in training]),
            "ownership_supervised_positions": sum(g["ownership_positions"] for g in training),
            "total_parameters": selected[0]["total_parameters"],
            "trainable_parameters": selected[0]["trainable_parameters"],
            "peak_online_tensor_bytes": max(g["peak_online_tensor_bytes"] for r in selected for g in r["games"]),
            "probe_ownership_mse": mean([p["ownership_mse"] for p in probes]),
            "probe_ownership_zero_mse": mean([p["ownership_zero_mse"] for p in probes]),
            "probe_nll": mean([p["nll"] for p in probes]),
            "probe_teacher_agreement": mean([p["agreement"] for p in probes]),
            "probe_query_action_changes": sum(p["action"] != p["no_query_action"] for p in probes),
            "probe_reset_action_changes": sum(p["action"] != p["reset_action"] for p in probes),
            "probe_query_rms": mean([p["query_rms"] for p in probes]),
            "probe_spatial_rms": mean([p["spatial_rms"] for p in probes]),
            "reference_anchor_violations": sum(a["violation"] > 1e-7 for a in anchor_rows),
            "reference_greedy_changes": sum(a["greedy_changed"] for a in anchor_rows),
            "max_reference_margin_violation": max(a["violation"] for a in anchor_rows),
            "max_reference_policy_kl": max(r["references"][method]["drift"]["max_policy_kl"] for r in references),
        }
    for method in ("baseline", "combined"):
        rows = []
        for budget in (0, 4, 8, 16):
            games = []
            for seed in SEEDS:
                run = indexed["frozen" if budget == 0 else method, seed]
                phase = "final" if budget in (0, 16) else f"curve_{budget}"
                games += [g for g in run["games"] if g["phase"] == phase and g["index"] < 24]
            rows.append({"online_games": budget, "evaluation_games": len(games),
                         "wins": sum(g["win"] for g in games), "win_rate": mean([g["win"] for g in games])})
        curves[method] = rows
    comparisons = {}
    for a, b in (("baseline", "frozen"), ("ownership", "baseline"), ("credit", "baseline"),
                 ("combined", "baseline"), ("combined", "ownership"), ("combined", "credit"),
                 ("combined", "unprotected"), ("combined", "frozen")):
        comparisons[f"{a}_minus_{b}"] = paired_interval(np.array(seed_rates[a]) - seed_rates[b])
    comparisons["ownership_credit_interaction"] = paired_interval(
        np.array(seed_rates["combined"]) - seed_rates["ownership"] - np.array(seed_rates["credit"]) + seed_rates["baseline"])
    return {"table": table, "learning_curves_matched_120_games": curves, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wait", action="store_true", help="audit formal results as running jobs finish")
    args = parser.parse_args()
    paths = [ROOT / f"analysis/results/go12_{m}_seed{s}.json" for s in SEEDS for m in METHODS]
    assert args.wait or all(p.exists() for p in paths), "full formal matrix required"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        remaining, pending, verified = list(paths), {}, []
        while remaining or pending:
            for path in list(remaining):
                if len(pending) >= args.workers:
                    break
                if path.exists():
                    pending[pool.submit(verify_cached, path)] = path
                    remaining.remove(path)
            if not pending:
                time.sleep(10)
                continue
            done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                verified.append(result)
                del pending[future]
                print(json.dumps({"audit": result, "runs_remaining": len(remaining) + len(pending)}), flush=True)
        references = list(pool.map(verify_base_and_references, SEEDS))
    runs = [json.loads(p.read_text()) for p in paths]
    summary = summarize(runs, references)
    summary.update(verified=True, runs_verified=verified, reference_audits=references,
                   source=source_hashes(), verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   total_complete_games=sum(r["games_replayed"] for r in verified) + 140)
    prior.save_json(summary, ROOT / "analysis/results/go12_summary.json")
    print(json.dumps({"verified": True, "complete_games": summary["total_complete_games"],
                      "wins": {m: row["wins"] for m, row in summary["table"].items()}}))


if __name__ == "__main__":
    main()
