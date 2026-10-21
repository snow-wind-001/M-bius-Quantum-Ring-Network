"""Independent Go14 replay, checkpoint/feedback audit, and seed-level summary."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import go14_readability as study
from experiments import go13_algorithm as old
from analysis.go12_verify import reference_score
from mqr import GoBoard, HeuristicGoTeacher
from mqr.go import DefensiveGoTeacher
from mqr.go_research import history_output, queried_logits
from mqr.go_search import policy_value_search

OUT = ROOT / "analysis/results"


def load(name):
    return json.loads((OUT / (name + ".json")).read_text())


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def assert_evaluation(actual, expected):
    """Require exact decisions/counts and float32-close recomputed scores."""
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        if name == "logits":
            torch.testing.assert_close(torch.tensor(value), torch.tensor(expected[name]), rtol=1e-6, atol=1e-6)
        elif name == "loss":
            assert math.isclose(value, expected[name], rel_tol=1e-6, abs_tol=1e-6)
        else:
            assert value == expected[name], (name, value, expected[name])


def verify_game(record, teacher=None, color=None):
    board = GoBoard(10, komi=5.5)
    decisions = {d["ply"]: d for d in record.get("decisions", [])}
    for index, move in enumerate(record["moves"]):
        assert not board.game_over and board.is_legal(move)
        if teacher is not None and index >= len(record["opening"]):
            if board.to_play == color:
                assert decisions[index]["action"] == move
                assert 0 <= decisions[index]["network_evaluations"] <= 16
            else:
                assert teacher.select_move(board) == move
        board.play(move)
    assert board.game_over and board.consecutive_passes == 2 and record["terminated"]
    assert record["moves"][-2:] == [100, 100]
    assert board.score() == record["score"] and board.winner() == record["winner"]
    assert reference_score(board) == record["score"]["margin_black"]
    if color is not None:
        assert record["win"] == int(board.winner() == color)
        assert record["margin"] == reference_score(board) * color


def audit_seed(seed):
    torch.set_num_threads(1)
    path = study.DIRECTORY / f"prepared-{seed}.pt"
    prepared = torch.load(path, weights_only=True)
    assert prepared["source"] == study.source_hashes()
    assert len(set(prepared["initial_hashes"].values())) == 1
    base_path = study.DIRECTORY / f"base-{seed}.pt"
    assert old.sha(base_path) == prepared["base_sha256"]
    base = torch.load(base_path, weights_only=True)
    assert len(base["training_games"]) == 24 and len(base["probe_games"]) == 4
    assert base["epochs"] == 24 and not bool(base["base"]["value.weight"].count_nonzero())
    for game in base["training_games"] + base["probe_games"]:
        verify_game(game)
    expected = {"a_train": 2048, "a_test": 256, "b_train": 1024, "b_test": 256}
    signatures, batches = {}, {}
    for split, records in prepared["data"].items():
        assert len(records) == expected[split]
        signatures[split] = set()
        for record in records:
            history = old.boards(record)
            assert len(history) == 2 and len(history[0].position_history) == 1
            for point, color in enumerate(history[0].board):
                if color:
                    assert history[0]._group_and_liberties(history[0].board, point)[1]
            assert history[-1].is_legal(record["query"]) == bool(record["label"])
            signatures[split].add(repr((history[-1].board, history[-1].to_play, record["query"])))
        batch = old.materialize(records)
        batches[split] = batch
        assert batch.labels.tolist() == [0., 1.] * (len(records) // 2)
        if split.startswith("a"):
            assert torch.equal(batch.features[-1, ::2], batch.features[-1, 1::2])
            assert not torch.equal(batch.features[0, ::2], batch.features[0, 1::2])
    assert not (signatures["a_train"] | signatures["b_train"]) & (signatures["a_test"] | signatures["b_test"])
    indices = prepared["anchor_indices"]
    assert len(indices) == 24 and len(set(indices)) == 24
    for layout in study.LAYOUTS:
        agent, _ = study.build(seed, layout)
        agent.load_state_dict(prepared["models"][layout])
        assert study.resources(agent) == prepared["resources"][layout]
        assert_evaluation(old.evaluate(agent, batches["a_test"]), prepared["curves"][layout][-1]["a"])
        assert len(prepared["updates"][layout]) == 128
        assert sum(u["feedback"] for u in prepared["updates"][layout]) == 2048
        for intervention in ("reset", "swap", "no_query"):
            assert_evaluation(old.evaluate(agent, batches["a_test"], intervention=intervention), prepared["interventions"][layout][intervention])
        assert prepared["interventions"][layout]["reset"]["accuracy"] == .5
        if layout in ("spatial", "similar_spatial", "kernel", "similar_kernel"):
            anchors = batches["a_train"].subset(indices)
            with torch.no_grad():
                margins = (2 * anchors.labels - 1) * queried_logits(history_output(agent, anchors.features), anchors.queries)
            assert bool((margins > .02).all())
        assert all(int(s["step"]) == 128 for s in prepared["optimizers"][layout]["state"].values())
        if layout == "spatial":
            selected = study.select_policy(agent, study.policy_records(base, training=True))
            assert set(selected) == set(prepared["policy"])
            for key in selected:
                torch.testing.assert_close(selected[key], prepared["policy"][key], rtol=0, atol=0)
    result = {"seed": seed, "verified": True, "base_complete_games": 28,
              "rule_histories": expected, "prepared_sha256": old.sha(path),
              "rule_anchors": len(indices), "policy_anchors": len(prepared["policy"].get("indices", []))}
    old.prior.save_json(result, OUT / f"go14_seed_audit_{seed}.json")
    return result


def audit_learning(key):
    method, seed = key
    torch.set_num_threads(1)
    run_path = OUT / f"go14_learning_{method}_seed{seed}.json"
    run = json.loads(run_path.read_text())
    assert run["source"] == study.source_hashes()
    prepared, agent, _, batches, learner, guard = study.restore(seed, method)
    for task in ("a", "b"):
        assert_evaluation(old.evaluate(agent, batches[task + "_test"]), run["before"][task])
    initial_hash = old.parameter_hash(agent)
    first = learner.step(batches["b_train"].subset(slice(0, 16)))
    first_errors = {}
    for name, value in first.items():
        reference = run["updates"][0][name]
        if name == "seconds":
            continue
        if isinstance(value, float):
            first_errors[name] = abs(value - reference)
            # Curvature divides a difference of float32 margins by a squared
            # step norm, amplifying last-bit forward variation. Decisions and
            # update norms retain the tighter checks below/above this branch.
            relative, absolute = (1e-4, 1e-5) if name == "curvature_estimate" else (1e-5, 2e-6)
            assert math.isclose(value, reference, rel_tol=relative, abs_tol=absolute), (name, value, reference)
        elif name == "pre_logits":
            first_errors[name] = float((torch.tensor(value) - torch.tensor(reference)).abs().max())
            torch.testing.assert_close(torch.tensor(value), torch.tensor(reference), rtol=1e-6, atol=1e-6)
        else:
            assert value == reference, (name, value, reference)
    assert len(run["updates"]) == 64 and sum(u["feedback"] for u in run["updates"]) == 1024
    protected = study.METHODS[method][1] not in ("free", "frozen")
    for index, update in enumerate(run["updates"]):
        logits = torch.tensor(update["pre_logits"])
        labels = batches["b_train"].labels[index * 16:index * 16 + 16]
        assert len(logits) == 16
        assert abs(float(F.binary_cross_entropy_with_logits(logits, labels)) - update["pre_loss"]) < 1e-6
        assert float(((logits >= 0) == labels.bool()).float().mean()) == update["pre_accuracy"]
        assert update["step_norm"] <= update["raw_step_norm"] + 1e-5
        assert update["raw_step_norm"] <= .5 + 1e-5 and update["backtracks"] <= 8
        if protected:
            assert update["actual_anchor_violations"] == 0
    path = study.DIRECTORY / f"learned-{method}-{seed}.pt"
    saved = torch.load(path, weights_only=True)
    assert saved["source"] == study.source_hashes() and saved["result"] == run
    learner.load_state_dict(saved["learner"])
    assert old.parameter_hash(agent) == run["parameter_hash"]
    if method.endswith("_frozen"):
        assert initial_hash == run["parameter_hash"]
    for task in ("a", "b"):
        assert_evaluation(old.evaluate(agent, batches[task + "_test"]), run["after"][task])
    policy = study.policy_probe(agent, prepared["policy_test"])
    reference = run["policy_after"]
    assert policy["actions"] == reference["actions"]
    assert policy["teacher_agreement"] == reference["teacher_agreement"]
    policy_error = float((torch.tensor(policy["probabilities"]) - torch.tensor(reference["probabilities"])).abs().max())
    torch.testing.assert_close(torch.tensor(policy["probabilities"]),
                               torch.tensor(reference["probabilities"]), rtol=1e-6, atol=1e-7)
    assert abs(policy["raw_pass_mean"] - reference["raw_pass_mean"]) < 1e-7
    anchor_error = 0.
    if guard is not None:
        with torch.no_grad():
            margins = guard.margins(agent)
        reference_margins = torch.tensor(run["anchor_margins"])
        anchor_error = float((margins - reference_margins).abs().max())
        torch.testing.assert_close(margins, reference_margins, rtol=1e-6, atol=2e-6)
        if protected:
            assert bool((margins >= guard.floors - 1e-5).all())
    result = {"method": method, "seed": seed, "verified": True,
              "checkpoint_sha256": old.sha(path), "learning_sha256": old.sha(run_path),
              "constraint_tensor_bytes": tensor_bytes(saved["learner"]["constraint"]),
              "optimizer_tensor_bytes": tensor_bytes(saved["learner"]["optimizer"]),
              "optimizer_steps": sorted(set(int(s["step"]) for s in saved["learner"]["optimizer"]["state"].values())),
              "policy_probability_max_error": policy_error,
              "policy_raw_pass_error": abs(policy["raw_pass_mean"] - reference["raw_pass_mean"]),
              "first_update_float_errors": first_errors,
              "anchor_margin_max_error": anchor_error,
              "verifier_sha256": old.sha(Path(__file__))}
    old.prior.save_json(result, OUT / f"go14_learning_audit_{method}_seed{seed}.json")
    return result


def audit_games(key):
    method, seed = key
    torch.set_num_threads(1)
    path = OUT / f"go14_games_{method}_seed{seed}.json"
    run = json.loads(path.read_text())
    assert run["completed"] and len(run["games"]) == 24
    assert run["source"] == study.source_hashes()
    _, agent, encoder, _, learner, _ = study.restore(seed, method)
    checkpoint = study.DIRECTORY / f"learned-{method}-{seed}.pt"
    assert old.sha(checkpoint) == run["checkpoint_sha256"]
    learner.load_state_dict(torch.load(checkpoint, weights_only=True)["learner"])
    assert old.parameter_hash(agent) == run["parameter_hash"]
    reproduced = 0
    for index, game in enumerate(run["games"]):
        assert game["index"] == index and game["student_color"] == (1 if index % 2 == 0 else -1)
        assert game["opening"] == study.prior.opening(seed + 291000 + index // 2, 10, (4, 24, 60)[(index // 2) % 3])
        assert game["moves"][:len(game["opening"])] == game["opening"]
        teacher = DefensiveGoTeacher() if index // 2 % 2 else HeuristicGoTeacher()
        verify_game(game, teacher, game["student_color"])
        if index == 0:
            board = GoBoard(10, komi=5.5)
            state = agent.core.zero_state(1, device=torch.device("cpu"), dtype=torch.float32)
            with torch.no_grad():
                for ply, action in enumerate(game["moves"]):
                    output, state = agent._transition(encoder.encode_board(board), state, slow_write=True)
                    if ply >= len(game["opening"]) and board.to_play == game["student_color"]:
                        search = policy_value_search(agent, encoder, board, root_state=state,
                            root_log_probs=output.policy_logits, simulations=16, max_depth=8, value_scale=0.)
                        assert search["action"] == action
                        reproduced += 1
                        if seed != study.SEEDS[0]:
                            break
                    board.play(action)
    assert old.parameter_hash(agent) == run["parameter_hash"]
    result = {"method": method, "seed": seed, "verified": True, "complete_games": 24,
              "full_matches_reproduced": int(seed == study.SEEDS[0]),
              "decisions_reproduced": reproduced, "games_sha256": old.sha(path),
              "verifier_sha256": old.sha(Path(__file__))}
    old.prior.save_json(result, OUT / f"go14_games_audit_{method}_seed{seed}.json")
    return result


def paired(values):
    values = np.asarray(values, dtype=float)
    means = [values[list(ix)].mean() for ix in itertools.product(range(len(values)), repeat=len(values))]
    return {"mean": float(values.mean()), "interval95": np.quantile(means, [.025, .975]).tolist(),
            "per_seed": values.tolist(), "method": "descriptive paired training-seed bootstrap; all 3125 resamples"}


def summarize():
    prepared = [load(f"go14_prepared_seed{s}") for s in study.SEEDS]
    layouts = {}
    for layout in study.LAYOUTS:
        accuracy = [p["curves"][layout][-1]["a"]["accuracy"] for p in prepared]
        pairs = [p["curves"][layout][-1]["a"]["both_correct"] / 128 for p in prepared]
        layouts[layout] = {"accuracy": paired(accuracy), "pair_accuracy": paired(pairs),
            "learned_gate": bool(np.mean(accuracy) >= .8 and np.mean(pairs) >= .6),
            "resources": prepared[0]["resources"][layout],
            "a_training_seconds": sum(p["seconds"][layout] for p in prepared),
            "curves": [{"feedback": p0["feedback"], "accuracy": float(np.mean([p["curves"][layout][i]["a"]["accuracy"] for p in prepared]))}
                       for i, p0 in enumerate(prepared[0]["curves"][layout])],
            "interventions": {name: float(np.mean([p["interventions"][layout][name]["accuracy"] for p in prepared]))
                              for name in ("reset", "swap", "no_query")}}
    table, all_runs = {}, {}
    for method in study.METHODS:
        runs = [load(f"go14_learning_{method}_seed{s}") for s in study.SEEDS]
        audits = [load(f"go14_learning_audit_{method}_seed{s}") for s in study.SEEDS]
        all_runs[method] = runs
        mean = lambda fn: float(np.mean([fn(r) for r in runs]))
        updates = [u for r in runs for u in r["updates"]]
        policy = []
        for r in runs:
            before = np.asarray(r["policy_before"]["probabilities"])
            after = np.asarray(r["policy_after"]["probabilities"])
            kl = float(np.sum(before * (np.log(np.maximum(before, 1e-300)) - np.log(np.maximum(after, 1e-300))), axis=1).mean())
            policy.append({"seed": r["seed"], "kl_before_after": kl,
                "greedy_changes": sum(a != b for a, b in zip(r["policy_before"]["actions"], r["policy_after"]["actions"]))})
        row = {"a_before": mean(lambda r: r["before"]["a"]["accuracy"]),
            "a_after": mean(lambda r: r["after"]["a"]["accuracy"]),
            "b_before": mean(lambda r: r["before"]["b"]["accuracy"]),
            "b_after": mean(lambda r: r["after"]["b"]["accuracy"]),
            "a_loss_change": mean(lambda r: r["after"]["a"]["loss"] - r["before"]["a"]["loss"]),
            "b_loss_gain": mean(lambda r: r["before"]["b"]["loss"] - r["after"]["b"]["loss"]),
            "feedback": sum(u["feedback"] for u in updates),
            "attempts": 0 if method.endswith("_frozen") else len(updates),
            "accepted": sum(u["accepted"] for u in updates),
            "active_updates": sum(u["active"] > 0 for u in updates),
            "raw_violation_updates": sum(u["raw_anchor_violations"] > 0 for u in updates),
            "backtracked_updates": sum(u["backtracks"] > 0 for u in updates),
            "backtracks": sum(u["backtracks"] for u in updates),
            "projection_trials": sum(u["projection_trials"] for u in updates),
            "projection_failures": sum(not u.get("projection_converged", True) for u in updates),
            "optimizer_rollbacks": sum(u["optimizer_rolled_back"] for u in updates),
            "anchor_violations": sum(r["anchor_violations"] or 0 for r in runs),
            "rule_anchors_per_seed": [r["rule_anchors"] for r in runs],
            "policy_anchors_per_seed": [r["policy_anchors"] for r in runs],
            "mean_raw_norm": float(np.mean([u["raw_step_norm"] for u in updates])),
            "mean_step_norm": float(np.mean([u["step_norm"] for u in updates])),
            "median_step_norm": float(np.median([u["step_norm"] for u in updates])),
            "minimum_positive_step_norm": min((u["step_norm"] for u in updates if u["step_norm"] > 0), default=0),
            "seconds": sum(r["seconds"] for r in runs),
            "constraint_tensor_bytes": [r["constraint_tensor_bytes"] for r in audits],
            "optimizer_tensor_bytes": audits[0]["optimizer_tensor_bytes"],
            "policy_greedy_changes": sum(p["greedy_changes"] for p in policy), "policy_probes": 120,
            "policy_kl": float(np.mean([p["kl_before_after"] for p in policy])),
            "policy_teacher_before": mean(lambda r: r["policy_before"]["teacher_agreement"]),
            "policy_teacher_after": mean(lambda r: r["policy_after"]["teacher_agreement"]),
            "raw_pass_before": mean(lambda r: r["policy_before"]["raw_pass_mean"]),
            "raw_pass_after": mean(lambda r: r["policy_after"]["raw_pass_mean"]),
            "per_seed": [{"seed": r["seed"], "a_before": r["before"]["a"]["accuracy"],
                          "a_after": r["after"]["a"]["accuracy"], "b_before": r["before"]["b"]["accuracy"],
                          "b_after": r["after"]["b"]["accuracy"], "policy": p}
                         for r, p in zip(runs, policy)]}
        row["joint_gate"] = bool(row["a_after"] >= .8 and row["a_before"] - row["a_after"] <= .05
                                 and row["b_after"] >= .9 and row["b_after"] - row["b_before"] >= .1)
        table[method] = row
    comparisons = {}
    for left, right in (("spatial_free", "shared_free"), ("spatial_current", "spatial_free"),
                        ("spatial_adaptive", "spatial_free"),
                        ("spatial_adaptive", "spatial_current"), ("spatial_rollback", "spatial_adaptive"),
                        ("spatial_aligned", "spatial_adaptive"), ("spatial_aligned", "spatial_rule24"),
                        ("spatial_current", "similar_spatial_current"), ("spatial_adaptive", "similar_spatial_adaptive"),
                        ("kernel_current", "similar_kernel_current"), ("kernel_adaptive", "similar_kernel_adaptive")):
        comparisons[left + "_minus_" + right] = {
            task: paired([a["after"][task]["accuracy"] - b["after"][task]["accuracy"]
                          for a, b in zip(all_runs[left], all_runs[right])]) for task in ("a", "b")}
    games = {}
    for method in study.GAME_METHODS:
        records = [load(f"go14_games_{method}_seed{s}") for s in study.SEEDS]
        rows = [g for r in records for g in r["games"]]
        games[method] = {"games": len(rows), "wins": sum(g["win"] for g in rows),
            "win_rate": float(np.mean([g["win"] for g in rows])),
            "mean_margin": float(np.mean([g["margin"] for g in rows])),
            "seconds": sum(g["seconds"] for g in rows), "plies": sum(len(g["moves"]) for g in rows),
            "network_evaluations": sum(d["network_evaluations"] for g in rows for d in g["decisions"]),
            "raw_legal_rate": float(np.mean([d["raw_legal"] for g in rows for d in g["decisions"]])),
            "per_seed_wins": [sum(g["win"] for g in r["games"]) for r in records]}
    base_audits = [load(f"go14_seed_audit_{s}") for s in study.SEEDS]
    game_audits = [load(f"go14_games_audit_{m}_seed{s}") for m in study.GAME_METHODS for s in study.SEEDS]
    result = {"verified": True, "source": study.source_hashes(), "verifier_sha256": old.sha(Path(__file__)),
        "layouts": layouts, "table": table, "comparisons": comparisons, "games": games,
        "base_audits": base_audits, "game_audits": game_audits,
        "complete_games": sum(a["base_complete_games"] for a in base_audits) + sum(a["complete_games"] for a in game_audits),
        "full_matches_reproduced": sum(a["full_matches_reproduced"] for a in game_audits),
        "decisions_reproduced": sum(a["decisions_reproduced"] for a in game_audits)}
    assert result["complete_games"] == 860
    old.prior.save_json(result, OUT / "go14_summary.json")
    print(json.dumps({"verified": True, "complete_games": result["complete_games"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("seeds", "learning", "games", "summary"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    study.check_lock()
    if args.stage == "summary":
        summarize()
    else:
        function = {"seeds": audit_seed, "learning": audit_learning, "games": audit_games}[args.stage]
        items = list(study.SEEDS) if args.stage == "seeds" else [
            (method, seed) for method in (study.METHODS if args.stage == "learning" else study.GAME_METHODS)
            for seed in study.SEEDS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(function, item) for item in items]):
                print(json.dumps(future.result()), flush=True)
