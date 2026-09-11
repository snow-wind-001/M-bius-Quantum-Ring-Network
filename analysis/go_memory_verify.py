"""Audit v2 source hashes, matched-history interventions and actual Go game records."""

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mqr import GoBoard
from mqr.go_history_tasks import generate_reachable_history_pairs


def interval(values):
    values = np.array(values, dtype=float)
    rng = np.random.default_rng(821)
    samples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "seed_values": values.tolist(),
            "descriptive_seed_bootstrap_95": np.quantile(samples, [0.025, 0.975]).tolist()}


def check_file(path):
    data = json.loads(path.read_text())
    expected = {(int(seed), method) for seed in data["config"]["seeds"].split(',')
                for method in data["config"]["methods"].split(',')}
    actual = [(run["seed"], run["method"]) for run in data["runs"]]
    assert len(actual) == len(set(actual)) and set(actual) == expected, "incomplete or duplicate run matrix"
    def verify_sources(record):
        for name, expected_hash in record["source_sha256"].items():
            revision = record.get("source_revisions", {}).get(name)
            source = ((ROOT / name).read_bytes() if revision is None else
                      subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT))
            assert hashlib.sha256(source).hexdigest() == expected_hash, f"source mismatch: {name}"
            if name.startswith("mqr/"):
                current = (ROOT / name).read_bytes()
                if hashlib.sha256(current).hexdigest() != expected_hash:
                    assert name == "mqr/agent.py", f"unexpected current core difference: {name}"
                    # The only post-run API repair delegates preview to the
                    # same transition used by commit. Benchmark paths never
                    # call preview. Verify all other AST nodes are identical.
                    def without_preview(raw):
                        tree = ast.parse(raw)
                        for node in tree.body:
                            if isinstance(node, ast.ClassDef) and node.name == "TemporalUtilityMQRAgent":
                                node.body = [item for item in node.body if not
                                             (isinstance(item, ast.FunctionDef) and item.name == "preview_step")]
                        return ast.dump(tree, include_attributes=False)
                    assert without_preview(source) == without_preview(current), "non-preview code changed after these runs"
    verify_sources(data)
    for run in data["runs"]:
        if "source_sha256" in run:
            verify_sources(run)
    return data


def history_summary(data):
    config = data["config"]
    for seed in map(int, config["seeds"].split(',')):
        hashes = []
        for offset, count in ((11000, config["history_train_pairs"]), (12000, config["history_test_pairs"])):
            pairs = generate_reachable_history_pairs(count, seed=seed + offset, size=config["size"], moves=config["history_moves"])
            values = {hashlib.sha256(repr(pair.moves).encode()).hexdigest() for pair in pairs}
            assert len(values) == count
            hashes.append(values)
        assert not hashes[0] & hashes[1], "history trajectory leakage"
        for run in (row for row in data["runs"] if row["seed"] == seed):
            for name, values in zip(("train", "test"), hashes):
                assert run[f"{name}_trajectory_hash"] == hashlib.sha256(''.join(sorted(values)).encode()).hexdigest()
            after = run["after"]
            assert after["reset"]["pair_choice_accuracy"] == 0.5, "reset baseline must be at the pair ceiling"
            assert abs(after["history"]["pair_choice_accuracy"] + after["swap"]["pair_choice_accuracy"] - 1) < 1e-6
            assert after["reset"]["paired_state_l2_distance"] == 0
            assert after["reset"]["paired_point_logit_l2_distance"] < 1e-4
    groups = defaultdict(list)
    for run in data["runs"]:
        groups[run["method"]].append(run)
    summary = {}
    for method, rows in groups.items():
        summary[method] = {"accuracy": interval([r["after"]["history"]["pair_choice_accuracy"] for r in rows]),
                           "point_nll": interval([r["after"]["history"]["point_nll"] for r in rows]),
                           "swap_accuracy": interval([r["after"]["swap"]["pair_choice_accuracy"] for r in rows]),
                           "reset_accuracy": interval([r["after"]["reset"]["pair_choice_accuracy"] for r in rows]),
                           "trainable_parameters": rows[0]["trainable_parameters"],
                           "ring_state_bytes": rows[0]["ring_state_bytes"],
                           "external_memory_bytes": rows[0]["observation_memory_bytes"],
                           "optimizer_tensor_bytes": rows[0]["optimizer_tensor_bytes"]}
    paired = {}
    for left, right in (("orthogonal", "orthogonal_short"), ("orthogonal", "identity"),
                        ("orthogonal", "shift"), ("conditional", "orthogonal"),
                        ("conditional_external", "conditional"), ("conditional_external", "stateless_external")):
        l = {r["seed"]: r["after"]["history"]["pair_choice_accuracy"] for r in groups[left]}
        paired[f"{left}_minus_{right}_accuracy"] = interval([l[r["seed"]] - r["after"]["history"]["pair_choice_accuracy"] for r in groups[right]])
    return {"methods": summary, "paired_comparisons": paired, "runs": len(data["runs"]),
            "task": "reachable history recall, not Go strength", "test_pairs_per_seed": config["history_test_pairs"]}


def online_summary(data):
    config = data["config"]
    groups = defaultdict(list)
    game_count = 0
    openings = {}
    before = {}
    for run in data["runs"]:
        groups[run["method"]].append(run)
        seed = run["seed"]
        values = [run["before"][p]["joint_nll"] for p in ("a", "b")]
        if seed in before:
            assert np.allclose(before[seed], values, atol=1e-7, rtol=0), "initial policies do not match"
        before[seed] = values
        for phase in ("train_a", "train_b", "matches"):
            record = run[phase]
            wins = 0
            for game in record["games"]:
                board = GoBoard(config["size"], komi=2.5)
                for action in game["moves"]:
                    board.play(action)
                assert board.game_over == game["terminated"]
                assert game["winner"] == (board.winner() if board.game_over else None)
                win = bool(board.game_over and board.winner() == game["student_color"])
                assert game["student_win"] == win
                wins += win
                game_count += 1
            assert record["wins"] == wins
            if record["feedback_positions"]:
                assert record["feedback_positions"] == sum(len(g["moves"]) for g in record["games"])
                assert record["updates"] == sum((len(g["moves"]) + config["window"] - 1) // config["window"] for g in record["games"])
        match = run["matches"]
        assert match["paired_color_openings"] and match["random_opening_moves"] == 4
        prefix = [game["moves"][:4] for game in match["games"]]
        for index in range(0, len(prefix) - 1, 2):
            assert prefix[index] == prefix[index + 1], "colors must share an opening"
        if seed in openings:
            assert openings[seed] == prefix, "methods must share evaluation openings"
        openings[seed] = prefix
        if "guard" in run["method"]:
            assert run["train_b"]["max_anchor_kl"] <= config["anchor_kl"] + 1e-7
            assert run["anchor_drift"]["max_policy_kl"] <= config["anchor_kl"] + 1e-7
    summary = {}
    for method, rows in groups.items():
        summary[method] = {
            "a_gain_after_a": interval([r["before"]["a"]["joint_nll"] - r["after_a"]["a"]["joint_nll"] for r in rows]),
            "b_gain_during_b": interval([r["after_a"]["b"]["joint_nll"] - r["after_b"]["b"]["joint_nll"] for r in rows]),
            "a_forgetting_during_b": interval([r["after_b"]["a"]["joint_nll"] - r["after_a"]["a"]["joint_nll"] for r in rows]),
            "a_final_gain_from_initial": interval([r["before"]["a"]["joint_nll"] - r["after_b"]["a"]["joint_nll"] for r in rows]),
            "b_point_gain_during_b": interval([r["after_a"]["b"]["nonpass_point_nll"] - r["after_b"]["b"]["nonpass_point_nll"] for r in rows]),
            "b_pass_gain_during_b": interval([r["after_a"]["b"]["pass_nll"] - r["after_b"]["b"]["pass_nll"] for r in rows]),
            "wins": sum(r["matches"]["wins"] for r in rows),
            "terminated_matches": sum(r["matches"]["terminated_games"] for r in rows),
            "matches": sum(len(r["matches"]["games"]) for r in rows),
            "peak_persistent_online_tensor_bytes": max(r["train_b"]["peak_persistent_online_tensor_bytes"] for r in rows),
            "mean_online_step_ms": float(np.mean([r["train_b"]["mean_observe_feedback_ms"] for r in rows])),
            "trainable_parameters": rows[0]["trainable_parameters"],
            "total_parameters": rows[0]["total_parameters"],
            "max_anchor_kl": max((r.get("anchor_drift", {}).get("max_policy_kl", 0) for r in rows)),
            "anchor_backtracked_updates": sum(r["train_b"]["backtracked_updates"] for r in rows),
            "anchor_bytes": max((r.get("anchor_bytes", 0) for r in rows)),
            "ogd_bytes": max((r.get("ogd_bytes", 0) for r in rows)),
        }
    return {"methods": summary, "runs": len(data["runs"]), "legally_replayed_games": game_count,
            "initial_policy_match": True, "paired_opening_match": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default="analysis/results/go_memory_history_5seed.json")
    parser.add_argument("--online", default="analysis/results/go_memory_online_5seed.json")
    parser.add_argument("--output", default="analysis/results/go_memory_summary.json")
    args = parser.parse_args()
    history = check_file(ROOT / args.history)
    online = check_file(ROOT / args.online)
    result = {"version": 2, "source_hashes_match": True,
              "source_verification": "recorded Git sources; current core AST identical outside the separately tested preview dispatcher",
              "history": history_summary(history), "online": online_summary(online),
              "artifact_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (args.history, args.online)},
              "inference_limits": ["five seeds; descriptive bootstrap intervals", "heuristic teacher",
                                   "no claim of equal parameter, compute or total memory budgets",
                                   "no terminal-reward RL; online Go uses post-prediction teacher feedback",
                                   "history data uses 64 explicitly reported replay epochs"]}
    (ROOT / args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"source_hashes_match": True, "history_runs": result["history"]["runs"],
                      "online_runs": result["online"]["runs"], "legally_replayed_games": result["online"]["legally_replayed_games"]}, indent=2))


if __name__ == "__main__":
    main()
