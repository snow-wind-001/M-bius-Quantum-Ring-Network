"""Verify complete five-seed constraint evidence and produce a compact summary."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.go_constraint_research import HISTORY_METHODS, ONLINE_METHODS
from mqr import GoBoard
from mqr.go_history_tasks import generate_reachable_history_pairs
from mqr.go_memory import GoHistoryEncoder, GoObservationMemory


SEEDS = {401, 409, 419, 431, 443}
GIT_SOURCE_CACHE = {}


def load_group(pattern, study, methods):
    paths = sorted((ROOT / "analysis/results").glob(pattern))
    if not paths:
        raise AssertionError(f"missing {study} evidence")
    rows, config, artifacts = [], None, {}
    for path in paths:
        raw = json.loads(path.read_text())
        current = raw["config"]
        assert current["study"] == study
        for name, expected in raw["source_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
            key = (raw["base_git_revision"], name)
            if key not in GIT_SOURCE_CACHE:
                blob = subprocess.check_output(["git", "show", f"{key[0]}:{name}"], cwd=ROOT)
                GIT_SOURCE_CACHE[key] = hashlib.sha256(blob).hexdigest()
            assert GIT_SOURCE_CACHE[key] == expected, f"recorded commit differs: {name}"
        if config is None:
            config = current
        else:
            for name, value in config.items():
                if name not in ("seeds", "output", "checkpoint_dir", "methods"):
                    assert current[name] == value, f"mixed protocol: {name}"
        rows.extend(raw["runs"])
        artifacts[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    keys = [(row["seed"], row["method"]) for row in rows]
    assert len(keys) == len(set(keys)), "duplicate method/seed"
    assert set(keys) == {(seed, method) for seed in SEEDS for method in methods}, "incomplete formal matrix"
    assert config["size"] == 5 and config["ring_dim"] == config["latent_dim"] == 16
    return config, rows, artifacts


def describe(values):
    return {"mean": float(np.mean(values)), "min": float(np.min(values)),
            "max": float(np.max(values)), "per_seed": list(map(float, values))}


def paired_difference(left, right):
    values = np.array(left) - np.array(right)
    rng = np.random.default_rng(20260911)
    means = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "per_seed": values.tolist(),
            "descriptive_paired_bootstrap_95": np.quantile(means, [0.025, 0.975]).tolist()}


def verify_online(config, rows):
    for key, expected in (("train_games", 8), ("test_games", 8), ("match_games", 8),
                          ("warmup_games", 24), ("pretrain_epochs", 24), ("rank", 8),
                          ("anchors", 8), ("window", 8), ("credit", 8)):
        assert config[key] == expected, key
    indexed = {(row["seed"], row["method"]): row for row in rows}
    game_count, summaries = 0, {}
    for seed in sorted(SEEDS):
        reference = indexed[seed, "autograd1"]
        for method in ONLINE_METHODS:
            row = indexed[seed, method]
            assert row["before"] == reference["before"], "initial policies differ"
            assert row["after_a"] == reference["after_a"], "phase A is not matched"
            assert row["train_a"]["games"] == reference["train_a"]["games"]
            assert row["trainable_parameters"] == reference["trainable_parameters"]
            assert row["total_parameters"] == reference["total_parameters"]
            assert row["max_adjoint_norm_error"] < 5e-5
            if method == "transport1_guard":
                assert row["train_b"]["max_anchor_kl"] <= config["anchor_kl"] + 1e-7
            for phase in ("train_a", "train_b", "matches"):
                phase_row = row[phase]
                assert len(phase_row["games"]) == 8
                wins = terminated = 0
                for index, game in enumerate(phase_row["games"]):
                    board = GoBoard(config["size"], komi=2.5)
                    for action in game["moves"]:
                        assert not board.game_over
                        assert board.is_legal(action), (method, seed, phase, action)
                        board.play(action)
                    assert board.game_over == game["terminated"]
                    assert (board.winner() if board.game_over else None) == game["winner"]
                    won = bool(board.game_over and board.winner() == game["student_color"])
                    assert won == game["student_win"]
                    wins += won
                    terminated += board.game_over
                    game_count += 1
                    if phase == "matches":
                        assert game["moves"][:4] == reference["matches"]["games"][index]["moves"][:4]
                        if index % 2:
                            assert game["moves"][:4] == phase_row["games"][index - 1]["moves"][:4]
                assert wins == phase_row["wins"]
                assert terminated == phase_row["terminated_games"]
    for method in ONLINE_METHODS:
        selected = [indexed[seed, method] for seed in sorted(SEEDS)]
        summaries[method] = {
            "b_gain": describe([r["after_a"]["b"]["joint_nll"] - r["after_b"]["b"]["joint_nll"] for r in selected]),
            "a_change_during_b": describe([r["after_b"]["a"]["joint_nll"] - r["after_a"]["a"]["joint_nll"] for r in selected]),
            "a_gain_from_initial": describe([r["before"]["a"]["joint_nll"] - r["after_b"]["a"]["joint_nll"] for r in selected]),
            "wins": sum(r["matches"]["wins"] for r in selected),
            "terminated_matches": sum(r["matches"]["terminated_games"] for r in selected),
            "max_anchor_kl": max(r["train_b"]["max_anchor_kl"] for r in selected),
            "mean_refresh_count": float(np.mean([r["refresh_count"] for r in selected])),
            "mean_refresh_seconds": float(np.mean([r["refresh_seconds"] for r in selected])),
            "mean_total_seconds": float(np.mean([r["seconds"] for r in selected])),
            "max_persistent_online_tensor_bytes": max(r["train_b"]["peak_persistent_online_tensor_bytes"] for r in selected),
            "max_constraint_state_trace_bytes": max(r["peak_constraint_state_trace_bytes"] for r in selected),
            "backtracked_updates": sum(r["train_b"]["backtracked_updates"] for r in selected),
            "mean_update_norm": float(np.mean([r["train_b"]["mean_update_norm"] for r in selected])),
            "mean_ogd_retained_norm": float(np.mean([r["train_b"]["mean_ogd_retained_norm"] for r in selected])),
        }
    contrasts = {
        f"{left}_minus_{right}": paired_difference(summaries[left]["b_gain"]["per_seed"], summaries[right]["b_gain"]["per_seed"])
        for left, right in (("autograd1", "autograd8"), ("transport1", "autograd1"),
                            ("transport1_guard", "transport1"), ("transport1", "norm1"))
    }
    backend_agreement = {
        phase: sum(indexed[s, "autograd1"][phase]["games"] == indexed[s, "transport1"][phase]["games"] for s in SEEDS)
        for phase in ("train_a", "train_b", "matches")
    }
    return {"methods": summaries, "b_gain_contrasts": contrasts, "replayed_games": game_count,
            "backend_identical_game_batches_out_of_5": backend_agreement}


def verify_history(config, rows):
    assert config["history_epochs"] == 64 and config["history_moves"] == 12
    assert config["history_train_pairs"] == 96 and config["history_test_pairs"] == 64
    by_seed = {}
    for seed in sorted(SEEDS):
        groups = [generate_reachable_history_pairs(count, seed=seed + offset, size=5, moves=12)
                  for count, offset in ((96, 11000), (64, 12000))]
        hashes = [{hashlib.sha256(repr(pair.moves).encode()).hexdigest() for pair in pairs} for pairs in groups]
        assert not hashes[0] & hashes[1]
        by_seed[seed] = [hashlib.sha256("".join(sorted(items)).encode()).hexdigest() for items in hashes]
    summaries = {}
    for row in rows:
        assert row["updates"] == 64 * 12
        assert [row["train_trajectory_hash"], row["test_trajectory_hash"]] == by_seed[row["seed"]]
        assert row["after"]["reset"]["pair_choice_accuracy"] == 0.5
        assert abs(row["after"]["history"]["pair_choice_accuracy"] + row["after"]["swap"]["pair_choice_accuracy"] - 1) < 1e-7
        assert row["actual_credit_steps"] == (5 if row["method"] == "autograd_short" else 13)
    for method in HISTORY_METHODS:
        selected = sorted((row for row in rows if row["method"] == method), key=lambda row: row["seed"])
        summaries[method] = {
            "accuracy": describe([r["after"]["history"]["pair_choice_accuracy"] for r in selected]),
            "point_nll": describe([r["after"]["history"]["point_nll"] for r in selected]),
            "mean_seconds": float(np.mean([r["seconds"] for r in selected])),
            "max_state_trace_bytes": max(r["peak_state_trace_bytes"] for r in selected),
            "actual_terminal_credit_steps": selected[0]["actual_credit_steps"],
        }
    return {"methods": summaries,
            "full_minus_short_accuracy": paired_difference(summaries["transport_full"]["accuracy"]["per_seed"],
                                                           summaries["autograd_short"]["accuracy"]["per_seed"])}


def verify_mechanisms(config, rows):
    assert config["lengths"] == [8, 16, 32, 64] and config["repeats"] == 3
    summaries = {}
    for run in rows:
        assert max(run["proof"].values()) < 1e-11
        assert [r["length"] for r in run["rows"]] == config["lengths"]
        encoder = GoHistoryEncoder(5, history_slots=config["slots"] if run["method"] == "conditional_external" else 0)
        for row in run["rows"]:
            board, memory, features = GoBoard(5, komi=2.5), GoObservationMemory(encoder), []
            for index in range(row["length"]):
                assert not board.game_over
                features.append(encoder.with_history(board, memory.slots))
                memory.write(board)
                if index < len(row["moves"]):
                    assert board.is_legal(row["moves"][index])
                    board.play(row["moves"][index])
            assert len(row["moves"]) == row["length"] - 1
            assert hashlib.sha256(torch.cat(features).numpy().tobytes()).hexdigest() == row["features_sha256"]
            assert row["max_objective_error"] == 0 and row["max_gradient_error"] < 3e-6
            assert row["relative_gradient_error"] < 3e-5
            for block in row["block_gradient_errors"].values():
                assert block["max_absolute_error"] < 3e-6
                if block["reference_norm"] > 1e-7:
                    assert block["relative_error"] < 1e-4
    for method in ("orthogonal", "conditional", "conditional_external"):
        summaries[method] = {}
        for length in config["lengths"]:
            selected = [row for run in rows if run["method"] == method for row in run["rows"] if row["length"] == length]
            summaries[method][str(length)] = {
                "max_gradient_error": max(r["max_gradient_error"] for r in selected),
                "max_relative_gradient_error": max(r["relative_gradient_error"] for r in selected),
                "autograd_saved_bytes": float(np.mean([r["costs"]["autograd"]["peak_saved_tensor_storage_bytes"] for r in selected])),
                "transport_saved_bytes": float(np.mean([r["costs"]["transport"]["peak_saved_tensor_storage_bytes"] for r in selected])),
                "transport_state_trace_bytes": selected[0]["costs"]["transport"]["state_trace_bytes"],
                "autograd_seconds": float(np.mean([r["costs"]["autograd"]["median_seconds"] for r in selected])),
                "transport_seconds": float(np.mean([r["costs"]["transport"]["median_seconds"] for r in selected])),
            }
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="analysis/results/go_constraint_summary.json")
    options = parser.parse_args()
    torch.set_num_threads(1)
    result = {"version": 1, "seeds": sorted(SEEDS), "source_hashes_match": True,
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "artifacts": {}}
    for study, pattern, methods, verify in (
        ("online", "go_constraint_online_seed*.json", ONLINE_METHODS, verify_online),
        ("history", "go_constraint_history_seed*.json", HISTORY_METHODS, verify_history),
        ("mechanisms", "go_constraint_mechanisms_5seed.json",
         ("orthogonal", "conditional", "conditional_external"), verify_mechanisms),
    ):
        config, rows, artifacts = load_group(pattern, study, methods)
        result[study] = verify(config, rows)
        result["artifacts"].update(artifacts)
    destination = ROOT / options.output
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verified": True, "replayed_online_games": result["online"]["replayed_games"],
                      "history_runs": 15, "mechanism_prefixes": 60}, indent=2))


if __name__ == "__main__":
    main()
