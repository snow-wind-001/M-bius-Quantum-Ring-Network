"""Check provenance, legal game reconstruction, causal updates and full ablations."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.go_policy_research import CONFIRMATION_SEEDS, METHODS, sources
from mqr import GoBoard


def describe(values):
    return {"mean": float(np.mean(values)), "per_seed": [float(value) for value in values],
            "min": float(min(values)), "max": float(max(values))}


def paired(left, right):
    delta = np.array(left) - np.array(right)
    rng = np.random.default_rng(20260911)
    resampled = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
    return {"mean": float(delta.mean()), "per_seed": delta.tolist(),
            "descriptive_seed_bootstrap_95": np.quantile(resampled, (0.025, 0.975)).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="analysis/results/go_policy_seed*.json")
    parser.add_argument("--output", default="analysis/results/go_policy_summary.json")
    args = parser.parse_args()
    paths = sorted(ROOT.glob(args.pattern))
    assert paths, "no confirmation artifacts"
    rows, config, hashes, revisions = [], None, {}, set()
    committed_sources = {}
    for path in paths:
        raw = json.loads(path.read_text())
        assert raw["completed"] and raw["config"]["confirmation"]
        assert raw["source_sha256"] == sources(), "current implementation differs from experiment"
        commit = raw["source_commit"]
        revisions.add(commit)
        for name, digest in raw["source_sha256"].items():
            if (commit, name) not in committed_sources:
                contents = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
                committed_sources[commit, name] = hashlib.sha256(contents).hexdigest()
            assert committed_sources[commit, name] == digest, f"uncommitted experiment source: {name}"
        if config is None:
            config = raw["config"]
        else:
            for key, value in config.items():
                if key not in ("seeds", "output"):
                    assert raw["config"][key] == value, f"mixed configuration: {key}"
        rows.extend(raw["runs"])
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(revisions) == 1, "confirmation source commit differs"
    keys = [(row["seed"], row["method"]) for row in rows]
    assert len(keys) == len(set(keys))
    assert set(keys) == {(seed, method) for seed in CONFIRMATION_SEEDS for method in METHODS}
    for key, expected in (
        ("size", 5), ("ring_dim", 16), ("latent_dim", 16), ("channels", 12),
        ("warmup_games", 24), ("pretrain_epochs", 24), ("train_games", 8), ("test_games", 8),
        ("match_games", 16), ("window", 8), ("credit", 8), ("rank", 8), ("anchors", 8),
        ("refresh", 1), ("max_game_moves", 100), ("simulations", 16), ("search_depth", 8),
        ("search_methods", "legacy,reliable"),
    ):
        assert config[key] == expected, key
    indexed = {(row["seed"], row["method"]): row for row in rows}
    reconstructed = 0
    for seed in CONFIRMATION_SEEDS:
        reference = indexed[seed, "query"]
        for method in METHODS:
            row = indexed[seed, method]
            if method != "legacy":
                assert row["before"] == reference["before"], "new model initial policies differ"
            if method == "reliable":
                assert row["train_a"]["games"] == indexed[seed, "outcome"]["train_a"]["games"]
                assert row["after_a"] == indexed[seed, "outcome"]["after_a"]
                assert row["anchor_selection"]["protected"] == row["anchor_selection"]["teacher_agreements"]
            assert row["max_adjoint_norm_error"] < 5e-5
            assert row["anchor_drift"]["max_policy_kl"] <= config["anchor_kl"] + 1e-7
            phases = ("train_a", "train_b", "matches") + (("search_matches",) if "search_matches" in row else ())
            for phase in phases:
                part = row[phase]
                training = phase.startswith("train")
                expected_count = 8 if training else 16
                assert len(part["games"]) == expected_count
                wins, terminal, observations, terminal_positions = 0, 0, 0, 0
                for index, game in enumerate(part["games"]):
                    board = GoBoard(5, komi=2.5)
                    for action in game["moves"]:
                        assert not board.game_over and board.is_legal(action), (seed, method, phase, action)
                        board.play(action)
                    assert board.game_over == game["terminated"]
                    assert (board.winner() if board.game_over else None) == game["winner"]
                    assert game["student_color"] == (1 if index % 2 == 0 else -1)
                    won = bool(board.game_over and board.winner() == game["student_color"])
                    assert game["student_win"] == won
                    if board.game_over:
                        assert game["student_margin"] == board.score()["margin_black"] * game["student_color"]
                        assert game["prequential_value_mse"] is not None
                        terminal_positions += len(game["moves"])
                    else:
                        assert len(game["moves"]) == config["max_game_moves"]
                        assert game["student_margin"] is None and game["prequential_value_mse"] is None
                    wins += int(won)
                    terminal += int(board.game_over)
                    observations += len(game["moves"])
                    reconstructed += 1
                    if not training:
                        assert game["moves"][:4] == indexed[seed, "legacy"]["matches"]["games"][index]["moves"][:4]
                        if index % 2:
                            assert game["moves"][:4] == part["games"][index - 1]["moves"][:4]
                assert wins == part["wins"] and terminal == part["terminated_games"]
                assert observations == part["observations"]
                assert part["feedback_positions"] == (observations if training else 0)
                replaying = training and method in ("replay", "outcome", "reliable")
                assert part["replayed_positions"] == (terminal_positions if replaying else 0)
                assert part["terminal_value_labels"] == (
                    terminal_positions if training and method in ("outcome", "reliable") else 0
                )
                assert part["max_anchor_kl"] <= config["anchor_kl"] + 1e-7
                if not training:
                    assert not part["updates"]
                for update in part["updates"]:
                    assert update["prediction_before_update"]
                    assert all(age == 0 for age in update.get("parameter_staleness", [0]))
                    assert update["max_unitary_error"] < 5e-5
                if phase == "search_matches":
                    assert part["search_simulations"] == 16
                    assert part["search_network_evaluations"] <= 16 * part["student_decisions"]
            path = ROOT / row["checkpoint"]
            assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == row["checkpoint_sha256"]
    summary = {
        "verified": True, "source_commit": next(iter(revisions)), "seeds": list(CONFIRMATION_SEEDS),
        "configuration": config, "artifacts_sha256": hashes, "reconstructed_legal_games": reconstructed,
        "methods": {}, "paired_differences": {},
    }
    for method in METHODS:
        group = [indexed[seed, method] for seed in CONFIRMATION_SEEDS]
        matches = [row["matches"] for row in group]
        metrics = {
            "b_nll": [row["after_b"]["b"]["full"]["joint_nll"] for row in group],
            "b_teacher_agreement": [row["after_b"]["b"]["full"]["teacher_agreement"] for row in group],
            "b_query_removal_action_rate": [row["after_b"]["b"]["no_query"]["action_differs_from_full"] for row in group],
            "b_query_centered_logit_rms": [row["after_b"]["b"]["query_centered_logit_rms"] for row in group],
            "b_reset_history_nll": [row["after_b"]["b"]["reset_history"]["joint_nll"] for row in group],
            "b_reset_history_agreement": [row["after_b"]["b"]["reset_history"]["teacher_agreement"] for row in group],
            "a_after_a_nll": [row["after_a"]["a"]["full"]["joint_nll"] for row in group],
            "a_after_b_nll": [row["after_b"]["a"]["full"]["joint_nll"] for row in group],
            "wins": [part["wins"] for part in matches],
            "terminal_student_margin": [part["mean_terminal_student_margin"] for part in matches],
            "value_mse": [part["game_mean_value_mse"] for part in matches],
            "anchor_kl": [row["anchor_drift"]["max_policy_kl"] for row in group],
            "anchor_count": [row["anchor_selection"]["protected"] for row in group],
            "online_ms": [row["train_b"]["mean_observe_feedback_ms"] for row in group],
            "inference_ms": [part["mean_observe_feedback_ms"] for part in matches],
            "online_tensor_bytes": [max(row[p]["peak_persistent_online_tensor_bytes"] for p in ("train_a", "train_b")) for row in group],
            "train_feedback_positions": [sum(row[p]["feedback_positions"] for p in ("train_a", "train_b")) for row in group],
            "replayed_positions": [sum(row[p]["replayed_positions"] for p in ("train_a", "train_b")) for row in group],
            "updates": [sum(len(row[p]["updates"]) for p in ("train_a", "train_b")) for row in group],
            "seconds": [row["seconds"] for row in group],
        }
        entry = {key: describe(values) for key, values in metrics.items()}
        entry.update(
            wins_total=sum(part["wins"] for part in matches), games_total=80,
            terminated_total=sum(part["terminated_games"] for part in matches),
            query_action_changes=sum(part["query_action_changes"] for part in matches),
            student_decisions=sum(part["student_decisions"] for part in matches),
            student_tactics={key: sum(part["student_tactics"][key] for part in matches)
                             for key in matches[0]["student_tactics"]},
            teacher_on_same_positions_tactics={
                key: sum(part["teacher_on_same_positions_tactics"][key] for part in matches)
                for key in matches[0]["teacher_on_same_positions_tactics"]
            },
            trainable_parameters=group[0]["trainable_parameters"],
            total_parameters=group[0]["total_parameters"],
        )
        if "search_matches" in group[0]:
            search = [row["search_matches"] for row in group]
            entry["search"] = {
                "wins": describe([part["wins"] for part in search]),
                "wins_total": sum(part["wins"] for part in search), "games_total": 80,
                "terminated_total": sum(part["terminated_games"] for part in search),
                "terminal_student_margin": describe([part["mean_terminal_student_margin"] for part in search]),
                "inference_ms": describe([part["mean_observe_feedback_ms"] for part in search]),
                "network_evaluations": sum(part["search_network_evaluations"] for part in search),
            }
        summary["methods"][method] = entry
    for left, right in zip(METHODS[1:], METHODS[:-1]):
        summary["paired_differences"][f"{left}_minus_{right}"] = {
            metric: paired(summary["methods"][left][metric]["per_seed"],
                           summary["methods"][right][metric]["per_seed"])
            for metric in ("wins", "b_nll", "b_teacher_agreement", "value_mse")
        }
    summary["paired_differences"]["reliable_minus_legacy"] = {
        metric: paired(summary["methods"]["reliable"][metric]["per_seed"],
                       summary["methods"]["legacy"][metric]["per_seed"])
        for metric in ("wins", "b_nll", "b_teacher_agreement", "value_mse")
    }
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"verified": True, "legal_games": reconstructed,
                      "wins": {key: value["wins_total"] for key, value in summary["methods"].items()}}))


if __name__ == "__main__":
    main()
