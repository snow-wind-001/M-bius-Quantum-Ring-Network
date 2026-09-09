"""Validate/merge Go runs and recompute every reported paired statistic."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.orthogonal_go_online import paired_interval, summarize
from mqr import GoBoard


def validate(payload: dict) -> dict:
    config = payload["config"]
    runs = payload["runs"]
    assert payload["schema_version"] == 1
    assert [run["seed"] for run in runs] == config["seeds"], "incomplete or reordered seeds"
    assert len(set(config["seeds"])) == len(runs)
    assert payload["summary"] == summarize(runs), "stored summary differs from recomputation"
    assert payload["protocol"]["value_targets_used"] is False
    assert payload["protocol"]["predict_before_feedback"] is True
    for run in runs:
        assert list(run["methods"]) == config["methods"]
        before = next(iter(run["methods"].values()))["before"]
        feedback_counts = set()
        for method, item in run["methods"].items():
            assert item["before"] == before, "initial held-out predictions differ"
            assert item["max_unitary_error"] < 1e-4
            for stage in ("before", "after_a", "after_b"):
                for task in ("a", "b"):
                    metrics = item[stage][task]
                    assert all(math.isfinite(value) for value in metrics.values())
                    assert metrics["policy_nll"] >= 0.0
                    for key in ("raw_legality", "teacher_agreement", "masked_teacher_agreement", "illegal_probability_mass"):
                        assert 0 <= metrics[key] <= 1
            for game_stage in ("games_before", "games_after"):
                games = item[game_stage]
                assert games["training_updates"] == 0
                assert games["external_legal_mask"] is True
                assert len(games["games"]) == config["eval_games"]
                for game in games["games"]:
                    board = GoBoard(config["size"], komi=config["komi"])
                    for action in game["move_history"]:
                        assert not board.game_over and board.is_legal(action)
                        board.play(action)
                    assert game["moves"] == len(board.move_history) <= config["max_game_moves"]
                    assert game["terminated"] == board.game_over
                    assert game["terminated"] != game["truncated"]
                    assert game["win"] is None if game["truncated"] else game["win"] in (0., .5, 1.)
                    assert game["area_margin_at_stop"] == game["student_color"] * board.score()["margin_black"]
                    if game["terminated"]:
                        assert game["move_history"][-2:] == [config["size"]**2] * 2
                        assert game["win"] == (float(board.winner() == game["student_color"])
                                               + 0.5 * float(board.winner() == 0))
                    else:
                        assert len(board.move_history) == config["max_game_moves"]
                records = games["games"]
                wins = [game["win"] for game in records if game["terminated"]]
                assert games["terminal_win_rate"] == (sum(wins) / len(wins) if wins else None)
                assert games["termination_rate"] == sum(game["terminated"] for game in records) / len(records)
                for aggregate, field in (("mean_area_margin_at_stop", "area_margin_at_stop"), ("raw_legality", "raw_legality")):
                    assert math.isclose(games[aggregate], sum(game[field] for game in records) / len(records), abs_tol=1e-12)
            if method == "frozen":
                assert item["after_a"] == before and item["after_b"] == before
                assert item["games_after"] == item["games_before"]
                assert item["resources"]["trainable_parameters"] == 0
                continue
            counts = []
            for stage in ("train_a", "train_b"):
                training = item[stage]
                assert training["parameter_updates"] > 0
                assert training["feedback_positions"] == training["gradient_samples"]
                assert training["ogd_overlap_max"] < 1e-5
                counts.append((training["feedback_positions"], training["parameter_updates"]))
            feedback_counts.add(tuple(counts))
            if method.endswith("_ogd"):
                consolidation = item["consolidation"]
                assert 0 < consolidation["rank"] <= config["ogd_rank"]
                assert consolidation["orthogonality_error"] < 1e-5
                assert consolidation["score"] == "current_target_log_probability"
                assert item["resources"]["ogd_storage_bytes"] == (
                    consolidation["rank"] * item["resources"]["trainable_parameters"] * 4
                )
        assert len(feedback_counts) == 1, "methods received unequal main feedback or update counts"
    paired = {}
    methods = runs[0]["methods"]
    if "orthogonal" in methods and "orthogonal_ogd" in methods:
        values = []
        for run in runs:
            without, with_ogd = run["methods"]["orthogonal"], run["methods"]["orthogonal_ogd"]
            without_change = without["after_b"]["a"]["policy_nll"] - without["after_a"]["a"]["policy_nll"]
            with_change = with_ogd["after_b"]["a"]["policy_nll"] - with_ogd["after_a"]["a"]["policy_nll"]
            values.append(without_change - with_change)
        paired["ogd_reduction_in_old_task_nll_change"] = paired_interval(values)
    if "identity_ogd" in methods and "orthogonal_ogd" in methods:
        paired["orthogonal_vs_identity_ogd_b_nll_advantage"] = paired_interval([
            run["methods"]["identity_ogd"]["after_b"]["b"]["policy_nll"]
            - run["methods"]["orthogonal_ogd"]["after_b"]["b"]["policy_nll"]
            for run in runs
        ])
    return {
        "valid": True, "seeds": config["seeds"], "methods": config["methods"],
        "paired_diagnostics": paired,
        "source_files_match_current_workspace": all(
            (ROOT / name).is_file() and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
            for name, digest in payload["source_sha256"].items()
        ),
        "general_go_strength_established": False,
        "all_recorded_games_replayed": True,
        "confidence_intervals": "descriptive paired bootstrap across seeds; 10000 resamples",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--merge-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    parts = [json.loads(path.read_text()) for path in args.inputs]
    for part in parts:
        validate(part)
    if len(parts) == 1:
        merged = parts[0]
    else:
        merged = copy.deepcopy(parts[0])
        ignored_config = {"seeds", "output", "checkpoint_dir"}
        reference_config = {key: value for key, value in merged["config"].items() if key not in ignored_config}
        for part in parts[1:]:
            assert {key: value for key, value in part["config"].items() if key not in ignored_config} == reference_config
            assert part["protocol"] == merged["protocol"]
            assert part["source_sha256"] == merged["source_sha256"]
            merged["runs"].extend(part["runs"])
        merged["runs"].sort(key=lambda run: run["seed"])
        merged["config"]["seeds"] = [run["seed"] for run in merged["runs"]]
        merged["summary"] = summarize(merged["runs"])
    merged["artifact_sources"] = [str(path) for path in args.inputs]
    audit = validate(merged)
    if args.merge_output:
        args.merge_output.parent.mkdir(parents=True, exist_ok=True)
        merged["config"]["output"] = str(args.merge_output)
        args.merge_output.write_text(json.dumps(merged, indent=2, allow_nan=False) + "\n")
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
