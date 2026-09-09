"""Replay recorded live Go games and audit feedback windows and continuation.

This checks recorded actions, scores, and version counters against the rule
engine. It does not reconstruct neural predictions or certify teacher strength.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mqr import GoBoard


def validate(payload: dict, *, size: int, komi: float, window: int, max_moves: int) -> dict:
    """Recompute game evidence from the full recorded move history."""
    assert payload["schema_version"] == 1
    assert payload["mode"] == "live_supervised_online_go"
    assert payload["predict_before_teacher_feedback"] is True
    assert payload["external_legal_mask"] is True
    assert payload["value_reward_training"] is False
    assert payload["held_out_evaluation"] is False
    games = payload["games"]
    first_game = payload["initial_completed_games"]
    assert games and [game["game_index"] for game in games] == list(range(first_game, first_game + len(games)))
    next_version = None
    for game in games:
        board = GoBoard(size, komi=komi)
        index = game["game_index"]
        assert game["student_color"] == (1 if index % 2 == 0 else -1)
        rng = random.Random(payload["seed"] + 10000 + index // 2)
        for _ in range(2):
            board.play(rng.choice(board.legal_moves(include_pass=False)))
        assert board.move_history == game["move_history"][:2]
        events = game["events"]
        assert events and len(events) == len(game["move_history"]) - 2
        start_version = events[0]["prediction_version"]
        if next_version is not None:
            assert start_version == next_version, "parameter versions do not continue across games"
        for offset, event in enumerate(events):
            assert not board.game_over
            assert event["ply"] == len(board.move_history)
            assert event["student_turn"] == (board.to_play == game["student_color"])
            assert event["prediction_version"] == start_version + offset // window
            assert math.isfinite(event["prequential_nll"]) and event["prequential_nll"] >= 0
            assert event["raw_legal"] == board.is_legal(event["raw_action"])
            assert board.is_legal(event["teacher_action"])
            action = event["played_action"]
            if not event["student_turn"]:
                assert action == event["teacher_action"]
            elif event["raw_legal"]:
                assert action == event["raw_action"]
            assert action == game["move_history"][event["ply"]]
            assert board.is_legal(action), "record contains an illegal played action"
            board.play(action)
        assert game["terminated"] == board.game_over
        assert game["truncated"] == (not board.game_over)
        assert len(board.move_history) <= max_moves
        if not board.game_over:
            assert len(board.move_history) == max_moves
            assert game["win"] is None
        else:
            expected_win = float(board.winner() == game["student_color"]) + 0.5 * float(board.winner() == 0)
            assert game["win"] == expected_win
        assert game["area_margin_at_stop"] == game["student_color"] * board.score()["margin_black"]
        assert math.isclose(game["prequential_nll"], statistics.mean(event["prequential_nll"] for event in events), abs_tol=1e-12)
        assert math.isclose(game["raw_legality"], statistics.mean(event["raw_legal"] for event in events if event["student_turn"]), abs_tol=1e-12)
        assert game["parameter_updates"] == math.ceil(len(events) / window)
        next_version = start_version + game["parameter_updates"]
    return {
        "valid": True, "teacher": payload["teacher"], "games": len(games),
        "first_game": first_game, "next_game": first_game + len(games),
        "first_parameter_version": games[0]["events"][0]["prediction_version"],
        "next_parameter_version": next_version,
        "feedback_positions": sum(len(game["events"]) for game in games),
        "parameter_updates": sum(game["parameter_updates"] for game in games),
        "terminated_games": sum(game["terminated"] for game in games),
        "terminal_wins": sum(game["win"] == 1.0 for game in games),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--komi", type=float, default=2.5)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--max-moves", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert args.window > 0 and args.max_moves >= 4
    results, continuations, previous = {}, [], {}
    for path in args.inputs:
        payload = json.loads(path.read_text())
        result = validate(payload, size=args.size, komi=args.komi, window=args.window, max_moves=args.max_moves)
        key = (payload["method"], payload["teacher"], payload["seed"], payload["readout"])
        if key in previous and result["first_game"] == previous[key][1]["next_game"]:
            before_path, before = previous[key]
            assert result["first_parameter_version"] == before["next_parameter_version"]
            continuations.append({"from": str(before_path), "to": str(path), "valid": True})
        previous[key] = (path, result)
        results[str(path)] = {**result, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    audit = {
        "valid": True, "rules": {"size": args.size, "komi": args.komi, "window": args.window, "max_moves": args.max_moves},
        "records": results, "continuations": continuations,
        "neural_predictions_recomputed": False, "held_out_strength_claim": False,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
