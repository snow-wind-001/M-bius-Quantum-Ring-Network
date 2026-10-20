"""Independently verify the additional old-cadence control and paired comparisons."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.go12_verify import paired_interval, reference_score
from experiments.go12_efficiency import SEEDS, build, parameter_hash, mechanism_probe, prior
from experiments.go12_legacy_control import make_session, source_hashes
from mqr import GoBoard, HeuristicGoTeacher
from mqr.go import DefensiveGoTeacher
from mqr.go_search import policy_value_search


def verify(seed):
    torch.set_num_threads(1)
    data = json.loads((ROOT / f"analysis/results/go12_legacy_seed{seed}.json").read_text())
    args = argparse.Namespace(**data["config"])
    assert data["completed"] and data["source"] == source_hashes()
    assert (args.size, args.train_games, args.eval_games, args.simulations, args.search_depth,
            args.warmup_games, args.epochs) == (10, 16, 48, 16, 8, 24, 24)
    games, train = [], []
    for game in data["games"]:
        training = game["phase"].startswith("train_")
        (train if training else games).append(game)
        board = GoBoard(10, komi=5.5)
        teacher = DefensiveGoTeacher() if game["index"] // 2 % 2 else HeuristicGoTeacher()
        decisions = {d["ply"]: d for d in game["decisions"]}
        assert len(decisions) == len(game["decisions"])
        assert game["moves"][:len(game["opening"])] == game["opening"]
        for ply, action in enumerate(game["moves"]):
            assert not board.game_over
            if ply >= len(game["opening"]):
                if training or board.to_play == game["student_color"]:
                    decision = decisions[ply]
                    assert decision["action"] == action
                    assert decision["raw_legal"] == board.is_legal(decision["raw_action"])
                    assert 0 <= decision["search_evaluations"] <= 16
                else:
                    assert action == teacher.select_move(board)
            board.play(action)
        assert board.game_over and board.consecutive_passes == 2 and game["terminated"]
        assert board.score() == game["score"] and board.winner() == game["winner"]
        assert reference_score(board) == game["score"]["margin_black"]
        assert game["win"] == (board.winner() == game["student_color"])
        assert game["observations"] == len(game["moves"]) - len(game["opening"])
        expected = game["observations"] if training else 0
        assert game["terminal_labels"] == game["replayed_positions"] == expected
        assert game["ownership_positions"] == 0
        assert len(game["updates"]) == (2 * ((expected + 15) // 16) if training else 0)
        for update in game["updates"]:
            assert update["anchor_drift"]["max_margin_violation"] <= 1.1e-7
    assert len(train) == 16 and len(games) == 48
    assert sorted(g["index"] for g in train) == list(range(16))
    assert sorted(g["index"] for g in games) == list(range(48))
    check = data["evaluation_parameter_checks"]["final"]
    assert check["parameters_before"] == check["parameters_after"]
    assert check["version_before"] == check["version_after"]
    checkpoint = ROOT / data["checkpoint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == data["checkpoint_sha256"]
    agent, encoder = build(args, seed)
    session = make_session(args, agent, encoder)
    session.load_state_dict(torch.load(checkpoint, weights_only=True)["session"])
    assert session.replay_window == session.replay_credit_horizon == 16
    assert session.ownership_weight == 0 and len(session.behavior_memory.anchors) == 4
    assert session.behavior_memory.drift(agent) == data["final_anchor_drift"]
    base = torch.load(Path(args.directory) / f"base-{seed}.pt", weights_only=True)
    assert mechanism_probe(agent, base) == data["mechanisms"]["after"]
    before = parameter_hash(agent)
    reproduced = 0
    for game in games[:2]:
        session.reset_game()
        board = GoBoard(10, komi=5.5)
        for ply, action in enumerate(game["moves"]):
            prediction = session.observe(board, learn=False)
            if ply >= len(game["opening"]) and board.to_play == game["student_color"]:
                decision = policy_value_search(agent, encoder, board, prediction["state"],
                    prediction["policy_logits"], simulations=16, max_depth=8)
                assert decision["action"] == action
                reproduced += 1
            board.play(action)
    assert parameter_hash(agent) == before
    paired = {}
    for method in ("frozen", "baseline", "combined"):
        other = json.loads((ROOT / f"analysis/results/go12_{method}_seed{seed}.json").read_text())
        assert data["base_sha256"] == other["base_sha256"]
        assert data["initial_parameters"] == other["initial_parameters"]
        final = [g for g in other["games"] if g["phase"] == "final"]
        assert [(g["index"], g["opening"], g["student_color"]) for g in games] == [
            (g["index"], g["opening"], g["student_color"]) for g in final]
        paired[method] = sum(g["win"] for g in final) / 48
    return {"seed": seed, "games_replayed": 64, "decisions_reproduced": reproduced,
            "wins": sum(g["win"] for g in games), "win_rate": sum(g["win"] for g in games) / 48,
            "mean_margin": float(np.mean([g["margin"] for g in games])),
            "attempted_updates": sum(len(g["updates"]) for g in train),
            "accepted_updates": sum(u["did_update"] for g in train for u in g["updates"]),
            "training_positions": sum(g["observations"] for g in train),
            "training_search_evaluations": sum(g["search_evaluations"] for g in train),
            "paired_rates": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    paths = [ROOT / f"analysis/results/go12_{m}_seed{s}.json"
             for s in SEEDS for m in ("legacy", "frozen", "baseline", "combined")]
    while not all(p.exists() for p in paths):
        if not args.wait:
            parser.error("all paired results required")
        time.sleep(10)
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(verify, SEEDS))
    comparisons = {f"{method}_minus_legacy": paired_interval(
        [r["paired_rates"][method] - r["win_rate"] for r in rows])
        for method in ("frozen", "baseline", "combined")}
    result = {
        "verified": True, "scope": __doc__, "rows": rows, "comparisons": comparisons,
        "wins": sum(r["wins"] for r in rows), "games": 240, "complete_games_replayed": 320,
        "mean_margin": float(np.mean([r["mean_margin"] for r in rows])),
        "attempted_updates": sum(r["attempted_updates"] for r in rows),
        "accepted_updates": sum(r["accepted_updates"] for r in rows),
        "training_positions": sum(r["training_positions"] for r in rows),
        "source": source_hashes(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    prior.save_json(result, ROOT / "analysis/results/go12_legacy_summary.json")
    print(json.dumps({k: result[k] for k in ("verified", "wins", "games", "comparisons")}))


if __name__ == "__main__":
    main()
