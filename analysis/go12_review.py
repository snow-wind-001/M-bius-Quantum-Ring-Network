"""Post-hoc review of saved training exposure and constraint activity.

Counts immediate single-stone-ko prohibitions only, not every long-cycle
superko case. No new game, label fitting, or parameter update is performed.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go12_efficiency import METHODS, SEEDS, prior
from mqr import GoBoard


def exposure(game):
    board = GoBoard(10, komi=5.5)
    previous_capture = None
    prohibited, candidates, positions = 0, 0, 0
    start = len(game.get("opening", []))
    for ply, action in enumerate(game["moves"]):
        if ply >= start:
            positions += 1
            if previous_capture is not None:
                candidates += 1
                try:
                    stones, _, _ = board._simulate_stone(previous_capture)
                except ValueError:
                    pass
                else:
                    if (tuple(stones), -board.to_play) in board.position_history:
                        assert not board.is_legal(previous_capture)
                        prohibited += 1
        old, color = list(board.board), board.to_play
        board.play(action)
        captured = [p for p in range(100) if old[p] == -color and board.board[p] == 0]
        previous_capture = captured[0] if len(captured) == 1 else None
    return {"positions": positions, "single_capture_candidates": candidates,
            "immediate_ko_forbidden_positions": prohibited}


def main():
    torch.set_num_threads(1)
    table, rows = {}, []
    warm = []
    for seed in SEEDS:
        base = torch.load(ROOT / f"checkpoints/go12_v1/base-{seed}.pt", weights_only=True)
        warm += [{"seed": seed, **exposure(game)} for game in base["training_games"]]
    for method in (*METHODS, "legacy"):
        training, updates, priors = [], [], []
        for seed in SEEDS:
            run = json.loads((ROOT / f"analysis/results/go12_{method}_seed{seed}.json").read_text())
            seen = []
            for game in run["games"]:
                if not game["phase"].startswith("train_"):
                    continue
                prior_winner = float(np.mean(seen)) if seen else 0
                priors.append((prior_winner - game["winner"]) ** 2)
                seen.append(game["winner"])
                training.append(game)
                updates.extend(game["updates"])
                rows.append({"method": method, "seed": seed, "index": game["index"], **exposure(game)})
        selected = [r for r in rows if r["method"] == method]
        projections = [u["constraint_projection"] for u in updates if "constraint_projection" in u]
        table[method] = {
            "training_positions": sum(r["positions"] for r in selected),
            "immediate_ko_forbidden_positions": sum(r["immediate_ko_forbidden_positions"] for r in selected),
            "single_capture_candidates": sum(r["single_capture_candidates"] for r in selected),
            "projected_updates": len(projections),
            "updates_with_active_constraints": sum(p["active"] > 0 for p in projections),
            "mean_norm_retention": float(np.mean([p["norm_retention"] for p in projections])) if projections else None,
            "attempted_updates": len(updates),
            "backtracked_updates": sum(u.get("anchor_backtracks", 0) > 0 for u in updates),
            "max_update_norm": max((u["update_norm"] for u in updates), default=0),
            "prequential_value_mse": float(np.mean([g["value_mse"] for g in training])) if training else None,
            "causal_color_prior_mse": float(np.mean(priors)) if priors else None,
        }
    result = {"scope": __doc__, "table": table, "training_exposure": rows,
              "shared_pretraining_exposure": warm,
              "shared_pretraining_totals": {key: sum(r[key] for r in warm) for key in
                  ("positions", "single_capture_candidates", "immediate_ko_forbidden_positions")}}
    prior.save_json(result, ROOT / "analysis/results/go12_review.json")
    print(json.dumps({"table": table, "shared_pretraining_totals": result["shared_pretraining_totals"]}))


if __name__ == "__main__":
    main()
