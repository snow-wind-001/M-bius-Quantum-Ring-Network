"""Read-only ownership transfer diagnosis on the existing teacher probes.

Added while the main matrix runs. Separates occupied/empty-point error and
compares a current-stones-only predictor. These are teacher-behavior outcomes,
not labels for the learner's on-policy value and not extra complete matches.
"""
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
from experiments.go12_efficiency import METHODS, SEEDS, build, make_session, parameter_hash, prior
from experiments.go12_legacy_control import make_session as make_legacy_session
from mqr import GoBoard


@torch.no_grad()
def diagnose(seed):
    torch.set_num_threads(1)
    base = torch.load(ROOT / f"checkpoints/go12_v1/base-{seed}.pt", weights_only=True)
    selected = {(r["game_index"], r["ply"]) for r in base["probes"]}
    results = []
    for method in (*METHODS, "legacy"):
        run = json.loads((ROOT / f"analysis/results/go12_{method}_seed{seed}.json").read_text())
        args = argparse.Namespace(**run["config"])
        agent, encoder = build(args, seed)
        session = (make_legacy_session if method == "legacy" else make_session)(args, agent, encoder)
        checkpoint = ROOT / run["checkpoint"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == run["checkpoint_sha256"]
        session.load_state_dict(torch.load(checkpoint, weights_only=True)["session"])
        before = parameter_hash(agent)
        rows = []
        for index, game in enumerate(base["probe_games"]):
            board, terminal = GoBoard(10, komi=5.5), GoBoard(10, komi=5.5)
            for action in game["moves"]:
                terminal.play(action)
            ownership = np.array(terminal.area_ownership())
            state = agent.core.zero_state(1, device="cpu", dtype=torch.float32)
            for ply, action in enumerate(game["moves"]):
                x = encoder.encode_board(board)
                output, state = agent._transition(x, state, slow_write=True)
                if (index, ply) in selected:
                    y = ownership * board.to_play
                    predicted = output.ownership[0].numpy().astype(float)
                    stones = np.array(board.board) * board.to_play
                    occupied = stones != 0
                    errors, copy_errors = (predicted - y) ** 2, (stones - y) ** 2
                    score = predicted.sum() - board.to_play * board.komi
                    target_score = board.to_play * terminal.score()["margin_black"]
                    rows.append({
                        "game_index": index, "ply": ply,
                        "occupied_points": int(occupied.sum()), "empty_points": int((~occupied).sum()),
                        "occupied_squared_error_sum": float(errors[occupied].sum()),
                        "empty_squared_error_sum": float(errors[~occupied].sum()),
                        "mse": float(errors.mean()), "copy_stones_mse": float(copy_errors.mean()),
                        "score_absolute_error": abs(float(score) - target_score),
                        "score_sign_correct": bool((score > 0) == (target_score > 0)),
                        "value_sign_correct": bool((float(output.value[0]) >= 0) == (target_score > 0)),
                        "always_white_correct": terminal.winner() == -1,
                    })
                board.play(action)
        assert parameter_hash(agent) == before
        results.append({"method": method, "seed": seed, "rows": rows, "parameters_unchanged": True})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    paths = [ROOT / f"analysis/results/go12_{m}_seed{s}.json" for m in (*METHODS, "legacy") for s in SEEDS]
    while not all(p.exists() for p in paths):
        if not args.wait:
            parser.error("complete paired checkpoints required")
        time.sleep(10)
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = [r for group in pool.map(diagnose, SEEDS) for r in group]
    summary = {}
    for method in (*METHODS, "legacy"):
        rows = [p for r in results if r["method"] == method for p in r["rows"]]
        summary[method] = {key: float(np.mean([p[key] for p in rows])) for key in (
            "mse", "copy_stones_mse", "score_absolute_error", "score_sign_correct",
            "value_sign_correct", "always_white_correct")}
        for kind in ("occupied", "empty"):
            summary[method][kind + "_mse"] = sum(p[kind + "_squared_error_sum"] for p in rows) / sum(
                p[kind + "_points"] for p in rows)
    prior.save_json({"scope": __doc__, "summary": summary, "results": results,
                     "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                    ROOT / "analysis/results/go12_ownership_diagnostic.json")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
