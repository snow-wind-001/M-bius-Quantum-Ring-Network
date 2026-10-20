"""Additional read-only ko diagnosis; not training or a complete-game result.

Introduced during the formal run, before the full matrix was available, to
separate learned legality/history discrimination from the environment mask.
Pairs begin at legal setup positions, not at empty boards. No tuning uses them.
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
from analysis.go12_verify import restore
from experiments.go12_efficiency import METHODS, SEEDS, source_hashes, parameter_hash, prior
from mqr.go_history_tasks import generate_ko_history_pairs


@torch.no_grad()
def diagnose(seed):
    torch.set_num_threads(1)
    pairs = generate_ko_history_pairs(100, seed=seed + 91000, size=10)
    results = []
    for method in METHODS:
        run = json.loads((ROOT / f"analysis/results/go12_{method}_seed{seed}.json").read_text())
        assert run["source"] == source_hashes()
        _, session = restore(run)
        agent = session.agent
        before = parameter_hash(agent)
        rows = []
        for pair in pairs:
            probabilities, reset_probabilities = [], []
            final_features = []
            state_differences = []
            for history in pair.histories:
                state = agent.core.zero_state(1, device="cpu", dtype=torch.float32)
                for board in history:
                    features = session.encoder.encode_board(board)
                    output, state = agent._transition(features, state, slow_write=True)
                final_features.append(features)
                state_differences.append(torch.cat(state.rings, 1))
                probabilities.append(float(output.legality_logits[0, pair.recapture].sigmoid()))
                reset, _ = agent._transition(features, agent.core.zero_state(1, device="cpu", dtype=features.dtype),
                                             slow_write=True)
                reset_probabilities.append(float(reset.legality_logits[0, pair.recapture].sigmoid()))
            assert torch.equal(*final_features)
            assert reset_probabilities[0] == reset_probabilities[1]
            assert tuple(h[-1].is_legal(pair.recapture) for h in pair.histories) == (False, True)
            rows.append({
                "illegal_probability": probabilities[0], "legal_probability": probabilities[1],
                "correct_positions": int(probabilities[0] < .5) + int(probabilities[1] >= .5),
                "both_correct": probabilities[0] < .5 <= probabilities[1],
                "rank_correct": probabilities[1] > probabilities[0],
                "probability_gap": probabilities[1] - probabilities[0],
                "state_l2_difference": float((state_differences[0] - state_differences[1]).norm()),
                "reset_probabilities_equal": True,
            })
        assert parameter_hash(agent) == before
        results.append({"seed": seed, "method": method, "pairs": rows, "parameters_unchanged": True})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    paths = [ROOT / f"analysis/results/go12_{m}_seed{s}.json" for s in SEEDS for m in METHODS]
    while not all(p.exists() for p in paths):
        if not args.wait:
            parser.error("the full formal matrix is required")
        time.sleep(10)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = [r for group in pool.map(diagnose, SEEDS) for r in group]
    summary = {}
    for method in METHODS:
        rows = [p for r in results if r["method"] == method for p in r["pairs"]]
        summary[method] = {
            "pairs": len(rows), "positions": 2 * len(rows),
            "accuracy": sum(p["correct_positions"] for p in rows) / (2 * len(rows)),
            "both_correct": sum(p["both_correct"] for p in rows),
            "rank_accuracy": float(np.mean([p["rank_correct"] for p in rows])),
            "mean_probability_gap": float(np.mean([p["probability_gap"] for p in rows])),
            "mean_state_l2_difference": float(np.mean([p["state_l2_difference"] for p in rows])),
        }
    prior.save_json({"scope": __doc__, "source": source_hashes(), "results": results, "summary": summary,
                     "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                    ROOT / "analysis/results/go12_history_diagnostic.json")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
