"""Check generated history aliases, disjoint splits, and the complete seed matrix."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.go10_history import materialize
from analysis.go10_verify import paired_interval
from experiments.go10_continual import SEEDS, save_json, source_hashes


def main():
    torch.set_num_threads(1)
    rows = []
    generated = 0
    for seed in SEEDS:
        data = json.loads((ROOT / f"analysis/results/go10_history_seed{seed}.json").read_text())
        assert data["completed"] and data["source"] == source_hashes()
        assert data["diagnostic_source_sha256"] == hashlib.sha256((ROOT / "analysis/go10_history.py").read_bytes()).hexdigest()
        for task in ("ko", "recall24"):
            xs, labels, queries, signatures = materialize(task, seed)
            torch.testing.assert_close(xs[-1, ::2], xs[-1, 1::2], rtol=0, atol=0)
            assert bool((labels[::2] != labels[1::2]).all())
            assert not set(signatures[:96]).intersection(signatures[96:])
            split = {"train": signatures[:96], "test": signatures[96:]}
            digest = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
            expected = {"orthogonal", "identity", "stateless"} | ({"short_credit"} if task == "recall24" else set())
            selected = [r for r in data["results"] if r["task"] == task]
            assert {r["method"] for r in selected} == expected
            for record in selected:
                assert record["split_sha256"] == digest
                assert record["seed"] == seed and record["training_pairs"] == 96 and record["test_pairs"] == 64
                assert record["epochs"] == 48 and record["reset"]["accuracy"] <= 0.5 + 1e-12
                assert record["full"]["accuracy"] + record["swap"]["accuracy"] <= 1 + 1e-12
            generated += len(signatures)
        rows.extend(data["results"])
    table, contrasts = {}, {}
    for task in ("ko", "recall24"):
        table[task] = {}
        for method in ("orthogonal", "identity", "stateless", "short_credit"):
            selected = [r for r in rows if r["task"] == task and r["method"] == method]
            if not selected:
                continue
            table[task][method] = {name: float(np.mean([r[name]["accuracy"] for r in selected]))
                                   for name in ("full", "reset", "swap")}
            table[task][method]["per_seed"] = [r["full"]["accuracy"] for r in selected]
            table[task][method]["mean_seconds"] = float(np.mean([r["seconds"] for r in selected]))
        a = table[task]["orthogonal"]["per_seed"]
        for baseline in ("identity", "stateless"):
            b = table[task][baseline]["per_seed"]
            contrasts[task + "_orthogonal_minus_" + baseline] = paired_interval([x-y for x,y in zip(a,b)])
    result = {"verified": True, "regenerated_pairs": generated, "seeds": list(SEEDS),
              "tables": table, "contrasts": contrasts, "source": source_hashes(),
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(result, ROOT / "analysis/results/go10_history_summary.json")
    print(json.dumps(result | {"source": "recorded"}))


if __name__ == "__main__":
    main()
