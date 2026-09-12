"""Compare final six-ring behaviors against identical initial anchor references."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go11_continual import SEEDS, build, initialize_memory, make_session, save_json, source_hashes
from analysis.go11_value_ablation import parameter_hash

METHODS = ("frozen6", "selfplay_cone6", "selfplay_free6")


def analyze(seed):
    torch.set_num_threads(1)
    runs = {method: json.loads((ROOT / f"analysis/results/go11_{method}_seed{seed}.json").read_text())
            for method in METHODS}
    reference = runs["selfplay_cone6"]
    args = argparse.Namespace(**reference["config"])
    base_path = Path(args.directory) / f"base-{seed}.pt"
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == reference["base_sha256"]
    base = torch.load(base_path, weights_only=True)
    initial, encoder = build(args, "selfplay_cone6", seed)
    initial.spatial_skip_heads.load_state_dict(base["base"])
    before_initial = parameter_hash(initial)
    reference_session = make_session(args, initial, encoder, "selfplay_cone6")
    initialize_memory(reference_session, base)
    assert parameter_hash(initial) == before_initial
    memory = reference_session.behavior_memory
    assert len(memory.anchors) == reference["anchor_count"]
    results = []
    for method, run in runs.items():
        assert run["completed"] and run["source"] == source_hashes()
        assert run["base_sha256"] == reference["base_sha256"]
        config = argparse.Namespace(**run["config"])
        agent, encoder = build(config, method, seed)
        session = make_session(config, agent, encoder, method)
        checkpoint = Path(run["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == run["checkpoint_sha256"]
        session.load_state_dict(torch.load(checkpoint, weights_only=True)["session"])
        before, version = parameter_hash(agent), int(agent.online_parameter_version)
        drift = memory.drift(agent)
        if method == "selfplay_cone6":
            for name, value in drift.items():
                assert abs(value - run["final_anchor_drift"][name]) <= 1e-6
        if method == "frozen6":
            assert before == before_initial and drift["max_margin_violation"] == 0
        rows = []
        with torch.no_grad():
            for stratum, anchor in ((index, anchor) for index, group in enumerate(memory.records)
                                    for anchor in group):
                scores = memory._output(agent, anchor)
                alternatives = scores.clone()
                alternatives[anchor["action"]] = -torch.inf
                margin = float(scores[anchor["action"]] - alternatives.max())
                rows.append({"action": anchor["action"], "stratum": stratum,
                    "observations": len(anchor["features"]), "margin_floor": anchor["margin_floor"],
                    "margin": margin, "violation": max(0, anchor["margin_floor"] - margin),
                    "changed_raw_greedy": int(scores.argmax()) != anchor["action"],
                    "reference_features_sha256": hashlib.sha256(anchor["features"].numpy().tobytes()).hexdigest(),
                    "reference_policy_sha256": hashlib.sha256(anchor["reference_log_probs"].numpy().tobytes()).hexdigest()})
        assert parameter_hash(agent) == before and int(agent.online_parameter_version) == version
        if method == "selfplay_cone6":
            assert all(r["violation"] <= 1.1e-7 for r in rows)
        results.append({"seed": seed, "method": method, "rows": rows, "drift": drift,
                        "parameters_unchanged": True, "checkpoint_sha256": run["checkpoint_sha256"]})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--wait", action="store_true", help="wait for already-running formal evaluations")
    parser.add_argument("--output", default=str(ROOT / "analysis/results/go11_reference_verify.json"))
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("use a new output path; do not overwrite this diagnosis")
    while not all((ROOT / f"analysis/results/go11_{m}_seed{s}.json").exists() for m in METHODS for s in SEEDS):
        if not args.wait:
            parser.error("the complete six-ring matrix is required")
        time.sleep(15)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = [row for group in pool.map(analyze, SEEDS) for row in group]
    summary = {}
    for method in METHODS:
        selected = [r for r in results if r["method"] == method]
        anchors = [row for r in selected for row in r["rows"]]
        summary[method] = {"anchors": len(anchors),
            "changed_raw_greedy": sum(r["changed_raw_greedy"] for r in anchors),
            "margin_violations_over_1e_minus_7": sum(r["violation"] > 1e-7 for r in anchors),
            "max_margin_violation": max(r["violation"] for r in anchors),
            "max_policy_kl": max(r["drift"]["max_policy_kl"] for r in selected),
            "mean_policy_kl": float(np.mean([r["drift"]["mean_policy_kl"] for r in selected]))}
    save_json({"verified": True, "results": results, "summary": summary, "source": source_hashes(),
               "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "post-hoc final-checkpoint comparison at identical initial training anchors; not full-history, pathwise or unseen-task protection"}, args.output)
    print(json.dumps(summary), flush=True)
