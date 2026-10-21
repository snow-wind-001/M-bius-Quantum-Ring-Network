"""Rescore existing probes: action change alone is not behavior degradation."""
from pathlib import Path
import json
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import go14_readability as study
from experiments import go13_algorithm as old


if __name__ == "__main__":
    torch.set_num_threads(1)
    study.check_lock()
    result = {"post_hoc_rescoring_only": True, "source": study.source_hashes(), "runs": [], "summary": {}}
    for seed in study.SEEDS:
        path = study.DIRECTORY / f"prepared-{seed}.pt"
        prepared = torch.load(path, weights_only=True)
        targets = [r["action"] for r in prepared["policy_test"]]
        for method in study.METHODS:
            run_path = ROOT / "analysis/results" / f"go14_learning_{method}_seed{seed}.json"
            run = json.loads(run_path.read_text())
            before = [a == target for a, target in zip(run["policy_before"]["actions"], targets)]
            after = [a == target for a, target in zip(run["policy_after"]["actions"], targets)]
            assert len(before) == len(after) == 24
            assert abs(sum(before) / 24 - run["policy_before"]["teacher_agreement"]) < 1e-7
            assert abs(sum(after) / 24 - run["policy_after"]["teacher_agreement"]) < 1e-7
            result["runs"].append({"method": method, "seed": seed, "rows": 24,
                "teacher_correct_before": sum(before), "teacher_correct_after": sum(after),
                "retained_teacher_correct": sum(a and b for a, b in zip(before, after)),
                "lost_teacher_correct": sum(a and not b for a, b in zip(before, after)),
                "new_teacher_correct": sum(not a and b for a, b in zip(before, after)),
                "learning_sha256": old.sha(run_path), "prepared_sha256": old.sha(path)})
    for method in study.METHODS:
        rows = [r for r in result["runs"] if r["method"] == method]
        result["summary"][method] = {key: sum(r[key] for r in rows) for key in (
            "rows", "teacher_correct_before", "teacher_correct_after", "retained_teacher_correct",
            "lost_teacher_correct", "new_teacher_correct")}
    old.prior.save_json(result, ROOT / "analysis/results/go14_policy_quality.json")
    print(json.dumps({m: result["summary"][m] for m in ("spatial_free", "spatial_adaptive", "spatial_aligned")}), flush=True)
