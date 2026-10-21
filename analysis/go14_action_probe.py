"""Post hoc, read-only action effects of history; no legal-mask oracle at readout."""
from __future__ import annotations
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import go14_readability as study
from experiments import go13_algorithm as old
from mqr.go_research import history_output


@torch.no_grad()
def measure(agent, batch):
    scores, reset_scores = [], []
    for indices in torch.arange(len(batch.labels)).split(32):
        features = batch.features[:, indices]
        scores.append(history_output(agent, features).policy_logits)
        reset_scores.append(history_output(agent, features, reset=True).policy_logits)
    scores, reset_scores = torch.cat(scores), torch.cat(reset_scores)
    actions, reset_actions = scores.argmax(1), reset_scores.argmax(1)
    probs = scores.softmax(1).gather(1, batch.queries[:, None]).squeeze(1)
    assert torch.equal(reset_actions[::2], reset_actions[1::2])
    return {"rows": len(actions), "pairs": len(actions) // 2,
            "paired_raw_greedy_changes": int((actions[::2] != actions[1::2]).sum()),
            "raw_greedy_changes_after_reset": int((actions != reset_actions).sum()),
            "query_selected_when_legal": int((actions[1::2] == batch.queries[1::2]).sum()),
            "query_selected_when_illegal": int((actions[::2] == batch.queries[::2]).sum()),
            "mean_legal_minus_illegal_query_probability": float((probs[1::2] - probs[::2]).mean()),
            "actions": actions.tolist(), "reset_actions": reset_actions.tolist(),
            "query_probabilities": probs.tolist()}


if __name__ == "__main__":
    torch.set_num_threads(1)
    study.check_lock()
    result = {"post_hoc_read_only": True, "source": study.source_hashes(),
              "probe_sha256": old.sha(Path(__file__)), "stage": "after_A_before_B", "runs": []}
    for seed in study.SEEDS:
        path = study.DIRECTORY / f"prepared-{seed}.pt"
        prepared = torch.load(path, weights_only=True)
        batch = old.materialize(prepared["data"]["a_test"])
        for layout in study.LAYOUTS:
            agent, _ = study.build(seed, layout)
            agent.load_state_dict(prepared["models"][layout])
            before = old.parameter_hash(agent)
            metrics = measure(agent, batch)
            assert before == old.parameter_hash(agent)
            result["runs"].append({"seed": seed, "layout": layout,
                "checkpoint_sha256": old.sha(path), "parameter_hash": before, **metrics})
        print(json.dumps({"action_probes_completed_seed": seed}), flush=True)
    result["summary"] = {}
    for layout in study.LAYOUTS:
        rows = [r for r in result["runs"] if r["layout"] == layout]
        result["summary"][layout] = {key: sum(r[key] for r in rows) for key in (
            "rows", "pairs", "paired_raw_greedy_changes", "raw_greedy_changes_after_reset",
            "query_selected_when_legal", "query_selected_when_illegal")}
        result["summary"][layout]["mean_legal_minus_illegal_query_probability"] = sum(
            r["mean_legal_minus_illegal_query_probability"] for r in rows) / len(rows)
    old.prior.save_json(result, ROOT / "analysis/results/go14_action_probe.json")
