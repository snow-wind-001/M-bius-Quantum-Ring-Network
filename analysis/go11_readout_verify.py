"""Post-hoc full-prefix readout audit; no training, search or new games."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go11_continual import SEEDS, build, make_session, save_json, source_hashes
from analysis.go11_value_ablation import parameter_hash
from mqr import GoBoard


@torch.no_grad()
def analyze(key):
    method, seed = key
    torch.set_num_threads(1)
    path = ROOT / f"analysis/results/go11_{method}_seed{seed}.json"
    record = json.loads(path.read_text())
    assert record["completed"] and record["source"] == source_hashes()
    args = argparse.Namespace(**record["config"])
    checkpoint = Path(record["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == record["checkpoint_sha256"]
    base_path = Path(args.directory) / f"base-{seed}.pt"
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == record["base_sha256"]
    base = torch.load(base_path, weights_only=True)
    agent, encoder = build(args, method, seed)
    session = make_session(args, agent, encoder, method)
    session.load_state_dict(torch.load(checkpoint, weights_only=True)["session"])
    before = parameter_hash(agent)
    version = int(agent.online_parameter_version)
    rows = []

    def fresh_state():
        return agent.core.zero_state(1, device="cpu", dtype=torch.float32)

    for game_index, game in enumerate(base["probe_games"]):
        selected = {p["ply"]: p for p in base["probes"] if p["game_index"] == game_index}
        board, state = GoBoard(10, komi=5.5), fresh_state()
        for ply, action in enumerate(game["moves"][:max(selected) + 1]):
            x = encoder.encode_board(board)
            full, state = agent._transition(x, state, slow_write=True)
            if ply in selected:
                probe = selected[ply]
                torch.testing.assert_close(x, probe["x"], rtol=0, atol=0)
                short_state = fresh_state()
                for item in probe["features"].split(1):
                    short, short_state = agent._transition(item, short_state, slow_write=True)
                reset, _ = agent._transition(x, fresh_state(), slow_write=True)
                latent = agent.core.readout_state(state)
                rebuilt = agent._query_output(latent, state, x)
                torch.testing.assert_close(rebuilt.policy_logits, full.policy_logits, rtol=0, atol=0)
                no_query = agent._query_output(latent, state, x, include_query=False)
                hidden, correction, addressing = agent.query_components(latent, state, x)
                spatial = agent.spatial_skip_heads.placement(hidden).flatten(1)
                spatial_rms = (spatial - spatial.mean(1, keepdim=True)).square().mean().sqrt()
                correction_rms = correction[..., 0].square().mean().sqrt()
                legal = torch.tensor(board.legal_moves())
                outputs = {"full": full, "short": short, "reset": reset, "no_query": no_query}
                actions = {name: int(legal[out.policy_logits[0, legal].argmax()])
                           for name, out in outputs.items()}
                scores = full.policy_logits[0, legal].sort(descending=True).values
                perturbation = (full.policy_logits - no_query.policy_logits)[0, legal]
                perturbation_span = float(perturbation.max() - perturbation.min())
                margin = float(scores[0] - scores[1]) if len(scores) > 1 else None
                certificate = margin is None or margin > perturbation_span
                if certificate:
                    assert actions["full"] == actions["no_query"]
                full_state = torch.cat(state.rings, dim=1)
                short_flat = torch.cat(short_state.rings, dim=1)
                rows.append({
                    "game_index": game_index, "ply": ply, "full_observations": ply + 1,
                    "short_observations": len(probe["features"]), "domain": probe["domain"],
                    "actions": actions, "teacher_action": probe["action"],
                    "teacher_nll": {name: float(-out.policy_logits[0, probe["action"]])
                                    for name, out in outputs.items()},
                    "full_minus_short_policy_kl": float((full.policy_logits.exp()
                        * (full.policy_logits - short.policy_logits)).sum()),
                    "full_vs_short_state_relative_l2": float((full_state - short_flat).norm()
                        / full_state.norm().clamp_min(1e-30)),
                    "ring_norms_full": [float(h.norm()) for h in state.rings],
                    "ring_norms_short": [float(h.norm()) for h in short_state.rings],
                    "latent_abs_over_095_fraction": float((full.latent.abs() > 0.95).float().mean()),
                    "latent_tanh_mean_derivative": float((1 - full.latent.square()).mean()),
                    "query_placement_rms": float(correction_rms),
                    "spatial_placement_rms": float(spatial_rms),
                    "query_to_spatial_rms_ratio": float(correction_rms / spatial_rms.clamp_min(1e-30)),
                    "addressing_position_rms": float((addressing - addressing.mean(1, keepdim=True))
                        .square().mean().sqrt()),
                    "legal_greedy_margin": margin,
                    "query_log_probability_perturbation_span": perturbation_span,
                    "margin_certifies_unchanged_greedy_without_query": certificate,
                })
            board.play(action)
    assert len(rows) == 24
    assert parameter_hash(agent) == before and int(agent.online_parameter_version) == version
    return {"method": method, "seed": seed, "rows": rows,
            "parameters_unchanged": True, "checkpoint_sha256": record["checkpoint_sha256"],
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", default=str(ROOT / "analysis/results/go11_readout_verify.json"))
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("use a new output path; do not overwrite the post-hoc audit")
    methods = ("frozen6", "selfplay_cone6")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(analyze, [(method, seed) for method in methods for seed in SEEDS]))
    summaries = {}
    for method in methods:
        rows = [row for result in results if result["method"] == method for row in result["rows"]]
        summaries[method] = {
            "positions": len(rows),
            "full_prefix_length_min_mean_max": [min(r["full_observations"] for r in rows),
                float(np.mean([r["full_observations"] for r in rows])), max(r["full_observations"] for r in rows)],
            "full_changes_greedy_vs_short": sum(r["actions"]["full"] != r["actions"]["short"] for r in rows),
            "reset_changes_full_greedy": sum(r["actions"]["full"] != r["actions"]["reset"] for r in rows),
            "query_changes_full_greedy": sum(r["actions"]["full"] != r["actions"]["no_query"] for r in rows),
            "query_greedy_invariance_certificates": sum(r["margin_certifies_unchanged_greedy_without_query"] for r in rows),
            "teacher_agreement": {name: sum(r["actions"][name] == r["teacher_action"] for r in rows)
                                  for name in ("full", "short", "reset", "no_query")},
            "means": {name: float(np.mean([r[name] for r in rows])) for name in (
                "full_vs_short_state_relative_l2", "full_minus_short_policy_kl",
                "latent_abs_over_095_fraction", "latent_tanh_mean_derivative",
                "query_placement_rms", "spatial_placement_rms", "query_to_spatial_rms_ratio",
                "addressing_position_rms")},
        }
    save_json({"verified": True, "source": source_hashes(), "results": results, "summary": summaries,
               "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "post-hoc full teacher-prefix readout diagnosis after the short-window finding; no training, search, new games or confirmatory strength claim"}, args.output)
    print(json.dumps(summaries), flush=True)
