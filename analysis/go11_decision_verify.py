"""Fixed-state value/readout interventions and trained 10x10 adjoint checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go11_continual import SEEDS, METHODS, build, make_session, save_json, source_hashes
from experiments.go_policy_research import no_query_output
from mqr import GoBoard
from mqr.constraint_transport import terminal_gradients
from mqr.go_search import policy_value_search
from mqr.go_outcome import SearchOutcomeGoSession
from mqr.go_cone import TaskMarginMemory


def analyze(key):
    seed, method = key
    torch.set_num_threads(1)
    result = json.loads((ROOT / f"analysis/results/go11_{method}_seed{seed}.json").read_text())
    assert result["completed"] and result["source"] == source_hashes()
    args = argparse.Namespace(**result["config"])
    base = torch.load(Path(args.directory) / f"base-{seed}.pt", weights_only=True)
    checkpoint = Path(result["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == result["checkpoint_sha256"]
    work = torch.load(checkpoint, weights_only=True)
    agent, encoder = build(args, method, seed)
    session = make_session(args, agent, encoder, method)
    session.load_state_dict(work["session"])
    initial_agent, _ = build(args, method, seed)
    initial_agent.spatial_skip_heads.load_state_dict(base["base"])
    parameter_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in agent.parameters())).hexdigest()
    rows = []
    with torch.no_grad():
        for record in base["probes"]:
            board = GoBoard(10, komi=5.5)
            for action in base["probe_games"][record["game_index"]]["moves"][:record["ply"]]:
                board.play(action)
            torch.testing.assert_close(encoder.encode_board(board), record["x"], rtol=0, atol=0)
            state = agent.core.zero_state(1, device="cpu", dtype=torch.float32)
            for x in record["features"].split(1):
                output, state = agent._transition(x, state, slow_write=True)
            initial_state = initial_agent.core.zero_state(1, device="cpu", dtype=torch.float32)
            for x in record["features"].split(1):
                initial_output, initial_state = initial_agent._transition(x, initial_state, slow_write=True)
            alternate = no_query_output(agent, output, record["x"])
            reset, _ = agent._transition(record["x"], agent.core.zero_state(1, device="cpu", dtype=torch.float32), slow_write=True)
            legal = torch.tensor(board.legal_moves())
            actions = [int(legal[o.policy_logits[0, legal].argmax()]) for o in (output, alternate, reset)]
            searches = [policy_value_search(agent, encoder, board, state, output.policy_logits,
                                           simulations=args.eval_simulations, max_depth=args.search_depth,
                                           value_scale=scale) for scale in (1, 0)]
            original_transition = agent._transition
            color_value = float(output.value[0]) * board.to_play
            def constant_color_value(features, previous, *, slow_write):
                prediction, following = original_transition(features, previous, slow_write=slow_write)
                return replace(prediction, value=(2 * features[:, 3 * agent.points] - 1) * color_value), following
            with patch.object(agent, "_transition", constant_color_value):
                constant_search = policy_value_search(agent, encoder, board, state, output.policy_logits,
                    simulations=args.eval_simulations, max_depth=args.search_depth)
            correction = output.placement_logits - alternate.placement_logits
            correction -= correction.mean(1, keepdim=True)
            target = record["action"]
            rows.append({"domain": record["domain"], "query_changes_greedy": actions[0] != actions[1],
                         "reset_changes_greedy": actions[0] != actions[2],
                         "value_changes_search": searches[0]["action"] != searches[1]["action"],
                         "position_dependent_value_changes_search": searches[0]["action"] != constant_search["action"],
                         "query_centered_rms": float(correction.square().mean().sqrt()),
                         "full_nll": float(-output.policy_logits[0, target]),
                         "initial_nll": float(-initial_output.policy_logits[0, target]),
                         "raw_entropy": float(-(output.policy_logits.exp() * output.policy_logits).sum()),
                         "initial_raw_entropy": float(-(initial_output.policy_logits.exp() * initial_output.policy_logits).sum()),
                         "teacher_visited_with_value": target in searches[0]["visits"],
                         "teacher_visited_without_value": target in searches[1]["visits"],
                         "reset_nll": float(-reset.policy_logits[0, target]),
                         "value": float(output.value[0]), "outcome": record["value"],
                         "color_prior": base["color_prior"][record["color"]]})
    assert parameter_hash == hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in agent.parameters())).hexdigest()
    proof = None
    retained_gradient, constraint_projections = [], []
    rotation_error = None
    if method in ("selfplay_cone3", "selfplay_cone6", "selfplay_ogd3"):
        session.refresh_protection()
        for record in base["probes"][:3]:
            board = GoBoard(10, komi=5.5)
            for action in base["probe_games"][record["game_index"]]["moves"][:record["ply"]]:
                board.play(action)
            with torch.no_grad():
                state = agent.core.zero_state(1, device="cpu", dtype=torch.float32)
                for x in record["features"].split(1):
                    output, state = agent._transition(x, state, slow_write=True)
                search = policy_value_search(agent, encoder, board, state, output.policy_logits,
                                             simulations=args.eval_simulations, max_depth=args.search_depth)
            target_distribution = SearchOutcomeGoSession.search_target(search, agent.action_size)
            objective = lambda out: agent.compute_go_loss(out, torch.tensor([search["action"]]),
                policy_target=target_distribution, legality_target=record["legal"])["total"]
            gradient = terminal_gradients(agent, record["features"], objective).jacobian[0]
            if isinstance(session.behavior_memory, TaskMarginMemory):
                candidate = -agent.task_lr * gradient
                candidate *= min(1.0, agent.max_update_norm / max(float(candidate.norm()), 1e-30))
                projected, diagnostic = session.behavior_memory.project_update(candidate, parameter_version=int(agent.online_parameter_version))
                constraint_projections.append(diagnostic)
                retained_gradient.append(float(projected.norm()/candidate.norm().clamp_min(1e-30)))
            else:
                basis = agent.task_gradient_memory._basis
                projected = gradient - basis.T @ (basis @ gradient) if agent.task_gradient_memory.rank else gradient
                retained_gradient.append(float(projected.norm()/gradient.norm().clamp_min(1e-30)))
        agent.double()
        record = base["probes"][0]
        target = record["action"]
        objective = lambda output: torch.stack((output.policy_logits[0, target], output.value[0]))
        a = terminal_gradients(agent, record["features"].double(), objective, backend="transport")
        b = terminal_gradients(agent, record["features"].double(), objective, backend="autograd")
        torch.testing.assert_close(a.jacobian, b.jacobian, rtol=1e-8, atol=1e-10)
        proof = {"max_absolute_error": float((a.jacobian-b.jacobian).abs().max()),
                 "relative_error": float((a.jacobian-b.jacobian).norm()/b.jacobian.norm().clamp_min(1e-30)),
                 "diagnostics": a.diagnostics}
        with torch.no_grad():
            xs = torch.cat([r["x"] for r in base["probes"][:4]]).double()
            errors = []
            for index in agent.core.active_transition_indices:
                rotation = agent.core.conditioned_transition(index, xs)
                errors.append(float((rotation.transpose(-1,-2) @ rotation - torch.eye(agent.core.ring_dim)).abs().max()))
            rotation_error = max(errors, default=0.0)
    return {"seed": seed, "method": method, "positions": len(rows), "rows": rows, "gradient_proof": proof,
            "parameters_unchanged_during_interventions": True,
            "current_ogd_gradient_norm_retention_on_three_terminal_search_losses": retained_gradient,
            "current_halfspace_projection_diagnostics": constraint_projections,
            "conditional_orthogonality_max_absolute_error_float64": rotation_error,
            "query_changes_greedy": sum(r["query_changes_greedy"] for r in rows),
            "reset_changes_greedy": sum(r["reset_changes_greedy"] for r in rows),
            "value_changes_search": sum(r["value_changes_search"] for r in rows),
            "position_dependent_value_changes_search": sum(r["position_dependent_value_changes_search"] for r in rows),
            "query_centered_rms": float(np.mean([r["query_centered_rms"] for r in rows])),
            "teacher_behavior_value_mse_diagnostic": float(np.mean([(r["value"]-r["outcome"])**2 for r in rows])),
            "teacher_behavior_prior_mse_diagnostic": float(np.mean([(r["color_prior"]-r["outcome"])**2 for r in rows]))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "analysis/results/go11_decision_verify.json"))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    keys = [(seed, method) for method in ((args.method,) if args.method else METHODS)
            for seed in ((args.seed,) if args.seed else SEEDS)]
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    inputs = {f"{method}-{seed}": hashlib.sha256(
        (ROOT / f"analysis/results/go11_{method}_seed{seed}.json").read_bytes()).hexdigest()
        for seed, method in keys}
    results, missing = [], []
    for seed, method in keys:
        cache_path = ROOT / f"analysis/results/go11_decision_{method}_seed{seed}.json"
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else None
        if (cached is not None and cached.get("verified") and cached.get("verifier_sha256") == digest
                and cached.get("input_sha256", {}).get(f"{method}-{seed}") == inputs[f"{method}-{seed}"]
                and cached.get("source") == source_hashes() and len(cached.get("results", [])) == 1
                and cached["results"][0]["method"] == method and cached["results"][0]["seed"] == seed):
            results.extend(cached["results"])
        else:
            missing.append((seed, method))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results.extend(pool.map(analyze, missing))
    results.sort(key=lambda row: (METHODS.index(row["method"]), SEEDS.index(row["seed"])))
    save_json({"verified": True, "results": results, "source": source_hashes(),
               "verifier_sha256": digest, "input_sha256": inputs,
               "scope": "common teacher trajectories for interventions only; teacher returns are not shared-policy value ground truth; changed actions do not establish improved actions"}, args.output)
    print(json.dumps({"verified": True, "runs": len(results), "positions": sum(r["positions"] for r in results)}))
