"""Fixed-state value/readout interventions and trained 10x10 adjoint checks."""
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
from experiments.go10_continual import SEEDS, build, make_session, save_json, source_hashes
from experiments.go_policy_research import no_query_output
from mqr import GoBoard
from mqr.constraint_transport import terminal_gradients
from mqr.go_search import policy_value_search
from mqr.go_outcome import SearchOutcomeGoSession


def analyze(key):
    seed, method = key
    torch.set_num_threads(1)
    result = json.loads((ROOT / f"analysis/results/go10_{method}_seed{seed}.json").read_text())
    assert result["completed"] and result["source"] == source_hashes()
    args = argparse.Namespace(**result["config"])
    base = torch.load(Path(args.directory) / f"base-{seed}.pt", weights_only=True)
    checkpoint = Path(result["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
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
                                           simulations=args.simulations, max_depth=args.search_depth,
                                           value_scale=scale) for scale in (1, 0)]
            correction = output.placement_logits - alternate.placement_logits
            correction -= correction.mean(1, keepdim=True)
            target = record["action"]
            rows.append({"domain": record["domain"], "query_changes_greedy": actions[0] != actions[1],
                         "reset_changes_greedy": actions[0] != actions[2],
                         "value_changes_search": searches[0]["action"] != searches[1]["action"],
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
    retained_gradient = []
    if method == "optimized":
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
                                             simulations=args.simulations, max_depth=args.search_depth)
            target_distribution = SearchOutcomeGoSession.search_target(search, agent.action_size)
            objective = lambda out: agent.compute_go_loss(out, torch.tensor([search["action"]]),
                policy_target=target_distribution, legality_target=record["legal"])["total"]
            gradient = terminal_gradients(agent, record["features"], objective).jacobian[0]
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
    return {"seed": seed, "method": method, "positions": len(rows), "rows": rows, "gradient_proof": proof,
            "parameters_unchanged_during_interventions": True,
            "current_ogd_gradient_norm_retention_on_three_terminal_search_losses": retained_gradient,
            "query_changes_greedy": sum(r["query_changes_greedy"] for r in rows),
            "reset_changes_greedy": sum(r["reset_changes_greedy"] for r in rows),
            "value_changes_search": sum(r["value_changes_search"] for r in rows),
            "query_centered_rms": float(np.mean([r["query_centered_rms"] for r in rows])),
            "fixed_behavior_value_mse": float(np.mean([(r["value"]-r["outcome"])**2 for r in rows])),
            "fixed_behavior_prior_mse": float(np.mean([(r["color_prior"]-r["outcome"])**2 for r in rows]))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "analysis/results/go10_decision_verify.json"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(analyze, [(seed, method) for method in ("frozen", "legacy", "optimized") for seed in SEEDS]))
    save_json({"verified": True, "results": results, "source": source_hashes(),
               "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "shared teacher-generated complete trajectories; actions changing under intervention does not establish better actions"}, args.output)
    print(json.dumps({"verified": True, "runs": len(results), "positions": sum(r["positions"] for r in results)}))
