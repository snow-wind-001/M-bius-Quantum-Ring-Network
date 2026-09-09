"""Inspect saved Go models without training or modifying their checkpoints.

Uses four fixed test trajectories per phase for state/gradient diagnostics.
Training-A windows are reconstructed only to inspect the stored OGD anchors.
These are retrospective diagnostics, not new held-out performance claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.orthogonal_go_online import _data_hash, build_agent
from mqr import GoVectorEncoder, HeuristicGoTeacher
from mqr.go_agent import generate_go_agent_trajectories, legality_target


def average(values: list[float]) -> float:
    return float(np.mean(values))


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(left.norm() * right.norm())
    return float(left @ right) / denominator if denominator > 1e-20 else 0.0


def zero_state(agent):
    reference = next(agent.parameters())
    return agent.core.zero_state(1, device=reference.device, dtype=reference.dtype)


def flat_gradient(loss: torch.Tensor, active: list, *, retain: bool = False) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, [p for _, p in active], allow_unused=True, retain_graph=retain)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for (_, p), g in zip(active, gradients)])


def block_fractions(vector: torch.Tensor, active: list) -> dict:
    blocks, offset = {}, 0
    for name, parameter in active:
        key = ("transition" if "unitary_params" in name else
               "injection" if name.startswith(("core.input_down", "core.input_up")) else
               "core_readout" if name.startswith("core.readout") else
               "modulation" if name.startswith("ring_modulation") else "heads")
        blocks[key] = blocks.get(key, 0.0) + float(vector[offset:offset + parameter.numel()].square().sum())
        offset += parameter.numel()
    total = sum(blocks.values())
    return {key: value / total if total else 0.0 for key, value in blocks.items()}


def replay(agent, encoder, examples, *, prefix=()):
    state = zero_state(agent)
    with torch.no_grad():
        for example in prefix:
            _, state = agent._transition(encoder.encode_board(example.board), state, slow_write=True)
    outputs, losses = [], []
    for example in examples:
        output, state = agent._transition(encoder.encode_board(example.board), state, slow_write=True)
        outputs.append(output)
        losses.append(agent.compute_go_loss(output, torch.tensor([example.target_action]),
                                           legality_target=legality_target(example.board)))
    return outputs, losses


def inspect_model(agent, initial, encoder, data: dict, config: dict, seed: int) -> dict:
    active = [(name, p) for name, p in agent._named_task_parameters() if p.requires_grad]
    transition_drift = [float((agent.core.transition_matrix(j) - initial.core.transition_matrix(j)).detach().norm())
                        / math.sqrt(agent.ring_dim) for j in (1, 2)]
    states, latent_derivatives, injection_derivatives = [], [], []
    current_cos, canonical_cos = [], []
    history_drop, teacher_history_changes = [], 0
    phase_metrics = {}
    with torch.no_grad():
        for phase in ("a", "b"):
            nlls, true_losses, passes, initial_nlls = [], [], [], []
            pass_gains, point_gains, action_changes = [], [], []
            for trace in data[phase]:
                state = zero_state(agent)
                prior, prior_canonical = None, None
                for example in trace.examples:
                    board, target = example.board, example.target_action
                    x = encoder.encode_board(board)
                    canonical = x.clone()
                    if board.to_play == -1:
                        canonical[:, :agent.points] = x[:, agent.points:2 * agent.points]
                        canonical[:, agent.points:2 * agent.points] = x[:, :agent.points]
                    if prior is not None:
                        current_cos.append(cosine(x[0, :2 * agent.points], prior[0, :2 * agent.points]))
                        canonical_cos.append(cosine(canonical[0, :2 * agent.points], prior_canonical[0, :2 * agent.points]))
                    prior, prior_canonical = x, canonical
                    output, state = agent._transition(x, state, slow_write=True)
                    before, _ = initial._transition(x, zero_state(initial), slow_write=True)
                    reset_output, _ = agent._transition(x, zero_state(agent), slow_write=True)
                    history_drop.append(float(-reset_output.policy_logits[0, target] + output.policy_logits[0, target]))
                    states.append(torch.cat(state.rings, dim=1)[0])
                    latent_derivatives.append((1 - output.latent.square())[0])
                    injection_derivatives.append((1 - agent.core.input_down(x).tanh().square())[0])
                    nlls.append(float(-output.policy_logits[0, target]))
                    initial_nlls.append(float(-before.policy_logits[0, target]))
                    action_changes.append(int(output.policy_logits.argmax()) != int(before.policy_logits.argmax()))
                    is_pass = target == agent.points
                    current_pass_nll = -F.logsigmoid(output.pass_logit if is_pass else -output.pass_logit)
                    initial_pass_nll = -F.logsigmoid(before.pass_logit if is_pass else -before.pass_logit)
                    pass_gain = float(initial_pass_nll - current_pass_nll)
                    pass_gains.append(pass_gain)
                    point_gains.append(initial_nlls[-1] - nlls[-1] - pass_gain)
                    loss = agent.compute_go_loss(output, torch.tensor([target]), legality_target=legality_target(board))
                    true_losses.append(float(loss["total"]))
                    passes.append(target == agent.points)
                    # This counterfactual only removes superko history; it is
                    # not a legal alternative trajectory or a new Go benchmark.
                    erased = board.copy()
                    erased.position_history = {board.position_key()}
                    teacher_history_changes += HeuristicGoTeacher().select_move(erased) != target
            phase_metrics[phase] = {"positions": len(nlls), "joint_nll": average(nlls),
                "initial_joint_nll": average(initial_nlls), "pass_component_nll_gain": average(pass_gains),
                "conditional_point_component_nll_gain": average(point_gains),
                "raw_action_change_fraction_from_initial": average(action_changes),
                "training_objective": average(true_losses), "teacher_pass_fraction": average(passes)}
            assert abs(average(initial_nlls) - average(nlls) - average(pass_gains) - average(point_gains)) < 1e-6
    stacked = torch.stack(states).double()
    centered = stacked - stacked.mean(dim=0)
    singular = torch.linalg.svdvals(centered)
    eigenvalues = singular.square()
    participation = float(eigenvalues.sum().square() / eigenvalues.square().sum())
    derivatives = torch.cat(latent_derivatives)
    initial_parameters = {name: parameter.detach() for name, parameter in initial.named_parameters()}
    parameter_changes = {}
    for prefix in ("core.input_down", "core.input_up", "core.readout", "ring_modulation"):
        matches = [(name, p) for name, p in agent.named_parameters() if name.startswith(prefix)]
        if matches:
            delta = sum(float((p.detach() - initial_parameters[name]).square().sum()) for name, p in matches)
            original = sum(float(initial_parameters[name].square().sum()) for name, _ in matches)
            parameter_changes[prefix] = {"delta_norm": math.sqrt(delta), "initial_norm": math.sqrt(original)}

    gradients = []
    for trace in data["b"][:2]:
        for start in range(0, len(trace.examples), config["window"]):
            examples = trace.examples[start:start + config["window"]]
            outputs, losses = replay(agent, encoder, examples, prefix=trace.examples[:start])
            total = sum(item["total"] for item in losses) / len(losses)
            nll = sum(-out.policy_logits[0, ex.target_action] for out, ex in zip(outputs, examples)) / len(examples)
            raw = flat_gradient(total, active, retain=True)
            joint = flat_gradient(nll, active)
            entries, offset = [], 0
            for name, parameter in active:
                entries.append((name, raw[offset:offset + parameter.numel()].reshape_as(parameter), agent.task_lr))
                offset += parameter.numel()
            projected, stats = agent.task_gradient_memory.project_preconditioned(entries)
            projected_flat = torch.cat([projected[name].flatten() for name, _ in active])
            gradients.append({
                "objective_joint_nll_gradient_cosine": cosine(raw, joint),
                "projected_joint_nll_gradient_cosine": cosine(projected_flat, joint),
                "raw_direction_joint_nll_first_order": float(-joint @ raw),
                "projected_direction_joint_nll_first_order": float(-joint @ projected_flat),
                "norm_matched_raw_direction_joint_nll_first_order": float(-joint @ raw) * stats["retained_norm"],
                "retained_norm": stats["retained_norm"], "raw_gradient_norm": float(raw.norm()),
                "raw_gradient_energy_by_block": block_fractions(raw, active),
                "projected_gradient_energy_by_block": block_fractions(projected_flat, active),
            })

    anchor_metrics = []
    if agent.task_gradient_memory.rank:
        indices = [(i, end) for i, trace in enumerate(data["train_a"]) for end in range(1, len(trace.examples) + 1)]
        random.Random(seed + 9000).shuffle(indices)
        basis = agent.task_gradient_memory._basis
        for trace_id, end in indices[:agent.task_gradient_memory.rank]:
            trace = data["train_a"][trace_id]
            start = max(0, end - config["window"])
            window = trace.examples[start:end]
            short, _ = replay(agent, encoder, window)
            full, _ = replay(agent, encoder, window, prefix=trace.examples[:start])
            target = window[-1].target_action
            short_gradient = flat_gradient(short[-1].policy_logits[0, target], active)
            full_gradient = flat_gradient(full[-1].policy_logits[0, target], active)
            def coverage(vector):
                return float((basis @ vector).square().sum() / vector.square().sum().clamp_min(1e-30))
            anchor_metrics.append({
                "start_ply_in_recorded_trace": start,
                "current_truncated_gradient_coverage_by_old_basis": coverage(short_gradient),
                "current_contextual_gradient_coverage_by_old_basis": coverage(full_gradient),
                "truncated_contextual_gradient_cosine": cosine(short_gradient, full_gradient),
                "contextual_minus_truncated_log_probability": float((full[-1].policy_logits[0, target] - short[-1].policy_logits[0, target]).detach()),
            })
    return {
        "phase_probe_metrics": phase_metrics,
        "transition_normalized_frobenius_drift_from_initial": transition_drift,
        "parameter_changes_from_initial": parameter_changes,
        "latent_tanh_derivative_mean": float(derivatives.mean()),
        "latent_tanh_derivative_below_0_1_fraction": float((derivatives < .1).float().mean()),
        "injection_tanh_derivative_mean": float(torch.cat(injection_derivatives).mean()),
        "centered_state_covariance_participation_rank": participation,
        "uncentered_state_rms": float(stacked.square().mean().sqrt()),
        "current_player_board_cosine_lag1": average(current_cos),
        "canonical_black_white_board_cosine_lag1": average(canonical_cos),
        "mean_history_reset_nll_increase": average(history_drop),
        "teacher_label_changes_when_superko_history_erased": teacher_history_changes,
        "gradient_windows": gradients, "ogd_anchor_diagnostics": anchor_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43, 71, 101])
    parser.add_argument("--probe-traces", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/results/go_root_cause_diagnostics.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    payload = {"schema_version": 1, "retrospective_diagnostic": True, "training_updates": 0,
               "seeds": args.seeds, "probe_traces_per_phase": args.probe_traces,
               "new_performance_claim": False, "runs": [], "checkpoint_sha256": {}}
    for seed in args.seeds:
        reference = torch.load(ROOT / f"checkpoints/orthogonal_go_online/orthogonal_ogd-{seed}.pt", weights_only=True)
        config = reference["config"]
        data = {}
        for name, count, offset, prefix in (("a", args.probe_traces, 4000, 0), ("b", args.probe_traces, 5000, config["size"]**2 // 2),
                                            ("train_a", config["train_games"], 2000, 0)):
            data[name] = generate_go_agent_trajectories(count, size=config["size"], komi=config["komi"], seed=seed + offset,
                start_random_moves=prefix, recorded_moves=config["moves"], random_move_probability=.5, task=name)
        canonical = json.loads((ROOT / "analysis/results/orthogonal_go_online_5seed.json").read_text())
        expected = next(run for run in canonical["runs"] if run["seed"] == seed)
        assert _data_hash(data["train_a"]) == expected["data_sha256"]["train_a"]
        record = {"seed": seed, "train_a_hash_matches_original": True, "methods": {}}
        for readout, directory in (("global", "orthogonal_go_online"), ("spatial", "orthogonal_go_spatial")):
            for method in ("orthogonal", "orthogonal_ogd", "identity_ogd", "stateless"):
                path = ROOT / f"checkpoints/{directory}/{method}-{seed}.pt"
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                checkpoint = torch.load(path, weights_only=True)
                agent = build_agent(SimpleNamespace(**checkpoint["config"]), method, seed)
                initial = build_agent(SimpleNamespace(**checkpoint["config"]), method, seed)
                agent.load_state_dict(checkpoint["session"]["agent"])
                # These methods freeze the spatial base throughout training.
                # Reuse it to recover their exact zero-adapter initial policy.
                initial.spatial_skip_heads.load_state_dict(agent.spatial_skip_heads.state_dict())
                encoder = GoVectorEncoder(config["size"], 3 * config["size"]**2 + 6, projection_mode="identity")
                record["methods"][f"{readout}/{method}"] = inspect_model(agent, initial, encoder, data, config, seed)
                assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
                payload["checkpoint_sha256"][str(path.relative_to(ROOT))] = digest
                print(f"seed={seed} inspected {readout}/{method}", flush=True)
        payload["runs"].append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    payload["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [Path(__file__), ROOT / "mqr/agent.py", ROOT / "mqr/go_online.py", ROOT / "mqr/temporal.py"]}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
