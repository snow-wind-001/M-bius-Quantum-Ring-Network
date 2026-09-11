"""Read-only rank/step audits on the learned v2 checkpoints and seen B games."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from go_memory_research import build, session_for
from mqr import GoBoard, HeuristicGoTeacher
from mqr.go_agent import legality_target
from mqr.go_history_tasks import generate_reachable_history_pairs
from mqr.go_memory import GoHistoryEncoder, GoObservationMemory


def flat_grad(score, active, retain=False):
    values = torch.autograd.grad(score, [p for _, p in active], allow_unused=True, retain_graph=retain)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for (_, p), g in zip(active, values)])


def audit_checkpoint(path, run):
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    args = argparse.Namespace(**checkpoint["config"])
    agent, encoder = build(args, checkpoint["method"], checkpoint["seed"])
    session = session_for(args, agent, encoder, checkpoint["method"])
    session.load_state_dict(checkpoint["session"])
    bank = session.behavior_memory
    active = [(n, p) for n, p in agent._named_task_parameters() if p.requires_grad]
    before = [p.detach().clone() for _, p in active]
    board = GoBoard(args.size, komi=2.5)
    memory = GoObservationMemory(encoder)
    examples = []
    # These observations were already seen in the first phase-B game. This is
    # a local diagnostic at theta_final, not a new learning benchmark.
    for action in run["train_b"]["games"][0]["moves"][:8]:
        examples.append((encoder.with_history(board, memory.slots),
                         torch.tensor([HeuristicGoTeacher().select_move(board)]), legality_target(board)))
        memory.write(board); board.play(action)

    def objective():
        reference = next(agent.parameters())
        state = agent.core.zero_state(1, device=reference.device, dtype=reference.dtype)
        losses = []
        for x, target, legal in examples:
            output, state = agent._transition(x, state, slow_write=True)
            losses.append(agent.compute_go_loss(output, target, legality_target=legal)["total"])
        return torch.stack(losses).mean()

    loss = objective()
    raw = flat_grad(loss, active)
    reference_outputs, target_gradients = [], []
    for record in bank.anchors:
        output = bank._output(agent, record)
        reference_outputs.append(output.detach())
        target_gradients.append(flat_grad(output[record["action"]], active))
    targets = torch.stack(target_gradients)
    rows = []
    for rank in (4, 8, 16):
        agent.task_gradient_memory.max_rank = rank
        refresh = bank.refresh(agent)
        entries, cursor = [], 0
        for name, p in active:
            entries.append((name, raw[cursor:cursor + p.numel()].reshape_as(p), agent.task_lr))
            cursor += p.numel()
        projected, stats = agent.task_gradient_memory.project_preconditioned(entries)
        pg = torch.cat([projected[name].flatten() for name, _ in active])
        sgd = -agent.task_lr * raw
        ogd = -agent.task_lr * pg
        cap = agent.max_update_norm
        for vector in (sgd, ogd):
            if cap is not None:
                vector.mul_(min(1.0, cap / max(float(vector.norm()), 1e-12)))
        matched = sgd * (ogd.norm() / sgd.norm().clamp_min(1e-12))
        for name, delta in (("sgd", sgd), ("ogd", ogd), ("norm_matched_sgd", matched)):
            with torch.no_grad():
                cursor = 0
                for (_, p), original in zip(active, before):
                    p.copy_(original + delta[cursor:cursor + p.numel()].reshape_as(p).to(p))
                    cursor += p.numel()
                after_outputs = [bank._output(agent, record) for record in bank.anchors]
                after_loss = float(objective())
                kl = [float((ref.exp() * (ref - value)).sum().clamp_min(0))
                      for ref, value in zip(reference_outputs, after_outputs)]
                changes = torch.stack([value[record["action"]] - ref[record["action"]]
                                       for record, ref, value in zip(bank.anchors, reference_outputs, after_outputs)])
                linear = targets @ delta
                rows.append({"rank_budget": rank, "step": name, "refresh": refresh,
                             "step_norm": float(delta.norm()), "retained_norm": stats["retained_norm"],
                             "basis_bytes": agent.task_gradient_memory.storage_bytes,
                             "training_b_loss_before": float(loss.detach()), "training_b_loss_after": after_loss,
                             "max_anchor_step_kl": max(kl),
                             "max_first_order_target_drift": float(linear.abs().max()),
                             "max_actual_target_drift": float(changes.abs().max()),
                             "max_taylor_remainder": float((changes - linear).abs().max())})
                for (_, p), original in zip(active, before):
                    p.copy_(original)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    return {"method": run["method"], "seed": run["seed"], "checkpoint_sha256": original_hash,
            "anchor_bytes": bank.storage_bytes, "local_steps": rows, "checkpoint_unchanged": True}


def eviction_audit():
    pair = generate_reachable_history_pairs(1, seed=829, moves=12)[0]
    result = []
    for slots in (2, 16):
        encoder = GoHistoryEncoder(5, history_slots=slots)
        inputs = []
        for history in pair.histories:
            memory = GoObservationMemory(encoder)
            for board in history[:-1]:
                memory.write(board)
            inputs.append(encoder.with_history(history[-1], memory.slots))
        result.append({"slots": slots, "bytes": memory.storage_bytes,
                       "identical_query_observation": torch.equal(*inputs),
                       "input_l2_distance": float((inputs[0] - inputs[1]).norm()),
                       "evictions": memory.evictions})
    assert result[0]["identical_query_observation"] and not result[1]["identical_query_observation"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="analysis/results/go_memory_online_5seed.json")
    parser.add_argument("--checkpoints", default="checkpoints/go_memory_v2_confirmed")
    parser.add_argument("--output", default="analysis/results/go_memory_mechanisms.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = json.loads((ROOT / args.results).read_text())
    for name, expected in data["source_sha256"].items():
        revision = data.get("source_revisions", {}).get(name)
        source = ((ROOT / name).read_bytes() if revision is None else
                  subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT))
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f"source mismatch: {name}")
    rows = []
    for run in data["runs"]:
        if run["method"] not in ("orthogonal_guard", "conditional_external_guard"):
            continue
        path = ROOT / args.checkpoints / f'{run["method"]}-{run["seed"]}.pt'
        rows.append(audit_checkpoint(path, run))
    result = {"version": 2, "checkpoints": rows, "eviction_audit": eviction_audit(),
              "source_sha256": data["source_sha256"],
              "source_revisions": data.get("source_revisions", {}),
              "analysis_driver_sha256": hashlib.sha256((ROOT / "experiments/go_memory_research.py").read_bytes()).hexdigest(),
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "interpretation": "rank controls are local post-training probes on seen B windows; no long-run rank claim"}
    (ROOT / args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checkpoints_audited": len(rows), "candidate_steps": sum(len(r["local_steps"]) for r in rows),
                      "eviction_audit": result["eviction_audit"]}, indent=2))


if __name__ == "__main__":
    main()
