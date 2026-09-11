"""Post-hoc readout diagnosis; probes do not change the evaluated checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.go_memory_research import build, materialize, recall_forward
from mqr.go_history_tasks import generate_reachable_history_pairs


def ridge_probe(train, labels, test, targets):
    """Fixed ridge=1; standardization uses training data only."""
    train, test = train.double(), test.double()
    mean, scale = train.mean(dim=0), train.std(dim=0).clamp_min(1e-6)
    train, test = (train - mean) / scale, (test - mean) / scale
    train = torch.cat((train, torch.ones(len(train), 1, dtype=train.dtype)), dim=1)
    test = torch.cat((test, torch.ones(len(test), 1, dtype=test.dtype)), dim=1)
    onehot = torch.nn.functional.one_hot(labels, num_classes=25).double()
    penalty = torch.eye(train.size(1), dtype=train.dtype)
    penalty[-1, -1] = 0
    weights = torch.linalg.solve(train.T @ train + penalty, train.T @ onehot)
    logits = test @ weights
    choices = targets.reshape(-1, 2).repeat_interleave(2, dim=0)
    predicted = choices.gather(1, logits.gather(1, choices).argmax(dim=1, keepdim=True)).squeeze(1)
    return float((predicted == targets).float().mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default="checkpoints/go_constraint_v1")
    parser.add_argument("--output", default="analysis/results/go_constraint_readout_diagnosis.json")
    options = parser.parse_args()
    torch.set_num_threads(1)
    rows = []
    for seed in (401, 409, 419, 431, 443):
        path = ROOT / options.checkpoints / f"history-transport_full-{seed}.pt"
        original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        args = argparse.Namespace(**checkpoint["config"])
        agent, encoder = build(args, "conditional", seed, history_task=True)
        agent.load_state_dict(checkpoint["model_state"])
        datasets = [
            materialize(generate_reachable_history_pairs(count, seed=seed + offset, size=5, moves=12), encoder)
            for count, offset in ((96, 11000), (64, 12000))
        ]
        vectors, outputs = [], []
        with torch.no_grad():
            for features, _ in datasets:
                logits, state = recall_forward(agent, features, return_state=True)
                vectors.append(torch.cat(state.rings, dim=1))
                outputs.append(logits)
        (train_x, train_y), (test_x, test_y) = datasets
        choices = test_y.reshape(-1, 2).repeat_interleave(2, dim=0)
        candidate_logits = outputs[1].gather(1, choices)
        margins = candidate_logits[:, 0] - candidate_logits[:, 1]
        probabilities = outputs[1].softmax(dim=1)
        predicted = choices.gather(1, candidate_logits.argmax(dim=1, keepdim=True)).squeeze(1)
        row = {
            "seed": seed, "checkpoint_sha256": original_hash,
            "model_accuracy": float((predicted == test_y).float().mean()),
            "paired_state_distance": float((vectors[1][::2] - vectors[1][1::2]).norm(dim=1).mean()),
            "paired_all_logit_distance": float((outputs[1][::2] - outputs[1][1::2]).norm(dim=1).mean()),
            "paired_candidate_margin_difference": float((margins[::2] - margins[1::2]).abs().mean()),
            "paired_point_probability_l1": float((probabilities[::2] - probabilities[1::2]).abs().sum(dim=1).mean()),
            "board_only_ridge_accuracy": ridge_probe(train_x[:, -1], train_y, test_x[:, -1], test_y),
            "ring_and_board_ridge_accuracy": ridge_probe(
                torch.cat((vectors[0], train_x[:, -1]), dim=1), train_y,
                torch.cat((vectors[1], test_x[:, -1]), dim=1), test_y,
            ),
        }
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "post_hoc": True, "checkpoint_parameters_unchanged": True, "probe_ridge": 1.0,
        "probe_fit": "96 training pairs only; fixed standardization and ridge; 64 held-out test pairs",
        "interpretation": "Separate train-only linear readouts diagnose retained information. They are not an improvement to the online agent.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "runs": rows,
    }
    (ROOT / options.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
