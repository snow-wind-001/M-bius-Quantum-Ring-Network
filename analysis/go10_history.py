"""Separate history diagnostics; these accuracies are not game win rates."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go10_continual import build, save_json, source_hashes
from mqr.go_history_tasks import generate_ko_history_pairs, generate_reachable_history_pairs
from mqr.go_memory import GoHistoryEncoder


def materialize(task, seed, train_pairs=96, test_pairs=64):
    encoder = GoHistoryEncoder(10)
    if task == "ko":
        pairs = generate_ko_history_pairs(train_pairs + test_pairs, seed=seed + 71000, size=10)
    else:
        pairs = generate_reachable_history_pairs(train_pairs + test_pairs, seed=seed + 72000,
                                                 size=10, moves=24)
    xs, labels, queries, signatures = [], [], [], []
    for pair in pairs:
        signatures.append(hashlib.sha256(repr(tuple(tuple(board.board) for history in pair.histories
                                                    for board in history)).encode()).hexdigest())
        for index, history in enumerate(pair.histories):
            xs.append(torch.cat([encoder.encode_board(board) for board in history]))
            labels.append(float(pair.legal[index]) if task == "ko" else pair.targets[index])
            queries.append(pair.recapture if task == "ko" else 0)
    if len(set(signatures)) != len(signatures):
        raise RuntimeError("history diagnostic contains duplicated trajectories")
    return torch.stack(xs, dim=1), torch.tensor(labels), torch.tensor(queries), signatures


def forward(agent, xs, *, reset=False, credit=None):
    state = agent.core.zero_state(xs.size(1), device=xs.device, dtype=xs.dtype)
    for index, x in enumerate(xs):
        if reset:
            state = agent.core.zero_state(xs.size(1), device=xs.device, dtype=xs.dtype)
        elif credit and index and index % credit == 0:
            state = state.detached()
        output, state = agent._transition(x, state, slow_write=True)
    return output


def loss_and_prediction(task, output, labels, queries):
    if task == "ko":
        logits = output.legality_logits.gather(1, queries[:, None]).squeeze(1)
        return F.binary_cross_entropy_with_logits(logits, labels.float()), (logits >= 0).long()
    return F.cross_entropy(output.placement_logits, labels.long()), output.placement_logits.argmax(1)


@torch.no_grad()
def evaluate(task, agent, xs, labels, queries, *, reset=False, swap=False):
    if swap:
        # Every pair's final observation is equal; swapping only the historical
        # prefix reverses the correct history-dependent label.
        xs = xs.clone()
        order = torch.arange(xs.size(1)).reshape(-1, 2).flip(1).flatten()
        xs[:-1] = xs[:-1, order]
    correct, losses = [], []
    for indices in torch.arange(len(labels)).split(32):
        output = forward(agent, xs[:, indices], reset=reset)
        loss, prediction = loss_and_prediction(task, output, labels[indices], queries[indices])
        correct.extend((prediction == labels[indices].long()).tolist())
        losses.append(float(loss))
    return {"accuracy": float(np.mean(correct)), "loss": float(np.mean(losses))}


def run(args):
    base = torch.load(Path(args.directory) / f"base-{args.seed}.pt", weights_only=True)
    if base["source"] != source_hashes():
        raise RuntimeError("base source mismatch")
    results = []
    for task in ("ko", "recall24"):
        xs, labels, queries, signatures = materialize(task, args.seed)
        boundary = 192
        for method in ("orthogonal", "identity", "stateless", "short_credit"):
            if task == "ko" and method == "short_credit":
                continue
            started = time.perf_counter()
            agent, _ = build(argparse.Namespace(size=10), "identity" if method == "identity" else "optimized", args.seed)
            agent.spatial_skip_heads.load_state_dict(base["base"])
            optimizer = torch.optim.Adam([p for p in agent.parameters() if p.requires_grad], lr=0.003)
            generator = torch.Generator().manual_seed(args.seed + 73000)
            epochs = 48
            for _ in range(epochs):
                # Shuffle complete pairs to preserve balanced aliases within a batch.
                order = torch.randperm(boundary // 2, generator=generator)
                for pair_indices in order.split(16):
                    indices = torch.stack((2 * pair_indices, 2 * pair_indices + 1), dim=1).flatten()
                    output = forward(agent, xs[:, indices], reset=method == "stateless",
                                     credit=8 if method == "short_credit" else None)
                    loss, _ = loss_and_prediction(task, output, labels[indices], queries[indices])
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()
            held = (xs[:, boundary:], labels[boundary:], queries[boundary:])
            result = {"task": task, "method": method, "seed": args.seed, "epochs": epochs,
                      "training_pairs": boundary // 2, "test_pairs": len(labels[boundary:]) // 2,
                      "credit_steps": 1 if method == "stateless" else (1 if method == "short_credit" else len(xs)),
                      "full": evaluate(task, agent, *held, reset=method == "stateless"),
                      "reset": evaluate(task, agent, *held, reset=True),
                      "swap": evaluate(task, agent, *held, swap=True, reset=method == "stateless"),
                      "seconds": time.perf_counter() - started}
            results.append(result)
            print(json.dumps(result), flush=True)
        split = {"train": signatures[:boundary // 2], "test": signatures[boundary // 2:]}
        for result in results:
            if result["task"] == task:
                result["split_sha256"] = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    save_json({"completed": True, "source": source_hashes(), "diagnostic_source_sha256":
               hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "results": results,
               "scope": "separate 48-epoch supervised diagnostic; not online game training or Go strength"}, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--directory", default=str(ROOT / "checkpoints/go10_v1"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args)
