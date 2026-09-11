"""Compare full execution with a mid-game pause and continuation, without truncation."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go10_continual import run, save_json


def compare(left, right, *, path="root", counters=None):
    counters = {"tensors": 0, "fields": 0} if counters is None else counters
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=path)
        counters["tensors"] += 1
    elif isinstance(left, dict):
        assert set(left) == set(right), path
        for key in left:
            if key in ("seconds", "refresh_seconds"):
                continue
            compare(left[key], right[key], path=path + "." + str(key), counters=counters)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, path=path + f"[{index}]", counters=counters)
    else:
        assert left == right, (path, left, right)
        counters["fields"] += 1
    return counters


def main(args):
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "work-optimized-2601.pt").exists():
        raise RuntimeError("use a fresh resume-verification directory")
    settings = argparse.Namespace(seed=2601, method="optimized", size=10, directory=str(directory),
        output=str(directory / "result.json"), warmup_games=4, epochs=4, train_games=1, eval_games=2,
        window=16, simulations=16, search_depth=12, max_plies=0, prepare_only=False, confirmation=False)
    run(settings)
    work_path = directory / "work-optimized-2601.pt"
    continuous = torch.load(work_path, weights_only=True)
    work_path.rename(directory / "continuous.pt")
    settings.max_plies = 37
    run(settings)
    paused = torch.load(work_path, weights_only=True)
    assert not paused["completed"] and paused["active_game"] is not None
    assert not paused["active_game"]["board"]["game_over"]
    assert paused["session"]["pending"], "pause must preserve unfinished feedback"
    settings.max_plies = 0
    run(settings)
    resumed = torch.load(work_path, weights_only=True)
    assert continuous["completed"] and resumed["completed"]
    counters = compare(continuous["session"], resumed["session"])
    compare(continuous["games"], resumed["games"], counters=counters)
    compare(continuous["probes"], resumed["probes"], counters=counters)
    result = {"passed": True, "pause_ply": len(paused["active_game"]["board"]["move_history"]),
              "pending_at_pause": len(paused["session"]["pending"]),
              "completed_games_each": len(resumed["games"]),
              "terminal_labels_each": sum(g["terminal_labels"] for g in resumed["games"]),
              "parameter_updates_each": int(resumed["session"]["agent"]["online_task_updates"]),
              "comparison": counters, "excluded": ["seconds", "refresh_seconds"],
              "source": resumed["source"], "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(result, args.output)
    print(json.dumps(result | {"source": "recorded"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", default=str(ROOT / "analysis/results/go10_resume_verify.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
