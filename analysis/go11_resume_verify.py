"""Verify full 10x10 self-play execution against a 37-ply pause/continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go11_continual import run, save_json, source_hashes


def compare(left, right, *, path="root", counters=None):
    counters = {"tensors": 0, "fields": 0, "nonidentical_tensors": 0,
                "max_tensor_absolute_error": 0.0, "max_scalar_absolute_error": 0.0} if counters is None else counters
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6, msg=path)
        counters["tensors"] += 1
        counters["nonidentical_tensors"] += int(not torch.equal(left, right))
        if left.numel():
            counters["max_tensor_absolute_error"] = max(counters["max_tensor_absolute_error"], float((left.double()-right.double()).abs().max()))
    elif isinstance(left, dict):
        assert set(left) == set(right), path
        for key in left:
            if key not in ("seconds", "refresh_seconds"):
                compare(left[key], right[key], path=path+"."+str(key), counters=counters)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for i, (a,b) in enumerate(zip(left,right)):
            compare(a,b,path=path+f"[{i}]",counters=counters)
    elif isinstance(left, float):
        error = abs(left-right)
        assert error <= 1e-6 + 1e-6*abs(right), (path,left,right)
        counters["max_scalar_absolute_error"] = max(counters["max_scalar_absolute_error"],error)
        counters["fields"] += 1
    else:
        assert left == right, (path,left,right)
        counters["fields"] += 1
    return counters


def main(args):
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = argparse.Namespace(seed=3701, method="selfplay_cone6", size=10, directory=str(directory),
        output="analysis/results/go11_development.json", warmup_games=24, epochs=24,
        train_games=1, eval_games=2, window=16, simulations=32, eval_simulations=64,
        search_depth=16, max_plies=37, prepare_only=False, confirmation=False)
    work_path = directory / "work-selfplay_cone6-3701.pt"
    resumed_path = directory / "resumed-verified.pt"
    if not resumed_path.exists():
        if not work_path.exists():
            run(settings)
        paused = torch.load(work_path, weights_only=True)
        assert not paused["completed"] and paused["active_game"] is not None
        assert len(paused["active_game"]["board"]["move_history"]) == 37
        assert len(paused["session"]["pending"]) == 5
        assert paused["source"] == source_hashes()
        settings.max_plies = 0
        run(settings)
        work_path.rename(resumed_path)
        run(settings)
    resumed = torch.load(resumed_path, weights_only=True)
    continuous = torch.load(work_path, weights_only=True)
    assert resumed["completed"] and continuous["completed"]
    assert resumed["source"] == continuous["source"] == source_hashes()
    assert resumed["config"] == continuous["config"]
    counters = compare(resumed["session"], continuous["session"])
    compare(resumed["games"], continuous["games"], counters=counters)
    compare(resumed["probes"], continuous["probes"], counters=counters)
    base = torch.load(directory / "base-3701.pt", weights_only=True)
    assert not bool(base["base"]["value.weight"].count_nonzero())
    assert not bool(base["base"]["value.bias"].count_nonzero())
    result = {"verified": True, "pause_ply": 37, "pending_at_pause": 5,
        "completed_games_each": len(resumed["games"]), "comparison": counters,
        "terminal_labels_each": sum(g["terminal_labels"] for g in resumed["games"]),
        "replayed_opening_observations_each": sum(g["replayed_context"] for g in resumed["games"]),
        "parameter_updates_each": resumed["parameter_version"] if "parameter_version" in resumed else int(resumed["session"]["agent"]["online_task_updates"]),
        "float_tolerances": {"rtol": 1e-6, "atol": 1e-6},
        "bitwise_identical": counters["nonidentical_tensors"] == 0 and counters["max_scalar_absolute_error"] == 0,
        "moves_and_integer_counters_identical": True,
        "source": source_hashes(), "excluded": ["seconds", "refresh_seconds"],
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(result, args.output)
    print(json.dumps(result | {"source": "recorded"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", default=str(ROOT / "analysis/results/go11_resume_verify.json"))
    torch.set_num_threads(1)
    main(parser.parse_args())
