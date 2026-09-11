"""Compare two continuous online games with save/restore between the same games."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def compare(left, right, path="", counts=None):
    if counts is None:
        counts = {"tensors": 0, "max_abs_error": 0.0}
    if path.rsplit("/", 1)[-1] == "refresh_seconds":
        return counts  # wall-clock measurements are not algorithmic state
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), f"tensor continuation mismatch: {path}"
        counts["tensors"] += 1
    elif isinstance(left, dict):
        assert set(left) == set(right), path
        for key in left:
            compare(left[key], right[key], path + "/" + str(key), counts)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, path + "/" + str(index), counts)
    else:
        assert left == right, f"continuation mismatch: {path}"
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--directory", default="checkpoints/go_policy_resume")
    parser.add_argument("--output", default="analysis/results/go_policy_resume_verify.json")
    args = parser.parse_args()
    original = ROOT / args.checkpoint
    directory = ROOT / args.directory
    directory.mkdir(parents=True, exist_ok=True)

    def command(source, name, games):
        return [sys.executable, "experiments/play_policy_go.py", str(source), "--games", str(games),
                "--phase", "b", "--save", str(directory / f"{name}.pt"),
                "--output", str(directory / f"{name}.json")]

    jobs = [(name, subprocess.Popen(command(original, name, games), cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True))
            for name, games in (("full", 2), ("part1", 1))]
    for name, process in jobs:
        output, _ = process.communicate()
        assert process.returncode == 0, output
        print(name + ": " + output.strip(), flush=True)
    second = subprocess.run(command(directory / "part1.pt", "part2", 1), cwd=ROOT,
                            capture_output=True, text=True)
    assert second.returncode == 0, second.stdout + second.stderr
    print("part2: " + second.stdout.strip(), flush=True)
    full = torch.load(directory / "full.pt", map_location="cpu", weights_only=True)
    split = torch.load(directory / "part2.pt", map_location="cpu", weights_only=True)
    compared = compare(full["session"], split["session"])
    assert full["continuation_games"] == split["continuation_games"] == 2
    reports = {name: json.loads((directory / f"{name}.json").read_text()) for name in ("full", "part1", "part2")}
    assert reports["full"]["games"] == reports["part1"]["games"] + reports["part2"]["games"]
    for key in ("feedback_positions", "replayed_positions", "terminal_value_labels"):
        assert reports["full"][key] == reports["part1"][key] + reports["part2"][key]
    evidence = {
        "verified": True, "method": full["method"], "seed": full["seed"], **compared,
        "continuous_games": 2, "split_games": [1, 1], "terminal_value_labels": reports["full"]["terminal_value_labels"],
        "replayed_positions": reports["full"]["replayed_positions"],
        "max_anchor_kl": reports["full"]["max_anchor_kl"],
        "parent_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "ignored_state_fields": ["refresh_seconds"], "source_sha256": full["source_sha256"],
    }
    (ROOT / args.output).write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({key: value for key, value in evidence.items() if key != "source_sha256"}))


if __name__ == "__main__":
    main()
