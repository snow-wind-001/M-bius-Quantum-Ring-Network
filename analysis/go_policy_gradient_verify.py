"""Verify full-size trained query/spatial checkpoint adjoints in float64."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go_policy_research import build, sources
from mqr import GoBoard, HeuristicGoTeacher
from mqr.constraint_transport import terminal_gradients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*", default=[
        "checkpoints/go_policy_v1/query-503.pt", "checkpoints/go_policy_v1/spatial-503.pt",
    ])
    parser.add_argument("--output", default="analysis/results/go_policy_gradient_verify.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {"verified": False, "post_hoc_identity_check": True, "source_sha256": sources(), "runs": []}
    for name in args.checkpoints:
        path = ROOT / name
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["source_sha256"] == sources()
        agent, encoder = build(argparse.Namespace(**checkpoint["config"]), checkpoint["method"], checkpoint["seed"])
        agent.load_state_dict(checkpoint["session"]["agent"])
        agent.double()
        board, features = GoBoard(5, komi=2.5), []
        for index in range(13):
            features.append(encoder.encode_board(board).double())
            if index < 12:
                board.play(HeuristicGoTeacher().select_move(board))
        x = torch.cat(features)
        objective = lambda output: torch.stack((output.policy_logits[0, 0], output.policy_logits[0, 24]))
        actual = terminal_gradients(agent, x, objective)
        reference = terminal_gradients(agent, x, objective, backend="autograd")
        maximum = float((actual.jacobian - reference.jacobian).abs().max())
        relative = float((actual.jacobian - reference.jacobian).norm() / reference.jacobian.norm())
        assert maximum < 2e-10 and relative < 1e-9
        result["runs"].append({
            "method": checkpoint["method"], "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "observations": 13, "dtype": "float64", "gradient_max_abs_error": maximum,
            "gradient_relative_error": relative, "transport_diagnostics": actual.diagnostics,
        })
    result["verified"] = True
    (ROOT / args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verified": True, "max_abs_error": max(r["gradient_max_abs_error"] for r in result["runs"])}))


if __name__ == "__main__":
    main()
