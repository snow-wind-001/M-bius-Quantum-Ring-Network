"""Continue a constraint-transport checkpoint against the local Go teacher."""

import argparse
import json
from pathlib import Path

import torch

from go_constraint_research import build, play_games, session_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--phase", choices=("a", "b"), default="a")
    parser.add_argument("--save", required=True)
    parser.add_argument("--output", default="analysis/results/go_constraint_continuation.json")
    options = parser.parse_args()
    if options.games < 1:
        parser.error("--games must be positive")
    torch.set_num_threads(1)
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
    args = argparse.Namespace(**checkpoint["config"])
    method, seed = checkpoint["method"], checkpoint["seed"]
    agent, encoder = build(args, "conditional_guard", seed)
    session = session_for(args, agent, encoder, method)
    session.load_state_dict(checkpoint["session"])
    args.train_games = options.games
    start = int(checkpoint.get("continuation_games", 0))
    report = play_games(args, session, seed + 31000, learn=True, phase=options.phase, start_game=start)
    report.update(method=method, seed=seed, first_game=start, next_game=start + options.games)
    checkpoint["session"] = session.state_dict()
    checkpoint["continuation_games"] = start + options.games
    destination = Path(options.save)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "games"}, indent=2))


if __name__ == "__main__":
    main()
