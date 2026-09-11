"""Continue a signed-query/outcome checkpoint with real online game feedback."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from go_policy_research import build, play_games, session_for, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--phase", choices=("a", "b"), default="b")
    parser.add_argument("--simulations", type=int, default=0)
    parser.add_argument("--save", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    if options.games < 1 or options.simulations < 0:
        parser.error("games must be positive and simulations non-negative")
    destination, output = Path(options.save), Path(options.output)
    if destination.exists() or output.exists():
        raise FileExistsError("use new destinations; existing checkpoints and reports are preserved")
    torch.set_num_threads(1)
    original = Path(options.checkpoint)
    checkpoint = torch.load(original, map_location="cpu", weights_only=True)
    if checkpoint["source_sha256"] != sources():
        raise ValueError("checkpoint implementation differs; use its recorded source version")
    args = argparse.Namespace(**checkpoint["config"])
    method, seed = checkpoint["method"], checkpoint["seed"]
    agent, encoder = build(args, method, seed)
    session = session_for(args, agent, encoder, method)
    session.load_state_dict(checkpoint["session"])
    args.train_games = options.games
    start = int(checkpoint.get("continuation_games", 0))
    report = play_games(args, session, seed + 31000, learn=True, phase=options.phase,
                        simulations=options.simulations, start_game=start)
    report.update(method=method, seed=seed, first_game=start, next_game=start + options.games,
                  parent_sha256=hashlib.sha256(original.read_bytes()).hexdigest())
    checkpoint["session"] = session.state_dict()
    checkpoint["continuation_games"] = start + options.games
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "method", "seed", "wins", "terminated_games", "feedback_positions", "replayed_positions",
        "terminal_value_labels", "max_anchor_kl", "first_game", "next_game",
    )}))


if __name__ == "__main__":
    main()
