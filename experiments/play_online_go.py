"""Continue a trained small Go model with genuine predict-then-feedback games.

Example:
    python3 experiments/play_online_go.py \
        --checkpoint checkpoints/orthogonal_go_online/orthogonal_ogd-17.pt \
        --games 8 --save-to checkpoints/live_go.pt

Resuming the saved checkpoint continues at the next game with the exact model
and OGD state. The benchmark checkpoint configuration supplies the architecture.
Only fully terminated games have win labels; move-cap positions have area scores.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.orthogonal_go_online import build_agent
from mqr import GoBoard, GoOnlineSession, GoVectorEncoder, HeuristicGoTeacher
from mqr.sayuri import SayuriGTPClient, SayuriTeacher, discover_sayuri_paths


def play_one_game(
    session: GoOnlineSession, *, size: int, komi: float, game_index: int,
    seed: int, max_moves: int, teacher: object,
) -> dict:
    """Learn on the student's evolving trajectory with no future labels."""
    board = GoBoard(size, komi=komi)
    student_color = 1 if game_index % 2 == 0 else -1
    rng = random.Random(seed + 10000 + game_index // 2)
    for _ in range(2):
        board.play(rng.choice(board.legal_moves(include_pass=False)))
    session.reset_game()
    start_version = int(session.agent.online_parameter_version)
    events = []
    while not board.game_over and len(board.move_history) < max_moves:
        result = session.step(board, teacher)
        # Opponent and teacher share one fixed policy in this live demonstration.
        # Student moves always come from the pre-update model prediction.
        student_turn = board.to_play == student_color
        action = result["action"] if student_turn else result["teacher_action"]
        events.append({
            "ply": len(board.move_history), "student_turn": student_turn,
            "prediction_version": result["parameter_version"],
            "raw_action": result["raw_action"], "played_action": action,
            "teacher_action": result["teacher_action"], "raw_legal": result["raw_legal"],
            "prequential_nll": float(-result["policy_logits"][0, result["teacher_action"]]),
        })
        board.play(action)
    session.flush()
    session.reset_game()
    return {
        "game_index": game_index, "student_color": student_color,
        "terminated": board.game_over, "truncated": not board.game_over,
        "area_margin_at_stop": student_color * board.score()["margin_black"],
        "win": (float(board.winner() == student_color) + 0.5 * float(board.winner() == 0))
               if board.game_over else None,
        "parameter_updates": int(session.agent.online_parameter_version) - start_version,
        "prequential_nll": float(np.mean([item["prequential_nll"] for item in events])),
        "raw_legality": float(np.mean([item["raw_legal"] for item in events if item["student_turn"]])),
        "move_history": board.move_history, "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--teacher", choices=("heuristic", "sayuri"), default="heuristic")
    parser.add_argument("--save-to", type=Path, default=ROOT / "checkpoints/live_go.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/results/live_online_go.json")
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("games must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    architecture = SimpleNamespace(**config)
    method, seed = checkpoint["method"], checkpoint["seed"]
    if method == "frozen":
        parser.error("the frozen control cannot perform online updates")
    agent = build_agent(architecture, method, seed)
    encoder = GoVectorEncoder(config["size"], 3 * config["size"]**2 + 6, projection_mode="identity")
    session = GoOnlineSession(agent, encoder, update_every=config["window"])
    session.load_state_dict(checkpoint["session"])
    first_game = int(checkpoint.get("completed_live_games", 0))
    client = None
    teacher = HeuristicGoTeacher()
    if args.teacher == "sayuri":
        binary, weights = discover_sayuri_paths()
        client = SayuriGTPClient(binary, weights, board_size=config["size"], komi=config["komi"],
                                playouts=config["sayuri_playouts"], threads=1)
        teacher = SayuriTeacher(client, mode="policy")
    report = {
        "schema_version": 1, "mode": "live_supervised_online_go", "method": method,
        "teacher": args.teacher, "seed": seed, "readout": config.get("readout", "global"),
        "initial_completed_games": first_game, "predict_before_teacher_feedback": True,
        "external_legal_mask": True, "value_reward_training": False,
        "held_out_evaluation": False, "games": [],
    }
    try:
        for index in range(first_game, first_game + args.games):
            game = play_one_game(
                session, size=config["size"], komi=config["komi"], game_index=index,
                seed=seed, max_moves=config["max_game_moves"], teacher=teacher,
            )
            report["games"].append(game)
            checkpoint.update(session=session.state_dict(), completed_live_games=index + 1)
            args.save_to.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.save_to.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(args.save_to)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary_report = args.output.with_suffix(".tmp")
            temporary_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary_report.replace(args.output)
            print(f"game={index} updates={game['parameter_updates']} "
                  f"NLL={game['prequential_nll']:.4f} raw_legal={game['raw_legality']:.3f} "
                  f"terminated={game['terminated']}", flush=True)
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
