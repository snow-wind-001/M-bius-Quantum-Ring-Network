"""Small Go model + signed rings + trajectory OGD, with measured controls.

This is a teacher-supervised online adaptation study, not AlphaZero training.
The default teacher is a local tactical heuristic. Optional Sayuri stays in a
separate process. Test games never update weights; move-limit area scores are
reported separately from wins of games terminated by two passes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mqr import (
    GoBoard, GoLossWeights, GoMultiHeadOutput, GoOnlineSession, GoSpatialSkipHeads,
    GoVectorEncoder, HeuristicGoTeacher, MultiTimescaleGRUCore, MultiTimescaleMQR,
    TemporalUtilityMQRAgent, SpatialRingGoAgent, consolidate_task_memory,
)
from mqr.agent_baselines import estimate_core_forward_macs
from mqr.go_agent import GoAgentTrajectory, generate_go_agent_trajectories, legality_target
from mqr.sayuri import SayuriGTPClient, SayuriTeacher, discover_sayuri_paths


METHODS = (
    "frozen", "spatial_sgd", "stateless", "identity", "unistochastic",
    "orthogonal", "orthogonal_ogd", "identity_ogd", "gru",
)
LOSS_WEIGHTS = GoLossWeights(value=0.0, illegal_mass=0.2)


def build_agent(args: argparse.Namespace, method: str, seed: int) -> TemporalUtilityMQRAgent:
    """Match non-recurrent blocks and use the same common spatial base."""
    torch.manual_seed(seed)
    input_dim = 3 * args.size**2 + 6
    rates = (1.0, 0.15, 0.03)
    mode = method.removesuffix("_ogd")
    if mode == "gru":
        core = MultiTimescaleGRUCore(
            input_dim, args.ring_dim, args.latent_dim,
            leak_rates=rates, input_rank=8,
        )
    else:
        if mode in ("frozen", "spatial_sgd", "stateless"):
            rates = (1.0, 1.0, 1.0)
            mode = "identity"
        # Under isotropic uncorrelated unit-variance forcing this scaling
        # balances stationary variance. It is not an energy guarantee for
        # correlated Go observations or nonlinear readout.
        writes = tuple(math.sqrt(1.0 - (1.0 - rate)**2) for rate in rates)
        core = MultiTimescaleMQR(
            input_dim, args.ring_dim, args.latent_dim,
            leak_rates=rates, write_scales=writes, injection_rank=8,
            state_activation="none", transition_mode=mode,
            transition_structure="cyclic_givens", base_unitary_init="random",
            base_unitary_seed=seed + 31, readout_bias=True,
        )
    # Core-specific parameter draws must not change shared head initialization.
    torch.manual_seed(seed + 100)
    agent_class = SpatialRingGoAgent if getattr(args, "readout", "global") == "spatial" else TemporalUtilityMQRAgent
    agent = agent_class(
        input_dim, board_size=args.size, ring_dim=args.ring_dim,
        latent_dim=args.latent_dim, core=core, task_lr=args.lr,
        max_update_norm=0.15, max_trace_horizon=args.window,
        ogd_max_rank=args.ogd_rank if method.endswith("_ogd") else 0,
        spatial_skip_channels=args.channels, legality_policy_scale=1.0,
        loss_weights=LOSS_WEIGHTS,
    )
    for head in (agent.placement_head, agent.legality_head, agent.pass_head, agent.value_head):
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    for parameter in agent.utility_gate.parameters():
        parameter.requires_grad_(False)
    if method in ("frozen", "spatial_sgd"):
        for parameter in agent.parameters():
            parameter.requires_grad_(False)
    for parameter in agent.spatial_skip_heads.parameters():
        parameter.requires_grad_(method == "spatial_sgd")
    return agent


def _base_output(base: GoSpatialSkipHeads, features: torch.Tensor) -> GoMultiHeadOutput:
    placement, legality, passing, value = base(features)
    return GoMultiHeadOutput(
        placement, legality, passing, value.tanh(), features.new_zeros(features.size(0), 1),
        legality_policy_scale=1.0,
    )


def pretrain_base(
    args: argparse.Namespace, data: Sequence[GoAgentTrajectory], encoder: GoVectorEncoder,
    seed: int,
) -> GoSpatialSkipHeads:
    """Train one common tiny spatial network on an independent warmup stream."""
    torch.manual_seed(seed + 200)
    base = GoSpatialSkipHeads(encoder.output_dim, args.size, args.channels)
    torch.nn.init.constant_(base.pass_decision.bias, -math.log(args.size**2))
    examples = [example for trajectory in data for example in trajectory.examples]
    features = torch.cat([encoder.encode_board(example.board) for example in examples])
    actions = torch.tensor([example.target_action for example in examples])
    legality = torch.cat([legality_target(example.board) for example in examples])
    optimizer = torch.optim.Adam(base.parameters(), lr=0.01)
    loss_agent = build_agent(args, "frozen", seed)
    generator = torch.Generator().manual_seed(seed + 201)
    for _ in range(args.pretrain_epochs):
        order = torch.randperm(len(examples), generator=generator)
        for batch in order.split(64):
            losses = loss_agent.compute_go_loss(
                _base_output(base, features[batch]), actions[batch],
                legality_target=legality[batch],
            )
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
    return base.eval()


def _data_hash(data: Sequence[GoAgentTrajectory]) -> str:
    digest = hashlib.sha256()
    for trajectory in data:
        digest.update(b"episode\0")
        for example in trajectory.examples:
            board = example.board
            digest.update(repr((board.board, board.to_play, board.move_history,
                                sorted(board.position_history), example.target_action)).encode())
    return digest.hexdigest()


def datasets(args: argparse.Namespace, seed: int, teacher: Any) -> Dict[str, List[GoAgentTrajectory]]:
    """Split by seeded games, with separate warmup, train, and test streams."""
    common = dict(size=args.size, komi=args.komi, teacher=teacher,
                  recorded_moves=args.moves, random_move_probability=0.5)
    result = {}
    for name, offset, count, prefix in (
        ("warmup", 1000, args.warmup_games, 0),
        ("train_a", 2000, args.train_games, 0),
        ("train_b", 3000, args.train_games, args.size**2 // 2),
        ("test_a", 4000, args.test_games, 0),
        ("test_b", 5000, args.test_games, args.size**2 // 2),
    ):
        result[name] = generate_go_agent_trajectories(
            count, seed=seed + offset, start_random_moves=prefix, task=name, **common,
        )
    train_hashes = {_data_hash([trace]) for name in ("warmup", "train_a", "train_b")
                    for trace in result[name]}
    for name in ("test_a", "test_b"):
        if any(_data_hash([trace]) in train_hashes for trace in result[name]):
            raise RuntimeError("a held-out trajectory exactly duplicates training data")
    return result


def _session(agent: TemporalUtilityMQRAgent, encoder: GoVectorEncoder, window: int) -> GoOnlineSession:
    return GoOnlineSession(agent, encoder, update_every=window)


@torch.no_grad()
def evaluate(
    agent: TemporalUtilityMQRAgent, encoder: GoVectorEncoder,
    data: Sequence[GoAgentTrajectory], *, reset_history: bool = False,
) -> Dict[str, float]:
    """Evaluate without changing weights, OGD, or the live training stream."""
    states = copy.deepcopy(agent._stream_states)
    histories = copy.deepcopy(agent._feature_history)
    observations = agent.online_observations.clone()
    session = _session(agent, encoder, 1)
    losses, nlls, raw_legal, correct, masked_correct, illegal_mass = [], [], [], [], [], []
    try:
        for trajectory in data:
            session.reset_game()
            for example in trajectory.examples:
                if reset_history:
                    session.reset_game()
                result = session.observe(example.board, learn=False)
                target = torch.tensor([example.target_action])
                legal = legality_target(example.board)
                losses.append(float(agent.compute_go_loss(
                    result["output"], target, legality_target=legal,
                )["total"]))
                nlls.append(float(-result["policy_logits"][0, example.target_action]))
                raw_legal.append(float(result["raw_legal"]))
                correct.append(float(result["raw_action"] == example.target_action))
                masked_correct.append(float(result["action"] == example.target_action))
                illegal_mass.append(float((result["policy_logits"].exp()[0, :-1] * (1.0 - legal[0])).sum()))
    finally:
        agent._stream_states = states
        agent._feature_history = histories
        agent.online_observations.copy_(observations)
    return {
        "loss": float(np.mean(losses)), "policy_nll": float(np.mean(nlls)),
        "raw_legality": float(np.mean(raw_legal)), "teacher_agreement": float(np.mean(correct)),
        "masked_teacher_agreement": float(np.mean(masked_correct)),
        "illegal_probability_mass": float(np.mean(illegal_mass)), "positions": len(losses),
    }


def train_phase(
    agent: TemporalUtilityMQRAgent, encoder: GoVectorEncoder,
    data: Sequence[GoAgentTrajectory], args: argparse.Namespace,
) -> Dict[str, Any]:
    session = _session(agent, encoder, args.window)
    nlls, correct, raw_legal, timings, updates = [], [], [], [], []
    peak_online_bytes = 0
    started = time.perf_counter()
    for trajectory in data:
        session.reset_game()
        for example in trajectory.examples:
            tick = time.perf_counter()
            result = session.observe(example.board)
            # These labels become visible only after the prediction above.
            session.feedback(int(result["ticket_id"]), example.target_action)
            peak_online_bytes = max(peak_online_bytes, session.online_tensor_bytes)
            nlls.append(float(-result["policy_logits"][0, example.target_action]))
            correct.append(result["raw_action"] == example.target_action)
            raw_legal.append(result["raw_legal"])
            if session.pending_count == args.window:
                updates.append(session.flush())
            timings.append((time.perf_counter() - tick) * 1000.0)
        tail = session.flush()
        if tail is not None:
            updates.append(tail)
    session.reset_game()
    return {
        "prequential_nll": float(np.mean(nlls)),
        "prequential_agreement": float(np.mean(correct)),
        "prequential_raw_legality": float(np.mean(raw_legal)),
        "feedback_positions": len(nlls), "parameter_updates": len(updates),
        "gradient_samples": len(nlls), "seconds": time.perf_counter() - started,
        "peak_online_tensor_bytes_excluding_autograd": peak_online_bytes,
        "step_ms_p50": float(np.quantile(timings, 0.5)),
        "step_ms_p95": float(np.quantile(timings, 0.95)),
        "earliest_future_gradient_mean": float(np.mean([
            update.get("earliest_input_future_gradient_norm", 0.0) for update in updates
        ])),
        "ogd_retained_norm_mean": float(np.mean([update["ogd_retained_norm"] for update in updates])),
        "ogd_overlap_max": max(update["ogd_max_abs_overlap"] for update in updates),
    }


def anchors_from_training(
    data: Sequence[GoAgentTrajectory], encoder: GoVectorEncoder, window: int, seed: int,
) -> List[List[tuple[torch.Tensor, int]]]:
    anchors = []
    for trajectory in data:
        # Every prefix has actually been observed during training. Replaying
        # only the last window is a declared truncation, shared with updates.
        for end in range(1, len(trajectory.examples) + 1):
            anchors.append([
                (encoder.encode_board(item.board), item.target_action)
                for item in trajectory.examples[max(0, end - window):end]
            ])
    random.Random(seed + 9000).shuffle(anchors)
    return anchors


@torch.no_grad()
def play_games(
    agent: TemporalUtilityMQRAgent, encoder: GoVectorEncoder,
    args: argparse.Namespace, seed: int, opponent: Any,
) -> Dict[str, Any]:
    session = _session(agent, encoder, 1)
    records = []
    for game_index in range(args.eval_games):
        board = GoBoard(args.size, komi=args.komi)
        student_color = 1 if game_index % 2 == 0 else -1
        rng = random.Random(seed + 6000 + game_index // 2)
        for _ in range(2):
            board.play(rng.choice(board.legal_moves(include_pass=False)))
        session.reset_game()
        raw_legal, model_passes = [], []
        while not board.game_over and len(board.move_history) < args.max_game_moves:
            result = session.observe(board, learn=False)
            if board.to_play == student_color:
                action = result["action"]
                raw_legal.append(result["raw_legal"])
                model_passes.append(action == board.pass_action)
            else:
                action = int(opponent.select_move(board.copy()))
            board.play(action)
        # No forced passes or fictitious wins at the move cap.
        margin = board.score()["margin_black"] * student_color
        records.append({
            "student_color": student_color, "moves": len(board.move_history),
            "terminated": board.game_over, "truncated": not board.game_over,
            "area_margin_at_stop": margin,
            "win": (float(board.winner() == student_color) + 0.5 * float(board.winner() == 0))
                   if board.game_over else None,
            "raw_legality": float(np.mean(raw_legal)),
            "model_pass_rate": float(np.mean(model_passes)),
            "move_history": board.move_history,
        })
    session.reset_game()
    wins = [item["win"] for item in records if item["terminated"]]
    return {
        "games": records, "termination_rate": float(np.mean([item["terminated"] for item in records])),
        "terminal_win_rate": float(np.mean(wins)) if wins else None,
        "mean_area_margin_at_stop": float(np.mean([item["area_margin_at_stop"] for item in records])),
        "raw_legality": float(np.mean([item["raw_legality"] for item in records])),
        "external_legal_mask": True, "training_updates": 0,
        "dead_stone_adjudication": False,
    }


def paired_interval(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return {"mean": float(array.mean()), "ci95": None, "n": len(array)}
    rng = np.random.default_rng(909)
    means = array[rng.integers(0, len(array), (10000, len(array)))].mean(axis=1)
    return {"mean": float(array.mean()), "ci95": np.quantile(means, [0.025, 0.975]).tolist(), "n": len(array)}


def summarize(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    result = {}
    for method in runs[0]["methods"]:
        values = [run["methods"][method] for run in runs]
        result[method] = {
            "a_nll_gain": paired_interval([item["before"]["a"]["policy_nll"] - item["after_b"]["a"]["policy_nll"] for item in values]),
            "b_nll_gain": paired_interval([item["before"]["b"]["policy_nll"] - item["after_b"]["b"]["policy_nll"] for item in values]),
            "a_forgetting_nll": paired_interval([item["after_b"]["a"]["policy_nll"] - item["after_a"]["a"]["policy_nll"] for item in values]),
            "history_nll_benefit_b": paired_interval([item["reset_b"]["policy_nll"] - item["after_b"]["b"]["policy_nll"] for item in values]),
            "game_margin_gain": paired_interval([item["games_after"]["mean_area_margin_at_stop"] - item["games_before"]["mean_area_margin_at_stop"] for item in values]),
        }
    if all("orthogonal_ogd" in run["methods"] for run in runs):
        result["paired_b_nll_advantage_of_orthogonal_ogd"] = {
            control: paired_interval([
                run["methods"][control]["after_b"]["b"]["policy_nll"]
                - run["methods"]["orthogonal_ogd"]["after_b"]["b"]["policy_nll"]
                for run in runs
            ]) for control in runs[0]["methods"] if control != "orthogonal_ogd"
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--komi", type=float, default=2.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--warmup-games", type=int, default=24)
    parser.add_argument("--train-games", type=int, default=24)
    parser.add_argument("--test-games", type=int, default=16)
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--moves", type=int, default=24)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--ring-dim", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--readout", choices=("global", "spatial"), default="global")
    parser.add_argument("--ogd-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--max-game-moves", type=int, default=120)
    parser.add_argument("--teacher", choices=("heuristic", "sayuri"), default="heuristic")
    parser.add_argument("--opponent", choices=("heuristic", "sayuri"), default="heuristic")
    parser.add_argument("--sayuri-playouts", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/results/orthogonal_go_online.json")
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    for key in ("warmup_games", "train_games", "test_games", "pretrain_epochs", "moves", "window",
                "ring_dim", "latent_dim", "channels", "ogd_rank", "eval_games", "max_game_moves", "threads"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not 3 <= args.size <= 25 or args.ring_dim % 2 or args.eval_games % 2:
        parser.error("size >= 3, an even ring dimension, and paired-color game count are required")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("seeds and methods must be unique")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("learning rate must be finite and positive")
    if not math.isfinite(args.komi) or args.max_game_moves < 4:
        parser.error("komi must be finite and the game move limit at least four")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    client = None
    if "sayuri" in (args.teacher, args.opponent):
        binary, weights = discover_sayuri_paths()
        client = SayuriGTPClient(binary, weights, board_size=args.size, komi=args.komi,
                                playouts=args.sayuri_playouts, threads=1)
    teacher = HeuristicGoTeacher() if args.teacher == "heuristic" else SayuriTeacher(client, mode="policy")
    opponent = HeuristicGoTeacher() if args.opponent == "heuristic" else SayuriTeacher(client, mode="mcts")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    payload: Dict[str, Any] = {
        "schema_version": 1, "study": "signed_orthogonal_ring_online_go", "config": config,
        "protocol": {
            "mode": "exploratory_supervised_online_adaptation", "predict_before_feedback": True,
            "shared_pretrained_spatial_initialization": True,
            "base_frozen_except": ["spatial_sgd"], "value_targets_used": False,
            "split_unit": "independent_seeded_trajectory; common board positions can recur",
            "ogd_anchor_source": "previously_seen_task_a_training_windows",
            "claims_general_go_strength": False, "all_methods_resource_matched": False,
            "state_activation": "linear for ring controls", "write_scale": "sqrt(1-rho^2)",
        },
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__), ROOT / "mqr/go_online.py", ROOT / "mqr/temporal.py", ROOT / "mqr/unitary.py"]},
        "runs": [],
    }
    try:
        for seed in args.seeds:
            print(f"seed={seed}: generate independent streams and pretrain shared base", flush=True)
            data = datasets(args, seed, teacher)
            encoder = GoVectorEncoder(args.size, 3 * args.size**2 + 6, projection_mode="identity")
            base = pretrain_base(args, data["warmup"], encoder, seed)
            run: Dict[str, Any] = {"seed": seed, "data_sha256": {key: _data_hash(value) for key, value in data.items()}, "methods": {}}
            initial_probe = None
            for method in args.methods:
                agent = build_agent(args, method, seed)
                agent.spatial_skip_heads.load_state_dict(base.state_dict())
                probe = agent.preview_step(encoder.encode_board(GoBoard(args.size)), external_write=True)["policy_logits"]
                if initial_probe is None:
                    initial_probe = probe.detach().clone()
                if not torch.equal(probe, initial_probe):
                    raise RuntimeError("methods do not share the exact pretrained initial policy")
                result: Dict[str, Any] = {
                    "before": {key: evaluate(agent, encoder, data[f"test_{key}"]) for key in ("a", "b")},
                    "games_before": play_games(agent, encoder, args, seed, opponent),
                    "resources": {
                        "trainable_parameters": agent.task_parameter_count,
                        "total_parameters": sum(p.numel() for p in agent.parameters()),
                        "persistent_state_bytes_per_game": 3 * args.ring_dim * 4,
                        "core_forward_macs_estimate": estimate_core_forward_macs(agent.core),
                        "spatial_base_parameters": sum(p.numel() for p in base.parameters()),
                    },
                }
                if method != "frozen":
                    result["train_a"] = train_phase(agent, encoder, data["train_a"], args)
                result["after_a"] = {key: evaluate(agent, encoder, data[f"test_{key}"]) for key in ("a", "b")}
                if method.endswith("_ogd"):
                    tick = time.perf_counter()
                    result["consolidation"] = consolidate_task_memory(
                        agent, anchors_from_training(data["train_a"], encoder, args.window, seed),
                    )
                    result["consolidation"]["seconds"] = time.perf_counter() - tick
                if method != "frozen":
                    result["train_b"] = train_phase(agent, encoder, data["train_b"], args)
                result["after_b"] = {key: evaluate(agent, encoder, data[f"test_{key}"]) for key in ("a", "b")}
                result["reset_b"] = evaluate(agent, encoder, data["test_b"], reset_history=True)
                result["games_after"] = play_games(agent, encoder, args, seed, opponent)
                result["resources"]["ogd_storage_bytes"] = agent.task_gradient_memory.storage_bytes
                result["max_unitary_error"] = agent.core.max_unitary_error()
                run["methods"][method] = result
                print(f"seed={seed} {method}: B NLL {result['before']['b']['policy_nll']:.4f} -> "
                      f"{result['after_b']['b']['policy_nll']:.4f}; A forgetting "
                      f"{result['after_b']['a']['policy_nll'] - result['after_a']['a']['policy_nll']:+.4f}", flush=True)
                if args.checkpoint_dir is not None:
                    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({"config": config, "method": method, "seed": seed,
                                "session": _session(agent, encoder, args.window).state_dict()},
                               args.checkpoint_dir / f"{method}-{seed}.pt")
            payload["runs"].append(run)
            payload["summary"] = summarize(payload["runs"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            temporary.replace(args.output)
    finally:
        if client is not None:
            client.close()
    print(f"results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
