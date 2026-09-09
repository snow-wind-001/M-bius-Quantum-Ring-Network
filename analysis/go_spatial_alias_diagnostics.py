"""Measure exact spatial aliases in the existing one-convolution Go readout.

Identical 3x3 input patches produce identical features, and shared FiLM cannot
separate them. This gives a hard raw-policy NLL lower bound log(alias count)
for each non-pass target. It is a function-class diagnostic, not a win bound.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.orthogonal_go_online import build_agent
from mqr import GoBoard, GoVectorEncoder, HeuristicGoTeacher
from mqr.go_agent import generate_go_agent_trajectories


def patches(encoder: GoVectorEncoder, board: GoBoard) -> torch.Tensor:
    planes = encoder.encode_board(board)[:, :3 * board.size**2].reshape(1, 3, board.size, board.size)
    return F.unfold(planes, kernel_size=3, padding=1)[0].T


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(1)
    teacher = HeuristicGoTeacher()
    encoder = GoVectorEncoder(5, 81, projection_mode="identity")
    runs = []
    for seed in (17, 29, 43, 71, 101):
        cp = torch.load(ROOT / f"checkpoints/orthogonal_go_spatial/orthogonal_ogd-{seed}.pt", weights_only=True)
        agent = build_agent(SimpleNamespace(**cp["config"]), cp["method"], seed)
        agent.load_state_dict(cp["session"]["agent"])
        board = GoBoard(5, komi=2.5)
        target = teacher.select_move(board)
        x = encoder.encode_board(board)
        reference = next(agent.parameters())
        output, _ = agent._transition(x, agent.core.zero_state(1, device=reference.device, dtype=reference.dtype), slow_write=True)
        assert target == 12
        assert float(output.placement_logits.max() - output.placement_logits.min()) == 0
        assert float(output.legality_logits.max() - output.legality_logits.min()) == 0
        # A arbitrary latent cannot break the identical-feature symmetry.
        generator = torch.Generator().manual_seed(seed)
        for _ in range(4):
            z = torch.randn(1, agent.latent_dim, generator=generator)
            arbitrary = agent._head_output(z, x)
            assert float(arbitrary.placement_logits.max() - arbitrary.placement_logits.min()) == 0
        phases = {}
        for phase, offset, prefix in (("a", 4000, 0), ("b", 5000, 12)):
            data = generate_go_agent_trajectories(4, size=5, komi=2.5, seed=seed + offset,
                start_random_moves=prefix, recorded_moves=24, random_move_probability=.5)
            alias_counts, bounds, score_ambiguities, positions = [], [], [], 0
            for trace in data:
                for ex in trace.examples:
                    positions += 1
                    if ex.target_action == 25:
                        bounds.append(0.0)
                        continue
                    local = patches(encoder, ex.board)
                    aliases = (local == local[ex.target_action]).all(dim=1).nonzero().flatten().tolist()
                    alias_counts.append(len(aliases))
                    bounds.append(math.log(len(aliases)))
                    scores = [teacher.score_move(ex.board, action) for action in aliases]
                    score_ambiguities.append(len(aliases) > 1 and min(scores) < max(scores))
            phases[phase] = {
                "positions": positions, "non_pass_positions": len(alias_counts),
                "fraction_non_pass_targets_with_spatial_aliases": float(np.mean(np.array(alias_counts) > 1)),
                "mean_target_alias_group_size": float(np.mean(alias_counts)),
                "unavoidable_raw_joint_nll_lower_bound": float(np.mean(bounds)),
                "fraction_non_pass_alias_groups_with_different_teacher_scores": float(np.mean(score_ambiguities)),
            }
        runs.append({"seed": seed, "empty_board_teacher_action": target,
                     "empty_board_raw_model_action": int(output.policy_logits.argmax()),
                     "empty_board_target_probability": float(output.policy_logits.exp()[0, target]),
                     "empty_board_point_logit_range": float(output.placement_logits.max() - output.placement_logits.min()),
                     "phase_diagnostics": phases})
    payload = {"schema_version": 1, "retrospective_diagnostic": True, "training_updates": 0,
               "probe_traces_per_phase_per_seed": 4, "empty_board_unavoidable_nll_lower_bound": math.log(25),
               "bound_applies_to": "raw policy of the current spatially shared single-convolution feature class",
               "bound_is_not_game_strength": True, "runs": runs,
               "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (Path(__file__), ROOT / "mqr/go_online.py", ROOT / "mqr/agent.py")}}
    path = ROOT / "analysis/results/go_spatial_alias_diagnostics.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
