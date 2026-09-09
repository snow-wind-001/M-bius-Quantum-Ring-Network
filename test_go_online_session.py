"""Online Go ordering, history credit, consolidation, and restart tests."""

from __future__ import annotations

import copy
import unittest

import torch

from mqr import GoBoard, GoOnlineSession, GoVectorEncoder, HeuristicGoTeacher
from mqr import SpatialRingGoAgent, TemporalUtilityMQRAgent, consolidate_task_memory


def make_session(seed: int = 81) -> GoOnlineSession:
    torch.manual_seed(seed)
    encoder = GoVectorEncoder(3, 33, projection_mode="identity")
    agent = TemporalUtilityMQRAgent(
        33, board_size=3, ring_dim=8, latent_dim=8,
        transition_mode="orthogonal", transition_structure="cyclic_givens",
        base_unitary_init="random", base_unitary_seed=81,
        max_trace_horizon=4, ogd_max_rank=4,
    )
    return GoOnlineSession(agent, encoder, update_every=3)


class GoOnlineSessionTests(unittest.TestCase):
    def test_spatial_modulation_preserves_initial_base_then_learns(self) -> None:
        torch.manual_seed(80)
        agent = SpatialRingGoAgent(
            33, board_size=3, ring_dim=8, latent_dim=8,
            transition_mode="orthogonal", transition_structure="cyclic_givens",
            spatial_skip_channels=4, base_unitary_init="random",
        )
        with torch.no_grad():
            agent.spatial_skip_heads.placement.weight.normal_(std=0.2)
        for parameter in agent.spatial_skip_heads.parameters():
            parameter.requires_grad_(False)
        encoder = GoVectorEncoder(3, 33, projection_mode="identity")
        board = GoBoard(3)
        features = encoder.encode_board(board)
        before = agent.preview_step(features, external_write=True)["output"]
        spatial = agent.spatial_skip_heads(features)
        self.assertTrue(torch.equal(before.placement_logits, spatial[0]))
        self.assertTrue(torch.equal(before.pass_logit, spatial[2]))
        frozen = [p.detach().clone() for p in agent.spatial_skip_heads.parameters()]
        session = GoOnlineSession(agent, encoder, update_every=2)
        for _ in range(2):
            result = session.step(board, HeuristicGoTeacher())
            board.play(result["teacher_action"])
        self.assertGreater(float(agent.ring_modulation.weight.detach().abs().max()), 0.0)
        for actual, expected in zip(agent.spatial_skip_heads.parameters(), frozen):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(session.agent.pending_ticket_count, 0)
        self.assertGreater(session.online_tensor_bytes, 3 * 8 * 4)

    def test_teacher_runs_after_commit_and_before_update(self) -> None:
        session = make_session()
        board = GoBoard(3)
        teacher = HeuristicGoTeacher()

        class InspectTeacher:
            def select_move(self, position: GoBoard) -> int:
                self_test.assertGreater(session.agent.online_observations.item(), 0)
                self_test.assertEqual(session.agent.online_task_updates.item(), 0)
                return teacher.select_move(position)

        self_test = self
        for index in range(3):
            result = session.step(board, InspectTeacher())
            self.assertTrue(board.is_legal(result["action"]))
            self.assertEqual(result["parameter_version"], 0)
            self.assertEqual(result["update"] is None, index < 2)
            board.play(result["action"])
        self.assertEqual(session.agent.online_task_updates.item(), 1)
        self.assertEqual(session.agent.pending_ticket_count, 0)

    def test_pending_window_restart_is_exact(self) -> None:
        left = make_session()
        teacher = HeuristicGoTeacher()
        board = GoBoard(3)
        for _ in range(2):
            result = left.step(board, teacher)
            board.play(result["teacher_action"])
        right = make_session(92)
        right.load_state_dict(left.state_dict())
        a, b = left.step(board, teacher), right.step(board, teacher)
        self.assertTrue(torch.equal(a["policy_logits"], b["policy_logits"]))
        self.assertEqual(a["action"], b["action"])
        for p, q in zip(left.agent.parameters(), right.agent.parameters()):
            self.assertTrue(torch.equal(p, q))
        for p, q in zip(
            left.agent._stream_states[left.stream_id].rings,
            right.agent._stream_states[right.stream_id].rings,
        ):
            self.assertTrue(torch.equal(p, q))

    def test_incomplete_feedback_cannot_be_discarded_by_reset(self) -> None:
        session = make_session()
        result = session.observe(GoBoard(3))
        with self.assertRaises(RuntimeError):
            session.flush()
        with self.assertRaises(RuntimeError):
            session.reset_game()
        session.feedback(result["ticket_id"], 4)
        with self.assertRaises(RuntimeError):
            session.feedback(result["ticket_id"], 4)
        session.flush()
        session.reset_game()
        self.assertEqual(session.pending_count, 0)

    def test_current_output_memory_is_local_and_nonmutating(self) -> None:
        session = make_session()
        teacher = HeuristicGoTeacher()
        anchors = []
        board = GoBoard(3)
        for _ in range(4):
            features = session.encoder.encode_board(board)
            action = teacher.select_move(board)
            anchors.append([(features, action)])
            session.step(board, teacher)
            board.play(action)
        session.flush()
        parameters = [p.detach().clone() for p in session.agent.parameters()]
        state = copy.deepcopy(session.agent._stream_states)
        version = session.agent.online_parameter_version.item()
        report = consolidate_task_memory(session.agent, anchors)
        self.assertGreater(report["rank"], 0)
        self.assertLess(report["orthogonality_error"], 1e-5)
        self.assertEqual(version, session.agent.online_parameter_version.item())
        for actual, expected in zip(session.agent.parameters(), parameters):
            self.assertTrue(torch.equal(actual, expected))
        for key, value in state.items():
            for actual, expected in zip(session.agent._stream_states[key].rings, value.rings):
                self.assertTrue(torch.equal(actual, expected))

        active = [(n, p) for n, p in session.agent._named_task_parameters() if p.requires_grad]
        entries = [(n, torch.randn_like(p), session.agent.task_lr) for n, p in active]
        _, stats = session.agent.task_gradient_memory.project_preconditioned(entries)
        self.assertLess(stats["max_abs_overlap"], 1e-5)


if __name__ == "__main__":
    unittest.main()
