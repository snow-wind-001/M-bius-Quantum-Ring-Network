"""Terminal ownership and matched-window temporal credit contracts."""
import copy
import unittest
from unittest.mock import patch

import torch

from mqr import GoBoard, GoLossWeights
from mqr.constraint_transport import terminal_gradients
from mqr.go_memory import GoHistoryEncoder, GoResearchCore
from mqr.go_policy import SignedQueryGoAgent
from mqr.go_selfplay import SharedPolicyGoSession
from test_go_policy import assert_tree


def make_session(*, ownership_weight=0.5, replay_credit=4):
    torch.manual_seed(3901)
    encoder = GoHistoryEncoder(3)
    core = GoResearchCore(
        encoder.output_dim, 8, 8, board_size=3, conditional_scale=0.2,
        transition_mode="orthogonal", transition_structure="cyclic_givens",
        base_unitary_init="random", base_unitary_seed=3901,
        state_activation="none", leak_rates=(1., 0.2, 0.04),
    )
    agent = SignedQueryGoAgent(
        encoder.output_dim, board_size=3, core=core, ring_dim=8, latent_dim=8,
        channels=4, query_dim=4, ownership_head=True, adapt_spatial=True,
        spatial_head_only=True, max_trace_horizon=8, task_lr=0.02,
        loss_weights=GoLossWeights(placement=0, value=0, policy_distillation=1),
    )
    agent.utility_gate.requires_grad_(False)
    return SharedPolicyGoSession(
        agent, encoder, update_every=2, credit_horizon=2, replay_window=4,
        replay_credit_horizon=replay_credit, project_with_memory=False,
        replay_mode="outcome", outcome_weight=0.5, ownership_weight=ownership_weight,
        episode_capacity=16,
    )


def play(session, board, actions):
    for action in actions:
        prediction = session.observe(board)
        target = torch.zeros(1, board.action_size)
        target[0, action] = 1
        session.feedback(prediction["ticket_id"], action, policy_target=target)
        if session.pending_count == session.update_every:
            session.flush()
        board.play(action)


class GoEfficiencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ownership_matches_area_score_and_neutral_regions(self):
        board = GoBoard(3, komi=0.5)
        self.assertEqual(board.area_ownership(), [0] * 9)
        board.board = [1, 1, 1, 1, 0, 1, 1, 1, 1]
        self.assertEqual(board.area_ownership(), [1] * 9)
        board.board = [1, 0, -1, 1, 0, -1, 1, 0, -1]
        self.assertEqual(board.area_ownership(), board.board)
        self.assertEqual(sum(board.area_ownership()) - board.komi, board.score()["margin_black"])
        board.board = [-v for v in board.board]
        self.assertEqual(board.area_ownership(), board.board)

    def test_head_detach_serialize_and_shared_ring_gradients(self):
        session = make_session()
        agent = session.agent
        x = session.encoder.encode_board(GoBoard(3))
        output, _ = agent._transition(x, agent.core.zero_state(1, device="cpu", dtype=x.dtype), slow_write=True)
        loss = agent.compute_go_loss(
            output, torch.tensor([0]), ownership_target=torch.ones(1, 9),
            weights=GoLossWeights(placement=0, legality=0, pass_decision=0, value=0, ownership=1),
        )["total"]
        gradients = torch.autograd.grad(loss, (agent.ring_keys, agent.ring_value.weight,
                                              agent.ownership_head.weight))
        for gradient in gradients:
            self.assertGreater(float(gradient.norm()), 1e-8)
        self.assertFalse(output.detached().ownership.requires_grad)
        restored = agent._restore_output(agent._serialize_output(output))
        torch.testing.assert_close(restored.ownership, output.ownership)
        with self.assertRaises(ValueError):
            agent.compute_go_loss(output, torch.tensor([0]), ownership_target=torch.full((1, 9), 2.))

    def test_ownership_adjoint_matches_full_autograd(self):
        session = make_session()
        session.agent.double()
        board = GoBoard(3)
        features = []
        for action in (0, 8, 1, 7):
            features.append(session.encoder.encode_board(board))
            board.play(action)
        features = torch.cat(features).double()
        objective = lambda output: output.ownership[0, :3]
        analytic = terminal_gradients(session.agent, features, objective, backend="transport")
        direct = terminal_gradients(session.agent, features, objective, backend="autograd")
        torch.testing.assert_close(analytic.jacobian, direct.jacobian, atol=1e-10, rtol=1e-9)

    def test_terminal_targets_have_correct_color_and_no_early_labels(self):
        session = make_session()
        board = GoBoard(3)
        calls = []
        original = session.agent.compute_go_loss

        def capture(*args, **kwargs):
            target = kwargs.get("ownership_target")
            calls.append(None if target is None else target.clone())
            return original(*args, **kwargs)

        with patch.object(session.agent, "compute_go_loss", side_effect=capture):
            play(session, board, (0, 8, 1))
            before = copy.deepcopy(session.state_dict())
            with self.assertRaises(ValueError):
                session.finish_game(board)
            assert_tree(self, before, session.state_dict())
            self.assertTrue(all(target is None for target in calls))
            play(session, board, (board.pass_action, board.pass_action))
            start = len(calls)
            result = session.finish_game(board)
        targets = [target for target in calls[start:] if target is not None]
        final = torch.tensor(board.area_ownership(), dtype=torch.float32).unsqueeze(0)
        self.assertEqual(len(targets), 5)
        for index, target in enumerate(targets):
            torch.testing.assert_close(target, final * (1 if index % 2 == 0 else -1))
        self.assertEqual(result["ownership_positions"], 5)
        self.assertEqual(len(result["replay_updates"]), 2)
        self.assertEqual(session.agent.loss_weights.ownership, 0)

    def test_replay_horizon_changes_credit_without_changing_update_count(self):
        short, long = make_session(replay_credit=2), make_session(replay_credit=4)
        for session in (short, long):
            board = GoBoard(3)
            play(session, board, (0, 8, 1, board.pass_action, board.pass_action))
            result = session.finish_game(board)
            self.assertEqual(len(result["replay_updates"]), 2)
        self.assertEqual(int(short.agent.online_parameter_version), int(long.agent.online_parameter_version))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(short.agent.parameters(), long.agent.parameters())))

    def test_pending_ownership_checkpoint_resumes_exactly(self):
        first, resumed = make_session(), make_session()
        board = GoBoard(3)
        play(first, board, (0,))
        resumed.load_state_dict(first.state_dict())
        for session in (first, resumed):
            current = board.copy()
            play(session, current, (8, current.pass_action, current.pass_action))
            session.finish_game(current)
        assert_tree(self, first.state_dict(), resumed.state_dict())
        wrong = make_session(ownership_weight=0)
        with self.assertRaises(ValueError):
            wrong.load_state_dict(first.state_dict())

    def test_zero_auxiliary_weight_is_exact_gradient_control(self):
        session = make_session()
        agent = session.agent
        x = session.encoder.encode_board(GoBoard(3))
        output, _ = agent._transition(x, agent.core.zero_state(1, device="cpu", dtype=x.dtype), slow_write=True)
        target = torch.ones(1, 9)
        a = agent.compute_go_loss(output, torch.tensor([0]))["total"]
        b = agent.compute_go_loss(output, torch.tensor([0]), ownership_target=target)["total"]
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        gradient = torch.autograd.grad(b, agent.ownership_head.weight, allow_unused=True)[0]
        self.assertEqual(float(gradient.norm()), 0)


if __name__ == "__main__":
    unittest.main()
