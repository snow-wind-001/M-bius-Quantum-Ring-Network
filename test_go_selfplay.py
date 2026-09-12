"""One-sided projection, shared-policy feedback and complete context contracts."""
import copy
import unittest

import torch

from mqr import GoBoard, GoLossWeights
from mqr.constraint_transport import terminal_gradients
from mqr.go_cone import TaskMarginMemory, project_halfspaces
from mqr.go_selfplay import SharedPolicyGoSession
from test_go_policy import make_session, assert_tree


def session_with_margin(seed=3701):
    old = make_session(seed, window=2)
    old.agent.loss_weights = GoLossWeights(placement=0, pass_decision=0,
        legality=0.25, value=0, policy_distillation=1)
    memory = TaskMarginMemory(4, strata=2, reliable_only=True, margin_fraction=0.5)
    x = old.encoder.encode_board(GoBoard(3))
    with torch.no_grad():
        action = int(memory._output(old.agent, {"features": x, "action": 0}).argmax())
    memory.add([x], action, stratum=0)
    memory.freeze(old.agent)
    session = SharedPolicyGoSession(old.agent, old.encoder, update_every=2, credit_horizon=2,
        behavior_memory=memory, project_with_memory=False, max_anchor_kl=None,
        refresh_every=1, replay_mode="outcome", outcome_weight=0.5, episode_capacity=12)
    session.refresh_protection()
    return session


class SharedPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_projection_preserves_improvement_and_uses_slack(self):
        g = torch.tensor([[1., 0.]], dtype=torch.float64)
        b = torch.tensor([-0.5], dtype=torch.float64)
        for initial, expected in [([3., 2.], [3., 2.]), ([-1., 2.], [-0.5, 2.])]:
            actual, info = project_halfspaces(torch.tensor(initial, dtype=g.dtype), g, b)
            torch.testing.assert_close(actual, torch.tensor(expected, dtype=g.dtype), atol=1e-10, rtol=0)
            self.assertTrue(info["converged"])

    def test_coupled_projection_matches_analytic_solution(self):
        g = torch.tensor([[1., 0.], [1., 1.]], dtype=torch.float64)
        d = torch.tensor([-1., -2.], dtype=torch.float64)
        actual, info = project_halfspaces(d, g, torch.zeros(2, dtype=g.dtype))
        torch.testing.assert_close(actual, torch.tensor([0.5, -0.5], dtype=g.dtype), atol=1e-7, rtol=0)
        self.assertTrue(info["converged"])
        self.assertLessEqual(float(actual.norm()), float(d.norm()))

    def test_invalid_or_stale_halfspaces_rejected(self):
        with self.assertRaises(ValueError):
            project_halfspaces(torch.ones(2), torch.ones(1, 2), torch.ones(1))
        session = session_with_margin()
        memory = session.behavior_memory
        with self.assertRaises(RuntimeError):
            memory.project_update(torch.zeros(memory.normals.size(1)), parameter_version=100)

    def test_margin_adjoint_matches_autograd(self):
        session = session_with_margin()
        session.agent.double()
        memory = session.behavior_memory
        memory.refresh(session.agent)
        record = memory.anchors[0]
        with torch.no_grad():
            scores = memory._output(session.agent, record)
            scores[record["action"]] = -torch.inf
            others = scores.topk(2).indices
        result = terminal_gradients(session.agent, record["features"].double(),
            lambda out: out.policy_logits[0, record["action"]] - out.policy_logits[0, others], backend="autograd")
        torch.testing.assert_close(memory.normals, result.jacobian, atol=1e-10, rtol=1e-9)

    def test_shared_policy_rejects_missing_target_without_consuming_ticket(self):
        session = session_with_margin()
        prediction = session.observe(GoBoard(3))
        before = session.state_dict()
        with self.assertRaises(ValueError):
            session.feedback(prediction["ticket_id"], prediction["action"])
        assert_tree(self, before, session.state_dict())

    def test_opening_context_is_replayed_without_tickets(self):
        session = session_with_margin()
        board = GoBoard(3)
        for action in (0, 8):
            session.observe_context(board)
            board.play(action)
        before = copy.deepcopy(session.agent._stream_states[session.stream_id])
        session._prime_replay("context-check")
        actual = session.agent._stream_states["context-check"]
        for a, b in zip(before.rings, actual.rings):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertEqual(session.agent.pending_ticket_count, 0)
        session.agent.reset_state("context-check")
        with self.assertRaises(ValueError):
            session.observe(GoBoard(3))

    def test_pending_context_and_projected_terminal_replay_roundtrip(self):
        session = session_with_margin()
        board = GoBoard(3)
        for action in (0, 8):
            session.observe_context(board)
            board.play(action)
        target = torch.zeros(1, 10)
        target[0, board.pass_action] = 1
        prediction = session.observe(board)
        session.feedback(prediction["ticket_id"], board.pass_action, policy_target=target)
        board.play(board.pass_action)
        restored = session_with_margin()
        restored.load_state_dict(session.state_dict())
        assert_tree(self, session.state_dict(), restored.state_dict())
        for current in (session, restored):
            prediction = current.observe(board)
            current.feedback(prediction["ticket_id"], board.pass_action, policy_target=target)
            terminal = board.copy()
            terminal.play(board.pass_action)
            result = current.finish_game(terminal)
            self.assertEqual(result["replayed_context"], 2)
            self.assertEqual(result["positions"], 2)
            self.assertEqual(result["value_semantics"], "shared_policy")
            self.assertEqual(current.agent.pending_ticket_count, 0)
            self.assertTrue(any("constraint_projection" in r for r in result["replay_updates"]))
            self.assertLessEqual(current.behavior_memory.drift(current.agent)["max_margin_violation"], 1e-7)
        # Timing is deliberately not part of exact continuation semantics.
        def remove_times(state):
            if isinstance(state, dict):
                return {k: remove_times(v) for k, v in state.items() if k != "refresh_seconds"}
            if isinstance(state, list):
                return [remove_times(v) for v in state]
            return state
        assert_tree(self, remove_times(session.state_dict()), remove_times(restored.state_dict()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
