"""Search supervision, stable adaptation, ko observability and finite guards."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from mqr import GoBoard, GoLossWeights
from mqr.go_history_tasks import generate_ko_history_pairs
from mqr.go_memory import GoHistoryEncoder
from mqr.go_outcome import SearchOutcomeGoSession, PolicyBehaviorMemory
from mqr.go_search import policy_value_search
from test_go_policy import make_session, assert_tree


class Go10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def search_session(self):
        old = make_session(mode="outcome", window=2)
        old.agent.loss_weights = GoLossWeights(placement=0, pass_decision=0, legality=0,
                                              value=0, policy_distillation=1)
        return SearchOutcomeGoSession(old.agent, old.encoder, update_every=2, credit_horizon=2,
                                      replay_mode="outcome", episode_capacity=12)

    def test_soft_policy_gradient_matches_joint_cross_entropy(self):
        session = self.search_session()
        agent = session.agent.double()
        x = session.encoder.encode_board(GoBoard(3)).double()
        output, _ = agent._transition(x, agent.core.zero_state(1, device=x.device, dtype=x.dtype), slow_write=True)
        target = torch.arange(1, 11, dtype=x.dtype).reshape(1, -1)
        target /= target.sum()
        loss = agent.compute_go_loss(output, torch.tensor([0]), policy_target=target)["total"]
        direct = -(target * output.policy_logits).sum()
        params = tuple(p for p in agent.parameters() if p.requires_grad)
        a = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
        b = torch.autograd.grad(direct, params, allow_unused=True)
        for left, right in zip(a, b):
            if left is not None and right is not None:
                torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-12)

    def test_teacher_free_terminal_feedback_and_checkpoint(self):
        session = self.search_session()
        board = GoBoard(3)
        result = session.observe(board)
        target = torch.zeros(1, 10)
        target[0, board.pass_action] = 1
        session.feedback(result["ticket_id"], board.pass_action, policy_target=target)
        board.play(board.pass_action)
        restored = self.search_session()
        restored.load_state_dict(session.state_dict())
        for current in (session, restored):
            result = current.observe(board)
            current.feedback(result["ticket_id"], board.pass_action, policy_target=target)
            terminal = board.copy()
            terminal.play(terminal.pass_action)
            audit = current.finish_game(terminal)
            self.assertEqual(audit["value_targets"], [-1.0, 1.0])
            self.assertEqual(audit["replayed_positions"], 2)
            self.assertEqual(current.agent.pending_ticket_count, 0)
            self.assertGreater(int(current.agent.online_task_updates), 0)
        assert_tree(self, session.state_dict(), restored.state_dict())

    def test_invalid_policy_target_does_not_consume_feedback(self):
        session = self.search_session()
        board = GoBoard(3)
        board.play(0)
        result = session.observe(board)
        before = session.state_dict()
        for target in (torch.ones(1, 10), torch.full((1, 10), float("nan")),
                       torch.nn.functional.one_hot(torch.tensor([0]), 10).float()):
            with self.assertRaises(ValueError):
                session.feedback(result["ticket_id"], 1, policy_target=target)
            assert_tree(self, before, session.state_dict())

    def test_head_adaptation_preserves_pretrained_features(self):
        from argparse import Namespace
        from experiments.go10_continual import build
        agent, encoder = build(Namespace(size=10), "optimized", 2601)
        frozen = {n: p.detach().clone() for n, p in agent.named_parameters()
                  if n.startswith(("spatial_skip_heads.local.", "spatial_skip_heads.local_second."))}
        self.assertTrue(frozen)
        session = SearchOutcomeGoSession(agent, encoder, update_every=1)
        board = GoBoard(10)
        prediction = session.observe(board)
        target = torch.zeros(1, 101)
        target[0, 0] = 1
        session.feedback(prediction["ticket_id"], 0, policy_target=target)
        before = agent.spatial_skip_heads.pass_decision.weight.detach().clone()
        session.flush()
        for name, parameter in agent.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(before, agent.spatial_skip_heads.pass_decision.weight))

    def test_ko_pairs_have_identical_observation_and_different_true_legality(self):
        encoder = GoHistoryEncoder(10)
        for pair in generate_ko_history_pairs(16, seed=2601):
            finals = []
            for history in pair.histories:
                previous, final = history
                rebuilt = previous.copy()
                rebuilt.play(final.move_history[-1])
                self.assertEqual(rebuilt.position_history, final.position_history)
                finals.append(encoder.encode_board(final))
            torch.testing.assert_close(finals[0], finals[1], rtol=0, atol=0)
            self.assertEqual(tuple(h[-1].is_legal(pair.recapture) for h in pair.histories), (False, True))

    def test_margin_guard_checks_actual_actions_and_roundtrips(self):
        session = make_session(window=1)
        agent = session.agent
        with torch.no_grad():
            agent.spatial_skip_heads.pass_decision.bias.fill_(2)
        board = GoBoard(3)
        x = session.encoder.encode_board(board)
        memory = PolicyBehaviorMemory(4, strata=2, reliable_only=True, margin_fraction=0.5)
        memory.add([x], board.pass_action, stratum=0)
        memory.freeze(agent)
        self.assertEqual(len(memory.anchors), 1)
        self.assertEqual(memory.drift(agent)["max_margin_violation"], 0)
        clone = PolicyBehaviorMemory(4, strata=2, reliable_only=True, margin_fraction=0.5)
        clone.load_state_dict(memory.state_dict())
        with torch.no_grad():
            agent.spatial_skip_heads.pass_decision.bias.fill_(-20)
        self.assertGreater(clone.drift(agent)["max_margin_violation"], 0)

    def test_zero_value_search_is_read_only_and_retains_terminal_returns(self):
        session = make_session()
        board = GoBoard(3)
        board.play(board.pass_action)
        result = session.observe(board, learn=False)
        scores = torch.full_like(result["policy_logits"], -20)
        scores[0, board.pass_action] = 0
        before = copy.deepcopy(session.state_dict())
        search = policy_value_search(session.agent, session.encoder, board, result["state"], scores,
                                     simulations=2, max_depth=4, value_scale=0)
        self.assertEqual(search["action"], board.pass_action)
        self.assertEqual(search["action_values"][board.pass_action], 1)
        assert_tree(self, before, session.state_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
