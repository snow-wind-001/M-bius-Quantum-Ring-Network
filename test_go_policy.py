"""Signed query, protected spatial updates, terminal replay and search contracts."""

import copy
import io
import unittest
from unittest.mock import patch

import torch

from mqr import GoBoard, GoLossWeights, HeuristicGoTeacher
from mqr.constraint_transport import terminal_gradients
from mqr.go_memory import GoHistoryEncoder, GoResearchCore
from mqr.go_history_tasks import generate_reachable_history_pairs
from mqr.go_outcome import OutcomeGoSession, PolicyBehaviorMemory
from mqr.go_policy import SignedQueryGoAgent
from mqr.go_search import policy_value_search
from test_constraint_transport import prefix


def make_session(seed=1801, *, mode="none", slots=0, adapt=True, reliable=False, window=2):
    torch.manual_seed(seed)
    encoder = GoHistoryEncoder(3, history_slots=slots)
    core = GoResearchCore(
        encoder.output_dim, 8, 8, board_size=3, conditional_scale=0.2,
        transition_mode="orthogonal", transition_structure="cyclic_givens",
        base_unitary_init="random", base_unitary_seed=seed,
        state_activation="none", leak_rates=(1.0, 0.2, 0.04),
    )
    agent = SignedQueryGoAgent(
        encoder.output_dim, board_size=3, ring_dim=8, latent_dim=8, core=core,
        channels=4, query_dim=4, history_slots=slots, adapt_spatial=adapt,
        max_trace_horizon=8, task_lr=0.1, max_update_norm=0.15, ogd_max_rank=4,
        loss_weights=GoLossWeights(value=0), legality_policy_scale=1.0,
    )
    agent.utility_gate.requires_grad_(False)
    return OutcomeGoSession(
        agent, encoder, update_every=window, credit_horizon=2,
        behavior_memory=PolicyBehaviorMemory(4, strata=2, reliable_only=reliable),
        refresh_every=1, max_anchor_kl=0.01, replay_mode=mode, episode_capacity=12,
    )


def assert_tree(test, left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_tree(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree(test, a, b)
    else:
        test.assertEqual(left, right)


class GoPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_signed_query_has_position_margin_and_initial_query_gradients(self):
        session = make_session()
        agent, encoder = session.agent, session.encoder
        x = encoder.encode_board(GoBoard(3))
        output, state = agent._transition(
            x, agent.core.zero_state(1, device=x.device, dtype=x.dtype), slow_write=True,
        )
        latent = agent.core.readout_state(state)
        _, correction, addressing = agent.query_components(latent, state, x)
        self.assertLess(abs(float(correction[..., 0].mean().detach())), 1e-7)
        self.assertGreater(float(correction[..., 0].std().detach()), 1e-5)
        self.assertLess(float(addressing.min().detach()), 0)
        self.assertGreater(float(addressing.max().detach()), 0)
        gradients = torch.autograd.grad(output.policy_logits[0, 0] - output.policy_logits[0, 8],
                                        (agent.geometry_query.weight, agent.ring_keys, agent.ring_value.weight))
        for gradient in gradients:
            self.assertGreater(float(gradient.norm()), 1e-6)

    def test_new_readout_transport_and_external_memory_match_autograd(self):
        for slots in (0, 2):
            session = make_session(slots=slots)
            agent = session.agent.double()
            xs = prefix(agent, session.encoder, steps=6)
            objective = lambda out: torch.stack((out.policy_logits[0, 0], out.policy_logits[0, 8], out.value[0]))
            a = terminal_gradients(agent, xs, objective)
            b = terminal_gradients(agent, xs, objective, backend="autograd")
            torch.testing.assert_close(a.jacobian, b.jacobian, rtol=1e-9, atol=1e-10)
            active = [(n, p) for n, p in agent._named_task_parameters() if p.requires_grad]
            self.assertEqual(len(active), len({id(p) for _, p in active}))
            cursor = 0
            for name, p in active:
                if name == "geometry_query.weight":
                    index = int(a.jacobian[0, cursor:cursor + p.numel()].abs().argmax())
                    original = float(p.detach().flatten()[index])
                    values = []
                    for sign in (1, -1):
                        with torch.no_grad():
                            p.flatten()[index] = original + sign * 1e-5
                        values.append(float(terminal_gradients(agent, xs, objective).values[0]))
                    with torch.no_grad():
                        p.flatten()[index] = original
                    self.assertAlmostEqual((values[0] - values[1]) / 2e-5,
                                           float(a.jacobian[0, cursor + index]), places=8)
                cursor += p.numel()

    def test_identical_current_boards_can_read_different_history_margins(self):
        from argparse import Namespace
        from experiments.go_policy_research import build
        args = Namespace(size=5, ring_dim=16, latent_dim=16, channels=12, slots=0,
                         rank=8, lr=0.08, history_moves=12)
        agent, encoder = build(args, "query", 1801)
        pair = generate_reachable_history_pairs(1, seed=1801, size=5, moves=8)[0]
        outputs, inputs = [], []
        with torch.no_grad():
            for history in pair.histories:
                x = encoder.encode_board(history[0])
                state = agent.core.zero_state(1, device=x.device, dtype=x.dtype)
                for board in history:
                    x = encoder.encode_board(board)
                    _, state = agent._transition(x, state, slow_write=True)
                _, correction, _ = agent.query_components(agent.core.readout_state(state), state, x)
                outputs.append(correction[0, :, 0])
                inputs.append(x)
        torch.testing.assert_close(inputs[0], inputs[1], rtol=0, atol=0)
        # Spatial backbone and current geometry are identical; this difference
        # must come from the historical ring contents. It is not a Go win test.
        contrast = outputs[0] - outputs[1]
        self.assertGreater(float((contrast - contrast.mean()).norm()), 1e-5)

    def test_spatial_adaptation_uses_the_joint_protected_update(self):
        session = make_session()
        agent, memory = session.agent, session.behavior_memory
        xs = prefix(agent, session.encoder)
        for i in range(4):
            memory.add(list(xs[:i + 2].split(1)), i, stratum=i % 2)
        memory.freeze(agent)
        before = agent.spatial_skip_heads.placement.weight.detach().clone()
        board = GoBoard(3)
        session.step(board, HeuristicGoTeacher())
        board.play(0)
        result = session.step(board, HeuristicGoTeacher())
        update = result["update"]
        self.assertTrue(update["ogd_projection_applied"])
        self.assertLessEqual(update["anchor_drift"]["max_policy_kl"], 0.0100001)
        self.assertGreater(float((agent.spatial_skip_heads.placement.weight - before).detach().norm()), 0)
        self.assertIn("spatial_skip_heads.placement.weight",
                      [name for name, _ in agent.task_gradient_memory._layout])
        session.abort_game()

    def test_empty_reliable_bank_is_explicit_and_restores(self):
        session = make_session(reliable=True)
        agent, memory = session.agent, session.behavior_memory
        xs = prefix(agent, session.encoder, steps=3)
        for i in range(2):
            record = {"features": xs[:i + 1], "action": 0}
            with torch.no_grad():
                wrong = (int(memory._output(agent, record).argmax()) + 1) % agent.action_size
            memory.add(list(xs[:i + 1].split(1)), wrong, stratum=i)
        memory.freeze(agent)
        self.assertEqual(memory.candidate_count, 2)
        self.assertEqual(len(memory.anchors), 0)
        self.assertEqual(memory.refresh(agent)["rank"], 0)
        self.assertEqual(memory.drift(agent)["max_policy_kl"], 0)
        resumed = make_session(reliable=True)
        resumed.load_state_dict(session.state_dict())
        self.assertIsInstance(resumed.behavior_memory, PolicyBehaviorMemory)
        self.assertEqual(resumed.behavior_memory.agreement_count, 0)

    def test_only_real_terminal_same_game_outcomes_are_accepted(self):
        session = make_session(mode="outcome")
        board = GoBoard(3)
        result = session.observe(board)
        with self.assertRaisesRegex(ValueError, "finish_game"):
            session.feedback(result["ticket_id"], 0, value_target=1)
        session.feedback(result["ticket_id"], 0)
        with self.assertRaisesRegex(ValueError, "truly terminal"):
            session.finish_game(board)
        alien = GoBoard(3)
        alien.play(9); alien.play(9)
        with self.assertRaisesRegex(ValueError, "same game"):
            session.finish_game(alien)
        with self.assertRaisesRegex(RuntimeError, "abort"):
            session.reset_game()
        result = session.abort_game()
        self.assertEqual(result["value_targets"], [])
        self.assertEqual(session.replayed_positions, 0)
        session.reset_game()

    def test_terminal_signs_fresh_replay_and_policy_update_count_control(self):
        runs = []
        for mode in ("policy", "outcome"):
            session = make_session(mode=mode)
            board = GoBoard(3, komi=0.5)
            before = session.agent.value_head.weight.detach().clone()
            for _ in range(2):
                result = session.observe(board)
                session.feedback(result["ticket_id"], board.pass_action)
                board.play(board.pass_action)
            result = session.finish_game(board)
            self.assertEqual(result["replayed_positions"], 2)
            self.assertEqual(session.agent.pending_ticket_count, 0)
            self.assertEqual(session.agent.loss_weights.value, 0)
            self.assertEqual(len(result["replay_updates"]), 1)
            self.assertEqual(result["replay_updates"][0]["parameter_version"],
                             result["online_update"]["parameter_version"] + 1)
            changed = float((before - session.agent.value_head.weight).detach().norm())
            self.assertGreater(changed, 0) if mode == "outcome" else self.assertEqual(changed, 0)
            if mode == "outcome":
                self.assertEqual(result["value_targets"], [-1.0, 1.0])
            with self.assertRaises(ValueError):
                session.finish_game(board)
            runs.append(int(session.agent.online_task_updates))
        self.assertEqual(runs[0], runs[1])

    def test_midgame_checkpoint_resumes_pending_labels_and_terminal_replay(self):
        first = make_session(mode="outcome")
        board = GoBoard(3)
        first.step(board, HeuristicGoTeacher())
        board.play(board.pass_action)
        buffer = io.BytesIO()
        torch.save(first.state_dict(), buffer)
        buffer.seek(0)
        second = make_session(mode="outcome")
        second.load_state_dict(torch.load(buffer, weights_only=True))
        for session in (first, second):
            result = session.observe(board)
            session.feedback(result["ticket_id"], board.pass_action)
        board.play(board.pass_action)
        a, b = first.finish_game(board), second.finish_game(board)
        self.assertEqual(a["value_targets"], b["value_targets"])
        assert_tree(self, first.state_dict(), second.state_dict())
        incompatible = make_session(mode="policy")
        with self.assertRaisesRegex(ValueError, "configuration"):
            incompatible.load_state_dict(first.state_dict())

    def test_search_is_read_only_and_uses_terminal_player_perspective(self):
        for komi, expected in ((0.5, 1.0), (-0.5, -1.0)):
            session = make_session()
            board = GoBoard(3, komi=komi)
            board.play(board.pass_action)  # white may now end the game
            result = session.observe(board, learn=False)
            snapshot = session.state_dict()
            root = torch.full((1, 10), -20.0)
            root[0, -1] = 0
            search = policy_value_search(session.agent, session.encoder, board,
                                         result["state"], root, simulations=1)
            self.assertEqual(search["action_values"][9], expected)
            self.assertEqual(search["action"], 9)
            assert_tree(self, snapshot, session.state_dict())
            self.assertFalse(board.game_over)
            general = policy_value_search(session.agent, session.encoder, board,
                                          result["state"], result["policy_logits"], simulations=12)
            self.assertEqual(sum(general["visits"].values()), 12)
            self.assertLessEqual(general["network_evaluations"], 12)
            assert_tree(self, snapshot, session.state_dict())

    def test_search_precedes_teacher_and_full_window_update(self):
        session = make_session(window=1)
        board = GoBoard(3)
        version = int(session.agent.online_parameter_version)
        events = []

        def search(*args, **kwargs):
            self.assertEqual(int(session.agent.online_parameter_version), version)
            events.append("search")
            return policy_value_search(*args, **kwargs)

        class Teacher:
            def select_move(inner, position):
                self.assertEqual(events, ["search"])
                self.assertEqual(int(session.agent.online_parameter_version), version)
                events.append("teacher")
                return 0

        with patch("mqr.go_outcome.policy_value_search", side_effect=search):
            result = session.step(board, Teacher(), simulations=2)
        self.assertEqual(events, ["search", "teacher"])
        self.assertTrue(result["update"]["did_update"])
        self.assertTrue(board.is_legal(result["action"]))
        session.abort_game()

    def test_search_zero_budget_and_unsupported_branch_memory(self):
        session = make_session()
        board = GoBoard(3)
        result = session.observe(board, learn=False)
        search = policy_value_search(session.agent, session.encoder, board,
                                     result["state"], result["policy_logits"], simulations=0)
        self.assertEqual(search["action"], result["action"])
        external = make_session(slots=2)
        other = external.observe(board, learn=False)
        with self.assertRaisesRegex(ValueError, "history_slots"):
            policy_value_search(external.agent, external.encoder, board,
                                other["state"], other["policy_logits"])

    def test_invalid_search_requests_do_not_commit_a_prediction(self):
        for slots, simulations in ((0, -1), (2, 4)):
            session = make_session(slots=slots)
            before = session.state_dict()
            with self.assertRaises(ValueError):
                session.step(GoBoard(3), HeuristicGoTeacher(), simulations=simulations)
            assert_tree(self, before, session.state_dict())


if __name__ == "__main__":
    unittest.main()
