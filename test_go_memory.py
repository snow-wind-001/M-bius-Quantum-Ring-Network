"""Representation, causal history, conditional rotations, and finite anchor drift."""

import copy
import unittest
from types import SimpleNamespace

import torch

from mqr import GoBoard, GoLossWeights, GoSpatialSkipHeads, HeuristicGoTeacher
from mqr.go_history_tasks import generate_reachable_history_pairs
from mqr.go_memory import (GoHistoryEncoder, GoObservationMemory, GoResearchCore,
                           HistoryGoSession, PositionQueryGoAgent, fixed_color_features)
from mqr.go_protection import GoBehaviorMemory, ProtectedGoSession


def make_session(seed=701, slots=4, protected=False, **kwargs):
    torch.manual_seed(seed)
    encoder = GoHistoryEncoder(3, history_slots=slots)
    core = GoResearchCore(
        encoder.output_dim, 8, 8, board_size=3, conditional_scale=0.2,
        transition_mode="orthogonal", transition_structure="cyclic_givens",
        base_unitary_init="random", base_unitary_seed=seed,
        state_activation="none", leak_rates=(1.0, 0.2, 0.04),
    )
    agent = PositionQueryGoAgent(
        encoder.output_dim, board_size=3, ring_dim=8, latent_dim=8,
        channels=4, query_dim=4, history_slots=slots, core=core,
        max_trace_horizon=8, task_lr=0.1, ogd_max_rank=4,
        loss_weights=GoLossWeights(value=0),
    )
    cls = ProtectedGoSession if protected else HistoryGoSession
    return cls(agent, encoder, update_every=4, **kwargs)


class GoMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_geometry_breaks_empty_board_alias_and_can_fit_center(self):
        torch.manual_seed(701)
        x = GoHistoryEncoder(5).encode_board(GoBoard(5))
        legacy = GoSpatialSkipHeads(81, 5, 6)
        hidden = legacy.spatial_features(x).flatten(2)
        self.assertTrue(torch.equal(hidden[:, :, :1].expand_as(hidden), hidden))
        base = GoSpatialSkipHeads(81, 5, 6, geometry=True, depth=2)
        hidden = base.spatial_features(x).flatten(2)
        self.assertGreater(float((hidden[:, :, 12] - hidden[:, :, 0]).detach().norm()), 0.01)
        optimizer = torch.optim.Adam(base.parameters(), lr=0.03)
        for _ in range(35):
            loss = torch.nn.functional.cross_entropy(base(x)[0], torch.tensor([12]))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertEqual(int(base(x)[0].argmax()), 12)
        self.assertLess(float(loss.detach()), 0.1)

    def test_shared_controls_start_from_the_same_nonuniform_policy(self):
        from experiments.go_memory_research import build
        args = SimpleNamespace(size=5, ring_dim=8, latent_dim=8, channels=6,
                               slots=4, rank=4, lr=0.08, history_moves=12)
        reference, encoder = build(args, "frozen", 809)
        with torch.no_grad():
            reference.spatial_skip_heads.legality.weight.normal_(std=0.5)
        board = GoBoard(5); board.play(0); board.play(3)
        policy = reference.preview_step(encoder.encode_board(board), external_write=True)["policy_logits"]
        self.assertGreater(float(policy[0, :-1].std()), 0.01)
        base = reference.spatial_skip_heads.state_dict()
        for method in ("spatial", "film", "identity", "orthogonal", "conditional_external_guard"):
            agent, current_encoder = build(args, method, 809)
            agent.spatial_skip_heads.load_state_dict(base)
            current = agent.preview_step(current_encoder.encode_board(board), external_write=True)["policy_logits"]
            torch.testing.assert_close(current, policy, rtol=0, atol=0, msg=method)

    def test_fixed_colors_preserve_stones_across_player_changes(self):
        encoder = GoHistoryEncoder(3)
        board = GoBoard(3); board.play(0)
        before = fixed_color_features(encoder.encode_board(board), 3)
        board.play(8)
        after = fixed_color_features(encoder.encode_board(board), 3)
        self.assertEqual(float(before[0, 0]), 1.0)
        self.assertEqual(float(after[0, 0]), 1.0)
        self.assertEqual(float(after[0, 9 + 8]), 1.0)
        self.assertEqual(float(after[0, 9]), 0.0)

    def test_conditional_operator_is_orthogonal_and_contracts_same_input(self):
        session = make_session()
        core = session.agent.core.double()
        with torch.no_grad():
            for controller in core.angle_controllers.values():
                controller.weight.normal_(std=0.3)
        x = torch.randn(2, core.input_dim, dtype=torch.float64)
        state = core.zero_state(2, device=x.device, dtype=x.dtype)
        other = type(state)(tuple(torch.randn_like(ring) for ring in state.rings))
        _, left, certificate = core.forward_step(x, state=state, return_certificate=True)
        _, right, certificate = core.forward_step(x, state=other, return_certificate=True)
        self.assertTrue(certificate["certified"])
        self.assertLessEqual(certificate["max_bound_violation"], certificate["numerical_tolerance"])
        self.assertGreater(float(certificate["transition_linf_gain"][:, 1:].min()), 1.0)
        for index in (1, 2):
            operator = core.conditioned_transition(index, x)
            self.assertLess(float((operator @ operator.transpose(-1, -2) - torch.eye(8)).detach().abs().max()), 1e-12)
            actual = (right.rings[index] - left.rings[index]).norm(dim=-1)
            expected = (1 - core.leak_rates[index]) * other.rings[index].norm(dim=-1)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            parameter = core.unitary_params[index]
            offsets = core.orthogonal_offsets(index, x)
            restored = parameter.apply_orthogonal(parameter.apply_orthogonal(other.rings[index], angle_offsets=offsets),
                                                   angle_offsets=offsets, transpose=True)
            torch.testing.assert_close(restored, other.rings[index], atol=1e-12, rtol=1e-12)
        loss = right.rings[-1].square().sum() + right.rings[-1][:, 0].sum()
        gradient = torch.autograd.grad(loss, core.angle_controllers["2"].weight)[0]
        self.assertGreater(float(gradient.norm()), 1e-7)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in certificate.values() if isinstance(value, torch.Tensor)))
        offsets = core.orthogonal_offsets(2, x).detach().requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda angles: core.unitary_params[2].apply_orthogonal(other.rings[2], angle_offsets=angles),
            (offsets,), atol=1e-5, rtol=1e-4,
        ))
        with self.assertRaises(ValueError):
            core.transition_references()

    def test_external_memory_reads_before_write_and_resume_is_exact(self):
        left = make_session()
        board = GoBoard(3)
        first = left.observe(board)
        ticket = left.agent._pending_tickets[first["ticket_id"]]
        self.assertEqual(float(ticket["x"][:, 33:].abs().sum()), 0.0)
        self.assertEqual(left.observation_memory.writes, 1)
        left.feedback(first["ticket_id"], 4); board.play(4)
        for _ in range(2):
            result = left.step(board, HeuristicGoTeacher()); board.play(result["teacher_action"])
        right = make_session(seed=703)
        right.load_state_dict(left.state_dict())
        a, b = left.step(board, HeuristicGoTeacher()), right.step(board, HeuristicGoTeacher())
        torch.testing.assert_close(a["policy_logits"], b["policy_logits"], rtol=0, atol=0)
        for p, q in zip(left.agent.parameters(), right.agent.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        memory = GoObservationMemory(left.encoder)
        for _ in range(9):
            memory.write(board)
        self.assertEqual(memory.evictions, 5)
        self.assertEqual(memory.storage_bytes, 4 * 20 * 4)
        left.reset_game()
        self.assertEqual(float(left.observation_memory.slots.abs().sum()), 0.0)
        self.assertGreater(float(right.observation_memory.slots.abs().sum()), 0.0)

    def test_history_pairs_are_reachable_and_match_every_visible_field(self):
        pairs = generate_reachable_history_pairs(6, seed=707, moves=12)
        encoder = GoHistoryEncoder(5)
        for pair in pairs:
            for history, moves, target in zip(pair.histories, pair.moves, pair.targets):
                board = GoBoard(5, komi=2.5)
                for action, expected in zip(moves, history[1:]):
                    board.play(action)
                    self.assertEqual(board.position_history, expected.position_history)
                    self.assertEqual(board.board, expected.board)
                self.assertTrue(board.is_legal(target))
                self.assertEqual(target, 24 - moves[0])
            self.assertNotEqual(pair.targets[0], pair.targets[1])
            self.assertTrue(torch.equal(*(encoder.encode_board(h[-1]) for h in pair.histories)))

    def test_credit_horizon_changes_gradient_without_changing_predictions(self):
        full, short = make_session(slots=0), make_session(slots=0)
        # Nonzero output paths make the credit test independent of zero-head startup.
        with torch.no_grad():
            full.agent.point_correction.weight.normal_(std=0.5)
        short.agent.load_state_dict(full.agent.state_dict())
        board = GoBoard(3)
        ids = [[], []]
        labels = []
        for _ in range(4):
            results = [s.observe(board) for s in (full, short)]
            torch.testing.assert_close(results[0]["policy_logits"], results[1]["policy_logits"])
            action = HeuristicGoTeacher().select_move(board)
            labels.append({"action": torch.tensor([action])})
            for group, result in zip(ids, results):
                group.append(result["ticket_id"])
            board.play(action)
        reports = [s.agent.apply_trajectory_feedback(
            tickets, labels, loss_scales=[0, 0, 0, 1], credit_horizon=horizon,
        ) for s, tickets, horizon in zip((full, short), ids, (4, 2))]
        self.assertGreater(reports[0]["earliest_input_future_gradient_norm"], 1e-8)
        self.assertEqual(reports[1]["earliest_input_future_gradient_norm"], 0.0)
        self.assertEqual([s.agent.online_task_updates.item() for s in (full, short)], [1, 1])

    def test_anchor_refresh_and_finite_step_guard_preserve_reference_behavior(self):
        memory = GoBehaviorMemory(4, strata=2)
        session = make_session(protected=True, behavior_memory=memory, max_anchor_kl=0.0)
        board = GoBoard(3)
        features = []
        for index in range(4):
            features.append(session._encode_observation(board).clone())
            result = session.step(board, HeuristicGoTeacher())
            memory.add(features, result["teacher_action"], stratum=index % 2)
            board.play(result["teacher_action"])
        session.flush()
        memory.freeze(session.agent)
        original = [p.detach().clone() for p in session.agent.parameters()]
        report = memory.refresh(session.agent)
        self.assertGreater(report["rank"], 0)
        self.assertLess(report["orthogonality_error"], 1e-5)
        self.assertEqual(memory.drift(session.agent)["max_policy_kl"], 0.0)
        for p, q in zip(session.agent.parameters(), original):
            self.assertTrue(torch.equal(p, q))
        session.reset_game()
        board = GoBoard(3)
        for _ in range(4):
            result = session.step(board, HeuristicGoTeacher()); board.play(result["teacher_action"])
        self.assertLessEqual(result["update"]["anchor_drift"]["max_policy_kl"], 1e-7)
        self.assertGreater(result["update"]["anchor_backtracks"], 0)
        checkpoint = session.state_dict()
        resumed = make_session(protected=True, max_anchor_kl=0.0)
        resumed.load_state_dict(checkpoint)
        self.assertEqual(resumed.behavior_memory.frozen, True)
        self.assertEqual(resumed.behavior_memory.storage_bytes, memory.storage_bytes)
        self.assertGreater(session.online_tensor_bytes, memory.storage_bytes)
        with self.assertRaises(RuntimeError):
            memory.add(features, 4, stratum=0)


if __name__ == "__main__":
    unittest.main()
