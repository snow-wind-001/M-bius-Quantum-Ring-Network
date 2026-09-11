"""Exact adjoints, behavioral semantics, causal updates and restoration."""

import copy
import unittest

import torch

from mqr import GoBoard, HeuristicGoTeacher
from mqr.constraint_transport import pullback_ring_covectors, terminal_gradients
from mqr.go_constraints import ConstraintGoBehaviorMemory, ConstraintGoSession
from mqr.go_memory import GoObservationMemory
from test_go_memory import make_session


def fixture(seed=1701, slots=4, dtype=torch.float64):
    session = make_session(seed, slots=slots)
    agent = session.agent.double() if dtype == torch.float64 else session.agent.float()
    with torch.no_grad():
        for module in (agent.ring_channel_gain, agent.point_correction, agent.pass_head, agent.value_head):
            module.weight.normal_(std=0.12)
        for parameter in agent.core.angle_controllers.parameters():
            parameter.normal_(std=0.04)
    return agent, session.encoder


def prefix(agent, encoder, steps=6):
    board = GoBoard(3)
    history = GoObservationMemory(encoder)
    values = []
    for index in range(steps):
        values.append(encoder.with_history(board, history.slots).to(next(agent.parameters())))
        history.write(board)
        if index < steps - 1:
            moves = [a for a in board.legal_moves() if a != board.pass_action]
            board.play(moves[index % len(moves)] if moves else board.pass_action)
    return torch.cat(values)


def objectives(output):
    return torch.stack((output.policy_logits[0, 0], output.policy_logits[-1, -1], output.value.sum()))


class ConstraintTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_readout_and_all_shared_parameter_gradients(self):
        for slots, dtype in ((0, torch.float64), (4, torch.float64), (4, torch.float32)):
            with self.subTest(slots=slots, dtype=dtype):
                agent, encoder = fixture(slots=slots, dtype=dtype)
                x = prefix(agent, encoder)
                active = [(n, p) for n, p in agent._named_task_parameters() if p.requires_grad]
                for _, p in active:
                    p.grad = torch.ones_like(p)
                snapshot = copy.deepcopy(agent.state_dict())
                result = terminal_gradients(agent, x, objectives)
                reference = terminal_gradients(agent, x, objectives, backend="autograd")
                tolerance = 2e-10 if dtype == torch.float64 else 2e-5
                torch.testing.assert_close(result.values, reference.values, rtol=0, atol=0)
                torch.testing.assert_close(result.jacobian, reference.jacobian, rtol=tolerance, atol=tolerance)
                cursor = 0
                for name, p in active:
                    block = result.jacobian[:, cursor:cursor + p.numel()]
                    if name in ("core.input_down.weight", "core.angle_controllers.1.weight",
                                "core.unitary_params.1.angles", "point_correction.weight"):
                        self.assertGreater(float(block.norm()), 1e-8, name)
                    cursor += p.numel()
                    self.assertTrue(torch.equal(p.grad, torch.ones_like(p)))
                for name, value in agent.state_dict().items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
                self.assertEqual(result.diagnostics["credit_steps"], len(x))
                self.assertLess(result.diagnostics["max_adjoint_norm_error"], tolerance)

    def test_finite_difference_includes_conditioned_rotation_and_early_injection(self):
        agent, encoder = fixture()
        x = prefix(agent, encoder)
        result = terminal_gradients(agent, x, lambda output: output.policy_logits[0, 0])
        active = [(n, p) for n, p in agent._named_task_parameters() if p.requires_grad]
        cursor = 0
        for name, p in active:
            if name in ("core.input_down.weight", "core.angle_controllers.1.weight"):
                index = int(result.jacobian[0, cursor:cursor + p.numel()].abs().argmax())
                original = float(p.detach().flatten()[index])
                values = []
                for sign in (1, -1):
                    with torch.no_grad():
                        p.flatten()[index] = original + sign * 1e-5
                        state = agent.core.zero_state(1, device=x.device, dtype=x.dtype)
                        for item in x.split(1):
                            output, state = agent._transition(item, state, slow_write=True)
                        values.append(float(output.policy_logits[0, 0]))
                with torch.no_grad():
                    p.flatten()[index] = original
                self.assertAlmostEqual((values[0] - values[1]) / 2e-5,
                                       float(result.jacobian[0, cursor + index]), places=8)
            cursor += p.numel()

    def test_batched_streams_truncation_and_frozen_transition_parameters(self):
        agent, encoder = fixture(slots=0)
        agent.core.angle_controllers.requires_grad_(False)
        x = prefix(agent, encoder)[:, None].repeat(1, 2, 1)
        x[:, 1, 0] += 0.1
        for horizon in (None, 1, 2, 4, 20):
            a = terminal_gradients(agent, x, objectives, backend="autograd", credit_horizon=horizon)
            b = terminal_gradients(agent, x, objectives, credit_horizon=horizon)
            torch.testing.assert_close(a.jacobian, b.jacobian, rtol=1e-9, atol=1e-10)
        full = terminal_gradients(agent, x, objectives)
        short = terminal_gradients(agent, x, objectives, credit_horizon=1)
        self.assertGreater(float((full.jacobian - short.jacobian).norm()), 1e-5)

    def test_covector_pullback_matches_actual_state_jacobian(self):
        agent, encoder = fixture()
        core = agent.core
        x = prefix(agent, encoder)[:2]
        state = core.zero_state(2, device=x.device, dtype=x.dtype)
        state = type(state)(tuple(torch.randn_like(h, requires_grad=True) for h in state.rings))
        _, next_state = core.forward_step(x, state=state)
        q = tuple(torch.randn(3, *h.shape, dtype=x.dtype) for h in state.rings)
        transported, error = pullback_ring_covectors(core, x, q)
        for row in range(3):
            expected = torch.autograd.grad(next_state.rings, state.rings,
                                           grad_outputs=tuple(v[row] for v in q),
                                           retain_graph=row < 2, allow_unused=True)
            for actual, wanted, h in zip(transported, expected, state.rings):
                torch.testing.assert_close(actual[row], torch.zeros_like(h) if wanted is None else wanted,
                                           rtol=1e-12, atol=1e-12)
        self.assertLess(error, 1e-12)
        self.assertEqual(float(transported[0].norm()), 0)

    def test_unsupported_nonlinear_or_squared_dynamics_are_rejected(self):
        agent, encoder = fixture()
        x = prefix(agent, encoder)
        agent.core.state_activation = "tanh"
        with self.assertRaisesRegex(ValueError, "state_activation"):
            terminal_gradients(agent, x, objectives)
        agent.core.state_activation = "none"
        agent.core.transition_mode = "unistochastic"
        with self.assertRaisesRegex(ValueError, "signed orthogonal"):
            terminal_gradients(agent, x, objectives)

    def test_identity_control_and_single_observation(self):
        agent, encoder = fixture(slots=0)
        agent.core.transition_mode = "identity"
        agent.core.conditional_scale = 0
        for count in (1, 5):
            x = prefix(agent, encoder, count)
            a = terminal_gradients(agent, x, objectives, backend="autograd")
            b = terminal_gradients(agent, x, objectives)
            torch.testing.assert_close(a.jacobian, b.jacobian, rtol=1e-10, atol=1e-11)

    def test_current_behavior_basis_and_global_update_match(self):
        agent, encoder = fixture()
        x = prefix(agent, encoder)
        memory = ConstraintGoBehaviorMemory(4, strata=2)
        for index in range(4):
            memory.add(list(x[:index + 2].split(1)), index, stratum=index % 2)
        memory.freeze(agent)
        other = ConstraintGoBehaviorMemory(4, strata=2, backend="autograd")
        raw = memory.state_dict()
        raw["constraint_transport"]["backend"] = "autograd"
        other.load_state_dict(raw)
        a = memory.refresh(agent)
        basis = agent.task_gradient_memory._basis.clone()
        b = other.refresh(agent)
        comparison = agent.task_gradient_memory._basis
        self.assertEqual(a["selected_candidates"], b["selected_candidates"])
        gradient = torch.randn(basis.size(1), dtype=basis.dtype)
        torch.testing.assert_close(gradient - basis.T @ (basis @ gradient),
                                   gradient - comparison.T @ (comparison @ gradient),
                                   rtol=1e-9, atol=1e-9)
        with torch.no_grad():
            agent.point_correction.weight.add_(0.05)
            agent.online_parameter_version.add_(1)
        memory.refresh(agent)
        self.assertEqual(memory.last_refresh_version, 1)
        self.assertGreater(float((basis - agent.task_gradient_memory._basis).norm()), 1e-4)

    def test_online_causality_guard_and_transport_resume(self):
        agent, encoder = fixture(dtype=torch.float32)
        memory = ConstraintGoBehaviorMemory(4, strata=2)
        x = prefix(agent, encoder)
        for index in range(4):
            memory.add(list(x[:index + 2].split(1)), index, stratum=index % 2)
        memory.freeze(agent)
        session = ConstraintGoSession(agent, encoder, behavior_memory=memory,
                                      refresh_every=1, update_every=2, max_anchor_kl=0.01)
        board = GoBoard(3)
        expected = agent.preview_step(session._encode_observation(board), external_write=True)["policy_logits"]
        result = session.step(board, HeuristicGoTeacher())
        torch.testing.assert_close(result["policy_logits"], expected, rtol=0, atol=0)
        board.play(result["action"])
        snapshot = copy.deepcopy(session.state_dict())
        resumed_agent, resumed_encoder = fixture(dtype=torch.float32)
        resumed = ConstraintGoSession(resumed_agent, resumed_encoder, refresh_every=1,
                                      update_every=2, max_anchor_kl=0.01)
        resumed.load_state_dict(snapshot)
        self.assertIsInstance(resumed.behavior_memory, ConstraintGoBehaviorMemory)
        a = session.step(board, HeuristicGoTeacher())
        b = resumed.step(board, HeuristicGoTeacher())
        torch.testing.assert_close(a["policy_logits"], b["policy_logits"], rtol=0, atol=0)
        self.assertLessEqual(a["update"]["anchor_drift"]["max_policy_kl"], 0.0100001)
        for name, value in agent.state_dict().items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, resumed_agent.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(session.behavior_memory.refresh_count, resumed.behavior_memory.refresh_count)


if __name__ == "__main__":
    unittest.main()
