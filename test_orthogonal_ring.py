"""Signed-ring norm, gradient, damping, and transaction contracts."""

from __future__ import annotations

import unittest

import torch

from mqr import CyclicGivensUnistochasticParam, MultiTimescaleMQR, TemporalMQRState
from mqr.agent import TemporalUtilityMQRAgent


class OrthogonalRingTests(unittest.TestCase):
    def test_sparse_rotation_matches_dense_forward_and_gradient(self) -> None:
        torch.manual_seed(901)
        parameter = CyclicGivensUnistochasticParam(8, layers=3, base_seed=7).double()
        x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
        weight = torch.randn_like(x)
        sparse = parameter.apply_orthogonal(x)
        dense = x @ parameter.unitary().real.T
        torch.testing.assert_close(sparse, dense, atol=1e-12, rtol=1e-12)
        sparse_grad = torch.autograd.grad((sparse * weight).sum(), (x, parameter.angles))
        dense_grad = torch.autograd.grad((dense * weight).sum(), (x, parameter.angles))
        for actual, expected in zip(sparse_grad, dense_grad):
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            parameter.apply_orthogonal(sparse.detach(), transpose=True),
            x.detach(), atol=1e-12, rtol=1e-12,
        )

    def test_signed_propagation_preserves_norm_before_damping(self) -> None:
        torch.manual_seed(902)
        parameter = CyclicGivensUnistochasticParam(8, base_seed=8).double()
        x = torch.randn(4, 8, dtype=torch.float64)
        torch.testing.assert_close(
            parameter.apply_orthogonal(x).norm(dim=1), x.norm(dim=1),
            atol=1e-12, rtol=1e-12,
        )
        # |U|² is a different operator, and need not preserve this norm.
        self.assertGreater(float((x.norm(dim=1) - (x @ parameter.unistochastic().T).norm(dim=1)).abs().max()), 0.01)

    def test_linear_carry_and_certificate_for_signed_states(self) -> None:
        torch.manual_seed(903)
        core = MultiTimescaleMQR(
            3, 8, 2, leak_rates=(0.1,), write_scales=(0.0,),
            transition_mode="orthogonal", transition_structure="cyclic_givens",
            base_unitary_init="random", base_unitary_seed=9,
            state_activation="none",
        ).double()
        initial = torch.randn(2, 8, dtype=torch.float64)
        state = TemporalMQRState((initial,))
        for _ in range(12):
            _, state, certificate = core.forward_step(
                torch.zeros(2, 3, dtype=torch.float64), state=state,
                return_certificate=True,
            )
            self.assertTrue(certificate["certified"])
        torch.testing.assert_close(
            state.rings[0].norm(dim=1), initial.norm(dim=1) * 0.9**12,
            atol=1e-12, rtol=1e-12,
        )
        reference = core.transition_references()[0]
        before = core.transition_matrix(0).detach().clone()
        with torch.no_grad():
            core.unitary_params[0].angles.add_(0.02)
        audit = core.transition_drift_diagnostics(0, reference)
        self.assertAlmostEqual(
            audit["transition_fro_drift"],
            float((core.transition_matrix(0) - before).norm()), places=12,
        )
        self.assertTrue(audit["bounds_certified"])

    def test_online_trace_reaches_earlier_input_and_keeps_predictions(self) -> None:
        torch.manual_seed(904)
        agent = TemporalUtilityMQRAgent(
            6, board_size=3, ring_dim=8, latent_dim=8,
            transition_mode="orthogonal", transition_structure="cyclic_givens",
            base_unitary_init="random", base_unitary_seed=10,
            max_trace_horizon=4, ogd_max_rank=2,
        )
        commits = [agent.commit_step(torch.randn(1, 6), external_write=True) for _ in range(3)]
        predictions = [item["policy_logits"].clone() for item in commits]
        update = agent.apply_trajectory_feedback(
            [int(item["ticket_id"]) for item in commits],
            [{"action": torch.tensor([index])} for index in range(3)],
            loss_scales=(0.0, 0.0, 1.0), remember_gradient=True,
            return_grad_features=True,
        )
        self.assertTrue(update["all_predictions_committed_before_update"])
        self.assertGreater(update["earliest_input_future_gradient_norm"], 0.0)
        self.assertEqual(agent.task_gradient_memory.rank, 1)
        self.assertLess(agent.core.max_unitary_error(), 1e-5)
        for prediction, commit in zip(predictions, commits):
            self.assertTrue(torch.equal(prediction, commit["policy_logits"]))


if __name__ == "__main__":
    unittest.main()
