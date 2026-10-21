"""Contracts for the explicit Go14 write/read and online-update research APIs."""
from __future__ import annotations

import copy
import unittest

import torch

from experiments.go13_algorithm import build as build_previous
from experiments.go14_readability import LAYOUTS, build, resources
from mqr.go_feasible import FeasibleRuleLearner, ReadableBehaviorConstraint
from mqr.go_readable import readable_terminal_gradients
from mqr.go_research import (OnlineRuleLearner, RuleBatch, history_output,
                            parameter_vector, queried_logits)
from mqr.temporal import TemporalMQRState


class ReadableGoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_equal_parameter_budget_and_exact_shared_control(self):
        old, _ = build_previous(4903)
        torch.manual_seed(44)
        features = torch.randn(3, 2, 306) * .1
        reference = history_output(old, features)
        for layout in LAYOUTS:
            model, _ = build(4903, layout)
            self.assertEqual(resources(model)["total_parameters"], 36191)
            self.assertEqual(resources(model)["trainable_parameters"], 32577)
            for actual, expected in zip(model.parameters(), old.parameters()):
                self.assertTrue(torch.equal(actual, expected))
            if layout == "shared":
                actual = history_output(model, features)
                self.assertTrue(torch.equal(actual.legality_logits, reference.legality_logits))

    def test_spatial_write_has_no_position_alias_or_cross_position_write(self):
        model, _ = build(4903, "spatial")
        first = torch.zeros(1, 306)
        first[:, 300] = 1  # fixed black-to-play coordinates
        second = first.clone()
        second[0, 73] = 1
        difference = (model.core.input_forcing(second) - model.core.input_forcing(first)).reshape(1, 6, 100, 16)
        self.assertGreater(float(difference[:, :, 73].norm().detach()), 0)
        difference[:, :, 73] = 0
        self.assertEqual(float(difference.norm().detach()), 0)

    def test_convolutional_write_shares_relative_spatial_coordinates(self):
        model, _ = build(4903, "kernel")
        first = torch.zeros(1, 306); first[:, 300] = 1; first[:, 44] = 1
        second = torch.zeros_like(first); second[:, 300] = 1; second[:, 54] = 1
        left = model.core.input_forcing(first).reshape(1, 6, 10, 10, 16)
        right = model.core.input_forcing(second).reshape(1, 6, 10, 10, 16)
        self.assertTrue(torch.allclose(left[:, :, :-1], right[:, :, 1:], atol=1e-7))

    def test_previous_observation_is_causal_and_resettable(self):
        model, _ = build(4903, "previous")
        x1 = torch.zeros(1, 306); x1[:, 300] = 1; x1[:, 51] = 1
        x2 = torch.zeros_like(x1); x2[:, 300] = 1; x2[:, 62] = 1
        _, first = model.core.forward_step(x1)
        self.assertEqual(float(first.rings[-2].norm()), 0)
        _, second = model.core.forward_step(x2, state=first)
        self.assertTrue(torch.equal(second.rings[-2][:, :-1], model.core.board_features(x1)))
        self.assertEqual(float(second.rings[-2][:, -1]), 1)
        _, reset = model.core.forward_step(x2)
        self.assertEqual(float(reset.rings[-2].norm()), 0)

    def test_spatial_state_carry_and_covector_transpose(self):
        for layout in ("spatial", "similar_spatial"):
            model, _ = build(4903, layout)
            model.double()
            torch.manual_seed(45)
            x = torch.randn(2, 306, dtype=torch.float64) * .1
            v = torch.randn(2, 1600, dtype=torch.float64)
            q = torch.randn_like(v)
            for index in range(6):
                carried = model.core.carry(index, v, x)
                pulled = model.core.carry(index, q, x, transpose=True)
                self.assertTrue(torch.allclose((carried * q).sum(), (v * pulled).sum(), atol=1e-12, rtol=1e-12))
                if layout == "spatial":
                    rho = 1 - model.core.leak_rates[index]
                    self.assertTrue(torch.allclose(carried.norm(dim=-1), rho * v.norm(dim=-1), atol=1e-12, rtol=1e-12))

    def test_shared_parameter_adjoint_matches_full_graph(self):
        for layout in LAYOUTS:
            model, _ = build(4903, layout)
            model.double()
            torch.manual_seed(46)
            x = torch.randn(3, 2, 306, dtype=torch.float64) * .1
            queries = torch.tensor([31, 68])
            objective = lambda output: queried_logits(output, queries)
            analytic = readable_terminal_gradients(model, x, objective)
            complete = readable_terminal_gradients(model, x, objective, backend="autograd")
            self.assertEqual(analytic.parameter_names, complete.parameter_names)
            self.assertTrue(torch.allclose(analytic.values, complete.values, atol=1e-12, rtol=1e-12))
            self.assertLess(float((analytic.jacobian - complete.jacobian).abs().max()), 1e-10)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_stream_state_and_checkpoint_keep_spatial_memory(self):
        for layout in ("spatial", "previous"):
            model, _ = build(4903, layout)
            torch.manual_seed(47)
            xs = torch.randn(3, 1, 306) * .1
            state = model.core.zero_state(1, device=xs.device, dtype=xs.dtype)
            for x in xs[:2]:
                expected, state = model._transition(x, state, slow_write=True)
                actual = model.commit_step(x, stream_id="test", external_write=True, issue_ticket=False)
                self.assertTrue(torch.allclose(actual["policy_logits"], expected.policy_logits, atol=1e-6))
            restored, _ = build(4903, layout)
            restored.load_state_dict(copy.deepcopy(model.state_dict()))
            left = model.commit_step(xs[-1], stream_id="test", external_write=True, issue_ticket=False)
            right = restored.commit_step(xs[-1], stream_id="test", external_write=True, issue_ticket=False)
            self.assertTrue(torch.equal(left["policy_logits"], right["policy_logits"]))

    def make_protected(self, variant):
        model, _ = build(4903, "spatial")
        torch.manual_seed(48)
        features = torch.randn(2, 4, 306) * .1
        queries = torch.tensor([21, 42, 53, 74])
        with torch.no_grad():
            logits = queried_logits(history_output(model, features), queries)
        labels = (logits >= 0).float()
        anchors = RuleBatch(features, queries, labels)
        constraint = ReadableBehaviorConstraint(anchors, .5 * logits.abs())
        learner = FeasibleRuleLearner(model, variant=variant, constraint=constraint)
        return learner, RuleBatch(features, queries, 1 - labels)

    def test_actual_margin_guard_under_conflicting_feedback(self):
        for variant in ("current", "reserve", "adaptive", "rollback"):
            learner, batch = self.make_protected(variant)
            before = queried_logits(history_output(learner.agent, batch.features), batch.queries).detach()
            result = learner.step(batch)
            self.assertTrue(torch.equal(torch.tensor(result["pre_logits"]), before))
            self.assertTrue(bool((learner.constraint.margins(learner.agent) >= learner.constraint.floors - 1e-5).all()))
            self.assertEqual(result["feedback"], 4)
            self.assertEqual(result["actual_anchor_violations"], 0)

    def test_finite_step_resume_preserves_parameters_and_optimizer_state(self):
        for variant in ("adaptive", "rollback"):
            left, batch = self.make_protected(variant)
            left.step(batch)
            state = left.state_dict()
            right, _ = self.make_protected(variant)
            right.load_state_dict(state)
            a, b = left.step(batch), right.step(batch)
            self.assertTrue(torch.equal(parameter_vector(left.agent), parameter_vector(right.agent)))
            self.assertEqual(a["accepted"], b["accepted"])
            self.assertEqual(a["step_norm"], b["step_norm"])
            self.assertEqual(left.trust_radius, right.trust_radius)
            for key, value in left.optimizer.state_dict()["state"].items():
                for name, tensor in value.items():
                    self.assertTrue(torch.equal(tensor, right.optimizer.state_dict()["state"][key][name]))

    def test_accepted_float32_boundary_remains_valid_at_next_window(self):
        learner, batch = self.make_protected("reserve")
        guard = learner.constraint
        guard.floors.fill_(0.6950657367706299)
        boundary = guard.floors - 1e-5
        # This concrete value triggered the interrupted seed 1031 run.
        self.assertTrue(bool((boundary - guard.floors < -1e-5).all()))
        guard.margins = lambda agent: boundary.clone()
        guard.normals = torch.zeros(len(guard.floors), parameter_vector(learner.agent).numel())
        guard.refresh = lambda agent, update: None
        for _ in range(2):
            result = learner.step(batch)
            self.assertEqual(result["actual_anchor_violations"], 0)

    def test_original_current_step_is_unchanged(self):
        left, batch = self.make_protected("current")
        right, _ = self.make_protected("current")
        original = OnlineRuleLearner(right.agent, mode="current", constraint=right.constraint)
        a, b = left.step(batch), original.step(batch)
        self.assertTrue(torch.equal(parameter_vector(left.agent), parameter_vector(right.agent)))
        for key in ("pre_logits", "step_norm", "raw_step_norm", "backtracks", "accepted"):
            self.assertEqual(a[key], b[key])

    def test_rejected_step_rolls_back_only_the_declared_optimizer_policy(self):
        class RejectNonzero:
            def __init__(self, agent):
                self.origin = parameter_vector(agent)
                self.floors = self.origin.new_tensor([1.])
                self.normals = self.origin.new_zeros(1, self.origin.numel())
                self.refreshes = 0
                self.last_refresh = -1

            def refresh(self, agent, update):
                self.last_refresh = update
                self.refreshes += 1

            def margins(self, agent):
                return self.floors - 1e8 * (parameter_vector(agent) - self.origin).norm()

        for variant in ("adaptive", "rollback"):
            model, _ = build(4903, "kernel")
            torch.manual_seed(50)
            batch = RuleBatch(torch.randn(2, 2, 306) * .1, torch.tensor([25, 63]),
                              torch.tensor([0., 1.]))
            constraint = RejectNonzero(model)
            learner = FeasibleRuleLearner(model, variant=variant, constraint=constraint)
            result = learner.step(batch)
            self.assertFalse(result["accepted"])
            self.assertTrue(torch.equal(parameter_vector(model), constraint.origin))
            self.assertEqual(result["optimizer_rolled_back"], variant == "rollback")
            self.assertEqual(bool(learner.optimizer.state), variant == "adaptive")

    def test_policy_constraint_covers_all_current_legal_competitors(self):
        model, _ = build(4903, "spatial")
        torch.manual_seed(49)
        features = torch.randn(2, 2, 306) * .1
        with torch.no_grad():
            output = history_output(model, features)
        legal = torch.ones(2, 101, dtype=torch.bool)
        actions = output.policy_logits.argmax(1)
        alternatives = legal.clone().scatter_(1, actions[:, None], False)
        margins = output.policy_logits.gather(1, actions[:, None]).squeeze(1) - output.policy_logits.masked_fill(~alternatives, -torch.inf).max(1).values
        rule = RuleBatch(features[:, :1], torch.tensor([31]), torch.tensor([1.]))
        constraint = ReadableBehaviorConstraint(rule, torch.tensor([.01]),
            policy={"features": features, "actions": actions, "legal": legal, "floors": .5 * margins})
        self.assertTrue(torch.equal(constraint.margins(model)[1:], margins))
        analytic = readable_terminal_gradients(model, constraint.features(), constraint.objectives)
        complete = readable_terminal_gradients(model, constraint.features(), constraint.objectives, backend="autograd")
        self.assertLess(float((analytic.jacobian - complete.jacobian).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
