"""Algorithm-study contracts: geometry, causal feedback and finite protection."""
import copy
import io
import unittest

import torch

from experiments.go13_algorithm import build, materialize, make_suicide_pairs, pack_history
from mqr.go_history_tasks import generate_ko_history_pairs
from mqr.go_research import (OnlineRuleLearner, RuleBatch, RuleConstraint, history_output,
                             parameter_vector, queried_logits)
from mqr.temporal import TemporalMQRState


class AlgorithmResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, seed=4903, count=2):
        rows = []
        for index, p in enumerate(generate_ko_history_pairs(count, seed=seed, size=10)):
            rows.extend(pack_history(h, p.recapture, v, "ko", index) for h, v in zip(p.histories, p.legal))
        return materialize(rows)

    def test_rule_pairs_are_verified_and_final_observations_alias(self):
        batch = self.fixture(count=4)
        self.assertTrue(torch.equal(batch.features[-1, ::2], batch.features[-1, 1::2]))
        self.assertFalse(torch.equal(batch.features[0, ::2], batch.features[0, 1::2]))
        self.assertEqual(batch.labels.tolist(), [0., 1.] * 4)
        ordinary = materialize(make_suicide_pairs(4, 4905))
        self.assertEqual(ordinary.labels.tolist(), [0., 1.] * 4)
        self.assertEqual(ordinary.features.shape, batch.features.shape)

    def test_similarity_has_same_parameters_and_bounded_nonorthogonal_products(self):
        orth, _ = build(4907, "orth"); similar, _ = build(4907, "similar")
        self.assertTrue(torch.equal(parameter_vector(orth), parameter_vector(similar)))
        self.assertEqual(sum(p.numel() for p in orth.parameters()), sum(p.numel() for p in similar.parameters()))
        similar.double(); orth.double()
        # Nonzero input controllers exercise varying (noncommuting) rotations.
        with torch.no_grad():
            for module in similar.core.angle_controllers.values():
                module.weight.normal_(0, .01)
        xs = self.fixture().features.double()
        product = torch.eye(16, dtype=torch.float64)
        for k in range(32):
            a = similar.core.conditioned_transition(5, xs[k % 2, :1])[0]
            product = a @ product
            self.assertLessEqual(float(torch.linalg.matrix_norm(product.detach(), 2)), 2. + 1e-10)
        self.assertGreater(float((a.T @ a - torch.eye(16)).detach().abs().max()), .01)
        state = similar.core.zero_state(1, device="cpu", dtype=torch.float64)
        h = torch.randn(1, 16, dtype=torch.float64, requires_grad=True)
        state = TemporalMQRState(state.rings[:-1] + (h,))
        _, nxt = similar.core.forward_step(xs[0, :1], state=state)
        jac = torch.stack([torch.autograd.grad(nxt.rings[-1][0, i], h, retain_graph=True)[0][0] for i in range(16)])
        expected = .99 * similar.core.conditioned_transition(5, xs[0, :1])[0]
        torch.testing.assert_close(jac, expected, rtol=1e-10, atol=1e-10)

    def test_current_transport_equals_full_autograd_for_raw_rule_behavior(self):
        agent, _ = build(4911); agent.double()
        batch = self.fixture(); batch.features = batch.features.double(); batch.labels = batch.labels.double()
        floors = torch.ones(len(batch.labels), dtype=torch.float64) * .01
        a = RuleConstraint(batch, floors, backend="transport")
        b = RuleConstraint(batch, floors, backend="autograd")
        a.refresh(agent, 0); b.refresh(agent, 0)
        torch.testing.assert_close(a.normals, b.normals, atol=2e-9, rtol=2e-7)

    def stress(self, mode):
        agent, _ = build(4913)
        batch = self.fixture(count=1)
        with torch.no_grad():
            logits = queried_logits(history_output(agent, batch.features), batch.queries)
        # A numerical conflict witness, deliberately not a research dataset:
        # protect the current sign, then request its opposite.
        anchors = RuleBatch(batch.features, batch.queries, (logits >= 0).float())
        floors = logits.abs() * .99
        c = RuleConstraint(anchors, floors, refresh_every=8 if mode == "stale" else 1)
        learner = OnlineRuleLearner(agent, mode=mode, lr=.03, max_step=.5, constraint=c)
        conflict = RuleBatch(batch.features, batch.queries, 1 - anchors.labels)
        return learner, conflict

    def test_projection_and_nonlinear_guard_preserve_actual_anchor_behavior(self):
        learner, batch = self.stress("current")
        rows = [learner.step(batch) for _ in range(3)]
        self.assertTrue(any(r["raw_anchor_violations"] for r in rows))
        self.assertTrue(any(r["active"] or r["backtracks"] for r in rows))
        self.assertTrue(all(r["actual_anchor_violations"] == 0 for r in rows))
        self.assertEqual(learner.constraint.refreshes, 3)

    def test_cached_normals_are_reported_and_still_use_current_behavior_guard(self):
        learner, batch = self.stress("stale")
        a, b = learner.step(batch), learner.step(batch)
        self.assertEqual(a["normal_age"], 0)
        self.assertEqual(b["normal_age"], 1)
        self.assertEqual(learner.constraint.refreshes, 1)
        self.assertEqual(b["actual_anchor_violations"], 0)

    def test_predictions_precede_labels_and_checkpoint_resumes_exactly(self):
        batch = self.fixture()
        first, _ = build(4917); second, _ = build(4917)
        a, b = OnlineRuleLearner(first), OnlineRuleLearner(second)
        left = a.step(batch)
        right = b.step(RuleBatch(batch.features, batch.queries, 1 - batch.labels))
        self.assertEqual(left["pre_logits"], right["pre_logits"])
        self.assertFalse(torch.equal(parameter_vector(first), parameter_vector(second)))
        learner, conflict = self.stress("stale"); learner.step(conflict)
        buffer = io.BytesIO(); torch.save(learner.state_dict(), buffer); buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        resumed, _ = self.stress("stale"); resumed.load_state_dict(loaded)
        expected, actual = learner.step(conflict), resumed.step(conflict)
        self.assertEqual(expected, actual)
        self.assertTrue(torch.equal(parameter_vector(learner.agent), parameter_vector(resumed.agent)))


if __name__ == "__main__":
    unittest.main()
