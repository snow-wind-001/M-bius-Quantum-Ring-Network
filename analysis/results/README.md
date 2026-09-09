# Experiment Result Index

The canonical Phase-IV contextual-topology evidence is:

- `mqr_contextual_topology_formal_10seed.json`: ten seeds, two unknown opaque
  context topologies per seed, exact event/distractor joint-marginal matching,
  96 predict-before-update route steps per context, and 64 held-out episodes.
  Sparse-Givens and block-Cayley MQR both reach `1.0000` final accuracy and
  topology recovery; identity is `0.7127` and a direct ring buffer is `0.5029`.
  The equal-resource learnable sparse permutation also reaches `1.0000`, so
  `contextual_route_mechanism_qualified=true` while
  `mqr_independent_algorithm_advantage=false`. SHA-256 is
  `6b6fa524cd58cea70eb1013320444569bc5f5c134d3cde0aefcd1be31298916f`.
- `mqr_contextual_topology_sparse_lr02_sensitivity_10seed.json`: post-hoc
  strong-baseline sensitivity. Increasing only the sparse-permutation learning
  rate from 0.1 to 0.2 lowers B prequential regret from `0.10564` to `0.07230`,
  below Givens `0.08826`, while final accuracy remains tied. This artifact does
  not replace the primary run. SHA-256 is
  `9d7bfcf4e7f244a6d691265a9b8ab744294c179577149bf58b7d7324b57bf35d`.

Verify either artifact with `analysis/mqr_contextual_topology_verify.py` and
interpret it with `analysis/mqr_contextual_topology_report.md`.

The canonical 2026-09-01 residual address-routing artifact is:

- `mqr_routed_memory_qualification_formal_10seed.json`: ten seeds, 64 held-out
  episodes per seed, 64-byte state-matched long-delay, context-reversal, and
  repeated-query qualification. Residual routed MQR scores `1.0000`; paired
  improvements over identity, GRU, and fast-weight are
  `0.43379/0.49248/0.35176`, all with positive 95% Student-t intervals.
  Identity replacement hurts, the event/distractor write gap is `1.000`, and
  all five one-sided resource ratios pass 1.05. A direct fixed ring buffer also
  scores `1.0000`, so `mqr_route_mechanism_qualified=true` while
  `mqr_independent_algorithm_advantage=false`. Verify with
  `analysis/mqr_routed_memory_verify.py` and interpret with
  `analysis/mqr_routed_memory_report.md`. SHA-256 is
  `4823ea94a75f395ce550206ed4feaef4795b3c2ce4a07dbc5577c95771d8dbea`.

The canonical equal-resource 10x10/13x13 competitive results are:

- `unified_temporal_mqr_go_competitive_10x10_formal_10seed.json`: ten
  pre-registered seeds, 128 primary updates per task, 64 probes per task, and
  eight unmasked games per stage for MQR, identity, GRU, LSTM, fast-weight,
  LoRA, OGD-LoRA, replay-LoRA, and implicit Sinkhorn. SHA-256 is
  `4a0b5da17ceba3d0197e668833112b4753d079e4dfa8c560c52960c4fc8b6af2`.
- `unified_temporal_mqr_go_competitive_13x13_formal_10seed.json`: the exact
  cross-scale replication, differing only in board-dependent input width and
  pre-registered resource ranks. SHA-256 is
  `1e2fd0fe36014301a72fadccc38ba18c8be46ed1c23b9ab0e127d6251d2ab338`.
- `unified_temporal_mqr_go_competitive_joint_verify.json`: independent
  fail-closed recomputation of protocol, resources, occupancy, statistics, and
  the cross-scale decision.

Every method uses exactly 48 recurrent-state bytes. Parameter/forward/update
ratios are `1.0431/1.0304/1.0130` on 10x10 and
`1.0398/1.0253/1.0127` on 13x13. Online task loss decreases decisively on both
scales, but history focus remains exactly 0.5 and MQR does not beat identity or
the recurrent controls. The joint decision is `mqr_effective=false`. Verify
with `analysis/unified_temporal_mqr_go_competitive_verify.py` and interpret with
`analysis/mqr_formal_competitive_go_report.md`.

Those ratios are the immutable run-time records. The current auditor no longer
counts the inactive Sinkhorn full-leak transition as a frozen parameter and
charges a Cayley refresh only after a committed transition update. It therefore
reconstructs parameter/state/forward/update ratios of
`1.0436/1.0000/1.0328/1.0114` on 10x10 and
`1.0403/1.0000/1.0243/1.0099` on 13x13. Both versions pass the same gate; the
seeded behavior path and archived result JSON are unchanged.

Files containing `smoke`, `qualification`, or `pilot` in this experiment
family are development diagnostics and are not substitutes for these formal
artifacts.

The canonical 2026-08-29 mechanism-isolation artifacts are:

- `mqr_mechanism_qualification_formal_10seed.json`: schema-v2, ten-seed,
  length-12 marker-free repeated-query and reversal qualification. The future
  loss reaches the earliest input, injection, and cyclic transition, but cyclic
  MQR history accuracy is `0.5375` versus identity `0.5875`; the paired
  cyclic-minus-identity 95% Student-t interval is
  `[-0.07890,-0.02110]`. Reversal, identity-replacement, forgetting, and full
  competitive-resource gates fail, so `mqr_effective=false`. Verify with
  `analysis/mqr_mechanism_qualification_verify.py`.
- `mqr_cyclic_capacity_sweep_formal_10seed.json`: ten seeds at exact recurrent
  state budgets 48/192/768 B for cyclic MQR versus identity. All three
  amortized 1.05 resource gates pass without dummy padding, while the paired
  accuracy effects are `-0.02875/-0.03250/-0.07125`; only the 768 B interval
  is decisively negative. Peak refresh ratios are
  `1.1596/1.7122/2.3110` and remain separately failed. Verify with
  `analysis/mqr_capacity_sweep_verify.py`.

Interpret these results together with the formal Go evidence in
`analysis/mqr_mechanism_and_capacity_report.md`. The capacity sweep isolates
MQR versus identity only; it does not replace the nine-method 10x10/13x13
comparison. Pre-v2 and single-seed mechanism files are retained only as
historical diagnostics and are not publication evidence.

The canonical frozen-backbone application-positioning artifacts are:

- `frozen_sidecar_application_qualification.json`: ten deterministic seeds of
  a synthetic, frozen-backbone session-personalization and error-correction
  qualification. The sidecar is initially an exact no-op, only its residual
  readout changes, session memory reaches 1.0 accuracy versus 0.5 with reset
  state, the protected-probe KL stays below `2.86e-16`, and all unsafe
  candidates roll back. This is mechanism evidence, not real-model efficacy or
  MQR superiority. Regenerate it with
  `experiments/frozen_sidecar_application_qualification.py`.
- `mqr_application_positioning_audit.json`: fail-closed synthesis of the
  application qualification, real MiniCPM connectivity smoke, and the formal
  10x10/13x13 joint competitive evidence. This schema-v2 audit confirms that
  all requested protocol, baseline, feedback, and resource conditions are now
  present while the joint performance gate remains negative. Regenerate it with
  `analysis/mqr_application_positioning_verify.py`. It records that the
  frozen-sidecar positioning is evidence-consistent while real-model gain,
  general capability enhancement, animal-like learning, and independent MQR
  advantage remain unestablished.

The canonical Phase-II sidecar safety certification is:

- `mqr_sidecar_safety_certification.json`: 20 signed-state bound cases, 40
  finite Cayley-drift cases, constrained-OGD masking, an exact initial
  residual no-op, and atomic agent/LoRA rollback. Regenerate it with
  `analysis/mqr_sidecar_safety_verify.py`. All checks pass. This artifact
  certifies numerical and transaction semantics only; it is not a task result
  and does not change `mqr_effective=false`.

The canonical 2026-08-27 unified-agent results are:

- `unified_temporal_mqr_go.json`: five-seed AWR → history-dependent superko →
  short actor-critic comparison of unistochastic MQR, identity MQR, GRU, and
  fast-weight cores. Parameters and analytical MACs are within 5%, raw state is
  not. Verify with `analysis/unified_temporal_mqr_go_verify.py`; SHA-256 is
  `397410ac507448ba99e32de225466f0694206386ca67f7cafe41034568592fba`.
  The strict MQR effectiveness gate is false.
- `unified_temporal_mqr_minicpm_go_smoke.json`: real local MiniCPM5-1B external
  feature-gradient smoke test with one scheduled LoRA update and zero sampled
  frozen-weight drift. Verify with
  `analysis/unified_temporal_mqr_minicpm_go_verify.py`; SHA-256 is
  `cff28f78fccc9a0cdd2564ef831c7f32b502111782d5c5d9c88221ccc644882f`.
  It is connectivity evidence, not a performance result.

Interpret both with `analysis/unified_temporal_mqr_go_report.md`.

The canonical 2026-08-25 competitive reversal and Go v2 runs are:

- `temporal_mqr_reversal_matched.json`: five fresh-seed A→B→A reversal runs
  comparing MQR utility variants with GRU, fast-weight, LoRA, OGD-LoRA, and
  equal-byte replay under a 288–320 adaptive-parameter range and 4096-byte
  tensor-state cap. Verify with `analysis/temporal_mqr_reversal_verify.py`;
  SHA-256 is
  `f170b1829c38c8981d930b9df5c695754e0f240f61b675e14d1ec74a856c2d16`.
  Total sidecar parameters and FLOPs are not matched, and the pre-registered
  independent-advantage gate is false.
- `minicpm_go_real_games_v2_5seed.json`: five fresh seeds for combined,
  MQR-only, LoRA-only, and no-learning conditions, with eight training games,
  eight paired pre/post games, exact placement/pass probes, raw legality, and
  probe curves. Verify with `analysis/minicpm_go_real_games_v2_verify.py`;
  SHA-256 is
  `11fb69ab06f2421346a105c39c7e0a721e790360db9f2dc323ebbe2b41ae7b04`.
  The masked-margin trend is inconclusive and internal Go metrics degrade.

Interpret both with `analysis/mqr_reversal_go_v2_report.md`. Externally masked
legality is never evidence that the model learned Go rules.

The canonical delayed future-utility gate experiment is:

- `temporal_mqr_utility_digits.json`: 21 paired seed-condition runs comparing
  learned utility regression, learned utility with OGD, a frozen controller,
  causal novelty, a clairvoyant target-position oracle, ungated writes, and
  random sparse writes. The marker-free target position is randomized; paired
  shadow trajectories supervise future write advantage only after the action.
  Verify it with `analysis/temporal_mqr_utility_verify.py` and interpret it with
  `analysis/utility_mqr_research_report.md`. The target remains causally
  discoverable from novelty, so this supports delayed utility learning on an
  observable pattern, not arbitrary future-relevance prediction or competitive
  advantage over GRU, fast-weight, LoRA, or replay.

The canonical causal auxiliary learned-gate experiment is:

- `temporal_mqr_learned_gate_digits.json`: 24 seed-condition runs comparing
  learned/oracle/ungated/frozen-initial gates, false-open/false-close corruption,
  and equal-state single-slow versus multiscale memory. Initial hashes, controller
  trajectories, counters, and zero held-out drift are independently audited by
  `analysis/temporal_mqr_learned_gate_verify.py`. Interpret it with
  `analysis/temporal_mqr_runtime_report.md`. It supports online learning from an
  explicit marker plus immediate post-decision event labels; it does not support
  unsupervised salience discovery, delayed reward learning, or a multiscale edge.

The canonical practical Temporal MQR runtime stress test is:

- `temporal_mqr_interference_digits.json`: 21 seed-condition runs with real
  digit distractors, paired ungated/oracle/random write policies, single-slow
  and multiscale state, exact initialization hashes, and zero-drift audits.
  Interpret it with `analysis/temporal_mqr_runtime_report.md` and verify it via
  `analysis/temporal_mqr_interference_verify.py`. It supports externally
  controlled anti-interference writes, not a learned gate or multiscale edge.

The canonical temporal-memory mechanism run is:

- `temporal_mqr_delayed_digits.json`: three-seed, predict-before-update delayed
  Digits experiment comparing equilibrium decay, fast/slow/multiscale state,
  coupled/decoupled writes, identity/unistochastic topology, and exact
  no-learning controls. Interpret it with
  `analysis/temporal_mqr_validation_report.md`; it supports temporal state and
  write/decay decoupling, not a Cayley-topology advantage.

The canonical 2026-08-24 MiniCPM5/Sayuri runs are:

- `minicpm_go_sayuri_game_lora_24.json`: one evolving game with LoRA;
- `minicpm_go_sayuri_dataset_lora_48.json`: eight repeated positions with LoRA;
- `minicpm_go_sayuri_dataset_no_lora_48.json`: matched no-LoRA control;
- `minicpm_go_sayuri_game_resume_2.json`: two-step continuation from the 24-step checkpoint;
- `online_digits_principle.json`: three-seed download-free continual-learning baseline.
- `minicpm_go_real_games_3seed.json`: canonical native-policy, causal multi-game
  comparison across three seeds and four learning/prompt conditions;
- `minicpm_go_real_games_curriculum_3seed.json`: matched pass-curriculum
  experiment evaluated against the curriculum teacher;
- `minicpm_go_real_games_curriculum_native_eval_3seed.json`: the same curriculum
  training design cross-evaluated against native Sayuri policy;
- `minicpm_go_real_games_prompt_controls_3seed.json`: correct-rules,
  wrong-rules, and board-only counterfactual prompt controls under native
  Sayuri cross-evaluation.

Earlier `minicpm_go_online.json`, `minicpm_go_smoke.json`,
`minicpm_go_dataset*.json`, and `minicpm_go_sayuri_smoke.json` are exploratory
pre-fix records. Their static `execution_mode` and `limitations` strings may not
match their actual teacher/dtype/LoRA settings, and dataset runs carried state
across independent positions. Keep them only as historical diagnostics; do not
use them for reported conclusions.

`minicpm_go_real_games_*smoke.json` and the one-seed 7×7 result are mechanism
diagnostics, not headline evidence. Interpret all multi-game metrics with
`analysis/minicpm_go_real_games_report.md`; externally masked played-move
legality is never evidence of internal rule learning.

JSON metrics contain model and teacher paths for local reproducibility but do
not embed either checkpoint. Trainable `.pt` states are written under the
git-ignored `.external/mqr-checkpoints/` directory.
