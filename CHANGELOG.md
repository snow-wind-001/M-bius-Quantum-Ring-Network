# Changelog

## 2026-09-09

- Added signed cyclic Givens propagation with sparse forward/inverse operations,
  correct L2 and drift diagnostics, and an identity path without dense identity multiplication.
- Added causal Go feedback windows, exact pending-window recovery, current-output
  OGD consolidation, and optional ring modulation of frozen spatial features.
- Added small-model Go comparisons and an evolving-game online entry point with
  checkpoint continuation. Five paired seeds show about 50% / 61% less old-task
  NLL increase with OGD in the global/spatial variants, at a cost to new-task adaptation.
  Signed rings do not beat identity controls; Go strength improvement is not established.
- Recorded eight live training games with 268 feedback positions and 38 updates,
  including checkpoint continuation and a Sayuri policy teacher; all games were lost.
- Added structural proofs, reachable-subspace diagnostics, ten passing root test
  suites, result recomputation, full recorded-game replay, and a plotted research report.
- Added a native CodeRecoder snapshot/verification helper. Fixed and tested a
  manifest sorting inconsistency in the local companion CodeRecoder checkout;
  that tool repair is separate from this repository.

## 2026-09-01

- **Unknown contextual topology gate**:
  - Added a bounded contextual route bank and a learnable sparse-permutation
    control on exactly the same local factor graph as sparse Givens.
  - Added an exact matched-marginal generator: event and distractor candidates
    are permutations of the same `(key,value)` multiset, while the write critic
    learns finite future write/no-write query-loss gain rather than event labels.
  - Ran ten A -> B -> A seeds. Sparse Givens and block Cayley recover all hidden
    routes and score `1.000`; identity and direct ring buffer score `0.7127` and
    `0.5029`. Inactive context parameters remain bitwise unchanged.
  - The equal-resource learnable sparse route also scores `1.000`. A learning-rate
    sensitivity check reverses the prequential-regret ranking in its favor, so
    the contextual mechanism gate passes but independent MQR advantage remains
    false. Go, MiniCPM, and policy RL stay outside the active experiment stage.
- **Evidence and tests**:
  - Added a fail-closed verifier that regenerates topologies and marginal audits,
    checks raw prequential curves/context isolation, and recomputes resources,
    Student-t intervals, and decisions.
  - Expanded routed-memory tests from eight to eleven and added the formal report,
    primary result, and strong-baseline sensitivity artifact.

## 2026-08-29

- **Cayley representation closure**:
  - Proved that although one Cayley chart omits unitary representatives with
    eigenvalue `-1`, global-phase invariance makes the composite
    `A -> Cayley(A) -> |U|^2` surjective onto the complete unistochastic set.
  - Added `set_from_unitary_representative_()`, which searches more than `N`
    deterministic phases, maximizes `sigma_min(I + zeta U)`, imports a finite
    inverse-Cayley coordinate, and reports reconstruction diagnostics.
  - Added unit and float64 proof checks; the seeded transition reconstruction
    error is `4.44e-16`.
- **Finite trajectory credit boundary**:
  - Formalized that same-version bounded trajectory replay gives the exact
    finite-BPTT gradient and that contraction provides only the upper bound
    `||(dh_t)/(dh_s)|| <= (1-lambda)^(t-s)`, never a positive lower bound.
  - Added a numerical Jacobian upper-bound check and a saturated-tanh exact-zero
    witness. Future-gradient connectivity is no longer described as useful
    credit by itself.
- **Mechanism experiment correction and formal rerun**:
  - Bumped the repeated-query experiment to schema 2, classified all three
    queries correctly, calibrated the critic on all causal query positions,
    replaced normal intervals with two-sided 95% Student-t intervals, and made
    the independent-advantage gate intersectional across identity, recurrent,
    reversal, forgetting, credit, gate, and resource criteria.
  - The independently verified ten-seed run reaches MQR history accuracy
    `0.5375`, but identity reaches `0.5875`; paired MQR-minus-identity CI is
    `[-0.07890,-0.02110]`. Future credit reaches injection and transition, yet
    reversal and identity-replacement gates fail, so `mqr_effective=false`.
- **Equal-state capacity and refresh accounting**:
  - Added a 48/192/768 B cyclic-MQR versus identity sweep with no dummy padding,
    real committed-transition refresh counts, and separate amortized and peak
    update gates. Every run records exactly five cyclic refreshes and zero
    identity refreshes under the interval-32 schedule.
  - Replaced Sinkhorn's unused full-leak transition parameter with a
    checkpoint-key-compatible buffer. It consumes the historical seeded draw
    for downstream reproducibility but contributes no trainable or frozen
    parameter capacity, matching MQR's parameter-free inactive transition.
    This restores the exact frozen-base resource gate without padding.
  - All amortized 1.05 gates pass, while paired history effects remain
    `-0.02875/-0.03250/-0.07125`; peak update ratios
    `1.1596/1.7122/2.3110` all fail. Both capacity-advantage decisions are
    false.
- **Evidence synchronization**:
  - Added `analysis/mqr_mechanism_and_capacity_report.md`, two fail-closed
    verifiers and their canonical formal JSON artifacts. Updated the theory,
    revision plan, result registry, repository summary, README, and paper to
    separate structural correctness, graph connectivity, and independent
    algorithmic advantage.

## 2026-08-28

- **Formal equal-resource 10x10/13x13 online Go study**:
  - Added LSTM, low-rank residual LoRA/OGD-LoRA/replay-LoRA, and implicitly
    differentiated Sinkhorn cores to the shared Temporal Utility Agent.
  - Replaced the old `9/12/9/48`-scalar state mismatch with exactly 12 float32
    scalars (48 B) for all nine methods. A shared frozen base, zero-initialized
    residual, local rule-neutral spatial path, heads, data, losses, feedback,
    and seeded content-independent write route are verified exactly.
  - Ten pre-registered seeds on each of 10x10 and 13x13 use 128 primary online
    updates per task, 64 held-out probes, and eight unmasked games per stage.
    Parameter/forward/update ratios are at most `1.0431/1.0304/1.0130` and
    `1.0398/1.0253/1.0127`, respectively; both strict resource gates pass.
  - MQR task-A held-out loss reduction is `2.8331` on 10x10 and `3.0168` on
    13x13 with bootstrap intervals strictly above zero, confirming functional
    online weight learning. History-focus accuracy remains `0.500`; MQR ties
    identity and several recurrent controls, and trails replay-LoRA on 13x13
    dynamic legality/return. The replicated decision remains
    `mqr_effective=false`.
  - Added `test_competitive_go.py`, a resumable formal experiment, a joint
    fail-closed verifier, two canonical JSON results, and
    `analysis/mqr_formal_competitive_go_report.md`.

- **Phase-II safe temporal sidecar**:
  - Added `TemporalMQRSidecar` with an exactly zero-initialized residual,
    optional residual clipping, and a signed-state infinity-norm certificate.
  - Added finite Cayley drift diagnostics with
    `||delta U||_F <= 2||delta A||_F` and
    `||delta H||_F <= 4||delta A||_F`, plus minimal Cayley coordinates for the
    temporal core.
  - Added candidate-update constancy guards for the unified agent and MiniCPM
    LoRA bridge. Policy KL, output/state drift, and transition drift can reject
    an update while restoring parameters and OGD memory atomically. The Agent
    additionally restores its parameter version and revokes external-gradient
    authorization; the LoRA bridge has no parameter-version counter.
  - Made the MiniCPM LoRA bridge fail closed when given recurrent-state or
    Cayley-transition limits that it cannot observe; that bridge accepts only
    policy-KL/output constancy limits, while the unified agent owns state and
    transition audits.
  - Added an exact OGD projection inside a scheduled parameter subspace. Fast
    blocks can update while Cayley coordinates remain zero between slow ticks;
    low retained norm emits a plasticity warning instead of forcing an update.
  - Added `analysis/mqr_sidecar_safety_verify.py` and its canonical JSON. All
    20 signed-state and 40 Cayley-drift cases pass; deliberate agent and LoRA
    violations restore zero parameter drift and zero OGD rank.

- **Certified fixed-point updates**:
  - Added optional forward/adjoint tolerances, iteration counts, absolute and
    relative residuals, and the contraction certificate `residual / alpha`.
  - Added a direct nonlinear-adjoint oracle. When certification is requested,
    a failed solve now rejects the complete parameter, OGD-memory, and external
    feature-gradient transaction unless `allow_inexact_update=True` is explicit.
- **Cayley representation audit**:
  - Proved that the historical projected coordinates cover all of `u(N)` but
    store `2N^2` scalars for `N^2` effective degrees of freedom.
  - Added checkpoint-compatible `projected` and orthonormal `minimal` modes,
    coordinate pullbacks, and local `A -> U -> |U|^2` Jacobian diagnostics.
    The seeded `N=4` check reaches rank 9 away from identity and rank 0 at it.
- **Fair mathematical baseline**:
  - Added log-space Sinkhorn normalization with an implicit pullback for the
    converged scaling problem. Float64 relative pullback errors are
    `1.17e-15` for Cayley and `1.97e-15` for Sinkhorn on the matched check.
- **Sidecar and Go corrections**:
  - Added zero-initialized residual readout, 10x10 lossless 306-dimensional
    board features, arbitrary-size ko/superko pairs, and large-action hashing.
  - Replaced direct pass/placement-logit concatenation with the normalized
    factorization `P(point)=P(not-pass)P(point|not-pass)`.
- **Validation and evidence boundary**:
  - All six test entry points pass, totaling 100 tests, and both finite-solver
    error bounds hold throughout the float64 scan.
  - The corrected 10x10 five-seed pilot remains negative: MQR dynamic useful
    legality is `0.0153`, return is `-17.125`, and history focus is `0.500`.
    It trails identity and GRU, while the persistent-state matching gate also
    fails. The formal effectiveness decision remains false.
  - Added `analysis/mqr_sidecar_revision_plan.md`, the corrected pilot report,
    and synchronized the mathematical audit and paper with these claim limits.

## 2026-08-27

- **Unified Temporal Utility MQR Agent**:
  - Added a causal multi-timescale Agent with placement, legality, pass, and
    value heads; learned legality-policy fusion; illegal probability-mass loss;
    offline AWR; PPO-style short actor-critic; GAE; task/utility tickets; and
    slow external LoRA consolidation.
  - Added arbitrary reduced `grad_logits` to the fixed-point implicit updater.
    Cross-entropy and generic-gradient transactions match exactly in tests.
  - Added matched future write/no-write rollouts, a bounded causal critic
    buffer, and post-phase recalibration. Removed per-sample advantage clipping,
    which can reverse the conditional-mean sign required by the optimal write
    decision. Task OGD remains shared; non-stationary utility targets no longer
    use invalid cross-stage OGD projection.
- **Go mechanism and controls**:
  - Added frozen vector and MiniCPM board-encoder contracts, deterministic legal
    trajectories, and paired superko/fresh-history examples whose current input
    is identical while the legal recapture target differs.
  - Added identity MQR, GRU, and fast-weight cores under the same routing,
    heads, loss, OGD, data, and update count, plus fail-closed parameter/state/
    MAC and performance gates.
  - Five seeds matched trainable parameters and analytical MACs within 5%, but
    raw state differed by 8.33x. MQR reached chance-level 0.500 history-focus
    legality and did not jointly beat all controls on useful legality, return,
    and forgetting. The formal effectiveness claim remains false.
- **Slow MiniCPM LoRA smoke test**:
  - Verified one scheduled external-gradient update of 21,504 LoRA parameters
    on the local MiniCPM5-1B checkpoint. Adapter drift was non-zero while sampled
    frozen-weight drift was exactly zero; two-example probe loss worsened, so
    this is connectivity evidence only.
- **Verification and reporting**:
  - Added `test_unified_mqr_agent.py`, two executable experiments, two independent
    result verifiers, and `analysis/unified_temporal_mqr_go_report.md` with the
    mathematical claim boundaries and complete negative result.

## 2026-08-25

- **Temporal MQR-v6 marker-free contextual utility experts**:
  - Added a causal past-volatility trace and two hard-routed rank-4 utility
    experts under a total rank-8 gate budget. Routing never consumes a task ID,
    current label, or future feedback; legacy v5 behavior remains the default.
  - Added exact selected-expert attribution to tickets/checkpoints and tests
    proving unselected-expert zero drift, including under block-supported OGD.
  - Added a five-seed A→B→A reversal benchmark against parameter/state-capped
    GRU, fast-weight, LoRA, OGD-LoRA, and equal-byte replay. Expert MQR+OGD
    reduced LoRA's A-after-B drop by `36.75 pp` (bootstrap 95% CI
    `[34.00,39.50]`) and improved final balanced accuracy by `11.63 pp`, but
    lost `8.58 pp` in mean post-task accuracy and was about 75× slower.
  - The pre-registered independent-advantage gate therefore remains false;
    adaptive parameters and tensor-state bytes are capped, while total sidecar
    parameters and FLOPs are explicitly not matched.
- **MiniCPM5/Sayuri Go v2 evaluation**:
  - Added a true LoRA-only update condition, exact placement/pass probe strata,
    conditional placement metrics, pass calibration, raw student legality, and
    probe learning curves.
  - Ran five new seeds with eight training and eight paired evaluation games per
    condition. Masked margin changed by `+5.75` with a 95% t interval crossing
    zero, while native probe loss and pass Brier degraded on all five seeds.
  - Combined and MQR-only discrete behavior was identical; LoRA-only parameters
    moved without changing games. External legality masking remains excluded
    from claims of internal Go-rule learning.
- **Verification and documentation**:
  - Expanded Utility tests to 11 and Go/MiniCPM tests to 21, including exact
    probe stratification/decomposition, curve arithmetic, and LoRA-only MQR
    parameter isolation.
  - Added independent result verifiers and
    `analysis/mqr_reversal_go_v2_report.md` with mathematical claim boundaries.

## 2026-08-24

- **Temporal MQR-v5 future-utility write learning**:
  - Added `UtilityDrivenMQR` and `FutureUtilityGate`: the fast ring always sees
    candidates, while a low-rank controller predicts slow-ring write advantage
    from strictly causal novelty, uncertainty, past surprise, saturation, and
    context-change features.
  - Candidate tickets maintain paired write/no-write shadow trajectories. Future
    task feedback supervises `loss(no write) - loss(write) - memory cost` through
    feature replay; no event-identity label or task gradient crosses the action.
  - Added bounded delayed promotion, explicit irreversible false-positive
    reporting, keyed state/ticket recovery, independent gate/readout OGD and
    clipping, staleness counters, and external forcing in the temporal core.
  - Added nine strict tests for promotion-preserving contraction, causal
    no-leakage, ticket capacity/single use, useful promotion, false positives,
    complete-vector OGD/clipping, prequential readout feedback, and exact resume.
  - A three-seed marker-free Digits mechanism screen reached
    `72.00±6.24%` with utility OGD versus `57.67±27.23%` with plain utility
    updates, `10.33±3.79%` without gate updates, `8.67±3.79%` ungated, and
    `17.00±3.00%` random sparse writes. The OGD gate matched a causal novelty
    rule and the task's clairvoyant oracle; its advantage over plain updates was
    driven by preventing one seed's false-open collapse, not a uniform final gain.
  - The task makes future relevance causally predictable through novelty. This
    establishes delayed utility learning on a controlled observable pattern,
    not arbitrary salience discovery or superiority to GRU/fast-weight/LoRA/replay.

- **Temporal MQR-v4 causal learned write gate**:
  - Added the low-rank `TemporalWriteGate`, optional independent event inputs,
    continuous per-timescale gates, external override, class-balanced auxiliary
    BCE, and a strict stop-gradient boundary from task loss to gate parameters.
  - Gate and task gradients use separate complete-vector OGD memories and norm
    caps, then commit together at the transaction tail with one parameter-version
    increment. Gate weights, OGD, labels/updates, state and outputs resume exactly.
  - Expanded the Temporal suite from 22 to 27 tests. Different current event
    labels provably leave the current gate/state/logits identical; BCE descent,
    OGD layout/clipping, external override, and legacy checkpoints are covered.
  - Added a 24-run, three-seed marked-Digits experiment and independent verifier.
    The 44-parameter learned gate reached `70.83±6.55%` held-out accuracy versus
    `19.86±1.68%` ungated and `21.53±0.87%` without gate updates. Oracle was
    `72.78±7.46%`; 10%/25% false-open cost `20.28±5.37`/`33.75±7.23 pp`.
  - The controller sees an explicit non-class marker and immediate post-decision
    event labels. This supports causal auxiliary event learning, not unsupervised
    salience discovery or delayed-reward credit assignment. A single slow ring
    still beat the equal-state multiscale model by `2.78±1.20 pp`.

- **Temporal MQR-v3 practical online runtime**:
  - Added external per-sample/per-timescale write gates without changing the
    recurrent contraction factor, plus a capacity-bounded keyed state bank
    with fail-closed or explicitly auditable LRU overflow.
  - Added `infer_step`/`apply_feedback` transactions. Single-use tickets cache
    issue-time readout features/logits, report parameter staleness, enforce
    capacity/TTL/duplicate errors, and use a separate readout OGD basis.
  - Runtime state, LRU order/usage, pending tickets, parameter versions, and
    both OGD memories now round-trip through `state_dict`; legacy one-state
    Temporal checkpoints remain loadable.
  - Expanded the strict Temporal suite from 14 to 22 tests, covering exact
    gating, gated contraction, real-interference query isolation, interleaved
    session zero-crosstalk, capacity/LRU, exact delayed gradients, ticket
    failures, and runtime resume.
  - Added a 21-run, three-seed real-distractor Digits experiment. Multiscale
    oracle cue-only gating beat matched ungated state by `53.33±6.17 pp` and a
    same-sparsity random gate by `55.31±6.27 pp`; evaluation drift was exactly
    zero. The gate is external, and multiscale still did not beat one slow ring.
  - Added an independent result verifier and
    `analysis/temporal_mqr_runtime_report.md` with strict perturbation, delayed
    readout-gradient, isolation, capacity, and 2B integration boundaries.

- **Temporal MQR-v2 and causal delayed-memory validation**:
  - Added one-step multi-timescale state dynamics with independently configured
    decay and write strength, a predict-before-update online wrapper, unified
    OGD/trust-region updates, state/gradient checkpointing, and cached frozen
    Cayley transitions.
  - Added 14 strict tests covering exact half-lives, contraction, paired
    topology/write ablations, stream isolation, label leakage, state-versus-
    parameter attribution, OGD, clipping, and exact resume.
  - Added a three-seed delayed-Digits stream. Decoupling write strength from
    decay improved held-out accuracy by `55.31±7.10 pp`; temporal MQR beat the
    equilibrium-decay control by `68.44±2.25 pp`.
  - The same experiment found no topology advantage: matched unistochastic
    rings trailed identity by `1.25±0.83 pp`, and one slow ring beat the fixed-
    budget multiscale bank by `3.54±0.18 pp`. These negative controls are part
    of the reported result, not discarded tuning runs.

- **Causal multi-game Go validation**:
  - Added a read-only `preview_step` so an old-policy move can be committed to
    the board before feedback updates parameters; regression tests require the
    deferred update to reproduce identical pre-update logits.
  - Added `experiments/minicpm_go_real_games.py` with alternating student
    colors, learner-induced trajectories, fixed probes, paired pre/post games,
    exact no-learning audits, parameter drift, prompt controls, and native
    Sayuri cross-evaluation.
  - Native Sayuri labels improved three-seed probe loss by `0.335±0.067` but
    worsened paired margin by `11.5±8.2`, exposing a repeatable pass-recency
    collapse rather than practical Go improvement.
  - Added an explicit pass-gated curriculum diagnostic. It produced a
    preliminary `+6.5±7.7` margin change against native Sayuri, while raw
    legality and fixed-probe imitation did not improve; this is not yet a
    statistically resolved strength claim.
  - Added correct-rules, wrong-rules, and board-only prompts. Correct rules did
    not reliably beat wrong rules, so semantic rule understanding remains
    unproven. LoRA again had no observable behavioral increment over MQR-only.
- **Research documentation**:
  - Added `analysis/minicpm_go_real_games_report.md` with the causal protocol,
    complete results, prompt non-identifiability proof, OGD limits, surrogate
    mismatch analysis, and a hierarchical legality/point/pass/value design.

- **MiniCPM5 bridge**:
  - Added verified decoding of the local MiniCPM5-1B AWQ INT4 `compressed-tensors` checkpoint into a frozen FP16 execution backbone without modifying the checkpoint.
  - Added FP32 LoRA on the final decoder layer's `q_proj/v_proj` and an external-gradient VJP bridge from the MQR implicit action loss; 43,008 LoRA parameters are trainable.
- **Go and Sayuri**:
  - Added a strict 5×5 Go environment with capture, suicide, situational superko, pass termination, Chinese area scoring, and GTP coordinates.
  - Added a timeout-aware subprocess-only GTP adapter for local Sayuri commit `396b1d07`; the GPLv3 engine remains separately built. Policy/MCTS selection and sampled legal-move equivalence are tested.
- **Online algorithm fixes**:
  - Added an atomic trust region over the complete MQR displacement and stationary LayerNorm/tanh features to prevent online divergence.
  - Disabled recurrent state carry for independent dataset positions while retaining it for evolving games.
  - Changed OGD retention from a global step modulus to per-context counters, eliminating black/white phase aliasing; added a regression test and per-ring rank diagnostics.
  - Added a formal and numerical proof that blockwise OGD remains globally first-order orthogonal under independent positive trust-region scaling.
  - Made online checkpoints board-aligned by persisting move history, context counters, cumulative steps, routing/ring state, and both OGD memories; legacy checkpoints now clear unmatched transient ring state.
- **Initial validation**:
  - Sayuri-policy 24-step game: one completed game, held-out loss `3.289→3.170`, approximately 2.20 GB peak CUDA allocation and 40 ms mean online latency.
  - Repeated eight-position 48-step stream: first/last-quarter loss `3.149→2.249`, 75% replay masked agreement, and zero legal-set mismatches in eight sampled positions.
  - The matched no-LoRA result was effectively identical (mean absolute step-loss difference `2.17e-5`), so the evidence demonstrates a connected LoRA gradient path but not a LoRA benefit.
- **Documentation**:
  - Added `analysis/minicpm_sayuri_online_report.md` and updated the README, proof, research plan, and project summary with reproducible commands and explicit evidence limits.

## 2026-08-23

- **Online continual learning**:
  - Added `OnlineMultiRingClassifier` with read-before-write prequential steps, persistent per-ring state, stable explicit-context routing, capacity-safe isolation, and an experimental cosine-novelty router.
  - Added `OrthogonalGradientMemory`; all active ring gradients are now collected before mutation and projected once in learning-rate-whitened coordinates. Staged collection can protect later contexts without suppressing learning inside the current one.
  - Persisted routing, runtime states, and gradient bases through `state_dict`.
- **Principle experiment**:
  - Added a download-free, single-pass Digits A→B stream comparing one shared ring, OGD-64, and two isolated rings across three seeds.
  - Mean A forgetting was `17.96`, `14.63`, and `0.00` percentage points respectively. The result also exposes the trade-offs: OGD reduced B plasticity and its dense rank-64 basis used 1.76 MB; exact isolation doubled parameters and reduced cross-context transfer.
- **Online verification and research documentation**:
  - Added six focused tests for prequential label isolation, heterogeneous-step OGD identities, exact multi-ring isolation, state carry, unified projection, routing, and checkpoint round-trips.
  - Added `analysis/online_learning_research_plan.md` and updated the strict proof with the weighted OGD theorem, actual Digits evidence, failure boundaries, and staged 2B validation gates.
- **Correctness (implicit/Cayley gradients)**:
  - Replaced the historical Lie-algebra heuristic with the exact chain rule through the equilibrium, `H=|U|²`, dual-unitary multiplication, and the Cayley coordinate.
  - Fixed the missing `(1-α)/α` adjoint scaling, duplicate batch normalization, and exact learnable-β gradient.
  - Applied the same exact Cayley pullback to complex unitary dynamics.
- **Verification**:
  - Added Autograd-equivalence tests for real and complex equilibrium gradients, a Cayley differential test, a descent-direction test, and a duplicated-batch invariance test (`16/16` total tests pass).
  - Rebuilt `analysis/mqr_proof_verify.py`; the reconstructed legacy rule is negatively aligned with the true gradient in 27/30 seeded cases, while the corrected gradient matches Autograd to floating-point precision.
- **Analysis**:
  - Rewrote `analysis/mqr_math_proof.md` with strict convergence/gradient proofs, the linear-ring = LoRA theorem, orthogonal-gradient continual-learning guarantees, and a costed 2B sidecar design.

## 2026-01-11

- **Paper (mqr_arxiv.tex) major revision based on ICML-level review feedback**:
  - Fixed Remark 3.7: corrected the "all 1/3 matrix is not unistochastic" example with a proper counter-example.
  - Fixed Cayley bijection claim: added remark about eigenvalue -1 exception (measure-zero set in practice).
  - Added rigorous gradient derivation (Proposition: Cayley Differential) for $\partial\mathcal{L}/\partial A$.
  - Added quantum metaphor disclaimer in Abstract (clarifying MQR is classical, not quantum computing).
  - Added Related Work comparison table (4 methods × 5 attributes).
  - Updated Algorithm 1 with stopping criterion (residual < ε or max K) and warm-start option.
  - Added remarks on theorem applicability (h non-negativity for L1 conservation, 1-Lipschitz for nonlinear extension).
  - Added architecture figure (network_architecture.pdf) to Section 4.
  - Added hyperparameter summary table for reproducibility.
  - Added training curves figure (side-by-side accuracy and loss curves).
  - Added ablation table (patch pooling: mean vs. flatten).
  - Toned down exaggerated claims ("powerful synthesis", "new paradigm" → "alternative design point").
  - Fixed undefined section references (sec:ext_hmix → sec:ext_beta, sec:eqprop → sec:learning).

## 2026-01-06

- **Refactor (engineering structure)**: added a minimal `mqr/` package and made `mobius_quantum_ring.py` a backward-compatible facade.
- **Algorithm (HTML reproduction)**: implemented the MQR/UHR-Net core dynamics from `Möbius Quantum Ring.html`:
  - Cayley unitary parameterization (skew-Hermitian \(A\)) and unistochastic connection \(H=|U|^2\)
  - Fixed-point relaxation inference \(h \leftarrow (1-\alpha)\,h\,H^T + \alpha\,\mathcal{J}(x)\)
  - LoRA-style Hamiltonian injection \(\mathcal{J}(x)=W_{up}W_{down}x\)
  - Local projective sampling readout \(y=W_{readout}\cdot h^\*_{\mathcal{S}}\)
  - Added utilities mirroring the HTML adjoint-state derivation (`compute_adjoint_state`, `approx_grad_H`)
- **Scripts updated**: `train_mobius_cifar100.py`, `quick_start.py`, `test_mobius_model.py`
- **Docs updated**: `README.md`
- **Tests**:
  - `python3 test_mobius_model.py` (pass)
  - `python3 quick_start.py` (pass)

- **Strict training (HTML)**:
  - Added **Holomorphic Equilibrium Propagation** training path (inference/learning synchronized; no BPTT) via:
    - `MoebiusQuantumRing.eqprop_update_step(...)`
    - CLI: `train_mobius_cifar100.py --use-eqprop`
  - Implemented the paper’s key update ingredients:
    - Adjoint fixed point \(h^\dagger\)
    - \(\partial \mathcal{L}/\partial H \approx h^\dagger \otimes h^\*\)
    - Lie algebra update \(\Delta A \propto \mathrm{skew}(U^\dagger \cdot ((\partial\mathcal{L}/\partial H)\odot U \odot \bar U))\)
  - Added test coverage: `test_eqprop_update_step` in `test_mobius_model.py`

## 2026-01-07

- **Algorithm (online dual-loop extension)**:
  - Added optional **dual-unitary** factorization: \(U_{total}=U_{policy}\,U_{base}\) with frozen \(U_{base}\) (world model) and learnable \(U_{policy}\) (policy).
  - Added optional **learnable goal equilibrium** (class prototypes \(P\in\mathbb{R}^{C\times N}\)) for state-level supervision.
  - Added optional **prototype-distance readout** (goal-aligned logits) on the sampled subspace \(S\).
- **Bug fix (dual-unitary learning)**:
  - Fixed the unitary-manifold update to correctly pull gradients back through \(U_{total}=U_{policy}\,U_{base}\) when \(U_{base}\) is frozen.
- **Algorithm (expressivity extensions, optional)**:
  - Added optional **patch embedding** image encoder (`image_encoder=patch`) to introduce local receptive fields before ring injection.
  - Added optional **self-retention mixing** \(H_{eff}=(1-\beta)I+\beta H\) (fixed or learnable \(\beta\)) to reduce over-mixing while preserving doubly stochasticity and contraction.
  - Added optional **complex unitary dynamics** (`dynamics_mode=unitary`) with measurement readout (default \(|h^*|\)), enabling phase to participate in inference.
  - Added optional **1-Lipschitz activations** for injection/state relaxation (ReLU/tanh) while preserving fixed-point existence (Banach contraction).
- **Scripts/Docs updated**:
  - `train_mobius_cifar100.py`: added CLI flags for patch encoder, H-mix beta, unitary dynamics mode/measurement, and activation switches.
  - `README.md`: documented new architecture options and added A→B→C example commands.
  - `paper/mqr_arxiv.tex`: added new sections with derivations for Nonlinear/Patch/Beta/Complex-Unitary extensions.
- **Tests**:
  - Added coverage for patch encoder, H-mix beta, and unitary dynamics mode sanity checks.

- **Engineering (C strategy: ViT backbone + MQR head)**:
  - Added `image_encoder=vit` option (a standard ViT backbone) so MQR can act as a drop-in replacement for the usual classification head.
  - In `--use-eqprop` mode, added optional encoder optimizer (`--eqprop-encoder-optim`) to train the image encoder with AdamW/SGD while keeping the ring strictly EQProp.
  - Updated docs: `README.md` now includes ViT backbone parameters and a 70%+ oriented C-strategy command template.

- **Training recipe (DeiT/ViT-style, optional)**:
  - Added warmup+cosine schedule controls (`--warmup-epochs`, `--min-lr-ratio`) for both EQProp and non-EQProp training.
  - Added label smoothing via soft targets (`--label-smoothing`) and CutMix (`--cutmix-prob`, `--cutmix-alpha`).

- **Documentation (Algorithm Reproduction Report)**:
  - Generated comprehensive algorithm reproduction analysis report (`report.html`)
  - Created network architecture SVG diagram (`network_architecture.svg`)
  - Created training flow SVG diagram (`training_flow.svg`)
  - **Verification Result**: 100% reproduction of `Möbius Quantum Ring.html` algorithm specification confirmed
  - Report includes: Executive Summary, Architecture Overview, Core Components Analysis, Mathematical Verification, Training Flow, HTML vs Code Comparison, Performance Metrics

## 2026-01-08

- **Paper (experiments + figures)**:
  - Added a new `Experiments` section to `paper/mqr_arxiv.tex` (CIFAR-100 setup, main result table, and controlled ablation).
  - Generated paper-ready figures under `paper/figures/`:
    - `arch_vit_mqr.png` (C-strategy: ViT backbone + MQR head diagram)
    - `acc_curve_vit_mqr.png`, `loss_curve_vit_mqr.png` (training curves for the best ViT+MQR run)
    - `ablation_patch_pool_10ep.png` (controlled patch pooling ablation)
  - Saved reproducibility metadata to `paper/results/experiment_summary.json`.
- **Experiments (controlled ablation, 10 epochs, no Mixup/CutMix)**:
  - Patch encoder pooling: `patch_pool=mean` best test **3.70%** vs `patch_pool=flatten` best test **9.97%**.
- **Build**:
  - `bash paper/compile.sh` (pass; paper builds with new figures and section).
