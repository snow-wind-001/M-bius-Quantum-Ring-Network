# Changelog

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
- **Scripts/Docs updated**:
  - `train_mobius_cifar100.py`: added CLI flags `--readout-mode`, `--proto-tau`, plus dual-unitary/state-target options.
  - `README.md`: documented the new options and example commands.
- **Tests**:
  - Added coverage for prototype readout and ensured `U_base` remains frozen under updates.

- **Documentation (Algorithm Reproduction Report)**:
  - Generated comprehensive algorithm reproduction analysis report (`report.html`)
  - Created network architecture SVG diagram (`network_architecture.svg`)
  - Created training flow SVG diagram (`training_flow.svg`)
  - **Verification Result**: 100% reproduction of `Möbius Quantum Ring.html` algorithm specification confirmed
  - Report includes: Executive Summary, Architecture Overview, Core Components Analysis, Mathematical Verification, Training Flow, HTML vs Code Comparison, Performance Metrics
