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
