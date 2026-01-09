# Möbius Quantum Ring - arXiv Paper

This directory contains the LaTeX source for the paper:

**"Möbius Quantum Ring: A Unitary Manifold Approach to Stable and Expressive Recurrent Dynamics"**

## Files

- `mqr_arxiv.tex` - Main LaTeX source file
- `compile.sh` - Compilation script
- `figures/` - Paper figures (auto-generated from logs / experiments)
- `results/` - Experiment summaries for reproducibility

## Compilation

### Prerequisites

You need a basic LaTeX distribution installed. On Ubuntu/Debian:

```bash
# Minimal (usually sufficient):
sudo apt-get install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended

# Full (recommended):
sudo apt-get install texlive-full
```

### Compile

```bash
cd paper/

# Manual compilation (recommended):
pdflatex mqr_arxiv.tex
pdflatex mqr_arxiv.tex  # Run twice for references

# Or use the script:
./compile.sh
```

**Note:** This paper uses only basic LaTeX packages (no `algorithm2e` or `algorithmic` required).

The output PDF will be `mqr_arxiv.pdf`.

## Paper Structure

1. **Abstract** - Summary of MQR architecture and contributions
2. **Introduction** - Motivation and research question
3. **Related Work** - Stabilizing deep networks, orthogonal/unitary networks, Birkhoff polytope
4. **Mathematical Preliminaries** - Unitary group, Cayley transform, doubly stochastic matrices
5. **The MQR Architecture** - State dynamics, Hamiltonian injection, local projective readout
6. **Theoretical Analysis** - Fixed-point existence, spectral analysis, energy conservation
7. **Holomorphic Equilibrium Propagation** - BPTT-free learning algorithm
8. **Computational Complexity** - Forward pass costs, comparison with mHC
9. **Experiments** - CIFAR-100 results and controlled ablations
10. **Discussion** - Quantum metaphor, limitations, future directions
11. **Conclusion** - Summary of contributions

## Key Contributions

1. **Unistochastic Manifold Parameterization**: $H_{ij} = |U_{ij}|^2$ analytically guarantees doubly stochastic properties
2. **Cayley Transform**: Efficient bijective mapping from skew-Hermitian to unitary matrices
3. **Holomorphic Equilibrium Propagation**: BPTT-free training via Lie algebra updates
4. **Contraction Mapping Proof**: Guarantees unique fixed-point convergence

## Citation

```bibtex
@article{mqr2026,
  title={M{\"o}bius Quantum Ring: A Unitary Manifold Approach to Stable and Expressive Recurrent Dynamics},
  author={Anonymous},
  journal={arXiv preprint},
  year={2026}
}
```
