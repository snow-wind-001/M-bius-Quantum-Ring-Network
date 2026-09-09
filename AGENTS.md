# Repository Guidelines

## Project Structure & Module Organization

Core PyTorch code lives in `mqr/`: `unitary.py` implements Cayley/unistochastic parameterization, `ring.py` contains fixed-point dynamics and implicit updates, and `online.py` provides multi-ring routing and orthogonal gradient memory. `mobius_quantum_ring.py` is the backward-compatible public facade. Keep runnable studies in `experiments/`, mathematical checks and generated JSON results in `analysis/`, and user-facing guidance in `README.md`. Root `test_*.py` files are the canonical test suites. Treat `paper/`, large rendered documents, and `UsedCode/` as reference material unless a task explicitly targets them.

## Build, Test, and Development Commands

This repository has no package build step. From the repository root, use:

```bash
python3 -m py_compile mqr/*.py
python3 test_mobius_model.py
python3 test_online_learning.py
python3 analysis/mqr_proof_verify.py
python3 experiments/online_digits_principle.py
python3 quick_start.py
```

The first command catches syntax/import failures. The two test scripts cover core and online behavior; the proof script checks mathematical identities; the Digits experiment requires `scikit-learn` and writes `analysis/results/online_digits_principle.json`.

## Coding Style & Naming Conventions

Use four-space indentation, type hints on public APIs, and concise docstrings that state tensor shapes and mathematical assumptions. Follow Python conventions: `snake_case` for functions/variables, `PascalCase` for classes, and uppercase symbols only where they mirror formulas such as `U`, `H`, or `A`. Preserve backward-compatible names and keyword defaults. No formatter is enforced, so keep imports grouped and avoid unrelated mechanical rewrites.

## Testing Guidelines

Name tests `test_<behavior>` and make numerical tests deterministic with explicit seeds. Verify structural invariants (unitarity, routing isolation, pre-update predictions) in addition to losses. Use tight tolerances in `float64` proof tests and realistic tolerances in `float32` integration tests. Dataset tests must run without network access.

## Commit & Pull Request Guidelines

History is too small to establish a strict convention. Use short imperative subjects, optionally scoped, for example `online: add staged OGD projection`. Pull requests should explain the mathematical or behavioral contract, list exact verification commands, link issues, and report accuracy/latency/memory trade-offs. Include plots or screenshots only when visual output changes; do not commit checkpoints or regenerated PDFs unless required.
