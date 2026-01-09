from __future__ import annotations

import torch
import torch.nn as nn


class CayleyUnistochasticParam(nn.Module):
    """
    Cayley parameterization on the unitary group U(N), and its induced unistochastic matrix.

    HTML spec (Möbius Quantum Ring.html):
      - A is skew-Hermitian: A^† = -A
      - U = (I - A)(I + A)^(-1)  (Cayley transform)
      - H = |U|^2               (element-wise modulus squared), which is doubly-stochastic

    Implementation detail:
      We store real/imag parts of an unconstrained matrix and construct a skew-Hermitian A:
        A = 0.5 * [ (R - R^T) + i * (I + I^T) ]
      so that A^† = -A holds by construction.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim

        # Unconstrained parameters (real and imaginary parts)
        self.A_real = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.A_imag = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def skew_hermitian_A(self) -> torch.Tensor:
        """Return A in C^{N×N} with A^H = -A."""
        # Real part skew-symmetric, imag part symmetric -> skew-Hermitian overall.
        return 0.5 * torch.complex(
            self.A_real - self.A_real.T,
            self.A_imag + self.A_imag.T,
        )

    def unitary(self) -> torch.Tensor:
        """Return U in U(N) via Cayley transform."""
        A = self.skew_hermitian_A()
        I = torch.eye(self.dim, device=A.device, dtype=A.dtype)
        # Solve (I + A) U = (I - A) for numerical stability.
        return torch.linalg.solve(I + A, I - A)

    def unistochastic(self) -> torch.Tensor:
        """Return H = |U|^2 (real, non-negative)."""
        U = self.unitary()
        # For complex tensors, abs() returns a real tensor.
        return U.abs().pow(2)

    @torch.no_grad()
    def unitary_error_fro(self) -> torch.Tensor:
        """||U^H U - I||_F (diagnostic only)."""
        U = self.unitary()
        I = torch.eye(self.dim, device=U.device, dtype=U.dtype)
        return torch.linalg.matrix_norm(U.conj().transpose(-2, -1) @ U - I, ord="fro")

    @torch.no_grad()
    def doubly_stochastic_errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (max|row_sum-1|, max|col_sum-1|) for H (diagnostic only)."""
        H = self.unistochastic()
        row_err = (H.sum(dim=1) - 1.0).abs().max()
        col_err = (H.sum(dim=0) - 1.0).abs().max()
        return row_err, col_err

