from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class InactiveUnistochasticParam(nn.Module):
    """Parameter-free identity placeholder for an exactly zero carry path."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.register_buffer(
            "_identity",
            torch.eye(self.dim, dtype=torch.cdouble),
            persistent=False,
        )

    def unitary(self) -> torch.Tensor:
        return self._identity

    def unistochastic(self) -> torch.Tensor:
        return self._identity.real

    @torch.no_grad()
    def unitary_error_fro(self) -> torch.Tensor:
        return self._identity.real.new_zeros(())

    @torch.no_grad()
    def doubly_stochastic_errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        zero = self._identity.real.new_zeros(())
        return zero, zero

    @torch.no_grad()
    def coordinate_diagnostics(self, *, include_jacobian: bool = False) -> dict:
        result = {
            "coordinate_mode": "inactive",
            "raw_parameter_count": 0,
            "effective_dof": 0,
            "redundant_parameter_count": 0,
            "unitary_distance_from_identity_fro": 0.0,
            "transition_distance_from_identity_fro": 0.0,
            "transition_off_diagonal_mass": 0.0,
            "modulus_square_jacobian_max_rank": 0,
        }
        if include_jacobian:
            result.update(
                {
                    "modulus_square_jacobian_rank": 0,
                    "modulus_square_jacobian_largest_singular_value": 0.0,
                    "modulus_square_jacobian_smallest_retained_singular_value": 0.0,
                }
            )
        return result

    @torch.no_grad()
    def drift_diagnostics(self, _reference: torch.Tensor) -> dict[str, float | bool]:
        return {
            "a_fro_drift": 0.0,
            "unitary_fro_drift": 0.0,
            "transition_fro_drift": 0.0,
            "unitary_fro_bound": 0.0,
            "transition_fro_bound": 0.0,
            "unitary_bound_violation": 0.0,
            "transition_bound_violation": 0.0,
            "bounds_certified": True,
        }


class CyclicGivensUnistochasticParam(nn.Module):
    """Literal local ring built from alternating nearest-neighbor rotations.

    For even ``N``, layer zero rotates edges ``(0,1),(2,3),...`` and layer one
    rotates ``(1,2),(3,4),...,(N-1,0)``.  Repeating those two matchings creates a
    periodic, headless ring.  The product is real orthogonal (hence unitary),
    and its entrywise square is exactly unistochastic.  A fixed nonzero base
    angle opens the modulus-square Jacobian while trainable policy angles start
    at zero.

    This is deliberately a structured subgroup, not a parameterization of all
    ``U(N)``.  Its purpose is a low-cost ring inductive bias and a fair mechanism
    control for the dense Cayley parameterization.
    """

    def __init__(
        self,
        dim: int,
        *,
        layers: int = 2,
        base_init: str = "random",
        base_scale: float = 0.25,
        base_seed: int | None = None,
    ) -> None:
        super().__init__()
        if dim < 2 or dim % 2 != 0:
            raise ValueError("cyclic Givens requires a positive even dimension")
        if isinstance(layers, bool) or int(layers) != layers or int(layers) <= 0:
            raise ValueError("layers must be a positive integer")
        if base_init not in ("identity", "random"):
            raise ValueError('base_init must be "identity" or "random"')
        if not math.isfinite(float(base_scale)) or float(base_scale) <= 0.0:
            raise ValueError("base_scale must be finite and positive")
        self.dim = int(dim)
        self.layers = int(layers)
        self.base_init = str(base_init)
        self.base_scale = float(base_scale)
        pairs_i = []
        pairs_j = []
        for layer in range(self.layers):
            offset = layer % 2
            if offset == 0:
                left = list(range(0, self.dim, 2))
                right = list(range(1, self.dim, 2))
            else:
                left = list(range(1, self.dim, 2))
                right = list(range(2, self.dim, 2)) + [0]
            pairs_i.append(left)
            pairs_j.append(right)
        self.register_buffer(
            "_pair_i", torch.tensor(pairs_i, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_pair_j", torch.tensor(pairs_j, dtype=torch.long), persistent=False
        )
        self.angles = nn.Parameter(torch.zeros(self.layers, self.dim // 2))
        if self.base_init == "random":
            generator: torch.Generator | None = None
            if base_seed is not None:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(base_seed))
            base_angles = torch.randn(
                self.layers,
                self.dim // 2,
                generator=generator,
                dtype=torch.double,
            ) * self.base_scale
        else:
            base_angles = torch.zeros(
                self.layers, self.dim // 2, dtype=torch.double
            )
        self.register_buffer(
            "base_angles",
            base_angles,
            persistent=self.base_init == "random",
        )

    @property
    def effective_dof(self) -> int:
        return int(self.angles.numel())

    @property
    def raw_parameter_count(self) -> int:
        return int(self.angles.numel())

    def _orthogonal_from_angles(self, angles: torch.Tensor) -> torch.Tensor:
        if angles.shape != (self.layers, self.dim // 2):
            raise ValueError("angles have an incompatible shape")
        result = torch.eye(self.dim, device=angles.device, dtype=angles.dtype)
        for layer in range(self.layers):
            left = self._pair_i[layer]
            right = self._pair_j[layer]
            cosine = torch.cos(angles[layer])
            sine = torch.sin(angles[layer])
            left_rows = result.index_select(0, left)
            right_rows = result.index_select(0, right)
            next_left = cosine.unsqueeze(1) * left_rows - sine.unsqueeze(1) * right_rows
            next_right = sine.unsqueeze(1) * left_rows + cosine.unsqueeze(1) * right_rows
            result = result.index_copy(0, left, next_left)
            result = result.index_copy(0, right, next_right)
        return result

    def unitary(self) -> torch.Tensor:
        base = self.base_angles.to(device=self.angles.device, dtype=self.angles.dtype)
        orthogonal = self._orthogonal_from_angles(self.angles + base)
        return torch.complex(orthogonal, torch.zeros_like(orthogonal))

    def apply_orthogonal(
        self, state: torch.Tensor, *, transpose: bool = False,
        angle_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply signed rotations to [..., dim] in O(layers * dim) work.

        Returns state @ U.T (or state @ U with transpose=True), without
        constructing a dense matrix or discarding signs with |U|².
        Each rotation preserves the L2 norm; damping and nonlinearities in a
        surrounding recurrent network need their own stability analysis.
        """
        if state.ndim < 1 or state.shape[-1] != self.dim:
            raise ValueError(f"state must have last dimension {self.dim}")
        if not state.is_floating_point() or torch.is_complex(state):
            raise TypeError("state must be a real floating-point tensor")
        angles = (self.angles + self.base_angles.to(self.angles)).to(state)
        if angle_offsets is not None:
            if angle_offsets.shape[-2:] != (self.layers, self.dim // 2):
                raise ValueError("angle_offsets must end in [layers, dim // 2]")
            if not angle_offsets.is_floating_point() or not bool(torch.isfinite(angle_offsets).all()):
                raise ValueError("angle_offsets must be finite real angles")
            if torch.broadcast_shapes(state.shape[:-1], angle_offsets.shape[:-2]) != state.shape[:-1]:
                raise ValueError("angle_offsets must broadcast to the state batch dimensions")
            angles = angles + angle_offsets.to(state)
        layers = range(self.layers - 1, -1, -1) if transpose else range(self.layers)
        value = state
        for layer in layers:
            left = self._pair_i[layer].to(device=state.device)
            right = self._pair_j[layer].to(device=state.device)
            angle = -angles[..., layer, :] if transpose else angles[..., layer, :]
            cosine, sine = angle.cos(), angle.sin()
            left_value = value.index_select(-1, left)
            right_value = value.index_select(-1, right)
            value = value.index_copy(
                -1, left, cosine * left_value - sine * right_value
            )
            value = value.index_copy(
                -1, right, sine * left_value + cosine * right_value
            )
        return value

    def unistochastic(self) -> torch.Tensor:
        orthogonal = self.unitary().real
        return orthogonal.square()

    @torch.no_grad()
    def reset_policy_identity_(self) -> None:
        self.angles.zero_()

    @torch.no_grad()
    def unitary_error_fro(self) -> torch.Tensor:
        unitary = self.unitary()
        identity = torch.eye(self.dim, device=unitary.device, dtype=unitary.dtype)
        return torch.linalg.matrix_norm(
            unitary.conj().transpose(-2, -1) @ unitary - identity,
            ord="fro",
        )

    @torch.no_grad()
    def doubly_stochastic_errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        transition = self.unistochastic()
        return (
            (transition.sum(dim=1) - 1.0).abs().max(),
            (transition.sum(dim=0) - 1.0).abs().max(),
        )

    @torch.no_grad()
    def modulus_square_jacobian_singular_values(self) -> torch.Tensor:
        with torch.enable_grad():
            angles = self.angles.detach().requires_grad_(True)
            base = self.base_angles.to(device=angles.device, dtype=angles.dtype)

            def mapping(value: torch.Tensor) -> torch.Tensor:
                return self._orthogonal_from_angles(value + base).square().reshape(-1)

            jacobian = torch.autograd.functional.jacobian(
                mapping,
                angles,
                create_graph=False,
                vectorize=True,
            ).reshape(self.dim * self.dim, -1)
        return torch.linalg.svdvals(jacobian.detach())

    @torch.no_grad()
    def coordinate_diagnostics(self, *, include_jacobian: bool = False) -> dict:
        unitary = self.unitary()
        transition = unitary.real.square()
        identity_u = torch.eye(self.dim, device=unitary.device, dtype=unitary.dtype)
        identity_h = torch.eye(
            self.dim, device=transition.device, dtype=transition.dtype
        )
        result = {
            "coordinate_mode": "cyclic_givens",
            "raw_parameter_count": self.raw_parameter_count,
            "effective_dof": self.effective_dof,
            "redundant_parameter_count": 0,
            "unitary_distance_from_identity_fro": float(
                torch.linalg.matrix_norm(unitary - identity_u, ord="fro").item()
            ),
            "transition_distance_from_identity_fro": float(
                torch.linalg.matrix_norm(transition - identity_h, ord="fro").item()
            ),
            "transition_off_diagonal_mass": float(
                (transition.sum() - transition.diagonal().sum()).item()
            ),
            "modulus_square_jacobian_max_rank": self.effective_dof,
            "literal_cycle_adjacency": True,
            "layers": self.layers,
        }
        if include_jacobian:
            singular_values = self.modulus_square_jacobian_singular_values()
            largest = float(singular_values.max().item())
            tolerance = (
                max(1, self.effective_dof)
                * torch.finfo(singular_values.dtype).eps
                * largest
            )
            rank = int((singular_values > tolerance).sum().item()) if largest else 0
            result.update(
                {
                    "modulus_square_jacobian_rank": rank,
                    "modulus_square_jacobian_largest_singular_value": largest,
                    "modulus_square_jacobian_smallest_retained_singular_value": (
                        float(singular_values[rank - 1].item()) if rank > 0 else 0.0
                    ),
                }
            )
        return result

    @torch.no_grad()
    def drift_diagnostics(
        self,
        reference_angles: torch.Tensor,
    ) -> dict[str, float | bool]:
        if not isinstance(reference_angles, torch.Tensor) or reference_angles.shape != self.angles.shape:
            raise ValueError("reference_angles have an incompatible shape")
        reference = reference_angles.detach().to(self.angles)
        base = self.base_angles.to(self.angles)
        before = self._orthogonal_from_angles(reference + base)
        after = self._orthogonal_from_angles(self.angles + base)
        before_h = before.square()
        after_h = after.square()
        delta = self.angles - reference
        parameter_l1 = float(delta.abs().sum().item())
        parameter_l2 = float(torch.linalg.vector_norm(delta).item())
        unitary_drift = float(
            torch.linalg.matrix_norm(after - before, ord="fro").item()
        )
        transition_drift = float(
            torch.linalg.matrix_norm(after_h - before_h, ord="fro").item()
        )
        unitary_bound = math.sqrt(2.0) * parameter_l1
        transition_bound = 2.0 * unitary_bound
        slack = 128.0 * self.dim * torch.finfo(self.angles.dtype).eps * (
            1.0 + transition_bound
        )
        return {
            "a_fro_drift": parameter_l2,
            "angle_l1_drift": parameter_l1,
            "unitary_fro_drift": unitary_drift,
            "transition_fro_drift": transition_drift,
            "unitary_fro_bound": unitary_bound,
            "transition_fro_bound": transition_bound,
            "unitary_bound_violation": max(0.0, unitary_drift - unitary_bound),
            "transition_bound_violation": max(
                0.0, transition_drift - transition_bound
            ),
            "bounds_certified": bool(
                unitary_drift <= unitary_bound + slack
                and transition_drift <= transition_bound + slack
            ),
        }


class CayleyUnistochasticParam(nn.Module):
    """
    Cayley parameterization on the unitary group U(N), and its induced unistochastic matrix.

    HTML spec (Möbius Quantum Ring.html):
      - A is skew-Hermitian: A^† = -A
      - U = (I - A)(I + A)^(-1)  (Cayley transform)
      - H = |U|^2               (element-wise modulus squared), which is doubly-stochastic

    ``coordinate_mode="projected"`` preserves the historical checkpoint
    layout: two unconstrained ``N x N`` real tensors are projected onto
    ``u(N)``.  It stores ``2N^2`` scalars for ``N^2`` effective degrees of
    freedom.  ``coordinate_mode="minimal"`` instead expands exactly ``N^2``
    coordinates in an orthonormal basis of real skew-symmetric and imaginary
    symmetric matrices.
    """

    def __init__(self, dim: int, *, coordinate_mode: str = "projected"):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        coordinate_mode = str(coordinate_mode)
        if coordinate_mode not in ("projected", "minimal"):
            raise ValueError('coordinate_mode must be "projected" or "minimal"')
        self.dim = int(dim)
        self.coordinate_mode = coordinate_mode

        real_indices = torch.triu_indices(dim, dim, offset=1)
        symmetric_indices = torch.triu_indices(dim, dim, offset=0)
        self.register_buffer("_real_i", real_indices[0], persistent=False)
        self.register_buffer("_real_j", real_indices[1], persistent=False)
        self.register_buffer("_symmetric_i", symmetric_indices[0], persistent=False)
        self.register_buffer("_symmetric_j", symmetric_indices[1], persistent=False)

        if self.coordinate_mode == "projected":
            # Backward-compatible over-complete coordinates.
            self.A_real = nn.Parameter(torch.randn(dim, dim) * 0.01)
            self.A_imag = nn.Parameter(torch.randn(dim, dim) * 0.01)
        else:
            # Orthonormal minimal coordinates.  Off-diagonal basis elements
            # contain +/-1/sqrt(2); imaginary diagonal basis elements contain i.
            self.A_real = nn.Parameter(torch.randn(real_indices.size(1)) * 0.01)
            self.A_imag = nn.Parameter(
                torch.randn(symmetric_indices.size(1)) * 0.01
            )

    def skew_hermitian_A(self) -> torch.Tensor:
        """Return A in C^{N×N} with A^H = -A."""
        if self.coordinate_mode == "projected":
            # Real part skew-symmetric, imaginary part symmetric.
            return 0.5 * torch.complex(
                self.A_real - self.A_real.T,
                self.A_imag + self.A_imag.T,
            )

        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        real = torch.zeros(
            self.dim,
            self.dim,
            device=self.A_real.device,
            dtype=self.A_real.dtype,
        )
        real_values = self.A_real * inv_sqrt2
        real = real.index_put((self._real_i, self._real_j), real_values)
        real = real.index_put((self._real_j, self._real_i), -real_values)

        imag = torch.zeros(
            self.dim,
            self.dim,
            device=self.A_imag.device,
            dtype=self.A_imag.dtype,
        )
        diagonal = self._symmetric_i == self._symmetric_j
        imag_scale = torch.where(
            diagonal,
            torch.ones_like(self.A_imag),
            torch.full_like(self.A_imag, inv_sqrt2),
        )
        imag_values = self.A_imag * imag_scale
        imag = imag.index_put(
            (self._symmetric_i, self._symmetric_j), imag_values
        )
        off_diagonal = ~diagonal
        imag = imag.index_put(
            (
                self._symmetric_j[off_diagonal],
                self._symmetric_i[off_diagonal],
            ),
            imag_values[off_diagonal],
        )
        return torch.complex(real, imag)

    @property
    def effective_dof(self) -> int:
        """Real dimension of the represented Lie algebra ``u(N)``."""

        return self.dim * self.dim

    @property
    def raw_parameter_count(self) -> int:
        """Number of stored trainable real scalars."""

        return int(self.A_real.numel() + self.A_imag.numel())

    @torch.no_grad()
    def set_from_skew_hermitian_(self, A: torch.Tensor) -> None:
        """Set either coordinate layout from a supplied skew-Hermitian matrix."""

        if A.shape != (self.dim, self.dim):
            raise ValueError(f"A must be [{self.dim}, {self.dim}], got {tuple(A.shape)}")
        if not torch.is_complex(A):
            raise TypeError("A must be a complex skew-Hermitian tensor")
        A = A.to(device=self.A_real.device)
        skew_error = (A + A.conj().transpose(-2, -1)).abs().amax()
        eps = torch.finfo(A.real.dtype).eps
        scale = A.abs().amax().clamp_min(1.0)
        if float((skew_error / scale).cpu().item()) > 100.0 * self.dim * eps:
            raise ValueError("A must satisfy A^H = -A within numerical tolerance")

        if self.coordinate_mode == "projected":
            self.A_real.copy_(A.real.to(dtype=self.A_real.dtype))
            self.A_imag.copy_(A.imag.to(dtype=self.A_imag.dtype))
            return

        sqrt2 = math.sqrt(2.0)
        real_values = sqrt2 * A.real[self._real_i, self._real_j]
        diagonal = self._symmetric_i == self._symmetric_j
        imag_scale = torch.where(
            diagonal,
            torch.ones_like(self.A_imag),
            torch.full_like(self.A_imag, sqrt2),
        )
        imag_values = (
            A.imag[self._symmetric_i, self._symmetric_j]
            .to(dtype=self.A_imag.dtype)
            * imag_scale
        )
        self.A_real.copy_(real_values.to(dtype=self.A_real.dtype))
        self.A_imag.copy_(imag_values)

    @torch.no_grad()
    def set_from_unitary_representative_(
        self,
        unitary: torch.Tensor,
        *,
        phase_candidates: int | None = None,
    ) -> dict[str, float | int]:
        """Import any unistochastic representative through a finite Cayley chart.

        Cayley coordinates exclude unitary matrices with eigenvalue ``-1``.
        The induced transition does not share that omission because
        ``|zeta U|^2 = |U|^2`` for every scalar phase ``|zeta|=1``.  This
        method evaluates more than ``N`` deterministic phase candidates, so
        at least one cannot belong to the at-most-``N`` forbidden phases.  It
        selects the candidate maximizing ``sigma_min(I + zeta U)`` before
        applying the inverse Cayley transform.

        Args:
            unitary: complex tensor with shape ``[N, N]``.
            phase_candidates: number of equally spaced phases.  It must exceed
                ``N``; the default ``2N+1`` also improves numerical margin.

        Returns:
            Diagnostics for the selected phase and transition reconstruction.
        """

        if not isinstance(unitary, torch.Tensor):
            raise TypeError("unitary must be a tensor")
        if unitary.shape != (self.dim, self.dim):
            raise ValueError(
                f"unitary must be [{self.dim}, {self.dim}], got "
                f"{tuple(unitary.shape)}"
            )
        if not torch.is_complex(unitary):
            raise TypeError("unitary must be complex")
        if not bool(torch.isfinite(unitary.real).all()) or not bool(
            torch.isfinite(unitary.imag).all()
        ):
            raise ValueError("unitary must contain only finite values")

        value = unitary.detach().to(
            device=self.A_real.device,
            dtype=(
                torch.complex128
                if self.A_real.dtype == torch.float64
                else torch.complex64
            ),
        )
        identity = torch.eye(self.dim, device=value.device, dtype=value.dtype)
        unitary_error = torch.linalg.matrix_norm(
            value.conj().transpose(-2, -1) @ value - identity,
            ord="fro",
        )
        eps = torch.finfo(value.real.dtype).eps
        tolerance = 256.0 * self.dim * eps
        if float(unitary_error.item()) > tolerance:
            raise ValueError("unitary must be unitary within numerical tolerance")

        if phase_candidates is None:
            candidate_count = 2 * self.dim + 1
        else:
            if (
                isinstance(phase_candidates, bool)
                or int(phase_candidates) != phase_candidates
            ):
                raise ValueError("phase_candidates must be an integer greater than dim")
            candidate_count = int(phase_candidates)
        if candidate_count <= self.dim:
            raise ValueError("phase_candidates must be an integer greater than dim")
        indices = torch.arange(
            candidate_count, device=value.device, dtype=value.real.dtype
        )
        phases = torch.exp(2j * math.pi * indices / float(candidate_count)).to(
            dtype=value.dtype
        )
        margins = torch.stack(
            [
                torch.linalg.svdvals(identity + phase * value).amin()
                for phase in phases
            ]
        )
        selected_index = int(margins.argmax().item())
        selected_phase = phases[selected_index]
        selected_margin = margins[selected_index]
        if not bool(selected_margin > 0.0):
            raise RuntimeError("failed to find a finite inverse Cayley chart")

        admissible = selected_phase * value
        coordinate = torch.linalg.solve(identity + admissible, identity - admissible)
        self.set_from_skew_hermitian_(coordinate)
        reconstructed_transition = self.unistochastic().to(value.real)
        transition_error = torch.linalg.matrix_norm(
            reconstructed_transition - value.abs().square(), ord="fro"
        )
        return {
            "phase_candidate_count": candidate_count,
            "selected_phase_index": selected_index,
            "selected_phase_real": float(selected_phase.real.item()),
            "selected_phase_imag": float(selected_phase.imag.item()),
            "selected_margin_sigma_min": float(selected_margin.item()),
            "input_unitary_error_fro": float(unitary_error.item()),
            "transition_reconstruction_error_fro": float(transition_error.item()),
        }

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

    @staticmethod
    @torch.no_grad()
    def _validate_right_unitary(
        right_unitary: torch.Tensor | None,
        *,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Validate an optional frozen right factor used after Cayley.

        Temporal MQR can use ``U(A)=Cayley(A) U_0`` with a fixed non-monomial
        ``U_0``.  Keeping the factor outside the trainable coordinates preserves
        exact unitarity while avoiding the zero Jacobian of ``|U|^2`` at the
        identity.  This helper is intentionally strict because the structural
        guarantees are invalid if a caller supplies an approximately unitary
        factor without checking it.
        """

        if right_unitary is None:
            return None
        if not isinstance(right_unitary, torch.Tensor):
            raise TypeError("right_unitary must be a tensor or None")
        if right_unitary.shape != (dim, dim):
            raise ValueError(
                f"right_unitary must be [{dim}, {dim}], got "
                f"{tuple(right_unitary.shape)}"
            )
        if not torch.is_complex(right_unitary):
            raise TypeError("right_unitary must be complex")
        value = right_unitary.detach().to(device=device, dtype=dtype)
        if not bool(torch.isfinite(value.real).all()) or not bool(
            torch.isfinite(value.imag).all()
        ):
            raise ValueError("right_unitary must contain only finite values")
        identity = torch.eye(dim, device=device, dtype=dtype)
        error = torch.linalg.matrix_norm(
            value.conj().transpose(-2, -1) @ value - identity,
            ord="fro",
        )
        tolerance = 256.0 * dim * torch.finfo(value.real.dtype).eps
        if float(error.item()) > tolerance:
            raise ValueError("right_unitary must be unitary within numerical tolerance")
        return value

    @torch.no_grad()
    def reset_cayley_identity_(self) -> None:
        """Set the trainable Cayley policy to the identity exactly."""

        self.A_real.zero_()
        self.A_imag.zero_()

    @torch.no_grad()
    def drift_diagnostics(
        self,
        reference_A: torch.Tensor,
        *,
        right_unitary: torch.Tensor | None = None,
    ) -> dict[str, float | bool]:
        """Certify finite Cayley and unistochastic drift from ``reference_A``.

        For skew-Hermitian ``A`` and ``B``, both resolvents have spectral norm
        at most one.  The resolvent identity therefore gives
        ``||U(A)-U(B)||_F <= 2 ||A-B||_F``.  Entrywise modulus-square is
        2-Lipschitz on unitary entries, yielding
        ``||H(A)-H(B)||_F <= 4 ||A-B||_F``.
        """

        if not isinstance(reference_A, torch.Tensor):
            raise TypeError("reference_A must be a tensor")
        if reference_A.shape != (self.dim, self.dim):
            raise ValueError(
                f"reference_A must be [{self.dim}, {self.dim}], "
                f"got {tuple(reference_A.shape)}"
            )
        if not torch.is_complex(reference_A):
            raise TypeError("reference_A must be complex skew-Hermitian")
        current_A = self.skew_hermitian_A()
        reference = reference_A.detach().to(device=current_A.device, dtype=current_A.dtype)
        if not bool(torch.isfinite(reference.real).all()) or not bool(
            torch.isfinite(reference.imag).all()
        ):
            raise ValueError("reference_A must contain only finite values")
        skew_error = (reference + reference.conj().transpose(-2, -1)).abs().amax()
        scale = reference.abs().amax().clamp_min(1.0)
        tolerance = 100.0 * self.dim * torch.finfo(reference.real.dtype).eps
        if float((skew_error / scale).item()) > tolerance:
            raise ValueError("reference_A must be skew-Hermitian")

        identity = torch.eye(self.dim, device=current_A.device, dtype=current_A.dtype)
        right = self._validate_right_unitary(
            right_unitary,
            dim=self.dim,
            device=current_A.device,
            dtype=current_A.dtype,
        )
        reference_U = torch.linalg.solve(identity + reference, identity - reference)
        current_U = torch.linalg.solve(identity + current_A, identity - current_A)
        if right is not None:
            reference_U = reference_U @ right
            current_U = current_U @ right
        reference_H = reference_U.abs().square()
        current_H = current_U.abs().square()
        a_drift = float(
            torch.linalg.matrix_norm(current_A - reference, ord="fro").item()
        )
        unitary_drift = float(
            torch.linalg.matrix_norm(current_U - reference_U, ord="fro").item()
        )
        transition_drift = float(
            torch.linalg.matrix_norm(current_H - reference_H, ord="fro").item()
        )
        unitary_bound = 2.0 * a_drift
        transition_bound = 4.0 * a_drift
        numerical_slack = 64.0 * self.dim * torch.finfo(reference.real.dtype).eps * (
            1.0 + transition_bound
        )
        return {
            "a_fro_drift": a_drift,
            "unitary_fro_drift": unitary_drift,
            "transition_fro_drift": transition_drift,
            "unitary_fro_bound": unitary_bound,
            "transition_fro_bound": transition_bound,
            "unitary_bound_violation": max(0.0, unitary_drift - unitary_bound),
            "transition_bound_violation": max(
                0.0, transition_drift - transition_bound
            ),
            "bounds_certified": bool(
                unitary_drift <= unitary_bound + numerical_slack
                and transition_drift <= transition_bound + numerical_slack
            ),
        }

    @torch.no_grad()
    def cayley_pullback(self, grad_U: torch.Tensor) -> torch.Tensor:
        """Pull a real-loss Euclidean gradient on ``U`` back to Cayley coordinate ``A``.

        With ``R = (I + A)^(-1)``, the exact differential is
        ``dU = -2 R (dA) R``.  Under the real Frobenius inner product the
        ambient pullback is ``-2 R^H grad_U R^H``; projecting it onto the
        skew-Hermitian matrices gives the constrained gradient for ``A``.

        The returned real/imaginary parts can be applied directly to
        ``A_real`` and ``A_imag`` because their construction is the orthogonal
        projection onto skew-Hermitian matrices.
        """
        if grad_U.shape != (self.dim, self.dim):
            raise ValueError(f"grad_U must be [{self.dim}, {self.dim}], got {tuple(grad_U.shape)}")

        A = self.skew_hermitian_A()
        grad_U = grad_U.to(device=A.device, dtype=A.dtype)
        I = torch.eye(self.dim, device=A.device, dtype=A.dtype)
        R = torch.linalg.solve(I + A, I)
        R_H = R.conj().transpose(-2, -1)
        grad_A = -2.0 * (R_H @ grad_U @ R_H)
        return 0.5 * (grad_A - grad_A.conj().transpose(-2, -1))

    @torch.no_grad()
    def coordinate_gradients(self, grad_A: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map a skew-Hermitian ``grad_A`` to the stored real coordinates."""

        if grad_A.shape != (self.dim, self.dim):
            raise ValueError(
                f"grad_A must be [{self.dim}, {self.dim}], got {tuple(grad_A.shape)}"
            )
        grad_A = grad_A.to(
            device=self.A_real.device,
            dtype=torch.complex128 if self.A_real.dtype == torch.float64 else torch.complex64,
        )
        grad_A = 0.5 * (grad_A - grad_A.conj().transpose(-2, -1))
        if self.coordinate_mode == "projected":
            return {
                "A_real": grad_A.real.to(dtype=self.A_real.dtype),
                "A_imag": grad_A.imag.to(dtype=self.A_imag.dtype),
            }

        sqrt2 = math.sqrt(2.0)
        diagonal = self._symmetric_i == self._symmetric_j
        imag_scale = torch.where(
            diagonal,
            torch.ones_like(self.A_imag),
            torch.full_like(self.A_imag, sqrt2),
        )
        return {
            "A_real": (
                sqrt2 * grad_A.real[self._real_i, self._real_j]
            ).to(dtype=self.A_real.dtype),
            "A_imag": (
                imag_scale * grad_A.imag[self._symmetric_i, self._symmetric_j]
            ).to(dtype=self.A_imag.dtype),
        }

    @torch.no_grad()
    def modulus_square_jacobian_singular_values(
        self,
        *,
        right_unitary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return singular values of ``dA -> d|Cayley(A)|^2`` on ``u(N)``.

        This dense ``N^2 x N^2`` diagnostic is intended only for small proof
        problems.  It exposes the zero differential at diagonal/permutation
        unitaries and the generic maximum rank ``(N-1)^2``.
        """

        A = self.skew_hermitian_A()
        identity = torch.eye(self.dim, device=A.device, dtype=A.dtype)
        R = torch.linalg.solve(identity + A, identity)
        U_policy = torch.linalg.solve(identity + A, identity - A)
        right = self._validate_right_unitary(
            right_unitary,
            dim=self.dim,
            device=A.device,
            dtype=A.dtype,
        )
        U = U_policy if right is None else U_policy @ right
        directions = []
        inv_sqrt2 = 1.0 / math.sqrt(2.0)

        for i, j in zip(self._real_i.tolist(), self._real_j.tolist()):
            direction = torch.zeros_like(A)
            direction[i, j] = inv_sqrt2
            direction[j, i] = -inv_sqrt2
            directions.append(direction)
        for i, j in zip(
            self._symmetric_i.tolist(), self._symmetric_j.tolist()
        ):
            direction = torch.zeros_like(A)
            if i == j:
                direction[i, i] = 1j
            else:
                direction[i, j] = 1j * inv_sqrt2
                direction[j, i] = 1j * inv_sqrt2
            directions.append(direction)

        columns = []
        for direction in directions:
            dU = -2.0 * (R @ direction @ R)
            if right is not None:
                dU = dU @ right
            dH = 2.0 * (U.conj() * dU).real
            columns.append(dH.reshape(-1))
        jacobian = torch.stack(columns, dim=1)
        return torch.linalg.svdvals(jacobian)

    @torch.no_grad()
    def modulus_square_jacobian_rank(
        self,
        *,
        relative_tolerance: float | None = None,
        right_unitary: torch.Tensor | None = None,
    ) -> int:
        """Numerical rank of the local ``U -> |U|^2``-composed Cayley map."""

        singular_values = self.modulus_square_jacobian_singular_values(
            right_unitary=right_unitary
        )
        largest = float(singular_values.max().item())
        if largest == 0.0:
            return 0
        if relative_tolerance is None:
            relative_tolerance = float(
                max(1, self.effective_dof)
                * torch.finfo(singular_values.dtype).eps
            )
        if relative_tolerance < 0:
            raise ValueError("relative_tolerance must be non-negative or None")
        return int((singular_values > largest * relative_tolerance).sum().item())

    @torch.no_grad()
    def coordinate_diagnostics(
        self,
        *,
        include_jacobian: bool = False,
        right_unitary: torch.Tensor | None = None,
    ) -> dict:
        """Return representation and near-identity diagnostics."""

        U = self.unitary()
        right = self._validate_right_unitary(
            right_unitary,
            dim=self.dim,
            device=U.device,
            dtype=U.dtype,
        )
        if right is not None:
            U = U @ right
        H = U.abs().square()
        identity_u = torch.eye(self.dim, device=U.device, dtype=U.dtype)
        identity_h = torch.eye(self.dim, device=H.device, dtype=H.dtype)
        diagnostics = {
            "coordinate_mode": self.coordinate_mode,
            "raw_parameter_count": self.raw_parameter_count,
            "effective_dof": self.effective_dof,
            "redundant_parameter_count": self.raw_parameter_count - self.effective_dof,
            "unitary_distance_from_identity_fro": float(
                torch.linalg.matrix_norm(U - identity_u, ord="fro").item()
            ),
            "transition_distance_from_identity_fro": float(
                torch.linalg.matrix_norm(H - identity_h, ord="fro").item()
            ),
            "transition_off_diagonal_mass": float(
                (H.sum() - H.diagonal().sum()).item()
            ),
            "modulus_square_jacobian_max_rank": (self.dim - 1) ** 2,
        }
        if include_jacobian:
            singular_values = self.modulus_square_jacobian_singular_values(
                right_unitary=right
            )
            rank = self.modulus_square_jacobian_rank(right_unitary=right)
            diagnostics.update(
                {
                    "modulus_square_jacobian_rank": rank,
                    "modulus_square_jacobian_largest_singular_value": float(
                        singular_values.max().item()
                    ),
                    "modulus_square_jacobian_smallest_retained_singular_value": (
                        float(singular_values[rank - 1].item()) if rank > 0 else 0.0
                    ),
                }
            )
        return diagnostics

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
