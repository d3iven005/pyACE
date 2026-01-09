# src/ace_invariants.py
import torch
from typing import Tuple, Optional


def _lm_block(A: torch.Tensor, l: int) -> torch.Tensor:
    """
    Slice the (2l+1) block for fixed l from flattened lm index.

    A: (..., nY) with nY = (lmax+1)^2
    returns: (..., 2l+1) corresponding to m=-l..+l in your ordering:
             lm_index = l^2 + (m+l)
    """
    start = l * l
    end = start + (2 * l + 1)
    return A[..., start:end]


# -------------------------
# K=1  (pair / 2-body level)
# -------------------------
def compute_B1_from_A(A: torch.Tensor) -> torch.Tensor:
    """
    K=1 invariants (2-body level):
      B1[i, s, n] = A[i, s, n, l=0,m=0]

    Input:
      A: (N, S, nmax, nY)

    Output:
      B1: (N, S, nmax)
    """
    # lm index for (l=0,m=0) is 0 in your flattened convention.
    return A[..., 0]


# -------------------------
# K=2  (3-body level)
# -------------------------
def compute_B2_from_A(
    A: torch.Tensor,
    lmax: int,
    symmetrize_species: bool = True,
    symmetrize_radial: bool = True,
) -> torch.Tensor:
    """
    K=2 invariants (3-body level) using real spherical harmonics "power spectrum" dot products:

      B2[i, s1, s2, n1, n2, l] = sum_{m=-l..l} A[i,s1,n1,lm] * A[i,s2,n2,lm]

    This is a rotational invariant for each l because it is the Euclidean inner product
    within the (2l+1)-dim irreducible SO(3) representation in a real orthonormal basis.

    Input:
      A: (N, S, nmax, nY) where nY=(lmax+1)^2
      lmax: maximum angular momentum

    Options:
      symmetrize_species: enforce B2(...,s1,s2,...) == B2(...,s2,s1,...)
      symmetrize_radial:  enforce B2(...,n1,n2,...) == B2(...,n2,n1,...)

    Output:
      B2: (N, S, S, nmax, nmax, lmax+1)
    """
    N, S, nmax, nY = A.shape
    expected_nY = (lmax + 1) ** 2
    if nY != expected_nY:
        raise ValueError(f"A last dim nY={nY} but expected {(lmax+1)**2} for lmax={lmax}")

    B2 = torch.zeros((N, S, S, nmax, nmax, lmax + 1), dtype=A.dtype, device=A.device)

    for l in range(lmax + 1):
        # Ab: (N,S,nmax,2l+1)
        Ab = _lm_block(A, l)

        # Dot over m:
        # (N,S,nmax,M) x (N,S,nmax,M) -> (N,S,S,nmax,nmax)
        # indices: i a n m, i b p m -> i a b n p
        Bl = torch.einsum("ianm,ibpm->iabnp", Ab, Ab)
        B2[..., l] = Bl

    if symmetrize_species:
        B2 = 0.5 * (B2 + B2.transpose(1, 2))

    if symmetrize_radial:
        B2 = 0.5 * (B2 + B2.transpose(3, 4))

    return B2


# -------------------------
# Feature flattening
# -------------------------
def flatten_B_features(
    B1: torch.Tensor,
    B2: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """
    Flatten B1 and B2 into per-atom feature vectors.

    Inputs:
      B1: (N, S, nmax)
      B2: (N, S, S, nmax, nmax, lmax+1)

    Returns:
      X: (N, n_features)
      n_features: int
    """
    if B1.shape[0] != B2.shape[0]:
        raise ValueError("B1 and B2 must have the same leading N dimension")

    N = B1.shape[0]
    x1 = B1.reshape(N, -1)
    x2 = B2.reshape(N, -1)
    X = torch.cat([x1, x2], dim=1)
    return X, X.shape[1]


# -------------------------
# Linear energy model
# -------------------------
def compute_energy_linear(
    X: torch.Tensor,
    center_species_id: torch.Tensor,
    coeff: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linear ACE energy model with per-center-species parameters:

      E_i = < coeff[mu_i], X_i > + bias[mu_i]  (optional)
      E_total = sum_i E_i

    Inputs:
      X: (N, n_features)
      center_species_id: (N,) int64, species of center atom i
      coeff: (S_center, n_features)  (often S_center == number of elements)
      bias:  (S_center,) optional

    Returns:
      E_atom: (N,)
      E_total: scalar tensor
    """
    if center_species_id.dtype != torch.int64:
        center_species_id = center_species_id.to(torch.int64)

    if X.dim() != 2:
        raise ValueError("X must be 2D (N, n_features)")

    N, nfeat = X.shape
    if coeff.shape[1] != nfeat:
        raise ValueError(f"coeff has n_features={coeff.shape[1]} but X has {nfeat}")

    # Gather per-atom coefficient vectors: (N, n_features)
    C = coeff[center_species_id]  # (N, nfeat)

    # Per-atom energy
    E_atom = torch.sum(C * X, dim=1)

    if bias is not None:
        if bias.dim() != 1 or bias.shape[0] != coeff.shape[0]:
            raise ValueError("bias must have shape (S_center,)")
        E_atom = E_atom + bias[center_species_id]

    return E_atom, torch.sum(E_atom)
