# src/ace_atomic.py
import torch
from typing import Optional

from src.radial import RadialBasis
from src.spherical import unit_vectors, real_sph_harm_torch
#####
#A = build_A_nlm_species(
#        n_atoms=N,
#        idx_i=idx_i,
#        idx_j=idx_j,
#        rij=rij,                                                 cutoff
#        dij=dij,                                               rj - ri 
#        species_id=species_id,                   torch.tensor shape = (N_atoms, )
#        n_species=n_species,
#        radial_basis=rb,
#        lmax=lmax,
#    )
#### build_A_nlm_species():
#### output: A.shape = (N_atoms, n_species, nmax_out, nY)
#####

def build_phi_nlm(
    rij: torch.Tensor,
    dij: torch.Tensor,
    radial_basis: RadialBasis,
    lmax: int,
) -> torch.Tensor:
    """
    Build edge features phi_{nlm}(e) = g_n(r_e) * Y_{lm}(rhat_e)

    Inputs:
      rij: (E,3) displacement vectors (MIC/shift already handled)
      dij: (E,) distances
      radial_basis: callable r -> (E, nmax_out)
      lmax: maximum angular momentum

    Returns:
      phi: (E, nmax_out, (lmax+1)^2)
    """
    if dij.numel() == 0:
        nmax = radial_basis.nmax + (1 if radial_basis.include_g0 else 0)
        nY = (lmax + 1) ** 2
        return torch.zeros((0, nmax, nY), dtype=dij.dtype, device=dij.device)

    g = radial_basis(dij)                  # (E, nmax_out)
    rhat = unit_vectors(rij)               # (E,3)
    Y = real_sph_harm_torch(rhat, lmax=lmax)  # (E, nY)

    # (E, nmax_out, nY)
    return g.unsqueeze(-1) * Y.unsqueeze(1)


def build_A_nlm(
    n_atoms: int,
    idx_i: torch.Tensor,
    rij: torch.Tensor,
    dij: torch.Tensor,
    radial_basis: RadialBasis,
    lmax: int,
) -> torch.Tensor:
    """
    Single-species atomic base (your original behavior):

      A[i, n, lm] = sum_{e: idx_i[e]==i} phi[e, n, lm]

    Returns:
      A: (N, nmax_out, (lmax+1)^2)
    """
    phi = build_phi_nlm(rij=rij, dij=dij, radial_basis=radial_basis, lmax=lmax)
    nmax = phi.shape[1]
    nY = phi.shape[2]

    A = torch.zeros((n_atoms, nmax, nY), dtype=phi.dtype, device=phi.device)
    if phi.numel() == 0:
        return A

    phi_flat = phi.reshape(phi.shape[0], -1)  # (E, nmax*nY)
    A_flat = A.reshape(n_atoms, -1)           # (N, nmax*nY)
    A_flat.index_add_(0, idx_i.to(torch.int64), phi_flat)
    return A_flat.reshape(n_atoms, nmax, nY)


def build_A_nlm_species(
    n_atoms: int,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    rij: torch.Tensor,
    dij: torch.Tensor,
    species_id: torch.Tensor,
    n_species: int,
    radial_basis: RadialBasis,
    lmax: int,
) -> torch.Tensor:
    """
    Multi-species atomic base (Appendix A style species channels):

      A[i, s, n, lm] = sum_{e: idx_i[e]==i and species(idx_j[e])==s} phi[e, n, lm]

    where phi[e,n,lm] = g_n(r_e) * Y_lm(rhat_e)

    Inputs:
      n_atoms: N
      idx_i: (E,) center indices
      idx_j: (E,) neighbor indices
      rij: (E,3)
      dij: (E,)
      species_id: (N,) int64 tensor, species_id[j] in [0, n_species-1]
      n_species: number of species S
      radial_basis: RadialBasis
      lmax: int

    Returns:
      A: (N, S, nmax_out, (lmax+1)^2)
    """
    if species_id.dtype != torch.int64:
        species_id = species_id.to(torch.int64)

    phi = build_phi_nlm(rij=rij, dij=dij, radial_basis=radial_basis, lmax=lmax)
    nmax = phi.shape[1]
    nY = phi.shape[2]

    A = torch.zeros((n_atoms, n_species, nmax, nY), dtype=phi.dtype, device=phi.device)
    if phi.numel() == 0:
        return A

    # species on each edge: (E,)
    s_edge = species_id[idx_j.to(torch.int64)]

    # flatten (nmax,nY) for fast index_add
    phi_flat = phi.reshape(phi.shape[0], -1)  # (E, nmax*nY)

    # loop over species channels (memory-efficient and simple)
    # if S is small (typical), this is perfectly fine.
    for s in range(n_species):
        mask = (s_edge == s)
        if not torch.any(mask):
            continue

        idx_i_s = idx_i[mask].to(torch.int64)
        phi_s = phi_flat[mask]  # (E_s, nmax*nY)

        A_s_flat = A[:, s, :, :].reshape(n_atoms, -1)
        A_s_flat.index_add_(0, idx_i_s, phi_s)

    return A


def lm_to_index(l: int, m: int) -> int:
    """
    Match spherical.py convention: index = l^2 + (m+l)
    """
    if abs(m) > l:
        raise ValueError("|m| must be <= l")
    return l * l + (m + l)
