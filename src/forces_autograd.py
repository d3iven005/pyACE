# src/forces_autograd.py
import torch
from typing import Optional, Tuple, Dict, Any

from src.neighbours import build_neighbour_list_with_shifts
from src.ace_atomic import build_A_nlm_species
from src.ace_invariants import compute_B1_from_A, compute_B2_from_A, flatten_B_features, compute_energy_linear
from src.radial import RadialBasis


def precompute_edges(
    pos: torch.Tensor,
    rc: float,
    cell: Optional[torch.Tensor] = None,
    pbc: Tuple[bool, bool, bool] = (False, False, False),
    include_self: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Build and freeze a neighbor list (edges). This step is NOT differentiable
    and should be done outside the force computation loop.

    Returns a dict containing:
      idx_i: (E,) center indices
      idx_j: (E,) neighbor indices
      shift: (E,3) integer lattice shifts (zeros for non-PBC)
    """
    idx_i, idx_j, shift, _rij, _dij = build_neighbour_list_with_shifts(
        pos=pos,
        rc=rc,
        cell=cell,
        pbc=pbc,
        include_self=include_self,
    )
    return {"idx_i": idx_i, "idx_j": idx_j, "shift": shift}


def compute_rij_dij_from_edges(
    pos: torch.Tensor,
    edges: Dict[str, torch.Tensor],
    cell: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable reconstruction of rij and dij from frozen edges.

    rij_e = (pos[j] + shift @ cell) - pos[i]
    dij_e = ||rij_e||

    Inputs:
      pos: (N,3) requires_grad allowed
      edges: dict with idx_i, idx_j, shift
      cell: (3,3) if periodic; if None, shift is assumed zero

    Returns:
      rij: (E,3)
      dij: (E,)
    """
    idx_i = edges["idx_i"].to(torch.int64)
    idx_j = edges["idx_j"].to(torch.int64)
    shift = edges["shift"].to(torch.int64)

    rij = pos[idx_j] - pos[idx_i]  # (E,3)

    if cell is not None:
        # shift_cart = shift @ cell (cell rows are lattice vectors)
        shift_cart = (shift.to(pos.dtype) @ cell)  # (E,3)
        rij = rij + shift_cart

    dij = torch.linalg.norm(rij, dim=-1)  # (E,)
    return rij, dij


def ace_total_energy_from_edges(
    pos: torch.Tensor,
    species_id: torch.Tensor,
    n_species: int,
    edges: Dict[str, torch.Tensor],
    radial_basis: RadialBasis,
    lmax: int,
    coeff: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    cell: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable ACE energy computation using fixed edges.

    Returns:
      E_atom: (N,)
      E_total: scalar
    """
    N = pos.shape[0]
    rij, dij = compute_rij_dij_from_edges(pos, edges, cell=cell)

    A = build_A_nlm_species(
        n_atoms=N,
        idx_i=edges["idx_i"],
        idx_j=edges["idx_j"],
        rij=rij,
        dij=dij,
        species_id=species_id,
        n_species=n_species,
        radial_basis=radial_basis,
        lmax=lmax,
    )  # (N,S,nr,nY)

    B1 = compute_B1_from_A(A)            # (N,S,nr)
    B2 = compute_B2_from_A(A, lmax=lmax) # (N,S,S,nr,nr,lmax+1)

    X, _ = flatten_B_features(B1, B2)    # (N,nfeat)

    E_atom, E_total = compute_energy_linear(X, species_id, coeff, bias)
    return E_atom, E_total


def ace_forces_autograd(
    pos: torch.Tensor,
    species_id: torch.Tensor,
    n_species: int,
    edges: Dict[str, torch.Tensor],
    radial_basis: RadialBasis,
    lmax: int,
    coeff: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    cell: Optional[torch.Tensor] = None,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute forces via autograd:
      F = - dE_total / dpos

    Inputs:
      pos: (N,3) torch tensor (will be copied to requires_grad=True inside)
      create_graph: True if you need higher derivatives (usually False)

    Returns:
      E_atom: (N,)
      E_total: scalar
      forces: (N,3)
    """
    pos_req = pos.clone().detach().requires_grad_(True)

    E_atom, E_total = ace_total_energy_from_edges(
        pos=pos_req,
        species_id=species_id,
        n_species=n_species,
        edges=edges,
        radial_basis=radial_basis,
        lmax=lmax,
        coeff=coeff,
        bias=bias,
        cell=cell,
    )

    grad_pos = torch.autograd.grad(
        E_total,
        pos_req,
        create_graph=create_graph,
        retain_graph=create_graph,
        only_inputs=True,
    )[0]

    forces = -grad_pos
    return E_atom.detach(), E_total.detach(), forces.detach()
