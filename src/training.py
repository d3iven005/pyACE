# src/training.py
import torch
from typing import Dict, Tuple, Optional, List

from src.io_xyz import System
from src.radial import RadialBasis
from src.neighbours import build_neighbour_list_with_shifts
from src.ace_atomic import build_A_nlm_species
from src.ace_invariants import (
    compute_B1_from_A, compute_B2_from_A, flatten_B_features, compute_energy_linear
)
from src.forces_autograd import precompute_edges, compute_rij_dij_from_edges


@torch.no_grad()
def compute_A_B_X_for_frame(
    sys: System,
    rc: float,
    lmax: int,
    rb: RadialBasis,
    species_map_global: Dict[str, int],
    n_species: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = sys.pos.to(device=device, dtype=dtype)
    cell = sys.cell.to(device=device, dtype=dtype) if sys.cell is not None else None
    pbc = sys.pbc
    N = pos.shape[0]

    species_id = torch.tensor([species_map_global[s] for s in sys.symbols],
                              dtype=torch.int64, device=device)

    idx_i, idx_j, shift, rij, dij = build_neighbour_list_with_shifts(
        pos=pos, rc=rc, cell=cell, pbc=pbc, include_self=False
    )

    A = build_A_nlm_species(
        n_atoms=N, idx_i=idx_i, idx_j=idx_j, rij=rij, dij=dij,
        species_id=species_id, n_species=n_species, radial_basis=rb, lmax=lmax
    )

    B1 = compute_B1_from_A(A)
    B2 = compute_B2_from_A(A, lmax=lmax)
    X, nfeat = flatten_B_features(B1, B2)

    count_by_species = torch.zeros((n_species,), dtype=dtype, device=device)
    Xsum_by_species = torch.zeros((n_species, nfeat), dtype=dtype, device=device)
    for s in range(n_species):
        mask = (species_id == s)
        count_by_species[s] = mask.sum().to(dtype)
        if mask.any():
            Xsum_by_species[s] = X[mask].sum(dim=0)

    return X, species_id, count_by_species, Xsum_by_species


def train_energy_only_ridge(
    frames: List[System],
    rc: float,
    lmax: int,
    rb: RadialBasis,
    species_map_global: Dict[str, int],
    n_species: int,
    device: torch.device,
    dtype: torch.dtype,
    ridge_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X0, _, _, _ = compute_A_B_X_for_frame(frames[0], rc, lmax, rb, species_map_global, n_species, device, dtype)
    nfeat = X0.shape[1]

    P = n_species * nfeat + n_species
    M_rows, y_rows = [], []

    for k, sys in enumerate(frames):
        if "energy" not in sys.info:
            raise ValueError(f"Frame {k} missing info['energy'] in xyz comment line.")
        y = float(sys.info["energy"])

        _, _, count_s, Xsum_s = compute_A_B_X_for_frame(
            sys, rc, lmax, rb, species_map_global, n_species, device, dtype
        )

        row = torch.zeros((P,), dtype=dtype, device=device)
        for s in range(n_species):
            start = s * nfeat
            row[start:start + nfeat] = Xsum_s[s]

        bias_start = n_species * nfeat
        row[bias_start:bias_start + n_species] = count_s

        M_rows.append(row)
        y_rows.append(torch.tensor(y, dtype=dtype, device=device))

    M = torch.stack(M_rows, dim=0)
    yv = torch.stack(y_rows, dim=0)

    I = torch.eye(P, dtype=dtype, device=device)
    A_mat = M.T @ M + ridge_lambda * I
    b_vec = M.T @ yv
    w = torch.linalg.solve(A_mat, b_vec)

    coeff = w[:n_species * nfeat].reshape(n_species, nfeat)
    bias = w[n_species * nfeat:].reshape(n_species)
    return coeff, bias


def forward_energy_forces_frame(
    sys: System,
    edges: Dict[str, torch.Tensor],
    coeff: torch.Tensor,
    bias: Optional[torch.Tensor],
    rc: float,
    lmax: int,
    rb: RadialBasis,
    species_map_global: Dict[str, int],
    n_species: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pos = sys.pos.to(device=device, dtype=dtype).clone().detach().requires_grad_(True)
    cell = sys.cell.to(device=device, dtype=dtype) if sys.cell is not None else None

    species_id = torch.tensor([species_map_global[s] for s in sys.symbols],
                              dtype=torch.int64, device=device)

    rij, dij = compute_rij_dij_from_edges(pos, edges, cell=cell)

    A = build_A_nlm_species(
        n_atoms=pos.shape[0],
        idx_i=edges["idx_i"],
        idx_j=edges["idx_j"],
        rij=rij,
        dij=dij,
        species_id=species_id,
        n_species=n_species,
        radial_basis=rb,
        lmax=lmax,
    )

    B1 = compute_B1_from_A(A)
    B2 = compute_B2_from_A(A, lmax=lmax)
    X, _ = flatten_B_features(B1, B2)

    E_atom, E_total = compute_energy_linear(X, species_id, coeff, bias)

    grad_pos = torch.autograd.grad(E_total, pos, create_graph=True, retain_graph=True)[0]
    forces = -grad_pos
    return E_total, forces


def train_energy_forces_adam(
    frames: List[System],
    rc: float,
    lmax: int,
    rb: RadialBasis,
    species_map_global: Dict[str, int],
    n_species: int,
    device: torch.device,
    dtype: torch.dtype,
    epochs: int,
    lr: float,
    w_energy: float,
    w_forces: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X0, _, _, _ = compute_A_B_X_for_frame(frames[0], rc, lmax, rb, species_map_global, n_species, device, dtype)
    nfeat = X0.shape[1]

    coeff = torch.zeros((n_species, nfeat), dtype=dtype, device=device, requires_grad=True)
    bias = torch.zeros((n_species,), dtype=dtype, device=device, requires_grad=True)

    edges_list = []
    for sys in frames:
        pos0 = sys.pos.to(device=device, dtype=dtype)
        cell0 = sys.cell.to(device=device, dtype=dtype) if sys.cell is not None else None
        edges = precompute_edges(pos0, rc=rc, cell=cell0, pbc=sys.pbc, include_self=False)
        edges = {k: v.to(device=device) for k, v in edges.items()}
        edges_list.append(edges)

    opt = torch.optim.Adam([coeff, bias], lr=lr)

    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)

        loss_E = torch.zeros((), dtype=dtype, device=device)
        loss_F = torch.zeros((), dtype=dtype, device=device)
        mae_E  = torch.zeros((), dtype=dtype, device=device)
        mae_F  = torch.zeros((), dtype=dtype, device=device)

        for sys, edges in zip(frames, edges_list):
            if "energy" not in sys.info:
                raise ValueError("Missing energy in xyz comment line.")
            if sys.forces is None:
                raise ValueError("forces not found in xyz; cannot run force training.")

            E_true = torch.tensor(float(sys.info["energy"]), dtype=dtype, device=device)
            F_true = sys.forces.to(device=device, dtype=dtype)

            E_pred, F_pred = forward_energy_forces_frame(
                sys=sys, edges=edges, coeff=coeff, bias=bias,
                rc=rc, lmax=lmax, rb=rb,
                species_map_global=species_map_global, n_species=n_species,
                device=device, dtype=dtype
            )

            dE = E_pred - E_true
            dF = F_pred - F_true

            loss_E = loss_E + dE**2
            loss_F = loss_F + torch.mean(dF**2)
            mae_E  = mae_E  + torch.abs(dE)
            mae_F  = mae_F  + torch.mean(torch.abs(dF))

        inv = 1.0 / len(frames)
        loss_E = loss_E * inv
        loss_F = loss_F * inv
        mae_E  = mae_E  * inv
        mae_F  = mae_F  * inv

        loss = w_energy * loss_E + w_forces * loss_F
        loss.backward()
        opt.step()

        print(
            f"[epoch {ep:4d}] "
            f"loss={float(loss):.6e}  "
            f"loss_E(MSE)={float(loss_E):.6e}  "
            f"loss_F(MSE)={float(loss_F):.6e}  "
            f"mae_E={float(mae_E):.6e}  "
            f"mae_F={float(mae_F):.6e}"
        )

    return coeff.detach(), bias.detach()
