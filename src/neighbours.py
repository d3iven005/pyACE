import torch
from typing import Tuple, Optional

def _cell_lengths(cell: torch.Tensor) -> torch.Tensor:
    """
    cell: (3,3) with row-vectors as lattice vectors in Cartesian
    returns: (3,) lengths of lattice vectors
    """
    return torch.linalg.norm(cell, dim=1)

def _shift_ranges(cell: torch.Tensor, rc: float, pbc: Tuple[bool, bool, bool]) -> Tuple[int, int, int]:
    """
    Compute integer shift range along each lattice vector direction.
    A safe choice: nmax_k = ceil(rc / |a_k|) + 1 (only for periodic axes).
    """
    lengths = _cell_lengths(cell)
    nmax = []
    for k in range(3):
        if pbc[k]:
            # avoid division by zero
            Lk = float(lengths[k])
            if Lk <= 1e-12:
                nmax.append(0)
            else:
                nmax.append(int(torch.ceil(torch.tensor(rc / Lk)).item()) + 1)
        else:
            nmax.append(0)
    return nmax[0], nmax[1], nmax[2]

def mic_displacement(
    rij: torch.Tensor,
    cell: torch.Tensor,
    pbc: Tuple[bool, bool, bool],
) -> torch.Tensor:
    """
    Apply MIC to displacement vectors rij under PBC.

    Convention:
      Cartesian r = frac @ cell   (cell rows are lattice vectors)
      frac = r @ inv(cell)
    """
    inv_cell = torch.linalg.inv(cell)
    frac = rij @ inv_cell  # (E,3)

    pbc_mask = torch.tensor(pbc, device=rij.device, dtype=torch.bool)
    if pbc_mask.any():
        frac = frac.clone()
        frac[:, pbc_mask] = frac[:, pbc_mask] - torch.round(frac[:, pbc_mask])

    rij_mic = frac @ cell
    return rij_mic

def _build_neighbour_list_mic(
    pos: torch.Tensor,
    rc: float,
    cell: torch.Tensor,
    pbc: Tuple[bool, bool, bool],
    include_self: bool,
):
    """
    Fast path when cell is large enough (no multiple images inside cutoff).
    Returns idx_i, idx_j, shift(zeros), rij, dij
    """
    N = pos.shape[0]
    diff = pos[None, :, :] - pos[:, None, :]  # (N,N,3) r_j - r_i

    diff_flat = diff.reshape(-1, 3)
    diff_flat = mic_displacement(diff_flat, cell, pbc)
    diff = diff_flat.reshape(N, N, 3)

    dist = torch.linalg.norm(diff, dim=-1)  # (N,N)

    if include_self:
        mask = dist < rc
    else:
        mask = (dist < rc) & (dist > 0.0)

    idx_i, idx_j = torch.nonzero(mask, as_tuple=True)
    rij = diff[idx_i, idx_j, :]
    dij = dist[idx_i, idx_j]

    shift = torch.zeros((idx_i.numel(), 3), dtype=torch.int64, device=pos.device)
    return idx_i, idx_j, shift, rij, dij

def _build_neighbour_list_with_shifts(
    pos: torch.Tensor,
    rc: float,
    cell: torch.Tensor,
    pbc: Tuple[bool, bool, bool],
    include_self: bool,
):
    """
    General path: enumerate lattice shifts so that all images within cutoff are included.
    Returns idx_i, idx_j, shift(int), rij, dij
    """
    device = pos.device
    dtype = pos.dtype
    N = pos.shape[0]

    # base pairwise displacement in the home cell
    diff = pos[None, :, :] - pos[:, None, :]  # (N,N,3), r_j - r_i

    n0, n1, n2 = _shift_ranges(cell, rc, pbc)

    # enumerate integer shifts
    s0 = torch.arange(-n0, n0 + 1, device=device, dtype=torch.int64)
    s1 = torch.arange(-n1, n1 + 1, device=device, dtype=torch.int64)
    s2 = torch.arange(-n2, n2 + 1, device=device, dtype=torch.int64)
    shift = torch.cartesian_prod(s0, s1, s2)  # (S,3)

    # Convert shifts to Cartesian translations: t = shift @ cell
    shift_cart = (shift.to(dtype) @ cell)  # (S,3)

    # rij for all shifts: (S,N,N,3)
    rij = diff.unsqueeze(0) + shift_cart[:, None, None, :]

    dij = torch.linalg.norm(rij, dim=-1)  # (S,N,N)

    # build mask
    if include_self:
        mask = dij < rc
    else:
        mask = (dij < rc)

        # exclude exact self in the same image: i==j & shift==(0,0,0)
        # (self interactions for shift != 0 are valid neighbours when cell small)
        zero_shift = (shift[:, 0] == 0) & (shift[:, 1] == 0) & (shift[:, 2] == 0)  # (S,)
        if zero_shift.any():
            # diagonal mask for i==j
            diag = torch.eye(N, dtype=torch.bool, device=device)  # (N,N)
            mask[zero_shift, :, :] &= ~diag

    # collect edges
    s_idx, idx_i, idx_j = torch.nonzero(mask, as_tuple=True)  # each is (E,)

    rij_e = rij[s_idx, idx_i, idx_j, :]  # (E,3)
    dij_e = dij[s_idx, idx_i, idx_j]     # (E,)
    shift_e = shift[s_idx, :]            # (E,3) int64

    return idx_i, idx_j, shift_e, rij_e, dij_e

def build_neighbour_list_with_shifts(
    pos: torch.Tensor,
    rc: float,
    cell: Optional[torch.Tensor] = None,
    pbc: Tuple[bool, bool, bool] = (False, False, False),
    include_self: bool = False,
):
    """
    Public API that always returns shifts.
    """
    if cell is None or not any(pbc):
        # non-PBC: only shift=(0,0,0)
        idx_i, idx_j, rij, dij = build_neighbour_list_naive(pos, rc, None, (False, False, False), include_self)
        shift = torch.zeros((idx_i.numel(), 3), dtype=torch.int64, device=pos.device)
        return idx_i, idx_j, shift, rij, dij

    # Decide fast (MIC) vs shift enumeration
    lengths = _cell_lengths(cell)
    if torch.min(lengths).item() > 2.0 * rc:
        return _build_neighbour_list_mic(pos, rc, cell, pbc, include_self)
    else:
        return _build_neighbour_list_with_shifts(pos, rc, cell, pbc, include_self)

def build_neighbour_list_naive(
    pos: torch.Tensor,
    rc: float,
    cell: Optional[torch.Tensor] = None,
    pbc: Tuple[bool, bool, bool] = (False, False, False),
    include_self: bool = False,
):
    """
    Backwards-compatible API: returns idx_i, idx_j, rij, dij.
    Internally uses:
      - MIC if cell is "large enough"
      - shift enumeration if cell is small (so images are needed)
    """
    if cell is None or not any(pbc):
        # non-PBC O(N^2)
        N = pos.shape[0]
        diff = pos[None, :, :] - pos[:, None, :]  # (N,N,3)
        dist = torch.linalg.norm(diff, dim=-1)

        if include_self:
            mask = dist < rc
        else:
            mask = (dist < rc) & (dist > 0.0)

        idx_i, idx_j = torch.nonzero(mask, as_tuple=True)
        rij = diff[idx_i, idx_j, :]
        dij = dist[idx_i, idx_j]
        return idx_i, idx_j, rij, dij

    # PBC: choose mode
    lengths = _cell_lengths(cell)
    if torch.min(lengths).item() > 2.0 * rc:
        idx_i, idx_j, _shift, rij, dij = _build_neighbour_list_mic(pos, rc, cell, pbc, include_self)
        return idx_i, idx_j, rij, dij
    else:
        idx_i, idx_j, _shift, rij, dij = _build_neighbour_list_with_shifts(pos, rc, cell, pbc, include_self)
        return idx_i, idx_j, rij, dij
