# src/evaluation.py
import math
import torch
from typing import Dict, List, Optional

from src.io_xyz import System
from src.radial import RadialBasis
from src.ace_invariants import compute_energy_linear
from src.training import compute_A_B_X_for_frame
from src.forces_autograd import precompute_edges, ace_forces_autograd


@torch.no_grad()
def eval_energy_only(
    frames: List[System],
    coeff: torch.Tensor,
    bias: torch.Tensor,
    rc: float,
    lmax: int,
    rb: RadialBasis,
    species_map_global: Dict[str, int],
    n_species: int,
    device: torch.device,
    dtype: torch.dtype,
    tag: str = "eval",
) -> None:
    """
    Evaluate total energy error over frames that contain info['energy'].
    Prints MAE and RMSE.
    """
    errs: List[float] = []

    coeff_d = coeff.to(device=device, dtype=dtype)
    bias_d = bias.to(device=device, dtype=dtype)

    for sys in frames:
        if "energy" not in sys.info:
            continue

        X, species_id, _, _ = compute_A_B_X_for_frame(
            sys, rc, lmax, rb, species_map_global, n_species, device, dtype
        )
        _, E_pred = compute_energy_linear(X, species_id, coeff_d, bias_d)
        E_true = float(sys.info["energy"])
        errs.append(float(E_pred) - E_true)

    if not errs:
        print(f"[{tag}] no energy labels found.")
        return

    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    mae = sum(abs(e) for e in errs) / len(errs)
    print(f"[{tag}] Energy MAE={mae:.6e}  RMSE={rmse:.6e}  over {len(errs)} frames")


def eval_forces(
    frames: List[System],
    coeff: torch.Tensor,
    bias: Optional[torch.Tensor],
    rc: float,
    lmax: int,
    rb: RadialBasis,
    species_map_global: Dict[str, int],
    n_species: int,
    device: torch.device,
    dtype: torch.dtype,
    tag: str = "eval",
) -> None:
    """
    Evaluate force error over frames that contain forces.

    IMPORTANT:
      - Must NOT be under torch.no_grad() because forces require dE/dR.
      - We explicitly wrap the force computation in torch.enable_grad().
    Prints component-wise MAE and RMSE.
    """
    coeff_d = coeff.to(device=device, dtype=dtype)
    bias_d = None if bias is None else bias.to(device=device, dtype=dtype)

    abs_err_sum = 0.0
    sq_err_sum = 0.0
    n_comp = 0

    for sys in frames:
        if sys.forces is None:
            continue

        pos0 = sys.pos.to(device=device, dtype=dtype)
        cell0 = sys.cell.to(device=device, dtype=dtype) if sys.cell is not None else None

        # freeze edges for this configuration
        edges = precompute_edges(pos0, rc=rc, cell=cell0, pbc=sys.pbc, include_self=False)
        edges = {k: v.to(device=device) for k, v in edges.items()}

        # species ids
        species_id = torch.tensor(
            [species_map_global[s] for s in sys.symbols],
            dtype=torch.int64,
            device=device,
        )

        # compute forces by autograd
        with torch.enable_grad():
            _E_atom, _E_total, F_pred = ace_forces_autograd(
                pos=pos0,
                species_id=species_id,
                n_species=n_species,
                edges=edges,
                radial_basis=rb,
                lmax=lmax,
                coeff=coeff_d,
                bias=bias_d,
                cell=cell0,
                create_graph=False,
            )

        F_true = sys.forces.to(device=device, dtype=dtype)
        dF = F_pred - F_true

        abs_err_sum += float(torch.sum(torch.abs(dF)))
        sq_err_sum += float(torch.sum(dF * dF))
        n_comp += dF.numel()

    if n_comp == 0:
        print(f"[{tag}] no forces found.")
        return

    mae = abs_err_sum / n_comp
    rmse = math.sqrt(sq_err_sum / n_comp)
    print(f"[{tag}] Forces component-wise MAE={mae:.6e}  RMSE={rmse:.6e}  over {n_comp} components")
