# src/predict.py
import torch
from typing import Dict, List, Optional

from src.io_xyz import System
from src.radial import RadialBasis
from src.forces_autograd import precompute_edges, ace_forces_autograd


def predict_system(
    sys: System,
    coeff: torch.Tensor,
    bias: Optional[torch.Tensor],
    species_map: Dict[str, int],
    rb: RadialBasis,
    rc: float,
    lmax: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, torch.Tensor]:
    """
    Predict total energy and forces for a single System.
    """
    pos = sys.pos.to(device=device, dtype=dtype)
    cell = sys.cell.to(device=device, dtype=dtype) if sys.cell is not None else None

    species_id = torch.tensor(
        [species_map[s] for s in sys.symbols],
        dtype=torch.int64,
        device=device,
    )
    n_species = len(species_map)

    # freeze edges
    edges = precompute_edges(
        pos=pos,
        rc=rc,
        cell=cell,
        pbc=sys.pbc,
        include_self=False,
    )
    edges = {k: v.to(device=device) for k, v in edges.items()}

    # autograd forces
    with torch.enable_grad():
        E_atom, E_total, forces = ace_forces_autograd(
            pos=pos,
            species_id=species_id,
            n_species=n_species,
            edges=edges,
            radial_basis=rb,
            lmax=lmax,
            coeff=coeff,
            bias=bias,
            cell=cell,
            create_graph=False,
        )

    return float(E_total), forces.detach().cpu()

def write_extxyz(
    path: str,
    systems: List[System],
    energies: List[float],
    forces_list: List[torch.Tensor],
) -> None:
    """
    Write predicted systems to extxyz format with forces and energies.
    """
    assert len(systems) == len(energies) == len(forces_list)

    with open(path, "w", encoding="utf-8") as f:
        for sys, E, F in zip(systems, energies, forces_list):
            N = len(sys.symbols)
            f.write(f"{N}\n")

            # lattice
            lattice_str = ""
            if sys.cell is not None:
                lat = sys.cell.reshape(-1).tolist()
                lattice_str = 'Lattice="' + " ".join(f"{x:.16f}" for x in lat) + '" '

            # pbc
            pbc_str = ""
            if sys.pbc is not None:
                pbc_str = f'pbc="{ " ".join("T" if x else "F" for x in sys.pbc) }" '

            comment = (
                f'{lattice_str}'
                f'Properties=species:S:1:pos:R:3:forces:R:3 '
                f'energy={E:.16f} free_energy={E:.16f} '
                f'{pbc_str}'
            )
            f.write(comment.strip() + "\n")

            for (sym, r, fvec) in zip(sys.symbols, sys.pos, F):
                f.write(
                    f"{sym} "
                    f"{r[0]:.16f} {r[1]:.16f} {r[2]:.16f} "
                    f"{fvec[0]:.16f} {fvec[1]:.16f} {fvec[2]:.16f}\n"
                )

