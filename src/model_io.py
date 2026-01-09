# src/model_io.py
import torch
from typing import Dict, Any


def save_ace_model(
    path: str,
    coeff: torch.Tensor,
    bias: torch.Tensor,
    species_map: Dict[str, int],
    species_list: list[str],
    rc: float,
    lmax: int,
    nmax: int,
    include_g0: bool,
    lam: float,
) -> None:
    """
    Save linear ACE model to disk.
    """
    torch.save(
        {
            "coeff": coeff.detach().cpu(),
            "bias": bias.detach().cpu(),
            "species_map": species_map,
            "species_list": species_list,
            "rc": rc,
            "lmax": lmax,
            "nmax": nmax,
            "include_g0": include_g0,
            "lam": lam,
        },
        path,
    )


def load_ace_model(path: str, device=None, dtype=None) -> Dict[str, Any]:
    """
    Load ACE model from disk.

    Returns dict with keys:
      coeff, bias, species_map, species_list, rc, lmax, nmax, include_g0, lam
    """
    ckpt = torch.load(path, map_location="cpu")

    coeff = ckpt["coeff"]
    bias = ckpt["bias"]

    if device is not None:
        coeff = coeff.to(device=device)
        bias = bias.to(device=device)

    if dtype is not None:
        coeff = coeff.to(dtype=dtype)
        bias = bias.to(dtype=dtype)

    ckpt["coeff"] = coeff
    ckpt["bias"] = bias
    return ckpt
