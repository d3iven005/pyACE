from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple
import re
import torch

# =========================
# Data structure
# .info.get("energy", None) .info.get("free_energy", None)
# =========================

@dataclass
class System:
    symbols: List[str]
    pos: torch.Tensor                    # (N,3)
    cell: Optional[torch.Tensor]         # (3,3) or None
    pbc: Tuple[bool, bool, bool]
    info: Dict[str, Any]
    forces: Optional[torch.Tensor] = None  # (N,3) or None


# =========================
# Helpers
# =========================

def _parse_keyvals(comment: str) -> Dict[str, str]:
    """
    Parse key="..." and key=value from extxyz comment line.
    """
    info: Dict[str, str] = {}
    for m in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', comment):
        info[m.group(1)] = m.group(2)
    for tok in comment.split():
        if "=" in tok and not re.search(r'=\s*"', tok):
            k, v = tok.split("=", 1)
            if k not in info:
                info[k] = v
    return info


def _parse_lattice(info: Dict[str, str]) -> Optional[torch.Tensor]:
    if "Lattice" not in info:
        return None
    nums = [float(x) for x in info["Lattice"].split()]
    if len(nums) != 9:
        raise ValueError("Lattice must have 9 numbers")
    return torch.tensor(nums, dtype=torch.float64).reshape(3, 3)


def _parse_pbc(info: Dict[str, str]) -> Tuple[bool, bool, bool]:
    if "pbc" not in info:
        return (False, False, False)

    parts = info["pbc"].replace(",", " ").split()
    if len(parts) < 3:
        return (False, False, False)

    def tf(x: str) -> bool:
        return x.lower() in ("t", "true", "1", "yes")

    return (tf(parts[0]), tf(parts[1]), tf(parts[2]))


def _parse_properties(prop: str) -> List[Tuple[str, str, int]]:
    """
    Parse Properties string into [(name, type, count), ...]
    Example:
      species:S:1:pos:R:3:forces:R:3
    """
    tokens = prop.split(":")
    if len(tokens) % 3 != 0:
        raise ValueError(f"Invalid Properties format: {prop}")

    out = []
    for i in range(0, len(tokens), 3):
        name = tokens[i]
        typ = tokens[i + 1]
        cnt = int(tokens[i + 2])
        out.append((name, typ, cnt))
    return out


# =========================
# Main reader
# =========================

def iter_xyz(path: str, dtype=torch.float64) -> Iterator[System]:
    """
    Stream extxyz / xyz file frame by frame.
    """
    with open(path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue

            n_atoms = int(line)
            comment = f.readline().strip()

            info_raw = _parse_keyvals(comment)
            info: Dict[str, Any] = {}

            # energy / free_energy (optional)
            for key in ("energy", "free_energy"):
                if key in info_raw:
                    info[key] = float(info_raw[key])

            cell = _parse_lattice(info_raw)
            pbc = _parse_pbc(info_raw)

            # Properties schema (optional!)
            prop_schema = None
            if "Properties" in info_raw:
                prop_schema = _parse_properties(info_raw["Properties"])

            symbols: List[str] = []
            pos = torch.zeros((n_atoms, 3), dtype=dtype)
            forces = None

            if prop_schema:
                # find indices
                col_offsets = {}
                offset = 0
                for name, _, cnt in prop_schema:
                    col_offsets[name] = (offset, offset + cnt)
                    offset += cnt

                if "forces" in col_offsets:
                    forces = torch.zeros((n_atoms, 3), dtype=dtype)

            for i in range(n_atoms):
                parts = f.readline().split()

                if prop_schema:
                    cursor = 0
                    for name, typ, cnt in prop_schema:
                        values = parts[cursor: cursor + cnt]
                        cursor += cnt

                        if name == "species":
                            symbols.append(values[0])
                        elif name == "pos":
                            pos[i] = torch.tensor([float(x) for x in values], dtype=dtype)
                        elif name == "forces" and forces is not None:
                            forces[i] = torch.tensor([float(x) for x in values], dtype=dtype)

                else:
                    # fallback: simple xyz
                    symbols.append(parts[0])
                    pos[i] = torch.tensor(parts[1:4], dtype=dtype)

            yield System(
                symbols=symbols,
                pos=pos,
                cell=cell,
                pbc=pbc,
                info=info,
                forces=forces,
            )


def read_xyz(path: str, dtype=torch.float64) -> List[System]:
    """Read all frames into memory (for small files)."""
    return list(iter_xyz(path, dtype=dtype))
