import torch
from dataclasses import dataclass
from typing import Optional

# rb = RadialBasis(rc=6.0, nmax=6, lam=5.0) nmax g0, g1...gn radial channel

def cutoff_envelope(r: torch.Tensor, rc: float) -> torch.Tensor:
    """
    Smooth cutoff envelope:
      f(r) = 1 + cos(pi r / rc)   for r < rc
             0                   for r >= rc
    This is the envelope used in Drautz ACE radial basis definition. :contentReference[oaicite:1]{index=1}
    """
    # mask inside cutoff
    inside = (r < rc)
    # compute only inside; outside set to 0
    out = torch.zeros_like(r)
    # avoid NaNs for r==rc by masking
    out[inside] = 0.5*(1.0 + torch.cos(torch.pi * r[inside] / rc))
    return out


def chebyshev_T(n: int, x: torch.Tensor) -> torch.Tensor:
    """
    Chebyshev polynomial of the first kind T_n(x), using recurrence:
      T_0 = 1
      T_1 = x
      T_{k} = 2x T_{k-1} - T_{k-2}
    """
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return x

    T0 = torch.ones_like(x)
    T1 = x
    for _k in range(2, n + 1):
        T0, T1 = T1, 2.0 * x * T1 - T0
    return T1


def scaled_x(r: torch.Tensor, rc: float, lam: float) -> torch.Tensor:
    """
    Drautz ACE scaled distance x(r):
      x = 1 - 2 * (exp(-lam*(r/rc - 1)) - 1) / (exp(lam) - 1)
    Defined for r in [0, rc]. We'll compute it for all r but you should
    always multiply by cutoff_envelope to zero out r>=rc. :contentReference[oaicite:2]{index=2}
    """
    # Use tensor for exp(lam) so device/dtype matches
    lam_t = torch.tensor(lam, device=r.device, dtype=r.dtype)
    denom = torch.exp(lam_t) - 1.0

    # avoid division by zero if lam is extremely small
    # (in practice lam ~ 4-8 is typical; keep it safe anyway)
    denom = torch.where(torch.abs(denom) < 1e-12, torch.full_like(denom, 1e-12), denom)

    exp_term = torch.exp(-lam_t * (r / rc - 1.0))
    x = 1.0 - 2.0 * ((exp_term - 1.0) / denom)
    return x


def radial_g(
    r: torch.Tensor,
    rc: float,
    nmax: int,
    lam: float = 5.0,
    include_g0: bool = False,
) -> torch.Tensor:
    """
    Compute Drautz ACE radial basis g_n(r):
      g_0 = 1
      g_1 = 1 + cos(pi r/rc)
      g_k = 1/4 * (1 - T_{k-1}(x(r))) * (1 + cos(pi r/rc)),  k>=2
    with the convention that all are 0 for r >= rc due to envelope. :contentReference[oaicite:3]{index=3}

    Args:
      r: tensor of distances, shape (...), must be >=0
      rc: cutoff radius
      nmax: maximum n (>=1)
      lam: lambda parameter in x(r)
      include_g0: if True, prepend g0=1 (but still masked to 0 outside cutoff)

    Returns:
      g: shape (..., n_out)
         where n_out = nmax if include_g0=False (g1..g_nmax)
               n_out = nmax+1 if include_g0=True (g0..g_nmax)
    """
    if nmax < 1:
        raise ValueError("nmax must be >= 1")

    env = cutoff_envelope(r, rc)   # (...), already 0 outside
    x = scaled_x(r, rc, lam)       # (...), meaningful inside

    gs = []

    if include_g0:
        # Strictly speaking g0=1, but we still mask outside cutoff so features vanish there.
        gs.append(torch.ones_like(r) * (env > 0).to(r.dtype))

    # g1 = envelope
    gs.append(env)

    # gk for k>=2
    for k in range(2, nmax + 1):
        Tk_1 = chebyshev_T(k - 1, x)
        gk = 0.5 * (1.0 - Tk_1) * env
        gs.append(gk)

    return torch.stack(gs, dim=-1)


# Optional: nice OO wrapper
@dataclass
class RadialBasis:
    rc: float
    nmax: int
    lam: float = 5.0
    include_g0: bool = False

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        return radial_g(r, rc=self.rc, nmax=self.nmax, lam=self.lam, include_g0=self.include_g0)


# -------------------------
# Simple self-test (run manually)
# -------------------------
if __name__ == "__main__":
    torch.set_printoptions(precision=6, sci_mode=False)

    rc = 6.0
    nmax = 5
    r = torch.tensor([0.0, 1.0, 5.9, 6.0, 7.0], dtype=torch.float64, requires_grad=True)

    g = radial_g(r, rc=rc, nmax=nmax, lam=5.0, include_g0=False)
    print("g shape:", g.shape)  # (5, nmax)
    print(g)

    # Check cutoff: r>=rc => all zeros
    assert torch.allclose(g[3], torch.zeros(nmax, dtype=g.dtype))
    assert torch.allclose(g[4], torch.zeros(nmax, dtype=g.dtype))

    # Check differentiability
    loss = g.sum()
    loss.backward()
    print("dr loss:", r.grad)
