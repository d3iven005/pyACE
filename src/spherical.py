# src/spherical.py
import math
import torch
from typing import Tuple

# -------------------------
# Geometry helpers (keep yours)
# -------------------------
def safe_norm(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1), min=eps))

def unit_vectors(rij: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r = safe_norm(rij, eps=eps)
    return rij / r.unsqueeze(-1)

def lm_to_index(l: int, m: int) -> int:
    if abs(m) > l:
        raise ValueError("Need |m| <= l")
    return l * l + (m + l)

# -------------------------
# Torch real spherical harmonics (differentiable)
# -------------------------

def _double_factorial(n: int) -> float:
    if n <= 0:
        return 1.0
    out = 1.0
    for k in range(n, 0, -2):
        out *= k
    return out

def _norm_lm(l: int, m: int, dtype, device) -> torch.Tensor:
    """
    Normalization for complex spherical harmonics:
      Y_l^m = N_lm P_l^m(cosθ) e^{i m φ}
      N_lm = sqrt((2l+1)/(4π) * ( (l-m)!/(l+m)! ))
    """
    # use lgamma for factorials: n! = exp(lgamma(n+1))
    l_t = torch.tensor(l, dtype=dtype, device=device)
    m_t = torch.tensor(m, dtype=dtype, device=device)
    logfac_lm = torch.lgamma(l_t - m_t + 1.0)
    logfac_lp = torch.lgamma(l_t + m_t + 1.0)
    coef = (2.0 * l_t + 1.0) / (4.0 * math.pi)
    return torch.sqrt(coef * torch.exp(logfac_lm - logfac_lp))

def _assoc_legendre_all_lm(x: torch.Tensor, lmax: int) -> list[list[torch.Tensor]]:
    """
    Compute associated Legendre P_l^m(x) for all l<=lmax, m<=l, using recursion.
    Returns P[l][m] tensors with same shape as x.

    Convention: Condon-Shortley phase INCLUDED:
      P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^{m/2}
    """
    dtype = x.dtype
    device = x.device

    P: list[list[torch.Tensor]] = [[None for _ in range(lmax + 1)] for __ in range(lmax + 1)]

    one = torch.ones_like(x)
    # P_0^0
    P[0][0] = one

    if lmax == 0:
        return P

    # compute sqrt(1-x^2)
    somx2 = torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))

    # diagonal P_m^m
    for m in range(1, lmax + 1):
        coeff = (-1.0) ** m * _double_factorial(2 * m - 1)
        P[m][m] = coeff * (somx2 ** m)

    # P_{m+1}^m
    for m in range(0, lmax):
        P[m + 1][m] = (2 * m + 1) * x * P[m][m]

    # upward recursion for l >= m+2
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            P[l][m] = ((2 * l - 1) * x * P[l - 1][m] - (l + m - 1) * P[l - 2][m]) / (l - m)

    return P

def real_sph_harm_torch(rhat: torch.Tensor, lmax: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Differentiable real spherical harmonics up to lmax.

    Input:
      rhat: (E,3) (need not be perfectly normalized; will normalize safely)
    Output:
      Y_real: (E, (lmax+1)^2) with ordering:
        for l=0..lmax, for m=-l..l, idx = l^2 + (m+l)

    Real construction matched to your SciPy version:
      m = 0:  Y_real(l,0)  = Re(Y_l^0)
      m > 0:  Y_real(l,m)  = sqrt(2) * (-1)^m * Re(Y_l^m)
      m < 0:  Y_real(l,m)  = sqrt(2) * (-1)^{|m|} * Im(Y_l^{|m|})
    """
    if lmax < 0:
        raise ValueError("lmax must be >= 0")

    # normalize
    r = safe_norm(rhat, eps=eps)
    x = rhat[:, 0] / r
    y = rhat[:, 1] / r
    z = rhat[:, 2] / r

    # spherical angles
    theta = torch.acos(torch.clamp(z, -1.0 + 1e-15, 1.0 - 1e-15))
    # avoid atan2(0,0) which can produce NaN gradients
    eps_phi = 1e-12
    x2 = torch.where((x.abs() < eps_phi) & (y.abs() < eps_phi), x + eps_phi, x)
    phi = torch.atan2(y, x2)
    phi = torch.remainder(phi, 2.0 * math.pi)

    # map phi to [0, 2π)
    two_pi = 2.0 * math.pi
    phi = torch.remainder(phi, two_pi)

    cos_theta = torch.cos(theta)

    # associated Legendre P_l^m(cosθ)
    P = _assoc_legendre_all_lm(cos_theta, lmax=lmax)

    E = rhat.shape[0]
    nY = (lmax + 1) ** 2
    out = torch.zeros((E, nY), dtype=rhat.dtype, device=rhat.device)

    # build real SH columns
    for l in range(lmax + 1):
        for m in range(0, l + 1):
            N_lm = _norm_lm(l, m, dtype=rhat.dtype, device=rhat.device)  # scalar tensor
            Y_lm_amp = N_lm * P[l][m]  # (E,)

            if m == 0:
                idx = lm_to_index(l, 0)
                out[:, idx] = Y_lm_amp  # real
            else:
                # complex Y_l^m = amp * (cos(mφ) + i sin(mφ))
                c = torch.cos(m * phi)
                s = torch.sin(m * phi)
                # real basis mapping consistent with your SciPy code
                # m>0 component uses Re(Y_l^m)
                idx_p = lm_to_index(l, m)
                out[:, idx_p] = math.sqrt(2.0) * ((-1.0) ** m) * (Y_lm_amp * c)
                # m<0 component uses Im(Y_l^{|m|})
                idx_n = lm_to_index(l, -m)
                out[:, idx_n] = math.sqrt(2.0) * ((-1.0) ** m) * (Y_lm_amp * s)

    return out
