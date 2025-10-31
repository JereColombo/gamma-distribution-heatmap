# Heatmap of Λ(ε, α, κ) for fixed ε, with α on x-axis and κ on y-axis.
# Stable near ε → 0 and works (via analytic continuation) for ε^2 ≥ 2 as well.

import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy.polynomial.laguerre import laggauss

# ====== USER PARAMS ======
eps = 0.6                   # <- FIXED ε (change me). Use any ε > 0.
alpha_min, alpha_max = 0.0, 1.2
kappa_min, kappa_max = 1e-6, 10.0

n_alpha, n_kap = 400, 300   # grid resolution
n_lag = 128                 # Laguerre order (64–128 is good)
mp.mp.dps = 70              # working precision for special functions
# =========================

rt2 = float(np.sqrt(2.0))
alphas = np.linspace(alpha_min, alpha_max, n_alpha)
kappas = np.linspace(kappa_min, kappa_max, n_kap)

# ---------- helpers ----------
nodes, weights = laggauss(n_lag)  # ∫_0^∞ e^{-y} f(y) dy ≈ Σ w_i f(x_i)

def logsumexp(a):
    m = np.max(a)
    if not np.isfinite(m):
        return -np.inf
    return m + np.log(np.sum(np.exp(a - m)))

def Ex_over_1px_laguerre(k, theta):
    """
    E[x/(1+x)] for X~Gamma(k,theta) using Gauss–Laguerre in log-space.
    After x = θ y:
      E[1/(1+X)] = (1/Γ(k)) ∫ y^{k-1} e^{-y} / (1 + θ y) dy
      => log E[1/(1+X)] = log Σ w_i * y_i^{k-1}/(1+θ y_i)  - log Γ(k)
    Return E[x/(1+x)] = 1 - E[1/(1+X)].
    """
    y = nodes
    w = weights
    terms = np.log(w) + (k - 1.0)*np.log(y) - np.log1p(theta * y)
    logI = logsumexp(terms)
    logEinv = logI - float(mp.log(mp.gamma(k)))
    Einv = np.exp(logEinv) if np.isfinite(logEinv) else np.nan
    return 1.0 - Einv

def log_gamma_upper(a, x):
    # log Γ(a, x) using regularized Q(a,x) when possible
    try:
        Q = mp.gammainc(a, x, mp.inf, regularized=True)
        if Q == 0:
            return -mp.inf
        return mp.log(Q) + mp.log(mp.gamma(a))
    except Exception:
        G = mp.gammainc(a, x, mp.inf)
        if G <= 0:
            return -mp.inf
        return mp.log(G)

def Ex_over_1px_analytic(eps, kappa):
    """
    Analytic continuation: E[1/(1+X)] = β^k e^β Γ(1-k, β),
    with k=2/ε^2-1, θ=(ε^2 κ)/2, β=1/θ. Return E[x/(1+x)] = 1 - E[1/(1+X)].
    """
    k     = 2.0/(eps**2) - 1.0
    theta = (eps**2)*kappa/2.0
    beta  = 1.0/theta
    a     = 1 - mp.mpf(k)
    b     = mp.mpf(beta)
    logEinv = k*mp.log(b) + b + log_gamma_upper(a, b)
    if not mp.isfinite(logEinv):
        return np.nan
    Einv = float(mp.e**(logEinv))
    if not np.isfinite(Einv) or abs(Einv) > 1e8:
        return np.nan
    return 1.0 - Einv
# -------------------------

# Precompute E[x/(1+x)] as a function of κ only (α just shifts by -α)
Ex_over_1px_vals = np.empty(n_kap, dtype=float)

if eps**2 < 2.0:
    k = 2.0/(eps**2) - 1.0
    for i, kap in enumerate(kappas):
        theta = (eps**2) * kap / 2.0
        Ex_over_1px_vals[i] = Ex_over_1px_laguerre(k, theta) if theta > 0 else np.nan
else:
    # analytic continuation region
    for i, kap in enumerate(kappas):
        Ex_over_1px_vals[i] = Ex_over_1px_analytic(eps, kap)

# Build Λ(ε, α, κ) = E[x/(1+x)] - α over the (α, κ) grid
# shape: (n_kap, n_alpha) to match imshow (y by x)
Z = Ex_over_1px_vals[:, None] - alphas[None, :]

# Plot
plt.figure(figsize=(8, 6))
im = plt.imshow(
    Z,
    origin='lower',
    extent=[alpha_min, alpha_max, kappa_min, kappa_max],
    aspect='auto'
)
cbar = plt.colorbar(im)
cbar.set_label(r'$\Lambda(\varepsilon,\alpha,\kappa)$', fontsize=14, labelpad=10)

# Λ = 0 contour
ALPHA, KAP = np.meshgrid(alphas, kappas)
CS1 = plt.contour(ALPHA, KAP, Z, levels=[0.0], colors='k', linewidths=1.5)

# Make contour labels horizontal
plt.clabel(
    CS1,
    fmt={0.0: r'$\Lambda=0$'},
    manual=[(0.6, 1.0)],   # example position
    inline=True,
    fontsize=9
)

plt.title(rf'Graph of $\Lambda(\varepsilon,\alpha,\kappa)$ at $\varepsilon={eps}$ fixed')
plt.xlabel(r'$\alpha$', fontsize=14)
plt.ylabel(r'$\kappa$', fontsize=14)

# Change the size of the graduation labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()