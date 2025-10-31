# gamma-distribution-heatmap
Heatmap &amp; contour of $\Lambda(\varepsilon, \alpha, \kappa)=\mathbb E \left [X/(1+X) \right] - \alpha$, where $X\sim\Gamma(k, \theta)$ ; Gauss–Laguerre quadrature (log-space) + analytic continuation, high-precision via mpmath.

# $\Lambda(\varepsilon, \alpha, \kappa)$ — Heatmap & Contour with Gauss–Laguerre Quadrature

This repository computes and visualizes
\[
\Lambda(\varepsilon,\alpha,\kappa) = \mathbb{E}\!\left[\frac{X}{1+X}\right] - \alpha,
\qquad X\sim \mathrm{Gamma}(k,\theta),
\]
on a 2D grid for fixed \(\varepsilon>0\) while \(\alpha\) (x-axis) and \(\kappa\) (y-axis) vary.  
It uses **Gauss–Laguerre quadrature in log-space** for numerical stability and **analytic continuation** when \(\varepsilon^2\ge 2\). High-precision special functions are handled with `mpmath`.

<p align="center">
  <em>Example output (heatmap with the Λ=0 contour)</em><br/>
  <img src="output/example_lambda_heatmap.png" width="520"/>
</p>

---

## Mathematical outline

Let
- \( k = \frac{2}{\varepsilon^2} - 1 \),  
- \( \theta = \frac{\varepsilon^2\,\kappa}{2} \).

Then
\[
\mathbb{E}\!\left[\frac{1}{1+X}\right]
= \frac{1}{\Gamma(k)}\int_0^\infty \frac{x^{k-1}e^{-x/\theta}}{1+x}\,\frac{dx}{\theta^k}
= \frac{1}{\Gamma(k)}\int_0^\infty \frac{y^{k-1}e^{-y}}{1+\theta y}\,dy,
\]
after the change \(x=\theta y\). We evaluate this integral by **Gauss–Laguerre** and compute
\(\mathbb{E}[X/(1+X)] = 1 - \mathbb{E}[1/(1+X)]\). For \(\varepsilon^2\ge 2\), we switch to the analytic form
\[
\mathbb{E}\!\left[\frac{1}{1+X}\right] = \beta^{k} e^{\beta} \Gamma(1-k,\beta), \quad \beta=\frac{1}{\theta},
\]
implemented via `mpmath` (regularized upper incomplete gamma) with care in log-space.

---

## Key features
- **Stable log-space** accumulation (`logsumexp`) for Gauss–Laguerre nodes/weights.
- **High precision** special functions (`mpmath`, configurable `mp.mp.dps`).
- **Automatic regime selection**: quadrature for \(\varepsilon^2<2\), analytic continuation for \(\varepsilon^2\ge 2\).
- Fast precomputation in \(\kappa\): \(\Lambda = \mathbb{E}[X/(1+X)] - \alpha\) reduces to a vertical shift over \(\alpha\).

