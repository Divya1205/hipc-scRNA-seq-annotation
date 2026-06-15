"""
utils_rpca.py — Robust PCA via Principal Component Pursuit (PCP).

Decomposes a matrix  X = L + S  where
    L is low-rank   (the underlying biological/clean signal)
    S is sparse     (outliers: dropout artifacts, doublets, ambient RNA,
                     contamination, or per-cell deviations from a manifold).

Reference: Candès, Li, Ma, Wright (2011). "Robust Principal Component Analysis?"
We use the inexact ALM (Augmented Lagrange Multiplier) solver which is the
standard fast solver and converges in 30–100 iters on scRNA-seq–sized blocks.

Usage:
    L, S, info = rpca_pcp(X, lam=None, mu=None, tol=1e-6, max_iter=200)

Notes for scRNA-seq:
- Run on log-normalized, scaled expression OR on PCA-reduced coordinates.
- Running on the full ~20k-gene matrix is wasteful; reduce to top HVGs (2k–5k)
  or to PCA space (50 dim) first.
- For local RPCA, call this once per coarse cluster.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Soft-thresholding operators
# -----------------------------------------------------------------------------
def _shrink(X: np.ndarray, tau: float) -> np.ndarray:
    """Elementwise soft-threshold (proximal op of L1)."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def _svd_shrink(X: np.ndarray, tau: float) -> tuple[np.ndarray, int]:
    """
    Singular value soft-threshold (proximal op of nuclear norm).
    Returns the thresholded matrix and the resulting rank.
    """
    # economy SVD — full_matrices=False is critical for tall/wide matrices
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_thr = np.maximum(s - tau, 0.0)
    rank = int((s_thr > 0).sum())
    if rank == 0:
        return np.zeros_like(X), 0
    return (U[:, :rank] * s_thr[:rank]) @ Vt[:rank, :], rank


# -----------------------------------------------------------------------------
# Main solver
# -----------------------------------------------------------------------------
@dataclass
class RPCAResult:
    L: np.ndarray            # low-rank component, same shape as X
    S: np.ndarray            # sparse component, same shape as X
    rank: int                # rank of L at convergence
    n_iter: int
    converged: bool
    residual: float          # ||X - L - S||_F / ||X||_F


def rpca_pcp(
    X: np.ndarray,
    lam: float | None = None,
    mu: float | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
    verbose: bool = False,
) -> RPCAResult:
    """
    Solve  min ||L||_* + lam * ||S||_1   s.t. L + S = X   via inexact ALM.

    Parameters
    ----------
    X : (n_samples, n_features) array of floats. Cells x genes (or cells x PCs).
    lam : L1 penalty. Default = 1/sqrt(max(n,m)) per Candès et al.
    mu  : ALM penalty. Default = n*m / (4 * ||X||_1).
    tol : Convergence threshold on Frobenius residual.
    max_iter : Hard cap on iterations.

    Returns
    -------
    RPCAResult dataclass.
    """
    X = np.asarray(X, dtype=np.float64)
    n, m = X.shape

    if lam is None:
        lam = 1.0 / np.sqrt(max(n, m))
    if mu is None:
        # Standard heuristic; works well across orders of magnitude
        mu = (n * m) / (4.0 * np.sum(np.abs(X)) + 1e-12)
    mu_inv = 1.0 / mu

    # Initialization
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = X / max(np.linalg.norm(X, 2),
                np.linalg.norm(X, np.inf) / lam)  # dual variable

    norm_X = np.linalg.norm(X, "fro") + 1e-12
    converged = False
    rank = 0
    residual = np.nan

    for it in range(1, max_iter + 1):
        # L-update: SVD shrinkage on (X - S + Y/mu) with threshold 1/mu
        L, rank = _svd_shrink(X - S + mu_inv * Y, mu_inv)
        # S-update: soft-threshold on (X - L + Y/mu) with threshold lam/mu
        S = _shrink(X - L + mu_inv * Y, lam * mu_inv)
        # Dual update
        Z = X - L - S
        Y = Y + mu * Z

        residual = np.linalg.norm(Z, "fro") / norm_X
        if verbose and (it % 10 == 0 or it == 1):
            print(f"  [rpca] iter {it:3d}  rank(L)={rank:3d}  ||S||_0/N={np.mean(S != 0):.3f}  res={residual:.2e}")
        if residual < tol:
            converged = True
            break

    return RPCAResult(
        L=L, S=S, rank=rank, n_iter=it, converged=converged, residual=residual
    )


# -----------------------------------------------------------------------------
# Per-cell outlier score from S
# -----------------------------------------------------------------------------
def outlier_scores(S: np.ndarray) -> np.ndarray:
    """
    Per-cell outlier score = row-wise L2 norm of S, z-scored.
    High score => cell deviates strongly from the local low-rank manifold
                  (likely doublet, ambient-contaminated, dying, or rare type).

    Use this to:
      - flag cells for hierarchy fallback (label at parent level)
      - mark candidate doublets
      - sanity-check that S isn't capturing real biology you care about
    """
    norms = np.linalg.norm(S, axis=1)
    mu = norms.mean()
    sd = norms.std() + 1e-12
    return (norms - mu) / sd


# -----------------------------------------------------------------------------
# Local RPCA: per-cluster decomposition
# -----------------------------------------------------------------------------
def local_rpca(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    min_cluster_size: int = 30,
    lam_scale: float = 1.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run RPCA independently within each cluster, then stitch L, S back together
    in the original cell order.

    This is the key trick: the global scRNA-seq manifold is *not* low-rank
    (T cells and platelets don't share a linear subspace), but it is locally
    low-rank within a coarse lineage cluster. Running RPCA per cluster lets
    the low-rank assumption actually hold.

    Parameters
    ----------
    X : (n_cells, n_features) — log-normalized expression (HVGs) OR PCs.
    cluster_labels : (n_cells,) integer or string cluster assignments.
    min_cluster_size : clusters smaller than this are passed through unchanged.
    lam_scale : multiply the default lam by this. >1 = sparser S (more
                forgiving of outliers); <1 = denser S (more aggressive denoise).

    Returns
    -------
    L_full, S_full, outlier_z   — all shape (n_cells, n_features) or (n_cells,)
    """
    n, m = X.shape
    L_full = np.zeros_like(X, dtype=np.float64)
    S_full = np.zeros_like(X, dtype=np.float64)
    z_full = np.zeros(n, dtype=np.float64)

    unique = np.unique(cluster_labels)
    for c in unique:
        idx = np.where(cluster_labels == c)[0]
        Xc = X[idx]
        if len(idx) < min_cluster_size:
            # Too small to denoise reliably — keep as is, no outliers flagged
            L_full[idx] = Xc
            continue
        lam = lam_scale * (1.0 / np.sqrt(max(Xc.shape)))
        res = rpca_pcp(Xc, lam=lam, verbose=verbose)
        L_full[idx] = res.L
        S_full[idx] = res.S
        z_full[idx] = outlier_scores(res.S)
        if verbose:
            print(f"[local_rpca] cluster {c}: n={len(idx)}, rank(L)={res.rank}, "
                  f"converged={res.converged}, iters={res.n_iter}")

    return L_full, S_full, z_full