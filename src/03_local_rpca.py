"""
03_local_rpca.py — local RPCA on HVG expression and on Harmony PCA space.

We run RPCA twice:
  (a) On log-normalized HVG expression, per coarse cluster.
      -> .layers['lognorm_L']      (denoised expression)
      -> .layers['lognorm_S']      (sparse outliers)
      -> .obs['rpca_outlier_z']    (per-cell outlier z-score)
      The denoised matrix feeds CellTypist in step 04.

  (b) On the Harmony PCA embedding, per coarse cluster.
      -> .obsm['X_pca_harmony_L']
      -> .obs['rpca_emb_outlier_z']
      Useful for: re-clustering, neighbor graphs, downstream label transfer.

Why both? Some annotators want expression (CellTypist), others want
embeddings (scANVI, foundation-model classifiers). We don't want to
commit to one downstream consumer.
"""

import argparse
import numpy as np
import scanpy as sc
from scipy import sparse

from utils_rpca import local_rpca
# from utils_rpca import local_rpca

def _densify_hvg(adata):
    """Return a dense (cells x HVG) array from .X (already log-normalized)."""
    mask = adata.var["highly_variable"].values
    X = adata.X[:, mask]
    if sparse.issparse(X):
        X = X.toarray()
    return X.astype(np.float64), mask


def run(input_path: str, output_path: str, lam_scale: float = 1.0,
        min_cluster_size: int = 30, verbose: bool = True) -> None:
    print(f"[03_rpca] loading {input_path}")
    adata = sc.read_h5ad(input_path)

    if "leiden_coarse" not in adata.obs:
        raise RuntimeError("Run 02_coarse_cluster.py first (need leiden_coarse).")

    clusters = adata.obs["leiden_coarse"].astype(str).values

    # --- (a) Local RPCA on HVG expression -----------------------------------
    print("[03_rpca] (a) local RPCA on HVG log-normalized expression")
    X_hvg, hvg_mask = _densify_hvg(adata)
    L_hvg, S_hvg, z_hvg = local_rpca(
        X_hvg, clusters,
        min_cluster_size=min_cluster_size,
        lam_scale=lam_scale,
        verbose=verbose,
    )

    # Reconstruct full-gene-width matrices (zeros outside HVG slots).
    # CellTypist only needs the genes it knows about, so HVG-only is fine,
    # but we keep the same var axis so .X stays consistent.
    L_full = np.zeros(adata.shape, dtype=np.float64)
    S_full = np.zeros(adata.shape, dtype=np.float64)
    L_full[:, hvg_mask] = L_hvg
    S_full[:, hvg_mask] = S_hvg

    # Store as sparse to save disk
    adata.layers["lognorm_L"] = sparse.csr_matrix(L_full)
    adata.layers["lognorm_S"] = sparse.csr_matrix(S_full)
    adata.obs["rpca_outlier_z"] = z_hvg

    # --- (b) Local RPCA on Harmony PCA embedding ----------------------------
    print("[03_rpca] (b) local RPCA on Harmony PCA embedding")
    X_emb = np.asarray(adata.obsm["X_pca_harmony"], dtype=np.float64)
    L_emb, S_emb, z_emb = local_rpca(
        X_emb, clusters,
        min_cluster_size=min_cluster_size,
        lam_scale=lam_scale,
        verbose=verbose,
    )
    adata.obsm["X_pca_harmony_L"] = L_emb
    adata.obs["rpca_emb_outlier_z"] = z_emb

    adata.write_h5ad(output_path, compression="gzip")
    print(f"[03_rpca] wrote {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lam_scale", type=float, default=1.0)
    ap.add_argument("--min_cluster_size", type=int, default=30)
    args = ap.parse_args()
    run(args.input, args.output,
        lam_scale=args.lam_scale,
        min_cluster_size=args.min_cluster_size)