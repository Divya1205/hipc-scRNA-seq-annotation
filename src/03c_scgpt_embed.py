"""
03c_scgpt_embed.py — scGPT embeddings + RPCA on the embedding space.

Third foundation model in the ensemble. Architecturally and training-data
distinct from Geneformer, so it's a genuinely independent opinion rather than
a correlated re-run.

Outputs:
  .obsm['X_scgpt']        — raw scGPT cell embedding
  .obsm['X_scgpt_L']      — RPCA-denoised embedding (per coarse cluster)
  .obs['scgpt_outlier_z'] — per-cell embedding outlier z-score

This step is OPTIONAL. If scGPT / flash-attn isn't installed, skip it; the
pipeline runs fine with CellTypist + Geneformer. 04b will use whichever
embeddings are present.

Run after 02_coarse_cluster.py (and it can run on the same input as 03b).
"""

import argparse
import os
import sys
import numpy as np
import scanpy as sc

sys.path.insert(0, os.path.dirname(__file__))
from utils_rpca import local_rpca


def _scgpt_embeddings(adata, model_dir: str, batch_size: int = 64,
                      gene_col: str | None = None):
    """
    Embed cells with scGPT. Uses scgpt.tasks.embed_data, the supported
    high-level API for cell embeddings.

    model_dir : path to the scGPT model directory (config + weights + vocab).
                Default human model dir name is 'scGPT_human'.
    gene_col  : var column holding gene symbols. If None, uses var_names.
    """
    import scgpt as scg

    ad_in = adata.copy()
    # scGPT expects raw counts and gene symbols
    if "counts" in ad_in.layers:
        ad_in.X = ad_in.layers["counts"].copy()
    if gene_col and gene_col in ad_in.var.columns:
        ad_in.var["gene_name"] = ad_in.var[gene_col].astype(str)
    else:
        ad_in.var["gene_name"] = ad_in.var_names.astype(str)

    embedded = scg.tasks.embed_data(
        ad_in,
        model_dir,
        gene_col="gene_name",
        batch_size=batch_size,
        return_new_adata=True,
    )
    # embed_data puts the embedding in .obsm['X_scGPT'] or returns it as .X
    if "X_scGPT" in embedded.obsm:
        return np.asarray(embedded.obsm["X_scGPT"], dtype=np.float64)
    return np.asarray(embedded.X, dtype=np.float64)


def run(input_path: str, output_path: str,
        model_dir: str = "scGPT_human",
        batch_size: int = 64,
        gene_col: str | None = None,
        lam_scale: float = 1.0,
        min_cluster_size: int = 30) -> None:
    print(f"[03c_scgpt] loading {input_path}")
    adata = sc.read_h5ad(input_path)

    if "leiden_coarse" not in adata.obs:
        raise RuntimeError("Run 02_coarse_cluster.py first (need leiden_coarse).")

    print(f"[03c_scgpt] extracting scGPT embeddings ({model_dir}) ...")
    try:
        emb = _scgpt_embeddings(adata, model_dir=model_dir,
                                batch_size=batch_size, gene_col=gene_col)
    except ImportError:
        print("[03c_scgpt] scGPT not installed — skipping. "
              "Pipeline will use CellTypist + Geneformer only.")
        adata.write_h5ad(output_path, compression="gzip")
        return
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and batch_size > 8:
            nb = max(8, batch_size // 2)
            print(f"[03c_scgpt] OOM at bs={batch_size}, retrying at {nb}")
            emb = _scgpt_embeddings(adata, model_dir=model_dir,
                                    batch_size=nb, gene_col=gene_col)
        else:
            raise

    adata.obsm["X_scgpt"] = emb
    print(f"[03c_scgpt] embedding shape: {emb.shape}")

    print("[03c_scgpt] local RPCA on scGPT embedding space")
    clusters = adata.obs["leiden_coarse"].astype(str).values
    L_emb, S_emb, z_emb = local_rpca(
        emb, clusters,
        min_cluster_size=min_cluster_size,
        lam_scale=lam_scale,
        verbose=True,
    )
    adata.obsm["X_scgpt_L"] = L_emb
    adata.obs["scgpt_outlier_z"] = z_emb

    adata.write_h5ad(output_path, compression="gzip")
    print(f"[03c_scgpt] wrote {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_dir", default="scGPT_human",
                    help="Path to scGPT model directory")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--gene_col", default=None)
    ap.add_argument("--lam_scale", type=float, default=1.0)
    ap.add_argument("--min_cluster_size", type=int, default=30)
    args = ap.parse_args()
    run(args.input, args.output,
        model_dir=args.model_dir, batch_size=args.batch_size,
        gene_col=args.gene_col, lam_scale=args.lam_scale,
        min_cluster_size=args.min_cluster_size)