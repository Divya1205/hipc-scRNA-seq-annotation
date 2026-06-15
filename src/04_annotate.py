# """
# 04_annotate.py — CellTypist annotation on RPCA-denoised expression.

# We run CellTypist twice:
#   (1) on the original log-normalized X (baseline)
#   (2) on the RPCA-denoised L (treatment)

# This gives us an ablation for the paper / submission writeup AND lets us
# flag cells where denoising changed the prediction — those are exactly
# the cells to inspect manually.

# Stores:
#   .obs['ct_label_raw']        — CellTypist on .X (logCP10k)
#   .obs['ct_conf_raw']
#   .obs['ct_label_rpca']       — CellTypist on .layers['lognorm_L']
#   .obs['ct_conf_rpca']
#   .obs['ct_changed']          — boolean: did denoising change the label?
#   .obs['final_label']         — high-confidence rpca label, else raw label,
#                                 with hierarchy fallback for high-outlier cells
#                                 (the actual fallback happens in 05_ontology_map.py)
# """

# import argparse
# import numpy as np
# import scanpy as sc
# import anndata as ad
# from scipy import sparse
# import celltypist
# from celltypist import models


# def _predict(adata: ad.AnnData, model_name: str, majority_voting: bool = True):
#     """Return (labels_df, conf_array). Confidence = max class probability per
#     cell from CellTypist's probability matrix (always available, unlike the
#     optional conf_score column which some versions don't populate)."""
#     model = models.Model.load(model_name)
#     pred = celltypist.annotate(adata, model=model, majority_voting=majority_voting)
#     labels_df = pred.predicted_labels
#     conf = None
#     # Preferred: derive from the probability matrix (robust across versions)
#     try:
#         P = pred.probability_matrix
#         conf = np.asarray(P.values, dtype=float).max(axis=1)
#     except Exception:
#         # Fallback: conf_score column if present
#         if "conf_score" in labels_df.columns:
#             conf = pd.to_numeric(labels_df["conf_score"], errors="coerce").values
#     if conf is None:
#         conf = np.full(adata.n_obs, np.nan)
#     return labels_df, conf


# def run(input_path: str, output_path: str,
#         model_name: str = "Immune_All_Low.pkl") -> None:
#     print(f"[04_annot] loading {input_path}")
#     adata = sc.read_h5ad(input_path)

#     # Make sure CellTypist model is available
#     try:
#         models.Model.load(model_name)
#     except Exception:
#         print(f"[04_annot] downloading CellTypist model {model_name}")
#         models.download_models(force_update=False, model=[model_name])

#     # ---- (1) CellTypist on the original log-normalized matrix -------------
#     print("[04_annot] (1) CellTypist on raw lognorm")
#     adata_raw = adata.copy()
#     adata_raw.X = adata.layers["lognorm"].copy()
#     pred_raw, conf_raw = _predict(adata_raw, model_name)
#     adata.obs["ct_label_raw"] = pred_raw["majority_voting"].values \
#         if "majority_voting" in pred_raw.columns else pred_raw["predicted_labels"].values
#     adata.obs["ct_conf_raw"] = conf_raw

#     # ---- (2) CellTypist on the RPCA-denoised matrix -----------------------
#     # RPCA reconstruction shifts values so they no longer sum to exactly
#     # log1p(10k) per cell, which trips CellTypist's input validator. We
#     # re-normalize the denoised signal back to proper log1p(10k) form so it
#     # passes the check: undo log -> renormalize to 10k -> re-log.
#     print("[04_annot] (2) CellTypist on RPCA-denoised lognorm_L")
#     adata_l = adata.copy()
#     L = adata.layers["lognorm_L"]
#     L = L.toarray() if sparse.issparse(L) else np.asarray(L)
#     # Denoised values can go slightly negative; clip before expm1.
#     counts_like = np.expm1(np.clip(L, 0, None))
#     adata_l.X = sparse.csr_matrix(counts_like)
#     sc.pp.normalize_total(adata_l, target_sum=1e4)
#     sc.pp.log1p(adata_l)
#     pred_l, conf_l = _predict(adata_l, model_name)
#     adata.obs["ct_label_rpca"] = pred_l["majority_voting"].values \
#         if "majority_voting" in pred_l.columns else pred_l["predicted_labels"].values
#     adata.obs["ct_conf_rpca"] = conf_l

#     # ---- Disagreement flag ------------------------------------------------
#     adata.obs["ct_changed"] = (adata.obs["ct_label_raw"].astype(str)
#                                != adata.obs["ct_label_rpca"].astype(str))
#     n_changed = int(adata.obs["ct_changed"].sum())
#     print(f"[04_annot] denoising changed {n_changed}/{adata.n_obs} labels "
#           f"({100*n_changed/adata.n_obs:.1f}%)")

#     # ---- Provisional final label (refined by 05_ontology_map.py) ----------
#     # Default to RPCA prediction; the ontology mapper handles hierarchy fallback.
#     adata.obs["final_label_provisional"] = adata.obs["ct_label_rpca"]

#     adata.write_h5ad(output_path, compression="gzip")
#     print(f"[04_annot] wrote {output_path}")


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--input", required=True)
#     ap.add_argument("--output", required=True)
#     ap.add_argument("--model", default="Immune_All_Low.pkl")
#     args = ap.parse_args()
#     run(args.input, args.output, args.model)

"""
04_annotate.py — CellTypist annotation on RPCA-denoised expression.

We run CellTypist twice:
  (1) on the original log-normalized X (baseline)
  (2) on the RPCA-denoised L (treatment)

This gives us an ablation for the paper / submission writeup AND lets us
flag cells where denoising changed the prediction — those are exactly
the cells to inspect manually.

Stores:
  .obs['ct_label_raw']        — CellTypist on .X (logCP10k)
  .obs['ct_conf_raw']
  .obs['ct_label_rpca']       — CellTypist on .layers['lognorm_L']
  .obs['ct_conf_rpca']
  .obs['ct_changed']          — boolean: did denoising change the label?
  .obs['final_label']         — high-confidence rpca label, else raw label,
                                with hierarchy fallback for high-outlier cells
                                (the actual fallback happens in 05_ontology_map.py)

MEMORY / CORRECTNESS NOTES
--------------------------
The earlier "frugal" version OOM'd and, separately, produced subtly wrong
RPCA predictions. Both came from the same place — how the denoised matrix was
prepped for run (2). This version fixes three things:

* OOM: the original kept a full `adata.copy()` for run (1) alive while run (2)
  made several more full-size dense copies (clip -> expm1 -> csr). With a dense
  ~43k x 26k layer that is many multiples of the matrix in RAM at once. Here
  neither run copies the whole AnnData, the denoised matrix is built once in
  float32 row-chunks, and intermediates are freed promptly.

* Normalization denominator: CellTypist expects values normalized over the WHOLE
  transcriptome (that is the distribution the model was trained on), and its
  input validator additionally requires each cell's expm1 row-sum to be ~1e4.
  Restricting to model genes *before* normalizing breaks both. So we normalize
  over all genes and keep all genes for the call; CellTypist intersects to its
  feature set internally.

* Over-clustering partition: majority voting smooths labels within over-clusters.
  If run (2) re-clusters on the denoised data, `ct_changed` conflates "denoising
  changed the per-cell prediction" with "denoising changed the partition". We
  capture run (1)'s over-clustering and pass it to run (2) so voting happens on
  identical subclusters — denoising is the only thing that varies.
"""

import argparse
import numpy as np
import pandas as pd            # was used by _predict's fallback but never imported
import scanpy as sc
import anndata as ad
from scipy import sparse
import celltypist
from celltypist import models


def _predict(adata: ad.AnnData, model_name: str,
             majority_voting: bool = True, over_clustering=None):
    """Return (labels_df, conf_array). Confidence = max class probability per
    cell from CellTypist's probability matrix (always available, unlike the
    optional conf_score column which some versions don't populate).

    `over_clustering` (array/series/column key, or None) is forwarded to
    CellTypist so the caller can pin the majority-voting partition.
    """
    model = models.Model.load(model_name)
    pred = celltypist.annotate(
        adata, model=model,
        majority_voting=majority_voting,
        over_clustering=over_clustering,
    )
    labels_df = pred.predicted_labels
    conf = None
    # Preferred: derive from the probability matrix (robust across versions)
    try:
        P = pred.probability_matrix
        conf = np.asarray(P.values, dtype=float).max(axis=1)
    except Exception:
        # Fallback: conf_score column if present
        if "conf_score" in labels_df.columns:
            conf = pd.to_numeric(labels_df["conf_score"], errors="coerce").values
    if conf is None:
        conf = np.full(adata.n_obs, np.nan)
    return labels_df, conf


def _denoised_full_logcp10k(L, n_obs: int,
                            target_sum: float = 1e4,
                            chunk: int = 4000) -> np.ndarray:
    """Build FULL-gene logCP10k from the RPCA-denoised layer `L`.

    Each cell is normalized by its full-transcriptome total, so (a) per-gene
    values match the model's training distribution and (b) every cell's expm1
    row-sum is exactly `target_sum`, which keeps CellTypist's validator happy.

    Done in float32 row-chunks into one preallocated array — the full dense
    matrix is never duplicated.  Equivalent math, per chunk:
        clip(<0 -> 0) -> expm1 -> divide by (full-row total) -> *1e4 -> log1p
    """
    is_sp = sparse.issparse(L)
    n_genes = L.shape[1]
    out = np.empty((n_obs, n_genes), dtype=np.float32)
    for start in range(0, n_obs, chunk):
        end = min(start + chunk, n_obs)
        block = L[start:end]
        # fresh, writable, float32 chunk (never mutate the source layer)
        block = block.toarray() if is_sp else np.array(block, dtype=np.float32)
        if block.dtype != np.float32:
            block = block.astype(np.float32, copy=False)
        np.clip(block, 0, None, out=block)        # denoised values can go negative
        np.expm1(block, out=block)                # -> pseudo-counts
        tot = block.sum(axis=1, keepdims=True)    # per-cell total over ALL genes
        tot /= target_sum                         # tot / 1e4
        tot[tot == 0] = 1.0                        # guard empty cells
        block /= tot                              # CP10k
        np.log1p(block, out=block)                # logCP10k
        out[start:end] = block
    return out


def _attach_graph(dst: ad.AnnData, src: ad.AnnData) -> None:
    """Copy an existing neighborhood graph from src onto dst so CellTypist's
    over-clustering reuses it (matches the original run-1 behavior)."""
    if "neighbors" in src.uns:
        dst.uns["neighbors"] = src.uns["neighbors"]
        for k in ("connectivities", "distances"):
            if k in src.obsp:
                dst.obsp[k] = src.obsp[k]


def run(input_path: str, output_path: str,
        model_name: str = "Immune_All_Low.pkl") -> None:
    print(f"[04_annot] loading {input_path}")
    adata = sc.read_h5ad(input_path)

    # The RPCA-denoised layer is dense. Halve its resident footprint up front;
    # this is the single largest steady-state memory saving for big studies and
    # is numerically harmless for annotation.
    L = adata.layers["lognorm_L"]
    if not sparse.issparse(L) and np.asarray(L).dtype != np.float32:
        adata.layers["lognorm_L"] = np.asarray(L).astype(np.float32)

    # Make sure CellTypist model is available
    try:
        models.Model.load(model_name)
    except Exception:
        print(f"[04_annot] downloading CellTypist model {model_name}")
        models.download_models(force_update=False, model=[model_name])

    # ---- (1) CellTypist on the original log-normalized matrix -------------
    # Minimal AnnData (no layer/obs/var baggage) instead of adata.copy(), so we
    # never duplicate the big dense layer. Reuse the dataset's neighbor graph
    # exactly as before.
    print("[04_annot] (1) CellTypist on raw lognorm")
    adata_raw = ad.AnnData(
        X=adata.layers["lognorm"].copy(),
        obs=adata.obs[[]].copy(),
        var=pd.DataFrame(index=adata.var_names),
    )
    _attach_graph(adata_raw, adata)
    pred_raw, conf_raw = _predict(adata_raw, model_name)
    adata.obs["ct_label_raw"] = (
        pred_raw["majority_voting"].values
        if "majority_voting" in pred_raw.columns
        else pred_raw["predicted_labels"].values
    )
    adata.obs["ct_conf_raw"] = conf_raw

    # Capture run-1's over-clustering so run-2 votes on identical subclusters.
    over_clust = (pred_raw["over_clustering"].values
                  if "over_clustering" in pred_raw.columns else None)
    del adata_raw, pred_raw          # free run-1 object before allocating run-2

    # ---- (2) CellTypist on the RPCA-denoised matrix -----------------------
    # Full-gene logCP10k, normalized over the whole transcriptome (see
    # _denoised_full_logcp10k docstring). Built once in float32; no full copies.
    print("[04_annot] (2) CellTypist on RPCA-denoised lognorm_L")
    Xfull = _denoised_full_logcp10k(adata.layers["lognorm_L"], adata.n_obs)
    adata_l = ad.AnnData(
        X=Xfull,
        obs=adata.obs[[]].copy(),
        var=pd.DataFrame(index=adata.var_names),
    )
    # Pass the captured partition; no graph needed when over_clustering is given.
    pred_l, conf_l = _predict(adata_l, model_name, over_clustering=over_clust)
    del adata_l, Xfull               # free the dense run-2 matrix immediately
    adata.obs["ct_label_rpca"] = (
        pred_l["majority_voting"].values
        if "majority_voting" in pred_l.columns
        else pred_l["predicted_labels"].values
    )
    adata.obs["ct_conf_rpca"] = conf_l

    # ---- Disagreement flag ------------------------------------------------
    adata.obs["ct_changed"] = (adata.obs["ct_label_raw"].astype(str)
                               != adata.obs["ct_label_rpca"].astype(str))
    n_changed = int(adata.obs["ct_changed"].sum())
    print(f"[04_annot] denoising changed {n_changed}/{adata.n_obs} labels "
          f"({100*n_changed/adata.n_obs:.1f}%)")

    # ---- Provisional final label (refined by 05_ontology_map.py) ----------
    # Default to RPCA prediction; the ontology mapper handles hierarchy fallback.
    adata.obs["final_label_provisional"] = adata.obs["ct_label_rpca"]

    # NOTE: lognorm_L is preserved in the output for downstream steps. If 05_*
    # does not need it and you are tight on RAM/disk, add `del
    # adata.layers["lognorm_L"]` right after building Xfull above.
    adata.write_h5ad(output_path, compression="gzip")
    print(f"[04_annot] wrote {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Immune_All_Low.pkl")
    args = ap.parse_args()
    run(args.input, args.output, args.model)