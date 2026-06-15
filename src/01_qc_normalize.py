"""
01_qc_normalize.py — QC, normalization, HVG selection, PCA, batch correction.

Input : raw or filtered .h5ad (cells x genes, raw counts in .X or .layers['counts'])
Output: .h5ad with .X = log-normalized, .obsm['X_pca'], .obsm['X_pca_harmony'],
        .obs['doublet_score'], .obs['n_genes'], .obs['pct_mt'].

Conservative defaults that match what most challenge teams will use.
Two non-obvious checks:
  1. Gene-symbol sanity (the challenge forum reported one dataset with
     integer indices in var_names).
  2. Mitochondrial prefix auto-detection (human MT- vs mouse mt-).
"""

import argparse
import numpy as np
import scanpy as sc
import anndata as ad


def _check_gene_symbols(adata: ad.AnnData) -> None:
    """Fail loudly if var_names look like indices, not gene symbols."""
    sample = list(adata.var_names[:20])
    looks_numeric = all(s.isdigit() for s in sample)
    if looks_numeric:
        raise ValueError(
            "var_names appear to be integer indices, not gene symbols. "
            "Check adata.var for a 'gene_symbols' or 'feature_name' column "
            "and assign it to var_names before continuing. "
            "(See HIPC forum: 'Missing gene names in dataset'.)"
        )


def _detect_mito_prefix(adata: ad.AnnData) -> str:
    """Return the mito gene prefix present in this dataset."""
    if any(g.startswith("MT-") for g in adata.var_names):
        return "MT-"
    if any(g.startswith("mt-") for g in adata.var_names):
        return "mt-"
    # Fallback: no mito genes detected. Don't fail; just warn.
    print("[warn] No mitochondrial gene prefix detected (MT- or mt-). "
          "pct_mt will be 0 for all cells.")
    return "__none__"


def run(input_path: str, output_path: str, batch_key: str | None = "sample",
        keep_all_cells: bool = True) -> None:
    print(f"[01_qc] loading {input_path}")
    adata = sc.read_h5ad(input_path)
    adata.var_names_make_unique()

    # --- Drop non-RNA features (CITE-seq / multimodal) -----------------------
    # Some studies are CITE-seq: the matrix concatenates surface-protein (ADT)
    # features with genes. Protein features break gene-based annotation
    # (CellTypist/Geneformer expect gene symbols), so keep only the RNA modality.
    # Detect by common ADT naming conventions.
    _ADT_SUFFIXES = ("_PROT", "_ADT", "_TotalSeqB", "_TotalSeqC", "_AB", "-ADT")
    _ADT_PREFIXES = ("ADT_", "PROT_", "AB_")
    vnames = adata.var_names.astype(str)
    is_adt = np.array([
        v.endswith(_ADT_SUFFIXES) or v.startswith(_ADT_PREFIXES)
        for v in vnames
    ])
    # Also honor an explicit modality/feature_type column if present
    for col in ("feature_types", "feature_type", "modality"):
        if col in adata.var.columns:
            ft = adata.var[col].astype(str).str.lower()
            is_adt = is_adt | ft.isin(
                ["antibody capture", "antibody_capture", "adt", "protein"]
            ).values
            break
    if is_adt.any():
        n_adt = int(is_adt.sum())
        print(f"[01_qc] CITE-seq detected: dropping {n_adt} protein/ADT "
              f"features, keeping {int((~is_adt).sum())} genes")
        adata = adata[:, ~is_adt].copy()

    _check_gene_symbols(adata)

    # --- Locate raw counts ---------------------------------------------------
    # This is a *_processed.h5ad, so .X may already be normalized. Geneformer
    # and HVG selection need raw counts. Look for a counts layer; if none and
    # .X looks like integers, treat .X as counts; otherwise warn.
    if "counts" in adata.layers:
        print("[01_qc] using existing 'counts' layer as raw counts")
    elif any(k in adata.layers for k in ("raw_counts", "raw", "umi")):
        src = next(k for k in ("raw_counts", "raw", "umi") if k in adata.layers)
        adata.layers["counts"] = adata.layers[src].copy()
        print(f"[01_qc] copied layer '{src}' -> 'counts'")
    elif adata.raw is not None:
        print("[01_qc] using adata.raw as counts source")
        adata.layers["counts"] = adata.raw[:, adata.var_names].X.copy()
    else:
        from scipy import sparse as _sp
        Xs = adata.X[:1000].toarray() if _sp.issparse(adata.X) else adata.X[:1000]
        looks_int = np.allclose(Xs, np.round(Xs))
        if looks_int:
            print("[01_qc] .X looks like raw counts — using as 'counts'")
            adata.layers["counts"] = adata.X.copy()
        else:
            print("[01_qc] WARNING: .X looks normalized and no counts layer found. "
                  "Using .X as-is for 'counts'; Geneformer results may degrade. "
                  "Check the QC report / data docs for where raw counts live.")
            adata.layers["counts"] = adata.X.copy()

    # --- QC metrics ----------------------------------------------------------
    mito_prefix = _detect_mito_prefix(adata)
    if mito_prefix != "__none__":
        mt_flags = adata.var_names.str.startswith(mito_prefix)
        # Coerce to plain numpy bool: some studies yield a pandas nullable
        # BooleanArray, which scipy sparse indexing cannot handle.
        mt_flags = np.asarray(mt_flags, dtype=bool)
    else:
        mt_flags = np.zeros(adata.n_vars, dtype=bool)
    adata.var["mt"] = mt_flags
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                               log1p=False, inplace=True)
    adata.obs.rename(columns={"pct_counts_mt": "pct_mt"}, inplace=True)

    # --- Cell + gene filtering ----------------------------------------------
    # IMPORTANT: the challenge requires a label for EVERY barcode in the
    # study's annotation template. Dropping cells here = missing barcodes in
    # the submission. Since the input is already a *_processed.h5ad (QC'd by
    # the organizers), default behavior is to NOT drop any cells.
    n_before = adata.n_obs
    if not keep_all_cells:
        sc.pp.filter_cells(adata, min_genes=200)
        adata = adata[adata.obs["pct_mt"] < 20].copy()
        adata = adata[adata.obs["n_genes_by_counts"] < 8000].copy()
        print(f"[01_qc] cells filtered: {n_before} -> {adata.n_obs}")
    else:
        print(f"[01_qc] keep_all_cells=True — no cells dropped ({n_before} kept)")
    # Gene filtering never removes cells, so it's always safe:
    sc.pp.filter_genes(adata, min_cells=3)

    # --- Doublet detection (keep, don't drop — ontology has a 'Doublet' label)
    try:
        sc.pp.scrublet(adata, batch_key=batch_key if batch_key in adata.obs else None)
    except Exception as e:
        print(f"[01_qc] scrublet failed: {e}. Skipping doublet detection.")
        adata.obs["doublet_score"] = 0.0
        adata.obs["predicted_doublet"] = False

    # --- Normalization -------------------------------------------------------
    # Always (re)derive normalization from the counts layer so the result is
    # correct whether the input .X was raw or already normalized.
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Store normalized for CellTypist (it reads .X)
    adata.layers["lognorm"] = adata.X.copy()

    # --- HVG -----------------------------------------------------------------
    # seurat_v3 is preferred but its loess fit can hit numerical singularities
    # on some datasets. Fall back to the default 'seurat' flavor (no loess),
    # which is robust, if v3 fails.
    bkey = batch_key if (batch_key and batch_key in adata.obs) else None
    try:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=3000, flavor="seurat_v3",
            layer="counts", batch_key=bkey,
        )
        print("[01_qc] HVG: seurat_v3")
    except Exception as e:
        print(f"[01_qc] seurat_v3 HVG failed ({str(e)[:80]}); falling back to 'seurat' flavor")
        # 'seurat' flavor expects log-normalized data in .X (already set above)
        sc.pp.highly_variable_genes(
            adata, n_top_genes=3000, flavor="seurat", batch_key=bkey,
        )
        print("[01_qc] HVG: seurat (fallback)")

    # --- PCA on HVG-scaled data ---------------------------------------------
    # Work on a HVG-subset view for PCA, but keep the full adata around.
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=50, random_state=0)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]

    # --- Batch correction with Harmony --------------------------------------
    if batch_key and batch_key in adata.obs:
        try:
            import harmonypy as hm
            ho = hm.run_harmony(
                adata.obsm["X_pca"],
                adata.obs,
                vars_use=[batch_key],
                random_state=0,
            )
            n_cells = adata.n_obs
            Z = np.asarray(ho.Z_corr)
            # harmonypy versions differ in orientation; pick whichever axis
            # equals the cell count so we always end up (n_cells, n_pcs).
            if Z.shape[0] == n_cells:
                Zc = Z
            elif Z.shape[1] == n_cells:
                Zc = Z.T
            else:
                raise ValueError(f"Unexpected Z_corr shape {Z.shape} for {n_cells} cells")
            adata.obsm["X_pca_harmony"] = Zc
            print(f"[01_qc] Harmony done on batch_key='{batch_key}' "
                  f"(shape {Zc.shape})")
        except Exception as e:
            print(f"[01_qc] Harmony failed ({e}); using raw PCA.")
            adata.obsm["X_pca_harmony"] = adata.obsm["X_pca"]
    else:
        adata.obsm["X_pca_harmony"] = adata.obsm["X_pca"]

    # --- Sanitize dtypes before writing ------------------------------------
    # Some studies store the barcode index / string columns as pandas
    # StringArray/ArrowStringArray (nullable), which older anndata refuses to
    # write. Coerce the index and any extension-string/boolean columns to plain
    # numpy object/bool so the writer accepts them. Values are unchanged.
    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)
    for _df in (adata.obs, adata.var):
        for _col in _df.columns:
            _dt = str(_df[_col].dtype)
            if "string" in _dt.lower() or _dt == "object":
                _df[_col] = _df[_col].astype(str)
            elif _dt == "boolean":  # pandas nullable boolean
                _df[_col] = _df[_col].astype(bool)

    adata.write_h5ad(output_path, compression="gzip")
    print(f"[01_qc] wrote {output_path}  ({adata.n_obs} cells x {adata.n_vars} genes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_key", default="sample")
    ap.add_argument("--allow_cell_filtering", action="store_true",
                    help="If set, drop low-quality cells. DEFAULT IS OFF because "
                         "the challenge requires a label for every template barcode.")
    args = ap.parse_args()
    run(args.input, args.output, args.batch_key,
        keep_all_cells=not args.allow_cell_filtering)