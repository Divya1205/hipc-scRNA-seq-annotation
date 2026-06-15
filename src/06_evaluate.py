"""
06_evaluate.py — ablation + agreement metrics for the writeup.

Two modes:

(A) WITH ground truth (--truth_col points to a .obs column, or --truth_tsv
    gives barcode->label). Computes per-method accuracy, macro-F1, and
    balanced accuracy at the ontology level AND collapsed to coarse lineage.
    Methods compared:
        - baseline   : CellTypist on raw lognorm        (ct_label_raw)
        - rpca       : CellTypist on RPCA-denoised L     (ct_label_rpca)
        - geneformer : Geneformer kNN                    (gf_label)
        - scgpt      : scGPT kNN                          (sg_label)
        - ensemble   : confidence-weighted vote          (ensemble_label)
        - final      : ensemble + hierarchy fallback     (final_label)
    This is your ablation table: shows whether RPCA and the ensemble each
    add value.

(B) WITHOUT ground truth (no truth args). Reports INTERNAL agreement metrics:
        - pairwise agreement between annotators (a proxy for confidence)
        - fraction of cells where RPCA changed the CellTypist call
        - fallback rate (how many cells got pushed to a parent label)
        - silhouette of final labels on the denoised embedding
    Useful before the gold standard exists, and the challenge itself scores
    on consistency, so these numbers matter.

Writes a CSV summary and prints a table.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(__file__))
from utils_ontology import load_ontology


def _load_truth(adata, truth_col, truth_tsv):
    if truth_col and truth_col in adata.obs:
        return adata.obs[truth_col].astype(str).values
    if truth_tsv:
        t = pd.read_csv(truth_tsv, sep="\t")
        # expect columns: cell_barcode, celltype
        m = dict(zip(t.iloc[:, 0].astype(str), t.iloc[:, 1].astype(str)))
        return np.array([m.get(bc, "NA") for bc in adata.obs_names.astype(str)])
    return None


def _supervised_metrics(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, f1_score,
                                 balanced_accuracy_score)
    mask = (y_true != "NA") & (y_pred != "NA") & (~pd.isna(y_true)) & (~pd.isna(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return dict(n=0, accuracy=np.nan, macro_f1=np.nan, balanced_acc=np.nan)
    return dict(
        n=int(len(yt)),
        accuracy=accuracy_score(yt, yp),
        macro_f1=f1_score(yt, yp, average="macro", zero_division=0),
        balanced_acc=balanced_accuracy_score(yt, yp),
    )


def _coarsen(labels, ont, target_levels_up=99):
    """Collapse each label up to the root lineage for coarse-level scoring."""
    out = []
    for l in labels:
        cur = str(l)
        anc = ont.ancestors(cur)
        # take the highest non-root ancestor (lineage level) if available
        out.append(anc[-2] if len(anc) >= 2 else (anc[-1] if anc else cur))
    return np.array(out)


def _pairwise_agreement(adata, cols):
    present = [c for c in cols if c in adata.obs]
    res = {}
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a = adata.obs[present[i]].astype(str).values
            b = adata.obs[present[j]].astype(str).values
            res[f"{present[i]}~{present[j]}"] = float(np.mean(a == b))
    return res


def run(input_path, out_csv, ontology_path=None,
        truth_col=None, truth_tsv=None):
    print(f"[06_eval] loading {input_path}")
    adata = sc.read_h5ad(input_path)
    ont = load_ontology(ontology_path) if ontology_path else None

    method_cols = {
        "baseline":   "ct_label_raw",
        "rpca":       "ct_label_rpca",
        "geneformer": "gf_label",
        "scgpt":      "sg_label",
        "ensemble":   "ensemble_label",
        "final":      "final_label",
    }

    y_true = _load_truth(adata, truth_col, truth_tsv)
    rows = []

    if y_true is not None:
        print("[06_eval] mode A: supervised (ground truth provided)")
        for name, col in method_cols.items():
            if col not in adata.obs:
                continue
            yp = adata.obs[col].astype(str).values
            m = _supervised_metrics(y_true, yp)
            m["method"] = name; m["level"] = "fine"
            rows.append(m)
            if ont is not None:
                mc = _supervised_metrics(_coarsen(y_true, ont), _coarsen(yp, ont))
                mc["method"] = name; mc["level"] = "coarse"
                rows.append(mc)
    else:
        print("[06_eval] mode B: internal agreement (no ground truth)")
        agree = _pairwise_agreement(adata, list(method_cols.values()))
        for pair, val in agree.items():
            rows.append(dict(method=pair, level="agreement", accuracy=val,
                             macro_f1=np.nan, balanced_acc=np.nan, n=adata.n_obs))
        if "ct_changed" in adata.obs:
            rows.append(dict(method="rpca_changed_calls", level="diagnostic",
                             accuracy=float(adata.obs["ct_changed"].mean()),
                             macro_f1=np.nan, balanced_acc=np.nan, n=adata.n_obs))
        if "ensemble_disagree" in adata.obs:
            rows.append(dict(method="ensemble_flagged", level="diagnostic",
                             accuracy=float(adata.obs["ensemble_disagree"].mean()),
                             macro_f1=np.nan, balanced_acc=np.nan, n=adata.n_obs))
        # silhouette of final labels on denoised embedding
        for emb in ("X_geneformer_L", "X_pca_harmony_L", "X_scgpt_L"):
            if emb in adata.obsm and "final_label" in adata.obs:
                try:
                    from sklearn.metrics import silhouette_score
                    labs = adata.obs["final_label"].astype(str).values
                    if len(set(labs)) > 1:
                        s = silhouette_score(adata.obsm[emb], labs, sample_size=min(5000, adata.n_obs), random_state=0)
                        rows.append(dict(method=f"silhouette[{emb}]", level="diagnostic",
                                         accuracy=float(s), macro_f1=np.nan,
                                         balanced_acc=np.nan, n=adata.n_obs))
                except Exception as e:
                    print(f"[06_eval] silhouette on {emb} failed: {e}")

    df = pd.DataFrame(rows)
    cols_order = ["method", "level", "n", "accuracy", "macro_f1", "balanced_acc"]
    df = df[[c for c in cols_order if c in df.columns]]
    df.to_csv(out_csv, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n[06_eval] wrote {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, help="CSV summary path")
    ap.add_argument("--ontology", default=None,
                    help="Ontology TSV/XLSX, enables coarse-level scoring")
    ap.add_argument("--truth_col", default=None,
                    help=".obs column with ground-truth labels (mode A)")
    ap.add_argument("--truth_tsv", default=None,
                    help="TSV of barcode->label ground truth (mode A)")
    args = ap.parse_args()
    run(args.input, args.output, ontology_path=args.ontology,
        truth_col=args.truth_col, truth_tsv=args.truth_tsv)