"""
05_ontology_map.py — final label assignment + submission TSV.

This is where most teams will lose points. We:
  1. Map every CellTypist label to the challenge ontology via alias table.
  2. Apply hierarchy fallback for low-confidence / high-outlier cells:
       - if CellTypist confidence < CONF_THRESHOLD  -> climb 1 level
       - if RPCA outlier z-score > Z_THRESHOLD      -> climb 1 level
       - if both                                    -> climb 2 levels
       - if label can't be mapped at all            -> 'Unknown' OR root
  3. Write a per-cell TSV in the challenge format.

Configurable via CLI; defaults are conservative.
"""

import argparse
import sys
import pandas as pd
import scanpy as sc

# Allow running as a script from any cwd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from utils_ontology import load_ontology, map_label, CELLTYPIST_ALIAS


def run(input_path: str, ontology_path: str, output_tsv: str,
        conf_threshold: float = 0.5,
        z_threshold: float = 2.5,
        unknown_label: str = "Leukocyte") -> None:
    print(f"[05_ont] loading {input_path}")
    adata = sc.read_h5ad(input_path)
    print(f"[05_ont] loading ontology {ontology_path}")
    ont = load_ontology(ontology_path)

    # Prefer the ensemble label (CellTypist + foundation model) if step 04b ran;
    # otherwise fall back to the CellTypist-only RPCA label.
    if "ensemble_label" in adata.obs:
        print("[05_ont] using ensemble_label (CellTypist + Geneformer)")
        labels_in = adata.obs["ensemble_label"].astype(str).values
    else:
        print("[05_ont] using ct_label_rpca (CellTypist only)")
        labels_in = adata.obs["ct_label_rpca"].astype(str).values

    conf      = adata.obs.get("ct_conf_rpca", pd.Series([1.0]*adata.n_obs)).values
    z_out     = adata.obs.get("rpca_outlier_z", pd.Series([0.0]*adata.n_obs)).values
    z_gf      = adata.obs.get("gf_outlier_z", pd.Series([0.0]*adata.n_obs)).values
    z_sg      = adata.obs.get("scgpt_outlier_z", pd.Series([0.0]*adata.n_obs)).values
    disagree  = adata.obs.get("ensemble_disagree", pd.Series([False]*adata.n_obs)).values

    final = []
    n_mapped = n_fallback1 = n_fallback2 = n_unknown = 0
    for lab, c, z, zg, zs, dis in zip(labels_in, conf, z_out, z_gf, z_sg, disagree):
        mapped = map_label(lab, ont, CELLTYPIST_ALIAS)
        if mapped is None:
            final.append(unknown_label)
            n_unknown += 1
            continue

        low_conf = (c is not None) and (c < conf_threshold)
        # Outlier in ANY space: expression-RPCA, Geneformer-RPCA, scGPT-RPCA
        outlier  = ((z is not None) and (z > z_threshold)) or \
                   ((zg is not None) and (zg > z_threshold)) or \
                   ((zs is not None) and (zs > z_threshold))
        conflict = bool(dis)

        triggers = sum([low_conf, outlier, conflict])
        if triggers >= 2:
            final.append(ont.fallback(mapped, levels_up=2))
            n_fallback2 += 1
        elif triggers == 1:
            final.append(ont.fallback(mapped, levels_up=1))
            n_fallback1 += 1
        else:
            final.append(mapped)
            n_mapped += 1

    adata.obs["final_label"] = final
    n = adata.n_obs
    print(f"[05_ont] kept granular:   {n_mapped}/{n} ({100*n_mapped/n:.1f}%)")
    print(f"[05_ont] parent fallback: {n_fallback1}/{n} ({100*n_fallback1/n:.1f}%)")
    print(f"[05_ont] gp fallback:     {n_fallback2}/{n} ({100*n_fallback2/n:.1f}%)")
    print(f"[05_ont] unknown:         {n_unknown}/{n} ({100*n_unknown/n:.1f}%)")

    # Validate every final label is in the ontology
    final_set = set(adata.obs["final_label"].unique())
    missing = final_set - ont.terms
    if missing:
        raise RuntimeError(f"[05_ont] FAIL — these labels are not in the ontology: {missing}. "
                           f"Fix the alias map or the fallback root.")

    # ---- Build a per-cell confidence score --------------------------------
    # Priority of signals (use the first that has real, non-degenerate values):
    #   1. CellTypist conf_score (ct_conf_rpca) — if the version emits it
    #   2. Geneformer kNN vote fraction (gf_conf) — from step 04b
    #   3. Method agreement: 0.9 if CellTypist & Geneformer agree, else 0.5
    # Then DOWN-WEIGHT cells that were pushed to a parent via fallback, and
    # cells flagged as ensemble disagreements.
    import numpy as _np

    def _usable(col):
        if col not in adata.obs:
            return None
        v = pd.to_numeric(adata.obs[col], errors="coerce").values
        if _np.all(_np.isnan(v)):
            return None
        v = _np.nan_to_num(v, nan=0.0)
        if _np.nanstd(v) < 1e-9:   # all identical -> not informative
            return None
        return _np.clip(v, 0.0, 1.0)

    base = _usable("ct_conf_rpca")
    src = "CellTypist conf_score"
    if base is None:
        base = _usable("gf_conf")
        src = "Geneformer kNN vote fraction"
    if base is None:
        # agreement-based fallback
        if "gf_label" in adata.obs and "ct_label_rpca" in adata.obs:
            agree = (adata.obs["gf_label"].astype(str).values
                     == adata.obs["ct_label_rpca"].astype(str).values)
            base = _np.where(agree, 0.9, 0.5)
            src = "CellTypist/Geneformer agreement (0.9/0.5)"
        else:
            base = _np.full(adata.n_obs, 0.5)
            src = "uniform 0.5 (no signal available)"
    print(f"[05_ont] confidence source: {src}")

    # Down-weight fallback + disagreement cells
    conf_score = base.astype(float).copy()
    final_arr = _np.asarray(final)
    pred_is_parent = final_arr != _np.asarray(
        [map_label(l, ont, CELLTYPIST_ALIAS) or unknown_label for l in labels_in])
    conf_score[pred_is_parent] *= 0.7          # parent fallback => less certain at leaf
    if "ensemble_disagree" in adata.obs:
        conf_score[adata.obs["ensemble_disagree"].values.astype(bool)] *= 0.8
    conf_score = _np.clip(conf_score, 0.0, 1.0).round(4)

    # ---- Write submission TSV ---------------------------------------------
    out_df = pd.DataFrame({
        "cell_barcode":        adata.obs_names.values,
        "predicted_cell_type": adata.obs["final_label"].values,
        "confidence_score":    conf_score,
    })
    out_df.to_csv(output_tsv, sep="\t", index=False)
    print(f"[05_ont] wrote {output_tsv} ({len(out_df)} rows)")
    print(f"[05_ont] confidence range: {conf_score.min():.3f}–{conf_score.max():.3f}, "
          f"mean {conf_score.mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--ontology", required=True,
                    help="Path to CT_Ontology_Spreadsheet_*.tsv or .xlsx")
    ap.add_argument("--output", required=True, help="Output .tsv path")
    ap.add_argument("--conf_threshold", type=float, default=0.5)
    ap.add_argument("--z_threshold",    type=float, default=2.5)
    ap.add_argument("--unknown_label",  default="Leukocyte",
                    help="Root-level label to use when a prediction can't be mapped.")
    args = ap.parse_args()
    run(args.input, args.ontology, args.output,
        conf_threshold=args.conf_threshold,
        z_threshold=args.z_threshold,
        unknown_label=args.unknown_label)