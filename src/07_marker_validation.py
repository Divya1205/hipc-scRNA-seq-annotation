"""
07_marker_validation.py — does each predicted label actually express its markers?

This is the real validation when there's no ground truth. For every predicted
cell type, we score its canonical marker genes (Scanpy score_genes) and check
that cells with that label score HIGHER for their own markers than the rest of
the population. A label whose own-marker enrichment is weak or negative is a
red flag that the assignment is wrong.

Outputs:
  - eval/<study>_marker_report.csv : per-label enrichment table
      label, n_cells, own_marker_z, top_other_label, flag
  - eval/<study>_marker_dotplot.png : Scanpy dotplot (markers x labels) for eyeballing
  - prints a summary, flagging suspicious labels

Inputs:
  --annotated  the annotated .h5ad (has .layers['lognorm'] and the labels)
  --submission the .tsv (cell_barcode, predicted_cell_type) — used for labels
  --label_col  obs column to use instead of submission (default: final_label)

Marker panels below are canonical immune markers mapped to the HIPC ontology
terms. Genes absent from the dataset are skipped automatically.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc

# Canonical markers per HIPC ontology term. Keep to well-established genes.
MARKERS = {
    "CD4 T Cell (ab)":                    ["CD3D", "CD3E", "CD4", "IL7R"],
    "CD4 Naive / T Central Memory":       ["CD3D", "CD4", "CCR7", "SELL", "TCF7"],
    "CD4 T Effector Memory":              ["CD3D", "CD4", "GZMK", "CCL5", "IL7R"],
    "Treg":                               ["CD3D", "CD4", "FOXP3", "IL2RA", "CTLA4"],
    "CD8 T Cell (ab)":                    ["CD3D", "CD8A", "CD8B"],
    "CD8 Naive / T Central Memory":       ["CD3D", "CD8A", "CCR7", "SELL", "TCF7"],
    "CD8 Cytotoxic / T Effector Memory":  ["CD3D", "CD8A", "GZMB", "GZMH", "NKG7", "GNLY"],
    "gdT Cell":                           ["CD3D", "TRGC1", "TRDC", "TRGV9"],
    "MAIT Cell":                          ["CD3D", "SLC4A10", "KLRB1", "TRAV1-2"],
    "NKT Cell":                           ["CD3D", "NKG7", "KLRD1", "GNLY"],
    "NK Cell":                            ["NCAM1", "KLRD1", "NKG7", "GNLY", "FCGR3A", "KLRF1"],
    "B Cell":                             ["CD79A", "CD79B", "MS4A1", "CD19"],
    "Naive B Cell":                       ["MS4A1", "CD79A", "TCL1A", "IGHD", "IL4R"],
    "Memory B Cell":                      ["MS4A1", "CD79A", "CD27", "TNFRSF13B"],
    "Effector B":                         ["CD79A", "PRDM1", "XBP1"],
    "Plasma Cell":                        ["MZB1", "XBP1", "PRDM1", "SDC1", "IGHG1"],
    "Plasmablast":                        ["MZB1", "XBP1", "MKI67", "PRDM1"],
    "Classical Monocyte":                 ["CD14", "LYZ", "S100A8", "S100A9", "VCAN"],
    "Non-Classical Monocyte":             ["FCGR3A", "MS4A7", "CDKN1C", "LYZ"],
    "Intermediate Monocyte":              ["CD14", "FCGR3A", "LYZ", "HLA-DRA"],
    "Monocyte":                           ["LYZ", "CD14", "FCGR3A"],
    "Myeloid Cell":                       ["LYZ", "CST3"],
    "DC":                                 ["FCER1A", "CST3", "HLA-DRA"],
    "Conventional DC 1":                  ["CLEC9A", "XCR1", "BATF3", "CADM1"],
    "Conventional DC 2":                  ["FCER1A", "CD1C", "CLEC10A", "HLA-DRA"],
    "Plasmacytoid DC":                    ["LILRA4", "GZMB", "IL3RA", "CLEC4C", "IRF7", "TCF4"],
    "Granulocyte":                        ["FCGR3B", "CSF3R"],
    "Neutrophil":                         ["FCGR3B", "CSF3R", "S100A8"],
    "Eosinophil":                         ["IL5RA", "CCR3", "PRG2"],
    "Basophil":                           ["CPA3", "MS4A2", "GATA2"],
    "Mast Cell":                          ["CPA3", "TPSAB1", "MS4A2", "KIT"],
    "Platelet":                           ["PPBP", "PF4", "ITGA2B", "GP9"],
    "RBC":                                ["HBB", "HBA1", "HBA2", "ALAS2"],
    "HSC":                                ["CD34", "KIT", "SPINK2", "PRSS57"],
    # parent / coarse terms: use broad lineage markers
    "T Cell":                             ["CD3D", "CD3E", "TRAC"],
    "Lymphoid Cell":                      ["PTPRC", "CD3D", "MS4A1", "NKG7"],
    "Leukocyte":                          ["PTPRC"],
    "Blood Cell":                         ["PTPRC", "HBB", "PPBP"],
    "Doublet":                            [],   # no marker check meaningful
}


def _load_labels(adata, submission, label_col):
    if submission:
        sub = pd.read_csv(submission, sep="\t")
        m = dict(zip(sub["cell_barcode"].astype(str),
                     sub["predicted_cell_type"].astype(str)))
        return np.array([m.get(bc, "NA") for bc in adata.obs_names.astype(str)])
    if label_col in adata.obs:
        return adata.obs[label_col].astype(str).values
    raise RuntimeError("Provide --submission or a valid --label_col.")


def run(annotated, submission, label_col, study_id, outdir="eval"):
    os.makedirs(outdir, exist_ok=True)
    print(f"[07_marker] loading {annotated}")
    adata = sc.read_h5ad(annotated)

    # Use log-normalized expression for scoring
    if "lognorm" in adata.layers:
        adata.X = adata.layers["lognorm"].copy()
    labels = _load_labels(adata, submission, label_col)
    adata.obs["label"] = labels

    present = set(adata.var_names)
    rows = []
    score_cols = []

    # Score each label's marker panel across ALL cells
    for label, genes in MARKERS.items():
        if label not in set(labels):
            continue
        g = [x for x in genes if x in present]
        if len(g) == 0:
            continue
        key = f"score::{label}"
        sc.tl.score_genes(adata, gene_list=g, score_name=key, ctrl_size=50)
        score_cols.append((label, key, g))

    # For each label, compare its own-marker score in its cells vs other cells
    for label, key, g in score_cols:
        in_mask = adata.obs["label"].values == label
        s = adata.obs[key].values
        in_mean = s[in_mask].mean()
        out_mean = s[~in_mask].mean()
        out_sd = s[~in_mask].std() + 1e-9
        own_z = (in_mean - out_mean) / out_sd   # how enriched is own marker in own cells

        # which OTHER label's cells score highest for THIS panel? (confusion check)
        other_scores = {}
        for lab2, key2, _ in score_cols:
            if lab2 == label:
                continue
            other_scores[lab2] = s[adata.obs["label"].values == lab2].mean()
        top_other = max(other_scores, key=other_scores.get) if other_scores else "NA"

        flag = ""
        if own_z < 1.0:
            flag = "WEAK own-marker enrichment"
        if own_z < 0:
            flag = "NEGATIVE — likely misassigned"
        rows.append(dict(
            label=label, n_cells=int(in_mask.sum()),
            n_markers_used=len(g), markers=",".join(g),
            own_marker_z=round(float(own_z), 3),
            in_mean=round(float(in_mean), 3),
            top_other_label=top_other, flag=flag,
        ))

    df = pd.DataFrame(rows).sort_values("own_marker_z", ascending=False)
    csv_path = os.path.join(outdir, f"{study_id}_marker_report.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== Marker enrichment per predicted label "
          "(own_marker_z high = good) ===")
    print(df[["label", "n_cells", "own_marker_z", "top_other_label", "flag"]]
          .to_string(index=False))

    flagged = df[df["flag"] != ""]
    print(f"\n[07_marker] {len(flagged)} label(s) flagged:")
    for _, r in flagged.iterrows():
        print(f"   {r['label']:34s} z={r['own_marker_z']:.2f}  {r['flag']}")

    # Dotplot for visual inspection: one representative marker set
    try:
        # build a compact marker dict limited to labels present, top 3 genes each
        dot = {}
        for label, key, g in score_cols:
            if label in ("Doublet",):
                continue
            dot[label] = g[:3]
        sc.pl.dotplot(adata, dot, groupby="label", standard_scale="var",
                      show=False, save=f"_{study_id}_markers.png")
        # scanpy writes to ./figures/dotplot_..., move it
        src = f"figures/dotplot__{study_id}_markers.png"
        if os.path.exists(src):
            os.replace(src, os.path.join(outdir, f"{study_id}_marker_dotplot.png"))
            print(f"[07_marker] dotplot -> {outdir}/{study_id}_marker_dotplot.png")
    except Exception as e:
        print(f"[07_marker] dotplot skipped: {e}")

    print(f"[07_marker] report -> {csv_path}")

    # --- Label distribution summary ----------------------------------------
    # How many cells got each label, what % of the study, and whether the
    # label is a coarse parent/fallback term (lower granularity = weaker score).
    PARENT_FALLBACK = {
        "Leukocyte", "Lymphoid Cell", "Myeloid Cell", "Blood Cell",
        "T Cell", "B Cell", "Monocyte", "DC", "Doublet",
    }
    vc = pd.Series(labels).value_counts()
    total = int(vc.sum())
    dist = pd.DataFrame({
        "predicted_cell_type": vc.index,
        "n_cells": vc.values,
        "percent": (vc.values / total * 100).round(2),
    })
    dist["is_fallback"] = dist["predicted_cell_type"].isin(PARENT_FALLBACK)
    dist_path = os.path.join(outdir, f"{study_id}_label_distribution.csv")
    dist.to_csv(dist_path, index=False)

    fallback_n = int(dist.loc[dist["is_fallback"], "n_cells"].sum())
    fallback_pct = round(fallback_n / total * 100, 1)
    n_specific = int((~dist["is_fallback"]).sum())

    print("\n=== Label distribution ===")
    print(dist.to_string(index=False))
    print(f"\n[07_marker] {total} cells | {dist.shape[0]} labels "
          f"({n_specific} specific, {dist['is_fallback'].sum()} fallback types)")
    print(f"[07_marker] cells on coarse/fallback labels: "
          f"{fallback_n} ({fallback_pct}%)  "
          f"<-- high % means many cells could not be confidently subtyped")
    print(f"[07_marker] distribution -> {dist_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--submission", default=None)
    ap.add_argument("--label_col", default="final_label")
    ap.add_argument("--study_id", required=True)
    ap.add_argument("--outdir", default="eval")
    a = ap.parse_args()
    run(a.annotated, a.submission, a.label_col, a.study_id, a.outdir)