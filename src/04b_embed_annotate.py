"""
04b_embed_annotate.py — multi-model ensemble annotator.

Combines up to THREE independent annotators:
  1. CellTypist        (expression space)        -> ct_label_rpca / ct_conf_rpca  [from step 04]
  2. Geneformer + kNN  (X_geneformer_L space)     -> gf_label / gf_conf            [computed here]
  3. scGPT + kNN       (X_scgpt_L space)          -> sg_label / sg_conf            [computed here, if present]

For each foundation-model embedding present, we do reference-based OR
self-trained kNN label transfer (self-trained from high-confidence CellTypist
seeds when no labeled reference is given).

Ensemble rule: confidence-weighted vote.
  - Each annotator casts a vote for its label, weighted by its confidence.
  - The label with the highest summed weight wins.
  - If the winning margin is thin (< --margin) OR fewer than 2 annotators
    agree, the cell is flagged (ensemble_disagree=True) so step 05 applies
    hierarchy fallback.

Outputs:
  .obs['gf_label'], .obs['gf_conf']
  .obs['sg_label'], .obs['sg_conf']           (only if scGPT embedding present)
  .obs['ensemble_label']                       reconciled label (pre-ontology)
  .obs['ensemble_disagree']                    bool -> triggers fallback in 05
  .obs['ensemble_n_models']                    how many annotators voted
"""

import argparse
import os
import sys
from collections import defaultdict
import numpy as np
import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.dirname(__file__))


def _knn_predict(X_train, y_train, X_pred, k=15):
    if len(X_train) == 0:
        raise ValueError("kNN got an empty training set — no seed labels available.")
    k = max(1, min(k, len(X_train)))
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_pred)
    classes = clf.classes_
    idx = proba.argmax(axis=1)
    return classes[idx], proba[np.arange(len(idx)), idx]


def _embed_annotator(adata, emb_key, seed_label, seed_conf_arr,
                     ref_emb, ref_labels, seed_conf, k):
    """Label-transfer in one embedding space. Returns (labels, conf)."""
    X = np.asarray(adata.obsm[emb_key], dtype=np.float64)
    if ref_emb and ref_labels:
        Xtr = np.load(ref_emb); ytr = np.load(ref_labels, allow_pickle=True).astype(str)
        return _knn_predict(Xtr, ytr, X, k=k)

    conf = np.asarray(seed_conf_arr, dtype=float)
    conf = np.nan_to_num(conf, nan=0.0)
    # Pick high-confidence seeds. If confidence is missing/uniform/too sparse,
    # progressively relax: threshold -> median -> ALL labeled cells.
    mask = conf > seed_conf
    if mask.sum() < max(k * 2, 10):
        med = np.median(conf)
        mask = conf > med
    if mask.sum() < max(k * 2, 10):
        # confidence unusable (all equal / all zero) — seed from every labeled cell
        mask = np.ones(len(conf), dtype=bool)
        print(f"[04b]   confidence scores unusable for {emb_key}; "
              f"seeding kNN from all {int(mask.sum())} CellTypist labels")
    return _knn_predict(X[mask], seed_label[mask], X, k=k)


def _weighted_vote(votes, margin):
    """votes: list of (label, weight). Returns (winner, disagree_flag, n_models)."""
    if not votes:
        return None, True, 0
    tally = defaultdict(float)
    for lab, w in votes:
        tally[lab] += max(w, 1e-6)
    ranked = sorted(tally.items(), key=lambda kv: kv[1], reverse=True)
    winner, top_w = ranked[0]
    second_w = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(tally.values())
    n_for_winner = sum(1 for lab, _ in votes if lab == winner)
    thin_margin = (top_w - second_w) / (total + 1e-9) < margin
    disagree = thin_margin or (len(votes) >= 2 and n_for_winner < 2)
    return winner, disagree, len(votes)


def run(input_path: str, output_path: str,
        ref_emb=None, ref_labels=None,
        seed_conf: float = 0.7, k: int = 15, margin: float = 0.15) -> None:
    print(f"[04b] loading {input_path}")
    adata = sc.read_h5ad(input_path)

    if "ct_label_rpca" not in adata.obs:
        raise RuntimeError("Need ct_label_rpca + ct_conf_rpca from step 04.")
    ct_lab = adata.obs["ct_label_rpca"].astype(str).values
    ct_conf = np.nan_to_num(adata.obs["ct_conf_rpca"].values, nan=0.0)

    have_gf = "X_geneformer_L" in adata.obsm
    if have_gf:
        print("[04b] Geneformer kNN annotator")
        gf_lab, gf_conf = _embed_annotator(
            adata, "X_geneformer_L", ct_lab, ct_conf,
            ref_emb, ref_labels, seed_conf, k)
        adata.obs["gf_label"] = gf_lab
        adata.obs["gf_conf"] = gf_conf
    else:
        print("[04b] no Geneformer embedding found — skipping that annotator")

    have_sg = "X_scgpt_L" in adata.obsm
    if have_sg:
        print("[04b] scGPT kNN annotator")
        sg_lab, sg_conf = _embed_annotator(
            adata, "X_scgpt_L", ct_lab, ct_conf,
            ref_emb, ref_labels, seed_conf, k)
        adata.obs["sg_label"] = sg_lab
        adata.obs["sg_conf"] = sg_conf
    else:
        print("[04b] no scGPT embedding found — skipping that annotator")

    ens, disagree, nmodels = [], [], []
    for i in range(adata.n_obs):
        votes = [(ct_lab[i], ct_conf[i])]
        if have_gf:
            votes.append((adata.obs["gf_label"].iat[i], adata.obs["gf_conf"].iat[i]))
        if have_sg:
            votes.append((adata.obs["sg_label"].iat[i], adata.obs["sg_conf"].iat[i]))
        win, dis, nn = _weighted_vote(votes, margin)
        ens.append(win if win is not None else ct_lab[i])
        disagree.append(dis)
        nmodels.append(nn)

    adata.obs["ensemble_label"] = ens
    adata.obs["ensemble_disagree"] = disagree
    adata.obs["ensemble_n_models"] = nmodels

    n = adata.n_obs
    n_flag = int(np.sum(disagree))
    print(f"[04b] models per cell: {np.mean(nmodels):.1f} | "
          f"flagged for fallback: {n_flag}/{n} ({100*n_flag/n:.1f}%)")

    adata.write_h5ad(output_path, compression="gzip")
    print(f"[04b] wrote {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ref_emb", default=None)
    ap.add_argument("--ref_labels", default=None)
    ap.add_argument("--seed_conf", type=float, default=0.7)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--margin", type=float, default=0.15)
    args = ap.parse_args()
    run(args.input, args.output,
        ref_emb=args.ref_emb, ref_labels=args.ref_labels,
        seed_conf=args.seed_conf, k=args.k, margin=args.margin)