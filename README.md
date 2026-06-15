
# HIPC scRNA-seq Annotation Pipeline

Automated cell-type annotation of immune single-cell RNA-seq studies to the
fixed 39-term **HIPC Cell Ontology** (HIPC scRNA-seq Annotation Benchmark
Challenge, ImmuneSpace). One annotation TSV is produced per study, labelling
every barcode in the study.

The pipeline pairs a marker-based classifier ( **CellTypist** ) with a
foundation-model embedding ( **Geneformer** ), denoises each per-lineage block
with  **Robust PCA** , combines the annotators by a confidence-weighted vote, and
maps the result to the HIPC ontology with confidence-gated hierarchy fallback.
All models are used **pre-trained** (no fine-tuning).

## Repository layout

```
.
├── environment.yml          # conda environment (see Setup)
├── run_all.sh               # full pipeline on ONE study
├── run_batch.sh             # multiple studies, unattended
├── prep_vacc_10.py          # helper: reconstruct logCP10k input for
│                            # vaccination_study_10 from its .raw layer
├── LICENSE
├── README.md
└── src/
    ├── 01_qc_normalize.py        # normalize, HVG, PCA, Harmony; Scrublet scoring
    ├── 02_coarse_cluster.py      # Leiden lineage clusters on the Harmony embedding
    ├── 03_local_rpca.py          # Robust PCA per cluster: HVG expression + Harmony embedding
    ├── 03b_foundation_embed.py   # Geneformer embedding (MPS/CPU) + RPCA on the embedding
    ├── 03c_scgpt_embed.py        # optional 3rd embedding (self-skips if scGPT absent)
    ├── 04_annotate.py            # CellTypist on raw and RPCA-denoised expression
    ├── 04b_embed_annotate.py     # kNN label transfer in embedding space; ensemble vote
    ├── 05_ontology_map.py        # map to 39-term ontology + hierarchy fallback -> TSV
    ├── 06_evaluate.py            # supervised metrics (with truth) or internal agreement
    ├── 07_marker_validation.py   # marker-enrichment QC (own_marker_z, dotplots)
    ├── utils_ontology.py         # ontology alias map + parent/child hierarchy
    └── utils_rpca.py             # Robust PCA (Principal Component Pursuit) engine
```

## Method (pipeline order)

1. **01_qc_normalize** drop CITE-seq/ADT protein features if present; locate
   raw counts (a `counts` layer, else `.raw`, else integer `.X`);
   `normalize_total(1e4)` + `log1p`; 3000 highly variable genes
   (`seurat_v3`, falling back to `seurat`); PCA(50) on scaled HVGs; Harmony
   batch correction on `sample_id`. Doublets are **scored** with Scrublet but
   **retained** . Cell filtering is **off by default** — every barcode is kept
   (an opt-in `--allow_cell_filtering` flag exists but is not used for
   submissions).
2. **02_coarse_cluster** neighbors + Leiden (res 0.4) on the Harmony
   embedding -> coarse lineage clusters (`leiden_coarse`).
3. **03_local_rpca** Robust PCA (`X = L + S`) per cluster on (a) the HVG
   log-expression -> `lognorm_L`, and (b) the Harmony PCA embedding ->
   `X_pca_harmony_L`. The row-norm of `S` gives a per-cell outlier z-score.
4. **03b_foundation_embed** Geneformer (V1-10M) per-cell embedding via a
   manual MPS/CPU forward pass (avoids Geneformer's CUDA-only EmbExtractor),
   then Robust PCA on the embedding per cluster -> `X_geneformer_L`.
   *(primary methodological contribution)*
5. **03c_scgpt_embed** optional scGPT embedding + RPCA, a third independent
   annotator. Self-skips if scGPT/flash-attn isn't installed.
6. **04_annotate** CellTypist (`Immune_All_Low`, majority voting) on both the
   raw and RPCA-denoised expression; per-cell confidence from the probability
   matrix.
7. **04b_embed_annotate** kNN label transfer in each denoised embedding,
   seeded from high-confidence CellTypist labels; a confidence-weighted vote
   reconciles the annotators into `ensemble_label` (+ a disagreement flag).
8. **05_ontology_map** map the ensemble label to the 39-term HIPC ontology
   via an alias table and case-insensitive match; for low-confidence,
   high-outlier, or disagreement cells, climb the hierarchy one level (parent)
   or two (grandparent); write the submission TSV. Unmappable labels fall back
   to a root term rather than being left unassigned.
9. **06_evaluate / 07_marker_validation** supervised metrics when ground
   truth is supplied, otherwise internal agreement/silhouette; plus
   marker-enrichment QC. No ground truth is required.

## Setup

```bash
conda env create -f environment.yml
conda activate hipc
```

**Foundation model.** Geneformer is loaded from a local clone at
`Geneformer/Geneformer-V1-10M` and is **not** redistributed here — obtain it
from the official source. Known-working pins (captured in `environment.yml`):
`transformers==4.40.2` + `peft==0.11.1`; newer `transformers` breaks Geneformer
imports. **scGPT** (step 03c) is optional and CUDA-only; the pipeline runs with
CellTypist + Geneformer alone if it is absent.

## Data

Challenge data is **not included** (file size and data-use restrictions). Place
each study at:

```
data/raw/recipe_data/<study>/<study>_processed.h5ad
```

and the ontology spreadsheet at `data/raw/CT_Ontology_Spreadsheet_*.xlsx`.

## Running

Single study, end to end:

```bash
bash run_all.sh \
  data/raw/recipe_data/<study>/<study>_processed.h5ad \
  data/raw/CT_Ontology_Spreadsheet_20260526.xlsx \
  <study>
```

All studies under `recipe_data/`:

```bash
bash run_batch.sh
```

Specific studies (failure-isolated, auto-skips completed ones):

```bash
bash run_batch.sh <study_a> <study_b>
```

**Output:** `submission/<study>.tsv` with columns `cell_barcode`,
`predicted_cell_type`, `confidence_score` — one row per input barcode. Metrics
land in `eval/`, QC figures in `figures/`.

## Notes & limitations

* **Every barcode is labelled.** Cell filtering is disabled by default
  (`keep_all_cells=True`), so the submission row count matches each study's
  template.
* **Doublets** are scored by Scrublet but retained (the ontology has a `Doublet`
  class and every barcode must be labelled), so this is doublet  *scoring* , not
  removal.
* **Count source.** Step 01 looks for a `counts` layer, then `.raw`, then an
  integer-valued `.X`, and warns if only normalized data is available. Some
  deposits ship without raw counts; `vaccination_study_10` is one, and
  `prep_vacc_10.py` reconstructs log-normalized input for it from the `.raw`
  layer (a documented manual intervention).
* **Annotation quality.** Myeloid, B, NK, DC, and plasma populations annotate
  reliably and are marker-validated. **T-cell fine-subtyping** (naive vs memory
  CD4/CD8) is the weakest area and drives most of the parent-level fallback.
* **Compute.** Tested on macOS Apple Silicon (16 GB, MPS). Studies exceeding
  ~150k cells could not be run and are intended for larger-RAM / GPU compute.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
