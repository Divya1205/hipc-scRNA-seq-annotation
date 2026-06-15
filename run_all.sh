#!/usr/bin/env bash
# run_all.sh — run the full pipeline on one dataset.
# Usage: bash run_all.sh path/to/study_01.h5ad path/to/CT_Ontology.tsv study_01

set -euo pipefail

INPUT="${1:?usage: bash run_all.sh INPUT.h5ad ONTOLOGY.tsv STUDY_ID}"
ONTOLOGY="${2:?usage: bash run_all.sh INPUT.h5ad ONTOLOGY.tsv STUDY_ID}"
STUDY_ID="${3:?usage: bash run_all.sh INPUT.h5ad ONTOLOGY.tsv STUDY_ID}"

mkdir -p data/qc data/clustered data/denoised data/embedded data/annotated submission eval

python src/01_qc_normalize.py      --input "$INPUT"  --batch_key sample_id   --output "data/qc/${STUDY_ID}.h5ad"
python src/02_coarse_cluster.py    --input "data/qc/${STUDY_ID}.h5ad"        --output "data/clustered/${STUDY_ID}.h5ad"
python src/03_local_rpca.py        --input "data/clustered/${STUDY_ID}.h5ad" --output "data/denoised/${STUDY_ID}.h5ad"
python src/03b_foundation_embed.py --input "data/denoised/${STUDY_ID}.h5ad"  --output "data/embedded/${STUDY_ID}.h5ad" --model_size "Geneformer/Geneformer-V1-10M"
# scGPT is optional; if not installed it self-skips and just passes data through.
python src/03c_scgpt_embed.py      --input "data/embedded/${STUDY_ID}.h5ad"  --output "data/embedded/${STUDY_ID}.h5ad" || \
    echo "[run_all] scGPT step skipped"
python src/04_annotate.py          --input "data/embedded/${STUDY_ID}.h5ad"  --output "data/annotated/${STUDY_ID}.h5ad"
python src/04b_embed_annotate.py   --input "data/annotated/${STUDY_ID}.h5ad" --output "data/annotated/${STUDY_ID}.h5ad"
python src/05_ontology_map.py      --input "data/annotated/${STUDY_ID}.h5ad" --ontology "$ONTOLOGY" \
                                   --output "submission/${STUDY_ID}.tsv"
# Evaluation (mode B internal metrics by default; add --truth_col/--truth_tsv when you have labels)
python src/06_evaluate.py          --input "data/annotated/${STUDY_ID}.h5ad" --ontology "$ONTOLOGY" \
                                   --output "eval/${STUDY_ID}_metrics.csv"

echo "Done. Submission: submission/${STUDY_ID}.tsv | Metrics: eval/${STUDY_ID}_metrics.csv"
