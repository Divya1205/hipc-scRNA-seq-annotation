#!/usr/bin/env bash
# run_batch.sh — run the full pipeline on MULTIPLE studies, unattended.
#
# Usage:
#   bash run_batch.sh study_01 study_02 ...           # specific studies
#   bash run_batch.sh                                  # all studies in recipe_data/
#
# Each study runs end-to-end (01 -> 05). Failures are logged and SKIPPED so one
# bad study doesn't kill the whole night's run. A summary prints at the end.

set -uo pipefail   # NOTE: no -e, so a single study failing doesn't abort the batch

RECIPE_DIR="data/raw/recipe_data"
ONTOLOGY="data/raw/CT_Ontology_Spreadsheet_20260526.xlsx"
MODEL_DIR="${MODEL_DIR:-Geneformer/Geneformer-V1-10M}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" submission eval data/qc data/clustered data/denoised data/embedded data/annotated

# Determine which studies to run
if [[ $# -gt 0 ]]; then
  STUDIES=("$@")
else
  STUDIES=()
  for d in "$RECIPE_DIR"/*/; do
    STUDIES+=("$(basename "$d")")
  done
fi

echo "=== BATCH START $(date) ==="
echo "Studies to run: ${STUDIES[*]}"
echo ""

declare -a DONE=()
declare -a FAILED=()

for STUDY in "${STUDIES[@]}"; do
  INPUT="${RECIPE_DIR}/${STUDY}/${STUDY}_processed.h5ad"
  LOG="${LOG_DIR}/${STUDY}.log"

  echo "----------------------------------------------------------------"
  echo "[$(date +%H:%M:%S)] START ${STUDY}"

  if [[ ! -f "$INPUT" ]]; then
    echo "  SKIP: input not found: $INPUT" | tee -a "$LOG"
    FAILED+=("$STUDY (no input)")
    continue
  fi

  # Skip if already finished
  if [[ -f "submission/${STUDY}.tsv" ]]; then
    echo "  SKIP: submission/${STUDY}.tsv already exists"
    DONE+=("$STUDY (already done)")
    continue
  fi

  {
    echo "=== ${STUDY} $(date) ==="
    python src/01_qc_normalize.py      --input "$INPUT"  --batch_key sample_id   --output "data/qc/${STUDY}.h5ad"          && \
    python src/02_coarse_cluster.py    --input "data/qc/${STUDY}.h5ad"          --output "data/clustered/${STUDY}.h5ad"   && \
    python src/03_local_rpca.py        --input "data/clustered/${STUDY}.h5ad"   --output "data/denoised/${STUDY}.h5ad"    && \
    python src/03b_foundation_embed.py --input "data/denoised/${STUDY}.h5ad"    --output "data/embedded/${STUDY}.h5ad" --model_size "$MODEL_DIR" && \
    python src/04_annotate.py          --input "data/embedded/${STUDY}.h5ad"    --output "data/annotated/${STUDY}.h5ad"   && \
    python src/04b_embed_annotate.py   --input "data/annotated/${STUDY}.h5ad"   --output "data/annotated/${STUDY}.h5ad"   && \
    python src/05_ontology_map.py      --input "data/annotated/${STUDY}.h5ad"   --ontology "$ONTOLOGY" --output "submission/${STUDY}.tsv"
  } >> "$LOG" 2>&1

  if [[ -f "submission/${STUDY}.tsv" ]]; then
    N=$(($(wc -l < "submission/${STUDY}.tsv") - 1))
    echo "  [$(date +%H:%M:%S)] DONE ${STUDY} -> submission/${STUDY}.tsv (${N} cells)"
    DONE+=("$STUDY")
  else
    echo "  [$(date +%H:%M:%S)] FAILED ${STUDY} — see ${LOG} (tail below):"
    tail -n 15 "$LOG" | sed 's/^/      /'
    FAILED+=("$STUDY")
  fi
done

echo ""
echo "=== BATCH COMPLETE $(date) ==="
echo "DONE  (${#DONE[@]}): ${DONE[*]:-none}"
echo "FAILED(${#FAILED[@]}): ${FAILED[*]:-none}"
echo "Submissions in: submission/    Logs in: ${LOG_DIR}/"
