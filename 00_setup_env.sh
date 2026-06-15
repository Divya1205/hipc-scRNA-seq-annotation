#!/usr/bin/env bash
# 00_setup_env.sh — one-time environment setup for HIPC scRNA-seq pipeline.
# Tested on Linux x86_64 + CUDA 12.x. CPU-only works too (skip the GPU lines).

set -euo pipefail

ENV_NAME="hipc"

# --- 1. Create conda env -----------------------------------------------------
# Miniforge / Miniconda assumed. Python 3.11 is the sweet spot for the
# scverse stack in 2026: 3.12 still has wheel gaps for some annotation libs.
conda create -y -n "${ENV_NAME}" python=3.11
# Activate inside this script:
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# --- 2. Core scientific + scverse stack via pip ------------------------------
# Pinning versions defensively. Bump only if you have a reason.
pip install --upgrade pip

pip install \
    "numpy>=1.26,<2.1" \
    "scipy>=1.13" \
    "pandas>=2.2" \
    "scikit-learn>=1.5" \
    "matplotlib>=3.8" \
    "seaborn>=0.13" \
    "anndata>=0.10.8" \
    "scanpy>=1.10.3" \
    "leidenalg>=0.10" \
    "igraph>=0.11" \
    "harmonypy>=0.0.10" #\
    # "scrublet>=0.2.3"

# --- 3. Annotation tools -----------------------------------------------------
# CellTypist is the workhorse PBMC/immune annotator.
pip install "celltypist>=1.6.3"

# scVI/scANVI for label transfer (CPU OK, GPU strongly recommended for large data)
# pip install "scvi-tools>=1.2"

# --- 4. Robust PCA -----------------------------------------------------------
# We use our own PCP implementation in src/utils_rpca.py (it's small and
# avoids a heavy dependency). If you want a reference impl to benchmark
# against, uncomment the next line:
# pip install "robustpca>=0.1"

# --- 5. Foundation model: Geneformer ----------------------------------------
# Geneformer is our default foundation model: HuggingFace-based, no flash-attn
# headache, runs on CPU (slow) or GPU (fast). It produces per-cell embeddings
# that we then clean with RPCA in step 03b.
#
# Torch first. If you have no GPU, drop the --extra-index-url line and the
# default CPU wheel will be used.
pip install "torch>=2.2" "torchvision" --extra-index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.44" "accelerate>=0.33" "datasets>=2.20"

# Geneformer itself. It ships as a git repo on HuggingFace Hub with the
# pretrained weights + the gene-token dictionary. Install the package:
pip install "geneformer @ git+https://huggingface.co/ctheodoris/Geneformer.git"
# pyarrow/loom support Geneformer's tokenizer:
pip install "loompy>=3.0" "pyarrow>=16.0"

# # Pull the pretrained model + token dictionaries into the HF cache up front
# # so the first pipeline run isn't doing a surprise multi-GB download.
# python - <<'PY'
# try:
#     from huggingface_hub import snapshot_download
#     # 6-layer 30M model is the lightweight default; bump to gf-12L-95M for accuracy
#     snapshot_download(repo_id="ctheodoris/Geneformer",
#                       allow_patterns=["*.pkl", "*.json", "gf-6L-30M-i2048/*"])
#     print("Geneformer assets cached.")
# except Exception as e:
#     print(f"[warn] could not pre-cache Geneformer ({e}); "
#           "it will download on first use.")
# PY

# --- scGPT (third annotator for the ensemble) -------------------------------
# scGPT is a transformer foundation model trained on 33M cells. It gives a
# THIRD independent opinion (different architecture + training data than
# Geneformer). It needs flash-attn, which is CUDA-only and can be slow to
# build. If you are CPU-only, skip this block and the pipeline will run with
# just CellTypist + Geneformer.
pip install "scgpt>=0.2.4"
# flash-attn build can fail on older CUDA; --no-build-isolation often helps:
pip install "flash-attn>=2.5" --no-build-isolation || \
    echo "[warn] flash-attn build failed; scGPT may run slower or error. Skip 03c if so."
# scGPT loads weights from a local dir or HF. Pre-cache the blood/PBMC model:
python - <<'PY'
try:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="scGPT/scGPT_human",
                      allow_patterns=["*.json", "*.pt", "vocab.json"])
    print("scGPT human model cached.")
except Exception as e:
    print(f"[warn] could not pre-cache scGPT ({e}); fetch manually if needed.")
PY

# --- 6. Notebook + repro -----------------------------------------------------
pip install "jupyterlab>=4.2" "papermill>=2.6" "snakemake>=8.0" "pyyaml>=6.0"

# --- 7. Sanity check ---------------------------------------------------------
python - <<'PY'
import scanpy as sc, anndata as ad, celltypist, harmonypy, numpy as np, scipy
print("scanpy", sc.__version__)
print("anndata", ad.__version__)
print("celltypist", celltypist.__version__)
print("numpy", np.__version__, "scipy", scipy.__version__)
try:
    import torch, transformers
    print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
    print("transformers", transformers.__version__)
    import geneformer
    print("geneformer importable: OK")
except Exception as e:
    print(f"[warn] foundation-model stack check failed: {e}")
print("ENV OK")
PY

echo "Done. Activate with: conda activate ${ENV_NAME}"
