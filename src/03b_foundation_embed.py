"""
03b_foundation_embed.py — Geneformer embeddings (Mac/MPS-friendly) + RPCA.

IMPORTANT: this version does NOT use Geneformer's EmbExtractor, which hardcodes
device="cuda" and therefore crashes on Macs (no CUDA). Instead we:
  1. Use Geneformer's TranscriptomeTokenizer (works fine, CPU) to rank-encode cells.
  2. Load the model weights with transformers (AutoModel) and run the forward
     pass ourselves on MPS (Apple Silicon GPU) or CPU.
  3. Mean-pool the last hidden layer over real tokens -> per-cell embedding.
Then local RPCA on the embedding space, exactly as before.

--model_size now means the PATH to a local Geneformer model directory, e.g.
   Geneformer/Geneformer-V1-10M   (smallest, best for Mac)
   Geneformer/Geneformer-V2-104M
"""

import argparse
import os
import sys
import tempfile
import numpy as np
import scanpy as sc
from scipy import sparse

# Let MPS fall back to CPU for any op it doesn't support, instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, os.path.dirname(__file__))
from utils_rpca import local_rpca


def _pick_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _geneformer_embeddings(adata, model_dir, batch_size=16, emb_layer=-1):
    """Tokenize with Geneformer, then embed via a manual forward pass."""
    import torch
    from transformers import AutoModel
    from datasets import load_from_disk
    from geneformer import TranscriptomeTokenizer

    ad_gf = adata.copy()
    # Geneformer wants raw counts in .X
    if "counts" in ad_gf.layers:
        ad_gf.X = ad_gf.layers["counts"].copy()
    if not sparse.issparse(ad_gf.X):
        ad_gf.X = sparse.csr_matrix(ad_gf.X)

    ad_gf.obs["n_counts"] = np.asarray(ad_gf.X.sum(axis=1)).ravel()
    ad_gf.obs["__cell_order__"] = np.arange(ad_gf.n_obs)

    if "ensembl_id" not in ad_gf.var.columns:
        for cand in ("gene_ids", "gene_id", "ensembl_ids", "feature_id"):
            if cand in ad_gf.var.columns:
                ad_gf.var["ensembl_id"] = ad_gf.var[cand].astype(str)
                break
        else:
            ad_gf.var["ensembl_id"] = ad_gf.var_names.astype(str)

    with tempfile.TemporaryDirectory() as tmp:
        h5_dir = os.path.join(tmp, "h5ad"); os.makedirs(h5_dir)
        tok_dir = os.path.join(tmp, "tok"); os.makedirs(tok_dir)
        ad_gf.write_h5ad(os.path.join(h5_dir, "data.h5ad"))

        # --- tokenize (CPU, fine) ---
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict={"__cell_order__": "__cell_order__"},
            nproc=os.cpu_count() or 4,
        )
        tk.tokenize_data(h5_dir, tok_dir, "gf", file_format="h5ad")
        ds = load_from_disk(os.path.join(tok_dir, "gf.dataset"))

        # cell ordering to realign embeddings to original adata order
        if "__cell_order__" in ds.column_names:
            order = np.asarray(ds["__cell_order__"], dtype=int)
        else:
            order = np.arange(len(ds))

        # --- load model + manual forward pass on MPS/CPU ---
        device = _pick_device()
        print(f"[03b_gf] running Geneformer forward pass on device: {device}")
        try:
            model = AutoModel.from_pretrained(model_dir, output_hidden_states=True)
        except Exception:
            from transformers import BertModel
            model = BertModel.from_pretrained(model_dir, output_hidden_states=True)
        model = model.to(device).eval()

        # Geneformer V1 max context = 2048; V2 = 4096. Read from model config
        # and truncate any longer sequences to avoid position-embedding overflow.
        max_ctx = getattr(model.config, "max_position_embeddings", 2048)
        print(f"[03b_gf] model max context length: {max_ctx}")

        input_ids_all = ds["input_ids"]   # list[list[int]]
        n = len(input_ids_all)
        embs = []
        from math import ceil
        n_batches = ceil(n / batch_size)
        with torch.no_grad():
            for bi, start in enumerate(range(0, n, batch_size)):
                batch = input_ids_all[start:start + batch_size]
                batch = [seq[:max_ctx] for seq in batch]   # truncate to max context
                maxlen = max(len(x) for x in batch)
                ids = torch.zeros(len(batch), maxlen, dtype=torch.long)
                mask = torch.zeros(len(batch), maxlen, dtype=torch.long)
                for i, seq in enumerate(batch):
                    L = len(seq)
                    ids[i, :L] = torch.as_tensor(seq, dtype=torch.long)
                    mask[i, :L] = 1
                ids = ids.to(device); mask = mask.to(device)
                out = model(input_ids=ids, attention_mask=mask)
                h = out.hidden_states[emb_layer]          # (B, L, H)
                m = mask.unsqueeze(-1).float()
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
                embs.append(pooled.detach().cpu().numpy().astype(np.float64))
                if bi % 25 == 0 or bi == n_batches - 1:
                    print(f"[03b_gf]   batch {bi+1}/{n_batches}")

    emb = np.concatenate(embs, axis=0)
    aligned = np.zeros_like(emb)
    aligned[order] = emb
    return aligned


def run(input_path, output_path,
        model_size="Geneformer/Geneformer-V1-10M",
        batch_size=16, lam_scale=1.0, min_cluster_size=30):
    print(f"[03b_gf] loading {input_path}")
    adata = sc.read_h5ad(input_path)
    if "leiden_coarse" not in adata.obs:
        raise RuntimeError("Run 02_coarse_cluster.py first (need leiden_coarse).")

    if not os.path.isdir(model_size):
        raise FileNotFoundError(
            f"Geneformer model dir not found: '{model_size}'. "
            f"Pass --model_size with the path to a local model folder, e.g. "
            f"Geneformer/Geneformer-V1-10M")

    print(f"[03b_gf] extracting Geneformer embeddings from {model_size} ...")
    emb = _geneformer_embeddings(adata, model_size, batch_size=batch_size)
    adata.obsm["X_geneformer"] = emb
    print(f"[03b_gf] embedding shape: {emb.shape}")

    print("[03b_gf] local RPCA on Geneformer embedding space")
    clusters = adata.obs["leiden_coarse"].astype(str).values
    L_emb, S_emb, z_emb = local_rpca(
        emb, clusters, min_cluster_size=min_cluster_size,
        lam_scale=lam_scale, verbose=True)
    adata.obsm["X_geneformer_L"] = L_emb
    adata.obs["gf_outlier_z"] = z_emb

    adata.write_h5ad(output_path, compression="gzip")
    print(f"[03b_gf] wrote {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_size", default="Geneformer/Geneformer-V1-10M",
                    help="Path to a local Geneformer model directory")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lam_scale", type=float, default=1.0)
    ap.add_argument("--min_cluster_size", type=int, default=30)
    args = ap.parse_args()
    run(args.input, args.output,
        model_size=args.model_size, batch_size=args.batch_size,
        lam_scale=args.lam_scale, min_cluster_size=args.min_cluster_size)