"""
02_coarse_cluster.py — neighbors graph + low-resolution Leiden.

We want LINEAGE-LEVEL clusters (~5-15), not subtypes. These are the
"local" neighborhoods inside which RPCA's low-rank assumption holds.
Within a coarse cluster you should see one major lineage (T, B, NK,
Mono, etc.) with within-lineage state variation — that's exactly the
regime where X = L + S is well-posed.
"""

import argparse
import scanpy as sc


def run(input_path: str, output_path: str, resolution: float = 0.4,
        use_rep: str = "X_pca_harmony") -> None:
    print(f"[02_cluster] loading {input_path}")
    adata = sc.read_h5ad(input_path)

    sc.pp.neighbors(adata, n_neighbors=15, use_rep=use_rep, random_state=0)
    sc.tl.umap(adata, random_state=0)
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden_coarse",
                 random_state=0, flavor="igraph", n_iterations=2, directed=False)

    n_clusters = adata.obs["leiden_coarse"].nunique()
    print(f"[02_cluster] coarse Leiden (res={resolution}) -> {n_clusters} clusters")

    adata.write_h5ad(output_path, compression="gzip")
    print(f"[02_cluster] wrote {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resolution", type=float, default=0.4)
    ap.add_argument("--use_rep", default="X_pca_harmony")
    args = ap.parse_args()
    run(args.input, args.output, args.resolution, args.use_rep)