import scanpy as sc, numpy as np
from scipy import sparse
p='data/raw/recipe_data/vaccination_study_10/vaccination_study_10_processed.h5ad'
a=sc.read_h5ad(p)
raw=a.raw.to_adata()                       # X=.raw.X (logCP10k), 14969 genes
X=raw.X
if sparse.issparse(X):
    np.clip(X.data, 0, None, out=X.data)
else:
    np.clip(X, 0, None, out=X)
raw.X=X
raw.obs=a.obs.copy()                       # keep barcodes / sample_id / tissue
out='data/raw/recipe_data/vaccination_study_10/vaccination_study_10_LOGNORM.h5ad'
raw.write_h5ad(out)
print('wrote', out, raw.shape)
