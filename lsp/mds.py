import numpy as np
from sklearn import manifold

class MDS:
    """
    TODO
    """
    def __init__(self, n_components: int = 2):
        self.mds = manifold.MDS(n_components=n_components)

    def fit_transform(self, hi_dim: np.ndarray):
        return self.mds.fit_transform(hi_dim)
