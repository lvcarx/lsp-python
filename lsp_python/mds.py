import numpy as np
from sklearn import manifold

class MDS:
    """
    This class implements the Multidimensional Scaling (MDS) dimensionality reduction method.

    Parameters
    ----------

    n_components: int (default 2)
        The dimension to project onto.

    """
    def __init__(self, n_components: int = 2):
        self.mds = manifold.MDS(n_components=n_components, random_state=1)

    def fit_transform(self, hi_dim: np.ndarray):
        return self.mds.fit_transform(hi_dim)
