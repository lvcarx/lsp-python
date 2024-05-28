import numpy as np

from sklearn.neighbors import NearestNeighbors
from lsp.mds import MDS



class LSP:
    """
    TODO
    """
    def __init__(self, n_components: int = 2, n_neighbors: int = 15, mds: MDS = MDS()):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.mds = mds

    def fit(self, hi_dim: np.ndarray):
        return self.mds.fit_transform(hi_dim)

    def transform(self, lo_dim_init: np.ndarray, lo_dim_indices: np.ndarray, hi_dim: np.ndarray):
        return self.lsp_(lo_dim_init, lo_dim_indices, hi_dim)

    def fit_transform(self):
        pass

    def lsp_(self, lo_dim_init: np.ndarray, lo_dim_indices: np.ndarray, hi_dim: np.ndarray):
        nc = lo_dim_init.shape[0]
        n = hi_dim.shape[0]
        n_neighbors = self.n_neighbors

        # Sklearn implementation of KNN search
        NN = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto', metric='euclidean', n_jobs=-1)
        neighbors = NN.fit(hi_dim).kneighbors(hi_dim, return_distance=False)[:, 1:]

        A = np.zeros(shape=(n + nc, n), dtype=float)
        L = np.zeros(shape=(n, n), dtype=float)
        C = np.zeros(shape=(nc, n), dtype=float)

        for i in range(len(hi_dim)):
            L[i, i] = 1.0
            L[i, neighbors[i]] = -(1.0 / n_neighbors)

        for l_idx, i in enumerate(lo_dim_indices):
            C[l_idx, i] = 1.0

        A[:n, :] = L
        A[n:, :] = C

        b = np.zeros(shape=(n + nc, self.n_components), dtype=float)
        for l_idx, i in enumerate(range(n, n + nc)):
            b[i] = lo_dim_init[l_idx]

        return np.linalg.lstsq(A, b, rcond=None)[0]