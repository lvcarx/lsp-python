import numpy as np

from sklearn.neighbors import NearestNeighbors
from lsp.mds import MDS



class LSP:
    """
    TODO
    """
    def __init__(self, n_components: int = 2, n_neighbors: int = 15, initial_layout_algorithm: MDS = MDS()):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.initial_layout_algorithm = initial_layout_algorithm

    def fit(self, hi_dim: np.ndarray):
        return self.initial_layout_algorithm.fit_transform(hi_dim)

    def transform(self, lo_dim_init: np.ndarray, lo_dim_indices: np.ndarray, hi_dim: np.ndarray):
        return self.lsp_(lo_dim_init, lo_dim_indices, hi_dim)

    def fit_transform(self):
        pass

    def lsp_(self, lo_dim_init: np.ndarray, lo_dim_indices: np.ndarray, hi_dim: np.ndarray):
        n_control = lo_dim_init.shape[0]
        n_points = hi_dim.shape[0]
        n_neighbors = self.n_neighbors

        nn_obj = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto', metric='euclidean', n_jobs=-1)
        neighbors = nn_obj.fit(hi_dim).kneighbors(hi_dim, return_distance=False)[:, 1:]

        A = np.zeros(shape=(n_points + n_control, n_points), dtype=np.float64)
        neighborhood_matrix = np.zeros(shape=(n_points, n_points), dtype=np.float64)
        control_matrix = np.zeros(shape=(n_control, n_points), dtype=np.float64)

        for i in range(len(hi_dim)):
            neighborhood_matrix[i, i] = 1.0
            neighborhood_matrix[i, neighbors[i]] = -(1.0 / n_neighbors)

        for l_idx, i in enumerate(lo_dim_indices):
            control_matrix[l_idx, i] = 1.0

        A[:n_points, :] = neighborhood_matrix
        A[n_points:, :] = control_matrix

        b = np.zeros(shape=(n_points + n_control, self.n_components), dtype=np.float64)
        for l_idx, i in enumerate(range(n_points, n_points + n_control)):
            b[i] = lo_dim_init[l_idx]

        return np.linalg.lstsq(A, b, rcond=None)[0]