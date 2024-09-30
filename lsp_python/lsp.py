import numpy as np

from sklearn.neighbors import NearestNeighbors
from lsp_python.mds import MDS



class LSP:
    """
    This class implements the Least Squares Projection (LSP) dimensionality reduction method.


    Parameters
    ----------
    n_components: int (default 2)
        The dimension to project onto

    n_neighbors: int (default 15)
        The number of neighbors to consider for the neighborhood matrix.
        Larger neighborhoods will lead to projections which are more alike to the initial_layout_algorithm.

    initial_layout_algorithm: object (default MDS())
        The algorithm to use for the initial layout of the low-dimensional space.
        This algorithm should have a fit_transform method which takes a high-dimensional array as input and returns a low-dimensional array.

    """
    def __init__(self, n_components: int = 2, n_neighbors: int = 15, initial_layout_algorithm: MDS = MDS()):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.initial_layout_algorithm = initial_layout_algorithm

    def fit(self, hi_dim: np.ndarray):
        """
        Projects the initial high-dimensional data points onto the low-dimensional space using the initial_layout_algorithm.

        :param hi_dim: initial high-dimensional data points
        :return: the low-dimensional representation
        """
        return self.initial_layout_algorithm.fit_transform(hi_dim)

    def transform(self, control_points: np.ndarray, control_point_indices: np.ndarray, hi_dim: np.ndarray):
        """
        Projects the high-dimensional data points onto the low-dimensional space using the Least Squares Projection method and the control points.

        :param control_points: initial projection created with the initial_layout_algorithm
        :param control_point_indices: indices of the control points inside the high-dimensional data point set "hi_dim"
        :param hi_dim: all high-dimensional data points (including the control points)
        :return: the low-dimensional representation
        """
        return self.__least_square_projection(control_points, control_point_indices, hi_dim)

    def fit_transform(self, control_point_indices: np.ndarray, hi_dim: np.ndarray):
        """
        Projects the high-dimensional data points onto the low-dimensional space using the Least Squares Projection method and the control points.

        :param control_point_indices: defines which points are used as control points
        :param hi_dim: all high-dimensional data points (including the control points)
        :return: the low dimensional representation
        """
        lo_dim = self.fit(hi_dim[control_point_indices])
        return self.transform(lo_dim, control_point_indices, hi_dim)

    def __least_square_projection(self, control_points: np.ndarray, control_point_indices: np.ndarray, hi_dim: np.ndarray):
        n_control = control_points.shape[0]
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

        for l_idx, i in enumerate(control_point_indices):
            control_matrix[l_idx, i] = 1.0

        A[:n_points, :] = neighborhood_matrix
        A[n_points:, :] = control_matrix

        b = np.zeros(shape=(n_points + n_control, self.n_components), dtype=np.float64)
        for l_idx, i in enumerate(range(n_points, n_points + n_control)):
            b[i] = control_points[l_idx]

        return np.linalg.lstsq(A, b, rcond=None)[0]