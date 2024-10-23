import lorannlib
import numpy as np

from ..base.module import BaseANN


class Lorann(BaseANN):

    def __init__(self, metric, quantization_bits, n_clusters, global_dim, rank, train_size):
        self._metric = metric
        self.quantization_bits = quantization_bits
        self.n_clusters = n_clusters
        self.global_dim = global_dim
        self.rank = rank
        self.train_size = train_size

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self._metric == "angular":
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        n_samples, dim = X.shape
        self._index = lorannlib.LorannIndex(X, n_samples, dim, self.quantization_bits, self.n_clusters,
                                            self.global_dim, self.rank, self.train_size,
                                            self._metric == "euclidean", False)
        self._index.build(n_samples > 500000)

    def set_query_arguments(self, clusters_to_search, points_to_rerank):
        self.clusters_to_search, self.points_to_rerank = clusters_to_search, points_to_rerank

    def query(self, q, n):
        return self._index.search(q, n, self.clusters_to_search, self.points_to_rerank, False)

    def __str__(self):
        str_template = "Lorann(q=%d, nc=%d, gd=%d, r=%d, ts=%d, cs=%d, pr=%d)"
        return str_template % (
                self.quantization_bits,
                self.n_clusters,
                self.global_dim,
                self.rank,
                self.train_size,
                self.clusters_to_search,
                self.points_to_rerank
        )
