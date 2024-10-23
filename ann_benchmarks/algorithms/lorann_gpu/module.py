import numpy as np
import functools

from ..base.module import BaseANN


class IVFGPU(BaseANN):

    def __init__(self, metric):
        self._metric = metric
        self._index = None
        self._index_type = None
        self._name = None

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self._metric == "angular":
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        self._index = self._index_type(X, self._n_clusters, self._metric == "euclidean")

    def set_query_arguments(self, leaves):
        self._clusters_to_search = leaves

    def __str__(self):
        str_template = "%s(nc=%d, cs=%d)"
        return str_template % (
            self._name,
            self._n_clusters,
            self._clusters_to_search,
        )

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.res = self._index.search(X, n, self._clusters_to_search)

    def get_batch_results(self):
        return [list(x[x != -1]) for x in self.res]


class LorannGPU(BaseANN):

    def __init__(self, metric):
        self._metric = metric
        self._index = None
        self._index_type = None
        self._name = None

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self._metric == "angular":
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        self._index = self._index_type(X, self._n_clusters, self._global_dim, self._rank,
                                       self._train_size, self._metric == "euclidean")

    def set_query_arguments(self, clusters_to_search, points_to_rerank):
        self._clusters_to_search, self._points_to_rerank = clusters_to_search, points_to_rerank

    def __str__(self):
        str_template = "%s(gd=%d, r=%d, nc=%d, cs=%d, pr=%d)"
        return str_template % (
            self._name,
            self._global_dim,
            self._rank,
            self._n_clusters,
            self._clusters_to_search,
            self._points_to_rerank
        )

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.res = self._index.search(X, n, self._clusters_to_search, self._points_to_rerank)

    def get_batch_results(self):
        return [list(x[x != -1]) for x in self.res]


class IVFJax(IVFGPU):

    def __init__(self, metric, n_clusters):
        super().__init__(metric)

        import lorann_gpu.jax
        import jax.numpy as jnp

        self._n_clusters = n_clusters
        self._index_type = functools.partial(lorann_gpu.jax.IVF.build, data_dtype=jnp.float16, dtype=jnp.float16)
        self._name = "IVFJax"

    def set_query_arguments(self, leaves):
        import jax
        jax.clear_caches()

        self._clusters_to_search = leaves


class IVFMLX(IVFGPU):

    def __init__(self, metric, n_clusters):
        super().__init__(metric)

        import lorann_gpu.mlx
        import mlx.core as mx

        mx.set_default_device(mx.gpu)

        self._n_clusters = n_clusters
        self._index_type = functools.partial(lorann_gpu.mlx.IVF, dtype=mx.float32)
        self._name = "IVFMLX"

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.res = np.zeros((len(X), n), dtype=np.int32)
        for i in range(0, len(X), 100):
            self.res[i:i + 100] = self._index.search(X[i:i + 100], n, self._clusters_to_search)


class IVFMLXCPU(IVFGPU):

    def __init__(self, metric, n_clusters):
        super().__init__(metric)

        import lorann_gpu.mlx
        import mlx.core as mx

        mx.set_default_device(mx.cpu)

        self._n_clusters = n_clusters
        self._index_type = functools.partial(lorann_gpu.mlx.IVF, dtype=mx.float32)
        self._name = "IVFMLXCPU"

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.res = np.zeros((len(X), n), dtype=np.int32)
        for i in range(0, len(X), 100):
            self.res[i:i + 100] = self._index.search(X[i:i + 100], n, self._clusters_to_search)


class LorannJax(LorannGPU):

    def __init__(self, metric, global_dim, rank, train_size, n_clusters, precision):
        super().__init__(metric)

        import lorann_gpu.jax
        import jax.numpy as jnp

        dtype = jnp.float16 if precision == 16 else jnp.float32

        self._global_dim = global_dim
        self._rank = rank
        self._train_size = train_size
        self._n_clusters = n_clusters
        self._index_type = functools.partial(lorann_gpu.jax.Lorann.build, data_dtype=jnp.float16, dtype=dtype)
        self._name = "LorannJax"

    def set_query_arguments(self, clusters_to_search, points_to_rerank):
        import jax
        jax.clear_caches()

        self._clusters_to_search, self._points_to_rerank = clusters_to_search, points_to_rerank


class LorannTorch(LorannGPU):

    def __init__(self, metric, global_dim, rank, train_size, n_clusters):
        super().__init__(metric)

        import lorann_gpu.torch
        import torch

        self._global_dim = global_dim
        self._rank = rank
        self._train_size = train_size
        self._n_clusters = n_clusters
        self._index_type = functools.partial(lorann_gpu.torch.Lorann, dtype=torch.float16)
        self._name = "LorannTorch"

    def set_query_arguments(self, clusters_to_search, points_to_rerank):
        import torch
        torch.compiler.reset()

        self._clusters_to_search, self._points_to_rerank = clusters_to_search, points_to_rerank


class LorannCupy(LorannGPU):

    def __init__(self, metric, global_dim, rank, train_size, n_clusters):
        super().__init__(metric)

        import lorann_gpu.cupy

        self._global_dim = global_dim
        self._rank = rank
        self._train_size = train_size
        self._n_clusters = n_clusters
        self._index_type = lorann_gpu.cupy.Lorann
        self._name = "LorannCupy"


class LorannMLX(LorannGPU):

    def __init__(self, metric, global_dim, rank, train_size, n_clusters):
        super().__init__(metric)

        import lorann_gpu.mlx
        import mlx.core as mx

        mx.set_default_device(mx.gpu)

        self._global_dim = global_dim
        self._rank = rank
        self._train_size = train_size
        self._n_clusters = n_clusters
        self._index_type = functools.partial(lorann_gpu.mlx.Lorann, dtype=mx.float32, approximate=True)
        self._name = "LorannMLX"

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.res = np.zeros((len(X), n), dtype=np.int32)
        for i in range(0, len(X), 100):
            self.res[i:i + 100] = self._index.search(X[i:i + 100], n, self._clusters_to_search, self._points_to_rerank)


class LorannMLXCPU(LorannGPU):

    def __init__(self, metric, global_dim, rank, train_size, n_clusters):
        super().__init__(metric)

        import lorann_gpu.mlx
        import mlx.core as mx

        mx.set_default_device(mx.cpu)

        self._global_dim = global_dim
        self._rank = rank
        self._train_size = train_size
        self._n_clusters = n_clusters
        self._index_type = functools.partial(lorann_gpu.mlx.Lorann, dtype=mx.float32, approximate=True)
        self._name = "LorannMLXCPU"

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.res = np.zeros((len(X), n), dtype=np.int32)
        for i in range(0, len(X), 100):
            self.res[i:i + 100] = self._index.search(X[i:i + 100], n, self._clusters_to_search, self._points_to_rerank)
