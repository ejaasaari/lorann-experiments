import sys

from pylibraft.common import DeviceResources
from pylibraft.neighbors import ivf_flat, ivf_pq, cagra, refine
import numpy as np
import cupy as cp

from ..base.module import BaseANN


class RAFTIVF(BaseANN):
    def __init__(self, metric, n_list):
        self._metric = metric
        self._n_list = n_list
        self._handle = DeviceResources()
        self._index = None

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self._metric == "angular":
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        n, d = X.shape
        index_params = ivf_flat.IndexParams(n_lists=self._n_list)
        self._index = ivf_flat.build(index_params, cp.array(X), handle=self._handle)

    def set_query_arguments(self, n_probe):
        self._n_probe = n_probe
        self._search_params = ivf_flat.SearchParams(n_probes=n_probe)

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        D, L = ivf_flat.search(self._search_params,
                               self._index,
                               cp.array(X),
                               n,
                               handle=self._handle)
        self._handle.sync()
        self.res = (cp.asarray(D).get(), cp.asarray(L).get())

    def get_batch_results(self):
        D, L = self.res
        return [list(x[x != -1]) for x in L]

    def __str__(self):
        return "RAFTIVF(n_list={}, n_probes={})".format(self._n_list, self._n_probe)


class RAFTIVFPQ(BaseANN):
    def __init__(self, metric, n_list, pq_dim, pq_bits):
        self._metric = metric
        self._n_list = n_list
        self._pq_dim = pq_dim
        self._pq_bits = pq_bits
        self._handle = DeviceResources()
        self._index = None

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self._metric == "angular":
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        n, d = X.shape
        index_params = ivf_pq.IndexParams(n_lists=self._n_list, pq_dim=self._pq_dim, pq_bits=self._pq_bits)
        self._dataset = cp.array(X)
        self._index = ivf_pq.build(index_params, self._dataset, handle=self._handle)

    def set_query_arguments(self, n_probe, lut_dtype, refine_ratio):
        self._n_probe = n_probe
        self._lut_dtype = lut_dtype
        self._refine_ratio = refine_ratio
        ldtype = np.float16 if lut_dtype == 16 else np.float32

        self._search_params = ivf_pq.SearchParams(n_probes=n_probe, lut_dtype=ldtype)

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self._refine_ratio > 1:
            queries = cp.array(X)
            _, C = ivf_pq.search(self._search_params,
                                 self._index,
                                 queries,
                                 self._refine_ratio * n,
                                 handle=self._handle)
            D, L = refine(self._dataset, queries, C, n, handle=self._handle)
        else:
            D, L = ivf_pq.search(self._search_params,
                                self._index,
                                cp.array(X),
                                n,
                                handle=self._handle)
        self._handle.sync()
        self.res = (cp.asarray(D).get(), cp.asarray(L).get())

    def get_batch_results(self):
        D, L = self.res
        return [list(x[(x >= 0) & (x <= 2147483647)]) for x in L]

    def __str__(self):
        return "RAFTIVFPQ(n_list={}, pq_dim={}, pq_bits={}, n_probes={}, ldtype={}, ratio={})".format(self._n_list, self._pq_dim, self._pq_bits, self._n_probe, self._lut_dtype, self._refine_ratio)


class RAFTCAGRA(BaseANN):
    def __init__(self, metric, graph_degree, intermediate_graph_degree):
        self._metric = metric
        self._graph_degree = graph_degree
        self._intermediate_graph_degree = intermediate_graph_degree
        self._handle = DeviceResources()
        self._index = None

        if graph_degree > intermediate_graph_degree:
            raise Exception()

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self._metric == "angular":
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        n, d = X.shape
        index_params = cagra.IndexParams(graph_degree=self._graph_degree,
                                         intermediate_graph_degree=self._intermediate_graph_degree,
                                         build_algo="nn_descent")
        self._index = cagra.build(index_params, cp.array(X), handle=self._handle)

    def set_query_arguments(self, itopk, search_width):
        self._itopk = itopk
        self._search_width = search_width
        self._search_params = cagra.SearchParams(itopk_size=itopk, search_width=search_width)

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if n > self._itopk:
            self.res = (np.full((len(X), n), -1), np.full((len(X), n), -1))
            return

        D, L = cagra.search(self._search_params,
                            self._index,
                            cp.array(X),
                            n,
                            handle=self._handle)
        self._handle.sync()
        self.res = (cp.asarray(D).get(), cp.asarray(L).get())

    def get_batch_results(self):
        D, L = self.res
        return [list(x[x != -1]) for x in L]

    def __str__(self):
        return "RAFTCAGRA(graph_deg={}, i_graph_deg={}, itopk={}, search_w={})".format(self._graph_degree, self._intermediate_graph_degree, self._itopk, self._search_width)