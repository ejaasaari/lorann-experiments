import sys

import faiss
import numpy as np

from ..base.module import BaseANN

# Implementation based on
# https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_sift1m.py  # noqa


class FaissGPU(BaseANN):

    def __init__(self, metric):
        self._metric = metric
        self._res = faiss.StandardGpuResources()
        self._index = None

    def query(self, v, n):
        return [label for label, _ in self.query_with_distances(v, n)]

    def query_with_distances(self, v, n):
        v = v.astype(np.float32).reshape(1, -1)
        distances, labels = self._index.search(v, n)
        r = []
        for l, d in zip(labels[0], distances[0]):
            if l != -1:
                r.append((l, d))
        return r

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.res = self._index.search(X, n)

    def get_batch_results(self):
        D, L = self.res
        return [list(x[x != -1]) for x in L]


class FaissGPUIVF(FaissGPU):
    def __init__(self, metric, n_list, float16):
        super().__init__(metric)
        self._n_list = n_list
        self._float16 = float16

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self._metric == "angular":
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        n, d = X.shape
        index = faiss.index_factory(d, f"IVF{self._n_list},Flat")

        co = faiss.GpuClonerOptions()
        co.useFloat16 = bool(self._float16)

        self._index = faiss.index_cpu_to_gpu(self._res, 0, index, co)
        self._index.train(X)
        self._index.add(X)

    def set_query_arguments(self, n_probe):
        self._n_probe = n_probe
        self._index.nprobe = n_probe

    def __str__(self):
        return "FaissGPUIVF(n_list={}, n_probes={}, float16={})".format(self._n_list, self._n_probe, self._float16)


class FaissGPUIVFPQ(FaissGPU):
    def __init__(self, metric, n_list, code_size):
        super().__init__(metric)
        self._n_list = n_list
        self._code_size = code_size

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self._metric == "angular":
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        n, d = X.shape
        index = faiss.index_factory(d, f"IVF{self._n_list},PQ{self._code_size}")

        co = faiss.GpuClonerOptions()
        # GpuIndexIVFPQ with 56 bytes per code or more requires use of the
        # float16 IVFPQ mode due to shared memory limitations
        co.useFloat16 = True

        self._index = faiss.index_cpu_to_gpu(self._res, 0, index, co)
        self._index.train(X)
        self._index.add(X)

    def set_query_arguments(self, n_probe):
        self._n_probe = n_probe
        self._index.nprobe = n_probe

    def __str__(self):
        return "FaissGPUIVFPQ(n_list={}, n_probes={}, code_size={})".format(self._n_list, self._n_probe, self._code_size)
