LoRANN Experiments
==================

Experiments for the paper

> Jääsaari, E., Hyvönen, V., & Roos, T. (2024). LoRANN: Low-Rank Matrix Factorization for Approximate Nearest Neighbor Search. Advances in Neural Information Processing Systems, 37.

The code implementing LoRANN is in a [separate repository](https://github.com/ejaasaari/lorann).

**This project is a fork of the [ANN-benchmarks](https://github.com/erikbern/ann-benchmarks/) project with new data sets and support for GPU experiments.**

-----------

**Requirements**:
- Python 3.10 or newer
- Docker
- For GPU experiments, an NVIDIA GPU is required

**Installation**:

`python3 -m pip install -r requirements.txt`

**Usage**:

To build an algorithm, run e.g. `python3 install.py --algorithm lorann`.

To build all algorithms, run

```sh
for algo in faiss faiss_gpu glass hnswlib lorann lorann_gpu mrpt pynndescent qsg_ngt raft scann; do
  python3 install.py --algorithm $algo
done
```

To run an algorithm for e.g. the data set _fashion-mnist-784-euclidean_, run

`python3 run.py --dataset fashion-mnist-784-euclidean --algorithm lorann --count 100 --parallelism 6`

To plot results for the data set, run

`python3 plot.py --dataset fashion-mnist-784-euclidean --count 100 --y-scale log`

For a list of all the data sets, refer to the end of the file [ann_benchmarks/datasets.py](ann_benchmarks/datasets.py).

Our main experiments are performed on AWS `r6i.4xlarge` instances using Intel Xeon 8375C (Ice Lake) processors with hyperthreading disabled. To run our GPU experiments, we use an AWS `g5.2xlarge` instance with an NVIDIA A10G GPU (24 GB VRAM) and a `mac2-m2pro.metal` instance with an Apple M2 Pro SoC.
