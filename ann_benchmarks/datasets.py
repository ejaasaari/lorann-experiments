import os
import random
import tarfile
from urllib.request import urlopen, urlretrieve

import h5py
import numpy
from typing import Any, Callable, Dict, Tuple

def download(source_url: str, destination_path: str) -> None:
    """
    Downloads a file from the provided source URL to the specified destination path
    only if the file doesn't already exist at the destination.
    
    Args:
        source_url (str): The URL of the file to download.
        destination_path (str): The local path where the file should be saved.
    """
    if not os.path.exists(destination_path):
        print(f"downloading {source_url} -> {destination_path}...")
        urlretrieve(source_url, destination_path)


def get_dataset_fn(dataset_name: str) -> str:
    """
    Returns the full file path for a given dataset name in the data directory.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        str: The full file path of the dataset.
    """
    if not os.path.exists("data"):
        os.mkdir("data")
    return os.path.join("data", f"{dataset_name}.hdf5")


def get_dataset(dataset_name: str) -> Tuple[h5py.File, int]:
    """
    Fetches a dataset by downloading it from a known URL or creating it locally
    if it's not already present. The dataset file is then opened for reading, 
    and the file handle and the dimension of the dataset are returned.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        Tuple[h5py.File, int]: A tuple containing the opened HDF5 file object and
            the dimension of the dataset.
    """
    hdf5_filename = get_dataset_fn(dataset_name)
    if not os.path.exists(hdf5_filename):
        if dataset_name in DATASETS:
            print("Creating dataset locally")
            DATASETS[dataset_name](hdf5_filename)

    hdf5_file = h5py.File(hdf5_filename, "r")

    # here for backward compatibility, to ensure old datasets can still be used with newer versions
    # cast to integer because the json parser (later on) cannot interpret numpy integers
    dimension = int(hdf5_file.attrs["dimension"]) if "dimension" in hdf5_file.attrs else len(hdf5_file["train"][0])
    return hdf5_file, dimension


def write_output(train: numpy.ndarray, test: numpy.ndarray, fn: str, distance: str, point_type: str = "float", count: int = 100) -> None:
    """
    Writes the provided training and testing data to an HDF5 file. It also computes 
    and stores the nearest neighbors and their distances for the test set using a 
    brute-force approach.
    
    Args:
        train (numpy.ndarray): The training data.
        test (numpy.ndarray): The testing data.
        filename (str): The name of the HDF5 file to which data should be written.
        distance_metric (str): The distance metric to use for computing nearest neighbors.
        point_type (str, optional): The type of the data points. Defaults to "float".
        neighbors_count (int, optional): The number of nearest neighbors to compute for 
            each point in the test set. Defaults to 100.
    """
    from ann_benchmarks.algorithms.bruteforce.module import BruteForceBLAS

    with h5py.File(fn, "w") as f:
        f.attrs["type"] = "dense"
        f.attrs["distance"] = distance
        f.attrs["dimension"] = len(train[0])
        f.attrs["point_type"] = point_type
        print(f"train size: {train.shape[0]} * {train.shape[1]}")
        print(f"test size:  {test.shape[0]} * {test.shape[1]}")
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)

        # Create datasets for neighbors and distances
        neighbors_ds = f.create_dataset("neighbors", (len(test), count), dtype=int)
        distances_ds = f.create_dataset("distances", (len(test), count), dtype=float)

        # Fit the brute-force k-NN model
        bf = BruteForceBLAS(distance, precision=train.dtype)
        bf.fit(train)

        for i, x in enumerate(test):
            if i % 1000 == 0:
                print(f"{i}/{len(test)}...")

            # Query the model and sort results by distance
            res = list(bf.query_with_distances(x, count))
            res.sort(key=lambda t: t[-1])

            # Save neighbors indices and distances
            neighbors_ds[i] = [idx for idx, _ in res]
            distances_ds[i] = [dist for _, dist in res]


def train_test_split(X: numpy.ndarray, test_size: int = 10000, dimension: int = None) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Splits the provided dataset into a training set and a testing set.
    
    Args:
        X (numpy.ndarray): The dataset to split.
        test_size (int, optional): The number of samples to include in the test set. 
            Defaults to 10000.
        dimension (int, optional): The dimensionality of the data. If not provided, 
            it will be inferred from the second dimension of X. Defaults to None.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the training set and the testing set.
    """
    from sklearn.model_selection import train_test_split as sklearn_train_test_split

    dimension = dimension if not None else X.shape[1]
    print(f"Splitting {X.shape[0]}*{dimension} into train/test")
    return sklearn_train_test_split(X, test_size=test_size, random_state=1)


def random_sample(X: numpy.ndarray, size: int = 1000, seed: int = 1) -> numpy.ndarray:
    numpy.random.seed(seed)
    random_idx = numpy.random.choice(numpy.arange(len(X)), size=size, replace=False)
    return X[random_idx]


def glove(out_fn: str, d: int) -> None:
    import zipfile
    from sklearn.model_selection import train_test_split

    fn = "glove.twitter.27B.zip"
    url = "http://nlp.stanford.edu/data/%s" % fn
    download(url, fn)

    with zipfile.ZipFile(fn) as z:
        print("preparing %s" % out_fn)
        z_fn = "glove.twitter.27B.%dd.txt" % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(numpy.array(v))
        X = numpy.array(X).astype(numpy.float32)
        X_train, X_test = train_test_split(X, test_size=1000)
        write_output(X_train, X_test, out_fn, "angular")


def _load_texmex_vectors(f: Any, n: int, k: int) -> numpy.ndarray:
    import struct

    v = numpy.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack("f" * k, f.read(k * 4))

    return v


def _get_irisa_matrix(t: tarfile.TarFile, fn: str) -> numpy.ndarray:
    import struct

    m = t.getmember(fn)
    f = t.extractfile(m)
    (k,) = struct.unpack("i", f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)


def sift(out_fn: str) -> None:
    import tarfile

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    fn = os.path.join("data", "sift.tar.tz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "sift/sift_base.fvecs").astype(numpy.float32)
        test = random_sample(_get_irisa_matrix(t, "sift/sift_query.fvecs")).astype(numpy.float32)
        write_output(train, test, out_fn, "euclidean")


def gist(out_fn: str) -> None:
    import tarfile

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
    fn = os.path.join("data", "gist.tar.tz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "gist/gist_base.fvecs").astype(numpy.float32)
        test = _get_irisa_matrix(t, "gist/gist_query.fvecs").astype(numpy.float32)
        write_output(train, test, out_fn, "euclidean")


def _load_mnist_vectors(fn: str) -> numpy.ndarray:
    import gzip
    import struct

    print("parsing vectors in %s..." % fn)
    f = gzip.open(fn)
    type_code_info = {
        0x08: (1, "!B"),
        0x09: (1, "!b"),
        0x0B: (2, "!H"),
        0x0C: (4, "!I"),
        0x0D: (4, "!f"),
        0x0E: (8, "!d"),
    }
    magic, type_code, dim_count = struct.unpack("!hBB", f.read(4))
    assert magic == 0
    assert type_code in type_code_info

    dimensions = [struct.unpack("!I", f.read(4))[0] for i in range(dim_count)]

    entry_count = dimensions[0]
    entry_size = numpy.prod(dimensions[1:])

    b, format_string = type_code_info[type_code]
    vectors = []
    for i in range(entry_count):
        vectors.append([struct.unpack(format_string, f.read(b))[0] for j in range(entry_size)])
    return numpy.array(vectors)


def mnist(out_fn: str) -> None:
    download("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", "mnist-train.gz")  # noqa
    download("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", "mnist-test.gz")  # noqa
    train = _load_mnist_vectors("mnist-train.gz").astype(numpy.float32)
    test = random_sample(_load_mnist_vectors("mnist-test.gz")).astype(numpy.float32)
    write_output(train, test, out_fn, "euclidean")


def fashion_mnist(out_fn: str) -> None:
    download(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",  # noqa
        "fashion-mnist-train.gz",
    )
    download(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",  # noqa
        "fashion-mnist-test.gz",
    )
    train = _load_mnist_vectors("fashion-mnist-train.gz").astype(numpy.float32)
    test = random_sample(_load_mnist_vectors("fashion-mnist-test.gz")).astype(numpy.float32)
    write_output(train, test, out_fn, "euclidean")


def _read_fbin(filename: str, start_row: int = 0, count_rows = None):
    import numpy as np

    dtype = np.float32
    scalar_size = 4

    with open(filename, 'rb') as f:
        rows, cols = np.fromfile(f, count=2, dtype=np.int32)
        rows = (rows - start_row) if count_rows is None else count_rows
        arr = np.fromfile(f, count=rows * cols, dtype=dtype, offset=start_row * scalar_size * cols)

    return arr.reshape(rows, cols)


def _read_u8bin(filename: str, start_row: int = 0, count_rows = None):
    import numpy as np

    dtype = np.uint8
    scalar_size = 1

    with open(filename, 'rb') as f:
        rows, cols = np.fromfile(f, count=2, dtype=np.int32)
        rows = (rows - start_row) if count_rows is None else count_rows
        arr = np.fromfile(f, count=rows * cols, dtype=dtype, offset=start_row * scalar_size * cols)

    return arr.reshape(rows, cols).astype(np.float32)


def _read_json(fn, field):
    import cysimdjson
    import numpy as np

    parser = cysimdjson.JSONParser()

    vectors = []
    for line in open(fn):
        try:
            json_parsed = parser.loads(line)
            vectors.append(np.array(json_parsed[field]))
        except:
            continue

    return np.array(vectors, dtype=np.float32)


def _read_csv(fn, field, delimiter=','):
    import csv
    import numpy as np

    data = []
    with open(fn) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            data.append(np.array([float(x) for x in row[field][1:-1].split(',')]))

    return np.array(data, dtype=np.float32)


def _extract_gzip(fname):
    import gzip
    import shutil

    with gzip.open(fname, 'rb') as f_in:
        with open(fname.rstrip('.gz'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def _download_kaggle(dataset: str) -> None:
    try:
        import kaggle
        kaggle.api.authenticate()
    except:
        raise Exception(
        """To download this dataset, you need an API token from Kaggle:
           1. Go to your account settings on Kaggle.
           2. Scroll down to the "API" section and click on "Create New API Token".
           3. This will download a file named kaggle.json. Place it in ~/.kaggle/kaggle.json""")

    kaggle.api.dataset_download_files(dataset, path='.', unzip=True)


# Pretrained sentence BERT embeddings for AG News
# https://data.dtu.dk/articles/dataset/Pretrained_sentence_BERT_models_AG_News_embeddings/21286923
def ag_news(out_fn: str, model_name: str) -> None:
    import h5py
    import zipfile
    import tempfile

    dataset_url = "https://data.dtu.dk/ndownloader/articles/21286923/versions/1"
    download(dataset_url, "ag-news.zip")

    with tempfile.TemporaryDirectory() as extraction_directory:
        with zipfile.ZipFile('ag-news.zip', 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if model_name in file_name and file_name.endswith('train.h5'):
                    zip_ref.extract(file_name, extraction_directory)
                    f = h5py.File(extraction_directory + '/' + file_name, 'r')
                    X = f['embeddings'][:]
                    break

    X_train, X_test = train_test_split(X, test_size=1000)
    write_output(X_train, X_test, out_fn, "angular")


def ann_unum(out_fn: str, data_name: str, file_name: str) -> None:
    dataset_url = f"https://huggingface.co/datasets/unum-cloud/{data_name}/resolve/main/{file_name}"
    download(dataset_url, file_name)

    X = _read_fbin(file_name)
    X_train, X_test = train_test_split(X, test_size=1000)
    write_output(X_train, X_test, out_fn, "angular")


# ArXiv paper abstracts embedded using the InstructorXL model
# https://huggingface.co/datasets/Qdrant/arxiv-abstracts-instructorxl-embeddings
def arxiv_instructxl(out_fn: str) -> None:
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset("Qdrant/arxiv-abstracts-instructorxl-embeddings")
    n = len(ds['train'])

    X = np.zeros((n, 768), dtype=np.float32)
    for i in range(n):
        X[i] = np.array(ds['train'][i]['vector'], dtype=np.float32)

    X_train, X_test = train_test_split(X, test_size=1000)
    write_output(X_train, X_test, out_fn, "angular")


# ArXiv paper abstracts embedded using the OpenAI text-embedding-ada-002 model
# https://www.kaggle.com/datasets/awester/arxiv-embeddings
def arxiv_openai(out_fn: str) -> None:
    _download_kaggle('awester/arxiv-embeddings')

    X = _read_json('ml-arxiv-embeddings.json', 'embedding')
    X_train, X_test = train_test_split(X, test_size=1000)
    write_output(X_train, X_test, out_fn, "angular")


def load_fasttext_vectors(fname):
    import io
    import numpy as np
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = []
    for line in fin:
        tokens = line.rstrip().split(' ')
        data.append(np.array(list(map(float, tokens[1:]))))
    return np.array(data, dtype=np.float32)


# fastText English word vectors
# https://fasttext.cc/docs/en/english-vectors.html
def fasttext(out_fn: str, name: str, distance: str = "euclidean") -> None:
    import zipfile
    import numpy as np

    fname = name + ".vec.zip"
    dataset_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/" + fname
    download(dataset_url, fname)

    with zipfile.ZipFile(fname, 'r') as zip_ref:
        zip_ref.extractall('.')

    X = load_fasttext_vectors(name + ".vec")
    X_train, X_test = train_test_split(X, test_size=1000)
    write_output(X_train, X_test, out_fn, distance)


# OpenAI, all-MiniLM-L6-v2 and GTE-small embeddings for Wikipedia Simple English
# https://huggingface.co/datasets/Supabase/wikipedia-en-embeddings
def wiki_en_embeddings(out_fn: str, file_name: str, col_name: str) -> None:
    dataset_url = "https://huggingface.co/datasets/Supabase/wikipedia-en-embeddings/resolve/main/" + file_name + ".gz"

    download(dataset_url, file_name + ".gz")
    _extract_gzip(file_name + ".gz")

    X = _read_json(file_name, col_name)
    X_train, X_test = train_test_split(X, test_size=1000)
    write_output(X_train, X_test, out_fn, "angular")


# Food images from the Wolt app embedded with the clip-ViT-B-32 model
# https://huggingface.co/datasets/Qdrant/wolt-food-clip-ViT-B-32-embeddings
def wolt_vit(out_fn: str) -> None:
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset("Qdrant/wolt-food-clip-ViT-B-32-embeddings")
    n = len(ds['train'])

    X = np.zeros((n, 512), dtype=np.float32)
    for i in range(n):
        X[i] = np.array([float(x) for x in ds['train'][i]['vector'][1:-1].split(b',')], dtype=np.float32)

    X_train, X_test = train_test_split(X, test_size=1000)
    write_output(X_train, X_test, out_fn, "angular")


# Yandex Text-to-Image 50M queries sample data
# https://big-ann-benchmarks.com/neurips21.html
def yandex_toi(out_fn: str, size: int) -> None:
    dataset_url = "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin"
    download(dataset_url, "yandex.fbin")

    query_url = "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin"
    download(query_url, "yandex_query.fbin")

    #query_learn_url = "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin"
    #download(query_learn_url, "yandex_query_learn.fbin")

    X_train = random_sample(_read_fbin("yandex.fbin", count_rows=12000000), size)
    X_test = random_sample(_read_fbin("yandex_query.fbin"))

    #Q = _read_fbin("yandex_query_learn.fbin", count_rows=8000000)
    #numpy.save("yandex_query_learn.npy", Q)

    write_output(X_train, X_test, out_fn, "angular")


DATASETS: Dict[str, Callable[[str], None]] = {
    "fashion-mnist-784-euclidean": fashion_mnist,
    "gist-960-euclidean": gist,
    "glove-200-angular": lambda out_fn: glove(out_fn, 200),
    "mnist-784-euclidean": mnist,
    "sift-128-euclidean": sift,
    "ag-news-distilbert-768-angular": lambda out_fn: ag_news(out_fn, "multi-qa-distilbert-cos-v1"),
    "ag-news-minilm-384-angular": lambda out_fn: ag_news(out_fn, "all-MiniLM-L12-v2"),
    "ann-arxiv-768-angular": lambda out_fn: ann_unum(out_fn, "ann-arxiv-2m", "abstract.e5-base-v2.fbin"),
    "ann-t2i-200-angular": lambda out_fn: ann_unum(out_fn, "ann-t2i-1m", "base.1M.fbin"),
    "arxiv-instructxl-768-angular": arxiv_instructxl,
    "arxiv-openai-1536-angular": arxiv_openai,
    "fasttext-wiki-300-euclidean": lambda out_fn: fasttext(out_fn, "wiki-news-300d-1M"),
    "wiki-gte-384-angular": lambda out_fn: wiki_en_embeddings(out_fn, "wiki_gte.ndjson", "embedding"),
    "wiki-openai-1536-angular": lambda out_fn: wiki_en_embeddings(out_fn, "wiki_openai.ndjson", "text-embedding-ada-002"),
    "wolt-vit-512-angular": wolt_vit,
    "yandex-400K-200-angular": lambda out_fn: yandex_toi(out_fn, 400_000),
    "yandex-5M-200-angular": lambda out_fn: yandex_toi(out_fn, 5_000_000),
}
