"""Загрузка MNIST, фиксированное разбиение и просмотр изображения в терминале."""

import argparse
import gzip
import hashlib
import struct
import tempfile
from functools import partial
from pathlib import Path
from urllib.request import urlopen

import numpy as np


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "mnist"
MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}
SPLIT_SEED = 42
VALIDATION_SIZE = 5_000


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(partial(stream.read, 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_mnist(data_dir):
    """Скачивает четыре файла и проверяет их контрольные суммы SHA-256."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, checksum in FILES.items():
        path = data_dir / filename
        if path.is_file() and _sha256(path) == checksum:
            continue

        print(f"Скачивание {filename}...", flush=True)
        with tempfile.TemporaryDirectory(dir=data_dir) as temp_dir:
            temporary_path = Path(temp_dir) / filename
            with urlopen(MIRROR + filename, timeout=60) as response:
                temporary_path.write_bytes(response.read())
            if _sha256(temporary_path) != checksum:
                raise ValueError(f"Контрольная сумма не совпала для {filename}.")
            temporary_path.replace(path)


def _read_idx(path, magic_number, shape):
    try:
        with gzip.open(path, "rb") as stream:
            header_format = ">" + "I" * (len(shape) + 1)
            header = stream.read(struct.calcsize(header_format))
            if header != struct.pack(header_format, magic_number, *shape):
                raise ValueError(f"Неверный заголовок IDX в {path}.")
            pixels = np.frombuffer(stream.read(), dtype=np.uint8)
            return pixels.reshape(shape)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Файл {path} отсутствует. Сначала выполните python -m mnist_numpy.data."
        ) from error


def _load_pair(data_dir, prefix, count):
    images = _read_idx(data_dir / f"{prefix}-images-idx3-ubyte.gz", 2051, (count, 28, 28))
    labels = _read_idx(data_dir / f"{prefix}-labels-idx1-ubyte.gz", 2049, (count,))
    if labels.max() > 9:
        raise ValueError(f"В MNIST встречена метка за пределами 0–9 ({prefix}).")
    return images, labels.astype(np.int64)


def _prepare(images):
    features = images.reshape(-1, 784).astype(np.float32)
    features /= np.float32(255)
    return features


def load_train_val(data_dir=DEFAULT_DATA_DIR, limit_train=None):
    """Возвращает (x_train, y_train), (x_val, y_val) с пикселями в [0, 1]."""
    images, labels = _load_pair(data_dir, "train", 60_000)
    indices = np.random.default_rng(SPLIT_SEED).permutation(len(labels))
    validation_indices = indices[:VALIDATION_SIZE]
    train_indices = indices[VALIDATION_SIZE:][:limit_train]
    train = _prepare(images[train_indices]), labels[train_indices]
    validation = _prepare(images[validation_indices]), labels[validation_indices]
    return train, validation


def load_test(data_dir=DEFAULT_DATA_DIR):
    """Возвращает исходную тестовую выборку MNIST в порядке датасета."""
    images, labels = _load_pair(data_dir, "t10k", 10_000)
    return _prepare(images), labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--index", type=int, help="Индекс изображения в исходном файле, с нуля")
    args = parser.parse_args()

    if args.index is None:
        try:
            download_mnist(args.data_dir)
        except (OSError, ValueError) as error:
            parser.exit(1, f"Ошибка загрузки MNIST: {error}\n")
        print(f"Файлы MNIST готовы в {args.data_dir}")
        return

    prefix, count = ("train", 60_000) if args.split == "train" else ("t10k", 10_000)
    if not 0 <= args.index < count:
        parser.error(f"--index должен быть в диапазоне от 0 до {count - 1}")
    try:
        images, labels = _load_pair(args.data_dir, prefix, count)
    except (OSError, EOFError, ValueError) as error:
        parser.exit(1, f"Ошибка чтения MNIST: {error}\n")
    print(f"{args.split}[{args.index}], цифра {labels[args.index]}")
    shades = np.array(list(" .:-=+*#%@"))
    for row in shades[images[args.index].astype(np.int32) * (len(shades) - 1) // 255]:
        print("".join(pixel * 2 for pixel in row))


if __name__ == "__main__":
    main()
