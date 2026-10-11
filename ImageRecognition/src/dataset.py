"""Download and validate the original MNIST IDX files from the CVDF mirror."""

import gzip
import hashlib
from pathlib import Path
import shutil
import struct
import urllib.request

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "mnist"
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
# Published MNIST archive checksums, also used by torchvision's MNIST loader.
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def checksum(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_mnist(data_dir=DEFAULT_DATA_DIR):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected in FILES.items():
        target = data_dir / filename
        if target.exists() and checksum(target) == expected:
            print(f"Verified {filename}", flush=True)
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            print(f"Downloading {BASE_URL + filename}", flush=True)
            with urllib.request.urlopen(BASE_URL + filename, timeout=60) as source:
                with temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            if checksum(temporary) != expected:
                raise ValueError(f"Checksum mismatch for {filename}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return data_dir


def read_idx(path):
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    if len(raw) < 4:
        raise ValueError(f"Truncated IDX header: {path}")
    zero, kind, dimensions = struct.unpack(">HBB", raw[:4])
    if zero != 0 or kind != 8 or dimensions not in (1, 3):
        raise ValueError(f"Invalid MNIST IDX header: {path}")
    offset = 4 + dimensions * 4
    if len(raw) < offset:
        raise ValueError(f"Truncated IDX dimensions: {path}")
    shape = struct.unpack(">" + "I" * dimensions, raw[4:offset])
    if any(size == 0 for size in shape) or np.prod(shape) != len(raw) - offset:
        raise ValueError(f"Invalid IDX payload size: {path}")
    return np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape(shape).copy()


def load_mnist(data_dir=DEFAULT_DATA_DIR, split="train"):
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    prefix, count = ("train", 60000) if split == "train" else ("t10k", 10000)
    paths = [Path(data_dir) / f"{prefix}-{suffix}" for suffix in
             ("images-idx3-ubyte.gz", "labels-idx1-ubyte.gz")]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run 'python -m ImageRecognition download'.")
        if checksum(path) != FILES[path.name]:
            raise ValueError(f"Corrupt MNIST file: {path}. Run the download command to repair it.")
    images, labels = map(read_idx, paths)
    if images.shape != (count, 28, 28) or labels.shape != (count,) or np.any(labels > 9):
        raise ValueError(f"Unexpected MNIST {split} shapes or labels")
    return images, labels


def prepare_mnist(data_dir=DEFAULT_DATA_DIR):
    """Export unmodified IDX pixels as PNGs for the original folder-based API."""
    data_dir = Path(data_dir)
    for split in ("train", "test"):
        images, labels = load_mnist(data_dir, split)
        destination = data_dir / "images" / split
        for digit in range(10):
            (destination / str(digit)).mkdir(parents=True, exist_ok=True)
        for index, (pixels, label) in enumerate(zip(images, labels)):
            target = destination / str(int(label)) / f"{index:05d}.png"
            if not target.exists():
                Image.fromarray(pixels).save(target)
        print(f"Prepared {len(images):,} original {split} images in {destination}", flush=True)
    return data_dir / "images"
