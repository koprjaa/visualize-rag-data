"""Reading a ChromaDB folder from disk.

Shared by precompute.py and main.py, which both need to find the database, and
by precompute.py alone for the HNSW index. The parsing is kept apart from the
file handling so it can be tested without a real database.
"""

import os
import struct
from pathlib import Path

import numpy as np

# Byte offsets inside header.bin. hnswlib writes the struct fields in this order,
# and the layout is not documented anywhere, so the numbers were read off a real
# index. entry_size sits at 28. It is not at 36, which is the obvious guess and
# the wrong one.
HEADER_NUM_VECTORS = 20
HEADER_ENTRY_SIZE = 28
HEADER_VECTOR_OFFSET = 44

FLOAT32_BYTES = 4


class ChromaIndexError(ValueError):
    """The HNSW index on disk is not shaped the way the reader expects."""

# Radius the point cloud is scaled to, so the camera framing does not depend on
# how spread out a particular collection happens to be.
SCENE_RADIUS = 3.0


def resolve_chroma_db_path() -> Path:
    """Where the ChromaDB folder lives.

    In order: the CHROMA_DB_PATH variable, the current default layout, then the
    layout used before the data folder moved. The current default is returned
    when neither exists, so the error names the path that was expected.
    """
    env = os.getenv("CHROMA_DB_PATH")
    if env:
        return Path(env).expanduser()

    repo_root = Path(__file__).resolve().parent.parent
    current = repo_root / "data" / "chroma-db" / "protext"
    legacy = repo_root / "chroma_db_PROTEXT"

    if current.exists():
        return current
    if legacy.exists():
        return legacy
    return current


def find_collection_folder(db_path: Path, marker: str) -> Path | None:
    """First collection folder under db_path that holds the given file."""
    if not db_path.is_dir():
        return None
    return next(
        (d for d in sorted(db_path.iterdir()) if d.is_dir() and (d / marker).exists()),
        None,
    )


def parse_hnsw_header(header: bytes) -> tuple[int, int, int]:
    """Vector count, entry size and vector offset out of header.bin."""
    if len(header) < HEADER_VECTOR_OFFSET + 4:
        raise ChromaIndexError(f"header.bin is {len(header)} bytes, too short to read")
    return (
        struct.unpack("<I", header[HEADER_NUM_VECTORS:HEADER_NUM_VECTORS + 4])[0],
        struct.unpack("<I", header[HEADER_ENTRY_SIZE:HEADER_ENTRY_SIZE + 4])[0],
        struct.unpack("<I", header[HEADER_VECTOR_OFFSET:HEADER_VECTOR_OFFSET + 4])[0],
    )


def extract_vectors(
    data: bytes, num_vectors: int, entry_size: int, vector_offset: int, dimensions: int
) -> dict[int, np.ndarray]:
    """Pull the float32 vectors out of data_level0.bin, keyed by HNSW label.

    Labels start at one and positions in the file start at zero, so label N sits
    at position N-1. An entry that would run past the end of the file is left
    out rather than returned truncated, which happens when the index is still
    being written or when the dimension count is wrong.
    """
    if entry_size <= 0:
        raise ChromaIndexError(f"entry_size must be positive, got {entry_size}")

    span = dimensions * FLOAT32_BYTES
    vectors = {}
    for label in range(1, num_vectors + 1):
        start = (label - 1) * entry_size + vector_offset
        if start + span <= len(data):
            vectors[label] = np.frombuffer(data[start:start + span], dtype=np.float32).copy()
    return vectors


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Center the cloud on the origin and scale it to a fixed radius."""
    coords = coords - coords.mean(axis=0)
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords * (SCENE_RADIUS / max_dist)
    return coords
