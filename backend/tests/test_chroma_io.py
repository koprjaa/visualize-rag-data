"""Tests for reading a ChromaDB folder and its HNSW index.

The index files are built here byte by byte, so no real database is needed. The
header layout is the part worth pinning down: it is undocumented and the offsets
were read off a live index.
"""

import struct

import numpy as np
import pytest
from chroma_io import (
    FLOAT32_BYTES,
    HEADER_ENTRY_SIZE,
    HEADER_NUM_VECTORS,
    HEADER_VECTOR_OFFSET,
    SCENE_RADIUS,
    ChromaIndexError,
    extract_vectors,
    find_collection_folder,
    normalize_coords,
    parse_hnsw_header,
    resolve_chroma_db_path,
)

DIMENSIONS = 4
VECTOR_OFFSET = 8


def make_header(num_vectors, entry_size, vector_offset, size=64):
    """A header.bin holding the three fields the reader looks for."""
    buf = bytearray(size)
    struct.pack_into("<I", buf, HEADER_NUM_VECTORS, num_vectors)
    struct.pack_into("<I", buf, HEADER_ENTRY_SIZE, entry_size)
    struct.pack_into("<I", buf, HEADER_VECTOR_OFFSET, vector_offset)
    return bytes(buf)


def make_data(vectors, entry_size=None, vector_offset=VECTOR_OFFSET):
    """A data_level0.bin holding the given vectors, one per entry slot."""
    span = len(vectors[0]) * FLOAT32_BYTES
    entry_size = entry_size or vector_offset + span + 4  # trailing bytes per entry
    buf = bytearray(entry_size * len(vectors))
    for position, vector in enumerate(vectors):
        start = position * entry_size + vector_offset
        buf[start:start + span] = np.asarray(vector, dtype=np.float32).tobytes()
    return bytes(buf), entry_size


# --- parse_hnsw_header ------------------------------------------------------


def test_header_fields_are_read_from_their_offsets():
    assert parse_hnsw_header(make_header(601, 4128, 32)) == (601, 4128, 32)


def test_entry_size_is_read_from_offset_28_not_36():
    """The offset is undocumented and 36 is the plausible wrong answer."""
    header = bytearray(make_header(10, 4128, 32))
    struct.pack_into("<I", header, 36, 999999)
    assert parse_hnsw_header(bytes(header))[1] == 4128


def test_a_truncated_header_is_rejected():
    with pytest.raises(ChromaIndexError):
        parse_hnsw_header(b"\x00" * 8)


def test_a_header_exactly_long_enough_is_accepted():
    header = make_header(1, 16, 0, size=HEADER_VECTOR_OFFSET + 4)
    assert parse_hnsw_header(header) == (1, 16, 0)


# --- extract_vectors --------------------------------------------------------


def test_vectors_come_back_keyed_by_label_starting_at_one():
    vectors = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    data, entry_size = make_data(vectors)
    got = extract_vectors(data, 2, entry_size, VECTOR_OFFSET, DIMENSIONS)
    assert sorted(got) == [1, 2]
    np.testing.assert_allclose(got[1], vectors[0])
    np.testing.assert_allclose(got[2], vectors[1])


def test_label_n_is_read_from_position_n_minus_one():
    """An off-by-one here shifts every document onto the wrong coordinates."""
    vectors = [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]
    data, entry_size = make_data(vectors)
    got = extract_vectors(data, 3, entry_size, VECTOR_OFFSET, DIMENSIONS)
    np.testing.assert_allclose(got[3], vectors[2])


def test_an_entry_running_past_the_end_of_the_file_is_left_out():
    vectors = [[1.0, 2.0, 3.0, 4.0]]
    data, entry_size = make_data(vectors)
    got = extract_vectors(data, 5, entry_size, VECTOR_OFFSET, DIMENSIONS)
    assert list(got) == [1]


def test_a_wrong_dimension_count_drops_entries_rather_than_truncating():
    vectors = [[1.0, 2.0, 3.0, 4.0]]
    data, entry_size = make_data(vectors)
    assert extract_vectors(data, 1, entry_size, VECTOR_OFFSET, 4096) == {}


def test_the_returned_array_does_not_share_memory_with_the_buffer():
    """np.frombuffer gives a read-only view onto bytes, which breaks later math."""
    vectors = [[1.0, 2.0, 3.0, 4.0]]
    data, entry_size = make_data(vectors)
    got = extract_vectors(data, 1, entry_size, VECTOR_OFFSET, DIMENSIONS)
    got[1] += 1  # would raise on a read-only view
    np.testing.assert_allclose(got[1], [2.0, 3.0, 4.0, 5.0])


@pytest.mark.parametrize("entry_size", [0, -8])
def test_a_nonsense_entry_size_is_rejected(entry_size):
    with pytest.raises(ChromaIndexError):
        extract_vectors(b"\x00" * 64, 1, entry_size, 0, DIMENSIONS)


def test_no_vectors_gives_an_empty_result():
    assert extract_vectors(b"\x00" * 64, 0, 16, 0, DIMENSIONS) == {}


# --- normalize_coords -------------------------------------------------------


def test_the_cloud_ends_up_centered_on_the_origin():
    coords = np.array([[10.0, 10.0, 10.0], [12.0, 14.0, 16.0], [8.0, 6.0, 4.0]])
    np.testing.assert_allclose(normalize_coords(coords).mean(axis=0), [0, 0, 0], atol=1e-6)


def test_the_furthest_point_lands_on_the_scene_radius():
    coords = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    assert np.max(np.linalg.norm(normalize_coords(coords), axis=1)) == pytest.approx(SCENE_RADIUS)


def test_scaling_does_not_change_the_shape_of_the_cloud():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    scaled = normalize_coords(coords)
    original_ratio = np.linalg.norm(coords[2] - coords[0]) / np.linalg.norm(coords[1] - coords[0])
    scaled_ratio = np.linalg.norm(scaled[2] - scaled[0]) / np.linalg.norm(scaled[1] - scaled[0])
    assert scaled_ratio == pytest.approx(original_ratio)


def test_a_single_point_does_not_divide_by_zero():
    np.testing.assert_allclose(normalize_coords(np.array([[5.0, 5.0, 5.0]])), [[0, 0, 0]])


def test_points_that_all_sit_together_do_not_divide_by_zero():
    coords = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    np.testing.assert_allclose(normalize_coords(coords), np.zeros((2, 3)))


# --- folder discovery -------------------------------------------------------


def test_the_collection_folder_is_the_one_holding_the_marker(tmp_path):
    (tmp_path / "empty").mkdir()
    wanted = tmp_path / "collection-a"
    wanted.mkdir()
    (wanted / "data_level0.bin").write_bytes(b"")
    assert find_collection_folder(tmp_path, "data_level0.bin") == wanted


def test_no_matching_folder_gives_none(tmp_path):
    (tmp_path / "empty").mkdir()
    assert find_collection_folder(tmp_path, "data_level0.bin") is None


def test_a_missing_database_folder_gives_none(tmp_path):
    assert find_collection_folder(tmp_path / "nope", "data_level0.bin") is None


def test_the_choice_of_folder_is_stable_when_several_match(tmp_path):
    for name in ("z-collection", "a-collection"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "header.bin").write_bytes(b"")
    assert find_collection_folder(tmp_path, "header.bin").name == "a-collection"


# --- resolve_chroma_db_path -------------------------------------------------


def test_the_environment_variable_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path))
    assert resolve_chroma_db_path() == tmp_path


def test_a_tilde_in_the_environment_variable_is_expanded(monkeypatch):
    monkeypatch.setenv("CHROMA_DB_PATH", "~/chroma")
    assert "~" not in str(resolve_chroma_db_path())


def test_without_the_variable_the_current_layout_is_used(monkeypatch):
    monkeypatch.delenv("CHROMA_DB_PATH", raising=False)
    assert resolve_chroma_db_path().parts[-3:] == ("data", "chroma-db", "protext")
