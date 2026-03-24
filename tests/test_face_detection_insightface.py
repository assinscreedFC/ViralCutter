"""Tests for face_detection_insightface Re-ID helpers (get_face_embedding, cosine_similarity)."""

import numpy as np
import pytest

from scripts.face_detection_insightface import get_face_embedding, cosine_similarity


class TestGetFaceEmbedding:
    """Tests for get_face_embedding()."""

    def test_returns_none_for_empty_dict(self):
        assert get_face_embedding({}) is None

    def test_returns_none_for_no_embedding_keys(self):
        face = {'bbox': [0, 0, 100, 100], 'det_score': 0.9}
        assert get_face_embedding(face) is None

    def test_returns_embedding_when_present(self):
        emb = np.random.randn(512).astype(np.float32)
        face = {'embedding': emb}
        result = get_face_embedding(face)
        assert result is not None
        np.testing.assert_array_equal(result, emb)

    def test_prefers_normed_embedding(self):
        raw = np.random.randn(512).astype(np.float32)
        normed = raw / np.linalg.norm(raw)
        face = {'embedding': raw, 'normed_embedding': normed}
        result = get_face_embedding(face)
        np.testing.assert_array_equal(result, normed)

    def test_falls_back_to_raw_if_normed_absent(self):
        raw = np.random.randn(512).astype(np.float32)
        face = {'embedding': raw, 'normed_embedding': None}
        result = get_face_embedding(face)
        np.testing.assert_array_equal(result, raw)

    def test_returns_none_for_empty_array(self):
        face = {'embedding': np.array([])}
        assert get_face_embedding(face) is None


class TestCosineSimilarity:
    """Tests for cosine_similarity()."""

    def test_identical_vectors(self):
        a = np.random.randn(512)
        assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.random.randn(512)
        assert cosine_similarity(a, -a) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.zeros(512)
        b = np.zeros(512)
        a[0] = 1.0
        b[1] = 1.0
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(512)
        b = np.random.randn(512)
        assert cosine_similarity(a, b) == 0.0

    def test_normalized_vectors(self):
        a = np.random.randn(512)
        b = np.random.randn(512)
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        # Cosine similarity of normalized vectors = dot product
        expected = float(np.dot(a_norm, b_norm))
        assert cosine_similarity(a_norm, b_norm) == pytest.approx(expected, abs=1e-6)

    def test_symmetry(self):
        a = np.random.randn(512)
        b = np.random.randn(512)
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a), abs=1e-9)

    def test_result_in_range(self):
        """Cosine similarity should always be in [-1, 1]."""
        for _ in range(10):
            a = np.random.randn(512)
            b = np.random.randn(512)
            sim = cosine_similarity(a, b)
            assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6
