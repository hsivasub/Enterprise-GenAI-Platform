"""
Tests for ingestion/vector_store.py — FAISSVectorStore
========================================================
Tests add, search, delete, persistence, and deduplication.
Uses tiny embedding dimension (8-d) for fast CPU testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from ingestion.chunker import DocumentChunk
from ingestion.vector_store import FAISSVectorStore


EMBEDDING_DIM = 8


def make_chunk(doc_id: str, chunk_id: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        chunk_index=0,
        total_chunks=1,
        char_start=0,
        char_end=len(content),
        strategy="recursive_character",
        metadata={"filename": "test.txt"},
    )


def make_normalized_vector(dim: int = EMBEDDING_DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def vector_store(tmp_dir: Path) -> FAISSVectorStore:
    return FAISSVectorStore(
        index_path=str(tmp_dir / "faiss_test"),
        dimension=EMBEDDING_DIM,
    )


# ── Initialization ────────────────────────────────────────────────

class TestInitialization:

    def test_creates_with_zero_vectors(self, vector_store):
        assert vector_store.total_vectors == 0

    def test_dimension_is_set(self, vector_store):
        assert vector_store.dimension == EMBEDDING_DIM

    def test_empty_search_returns_empty_list(self, vector_store):
        q = make_normalized_vector()
        results = vector_store.search(q, top_k=5)
        assert results == []


# ── Add chunks ────────────────────────────────────────────────────

class TestAddChunks:

    def test_add_single_chunk(self, vector_store):
        chunk = make_chunk("doc-1", "chunk-001", "Apple revenue Q3 2024")
        vector = make_normalized_vector(seed=1)
        vector_store.add_chunks([(chunk, vector)])
        assert vector_store.total_vectors == 1

    def test_add_multiple_chunks(self, vector_store):
        chunks = [
            make_chunk("doc-1", f"chunk-{i:03d}", f"Content {i}")
            for i in range(5)
        ]
        vectors = [make_normalized_vector(seed=i) for i in range(5)]
        vector_store.add_chunks(list(zip(chunks, vectors)))
        assert vector_store.total_vectors == 5

    def test_duplicate_chunk_not_added_twice(self, vector_store):
        chunk = make_chunk("doc-1", "chunk-dup", "Duplicate content")
        vector = make_normalized_vector(seed=99)
        vector_store.add_chunks([(chunk, vector)])
        vector_store.add_chunks([(chunk, vector)])  # Second add - duplicate
        assert vector_store.total_vectors == 1


# ── Search ────────────────────────────────────────────────────────

class TestSearch:

    def test_search_returns_results_after_adding(self, vector_store):
        chunk = make_chunk("doc-1", "chunk-001", "Apple revenue Q3 2024")
        vector = make_normalized_vector(seed=1)
        vector_store.add_chunks([(chunk, vector)])

        results = vector_store.search(vector, top_k=1)
        assert len(results) == 1

    def test_search_result_has_chunk_id(self, vector_store):
        chunk = make_chunk("doc-1", "chunk-001", "Test content")
        vector = make_normalized_vector(seed=2)
        vector_store.add_chunks([(chunk, vector)])

        results = vector_store.search(vector, top_k=1)
        assert results[0]["chunk_id"] == "chunk-001"

    def test_search_result_has_score(self, vector_store):
        chunk = make_chunk("doc-1", "chunk-001", "Test content")
        vector = make_normalized_vector(seed=3)
        vector_store.add_chunks([(chunk, vector)])

        results = vector_store.search(vector, top_k=1)
        assert "score" in results[0]
        assert isinstance(results[0]["score"], float)

    def test_identical_vector_has_score_near_1(self, vector_store):
        chunk = make_chunk("doc-1", "chunk-001", "Test content")
        vector = make_normalized_vector(seed=4)
        vector_store.add_chunks([(chunk, vector)])

        results = vector_store.search(vector, top_k=1)
        # Inner product of identical normalized vectors ≈ 1.0
        assert results[0]["score"] > 0.99

    def test_top_k_limits_results(self, vector_store):
        for i in range(10):
            chunk = make_chunk("doc-1", f"chunk-{i:03d}", f"Content {i}")
            vector_store.add_chunks([(chunk, make_normalized_vector(seed=i))])

        results = vector_store.search(make_normalized_vector(seed=0), top_k=3)
        assert len(results) <= 3

    def test_results_sorted_by_score_descending(self, vector_store):
        for i in range(5):
            chunk = make_chunk("doc-1", f"chunk-{i:03d}", f"Content {i}")
            vector_store.add_chunks([(chunk, make_normalized_vector(seed=i))])

        results = vector_store.search(make_normalized_vector(seed=0), top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ── Delete ────────────────────────────────────────────────────────

class TestDelete:

    def test_delete_by_doc_id_removes_chunks(self, vector_store):
        chunks = [make_chunk("doc-A", f"chunk-A-{i}", f"Content {i}") for i in range(3)]
        vectors = [make_normalized_vector(seed=i) for i in range(3)]
        vector_store.add_chunks(list(zip(chunks, vectors)))

        removed = vector_store.delete_by_doc_id("doc-A")
        assert removed == 3
        assert vector_store.total_vectors == 0

    def test_delete_nonexistent_doc_returns_zero(self, vector_store):
        removed = vector_store.delete_by_doc_id("doc-does-not-exist")
        assert removed == 0

    def test_delete_only_removes_target_doc(self, vector_store):
        for doc_id, seed in [("doc-A", 1), ("doc-B", 2)]:
            c = make_chunk(doc_id, f"chunk-{doc_id}", f"Content for {doc_id}")
            vector_store.add_chunks([(c, make_normalized_vector(seed=seed))])

        vector_store.delete_by_doc_id("doc-A")
        assert vector_store.total_vectors == 1


# ── Persistence ───────────────────────────────────────────────────

class TestPersistence:

    def test_save_and_load(self, tmp_dir: Path):
        index_path = str(tmp_dir / "test_index")
        store1 = FAISSVectorStore(index_path=index_path, dimension=EMBEDDING_DIM)

        chunk = make_chunk("doc-1", "chunk-persist", "Persistent content")
        vector = make_normalized_vector(seed=42)
        store1.add_chunks([(chunk, vector)])
        store1.save()

        # Load into a new instance
        store2 = FAISSVectorStore(index_path=index_path, dimension=EMBEDDING_DIM)
        assert store2.total_vectors == 1
        results = store2.search(vector, top_k=1)
        assert results[0]["chunk_id"] == "chunk-persist"
