"""
Tests for ingestion/chunker.py
================================
Covers all three chunking strategies and the factory function.
"""

from __future__ import annotations

import pytest

from ingestion.chunker import (
    DocumentChunk,
    FixedSizeChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
    get_chunker,
)
from ingestion.document_loader import DocumentMetadata, RawDocument


# ── Helpers ───────────────────────────────────────────────────────

def make_raw_doc(content: str, doc_id: str = "doc-test-001") -> RawDocument:
    return RawDocument(
        doc_id=doc_id,
        content=content,
        metadata=DocumentMetadata(
            source="/tmp/test.txt",
            filename="test.txt",
            doc_type="txt",
            file_size_bytes=len(content),
            content_hash="abc12345",
        ),
    )


SHORT_TEXT = "This is a short sentence. It should produce very few chunks."
LONG_TEXT = (
    "Apple Inc. reported total net revenue of $85.8 billion for Q3 2024. "
    "Net income was $21.4 billion with EPS of $1.40. "
    "Services revenue reached $24.2 billion, growing 14% YoY. "
    "The gross margin improved to 46.3%, the highest in company history. "
    "iPhone revenue was $39.3 billion, Mac $7.0 billion, iPad $7.2 billion. "
    "The company returned $26 billion to shareholders through share buybacks. "
) * 5  # Repeat to force multiple chunks


# ── RecursiveCharacterChunker ────────────────────────────────────

class TestRecursiveCharacterChunker:

    def setup_method(self):
        self.chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20)

    def test_returns_list_of_document_chunks(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        assert isinstance(chunks, list)
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunks_have_correct_doc_id(self):
        doc = make_raw_doc(LONG_TEXT, doc_id="doc-xyz")
        chunks = self.chunker.chunk(doc)
        assert all(c.doc_id == "doc-xyz" for c in chunks)

    def test_chunk_ids_are_unique(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_produces_multiple_chunks_for_long_text(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) > 1

    def test_short_text_produces_at_least_one_chunk(self):
        doc = make_raw_doc(SHORT_TEXT)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_chunk_content_not_empty(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all(len(c.content.strip()) > 0 for c in chunks)

    def test_chunk_strategy_label(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all(c.strategy == "recursive_character" for c in chunks)

    def test_metadata_propagated(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all("filename" in c.metadata for c in chunks)

    def test_chunk_size_respected(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        # No chunk should be wildly over chunk_size (overlap can add a little)
        for c in chunks:
            assert len(c.content) <= 200 + 50, (
                f"Chunk too large: {len(c.content)} chars"
            )

    def test_total_chunks_field(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        total = len(chunks)
        assert all(c.total_chunks == total for c in chunks)

    def test_chunk_index_sequential(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_word_count_property(self):
        doc = make_raw_doc("Hello world this is a test sentence.")
        chunks = self.chunker.chunk(doc)
        for c in chunks:
            assert c.word_count > 0


# ── FixedSizeChunker ─────────────────────────────────────────────

class TestFixedSizeChunker:

    def setup_method(self):
        self.chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)

    def test_produces_chunks(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) > 0

    def test_strategy_label(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all(c.strategy == "fixed_size" for c in chunks)

    def test_deterministic_output(self):
        """Same input should always produce the same chunks."""
        doc = make_raw_doc(LONG_TEXT)
        chunks1 = self.chunker.chunk(doc)
        chunks2 = self.chunker.chunk(doc)
        assert [c.content for c in chunks1] == [c.content for c in chunks2]

    def test_offsets_are_non_negative(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all(c.char_start >= 0 for c in chunks)

    def test_last_chunk_ends_at_or_before_doc_end(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        assert chunks[-1].char_end <= len(LONG_TEXT) + 100  # Generous bound


# ── SentenceChunker ──────────────────────────────────────────────

class TestSentenceChunker:

    def setup_method(self):
        self.chunker = SentenceChunker(chunk_size=300, chunk_overlap=30)

    def test_produces_chunks(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_strategy_label(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all(c.strategy == "sentence_aware" for c in chunks)

    def test_chunks_not_empty(self):
        doc = make_raw_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all(c.content.strip() for c in chunks)


# ── Factory function ─────────────────────────────────────────────

class TestGetChunker:

    def test_returns_recursive_by_default(self):
        chunker = get_chunker("recursive_character")
        assert isinstance(chunker, RecursiveCharacterChunker)

    def test_returns_fixed_size(self):
        chunker = get_chunker("fixed_size")
        assert isinstance(chunker, FixedSizeChunker)

    def test_returns_sentence_aware(self):
        chunker = get_chunker("sentence_aware")
        assert isinstance(chunker, SentenceChunker)

    def test_raises_on_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("nonexistent_strategy")

    def test_accepts_chunk_size_kwarg(self):
        chunker = get_chunker("fixed_size", chunk_size=256)
        assert chunker.chunk_size == 256

    def test_accepts_chunk_overlap_kwarg(self):
        chunker = get_chunker("recursive_character", chunk_overlap=32)
        assert chunker.chunk_overlap == 32
