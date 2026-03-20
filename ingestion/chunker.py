"""
Chunking Strategies — Ingestion Layer
=======================================
Splits documents into chunks optimized for vector search and LLM context windows.

Three strategies implemented:
1. RecursiveCharacterChunker — LangChain-style, sentence-aware (default)
2. SemanticChunker — groups semantically related sentences
3. FixedSizeChunker — deterministic window-based (for baselines/testing)

Design notes:
- Overlap is critical: prevents context loss at chunk boundaries
- Metadata propagation: each chunk knows its parent document + position
- Chunk IDs are deterministic for deduplication
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Field

from config.settings import settings
from ingestion.document_loader import RawDocument
from observability.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Domain models
# ─────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A text chunk ready for embedding and vector storage."""
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int              # Position within the parent document
    total_chunks: int             # Total chunks from parent document
    char_start: int               # Character offset in original document
    char_end: int
    strategy: str                 # Which chunking strategy produced this
    metadata: dict = Field(default_factory=dict)  # Inherited doc metadata

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        return len(self.content)


def _make_chunk_id(doc_id: str, chunk_index: int, content: str) -> str:
    h = hashlib.md5(f"{doc_id}:{chunk_index}:{content[:50]}".encode()).hexdigest()[:8]
    return f"chunk-{h}"


# ─────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────

class ChunkerBase(ABC):
    strategy_name: str = "base"

    @abstractmethod
    def chunk(self, document: RawDocument) -> List[DocumentChunk]:
        raise NotImplementedError

    def _build_chunks(
        self, doc: RawDocument, text_segments: List[str], offsets: List[int]
    ) -> List[DocumentChunk]:
        """Convert raw text segments into DocumentChunk objects."""
        total = len(text_segments)
        chunks: List[DocumentChunk] = []
        for idx, (text, offset) in enumerate(zip(text_segments, offsets)):
            stripped = text.strip()
            if not stripped:
                continue
            chunk = DocumentChunk(
                chunk_id=_make_chunk_id(doc.doc_id, idx, stripped),
                doc_id=doc.doc_id,
                content=stripped,
                chunk_index=idx,
                total_chunks=total,
                char_start=offset,
                char_end=offset + len(text),
                strategy=self.strategy_name,
                metadata={
                    **doc.metadata.model_dump(),
                    "chunk_index": idx,
                },
            )
            chunks.append(chunk)
        return chunks


# ─────────────────────────────────────────────────────────────
# Strategy 1: Recursive Character Splitter (default)
# ─────────────────────────────────────────────────────────────

class RecursiveCharacterChunker(ChunkerBase):
    """
    Splits on paragraph → sentence → word boundaries in order.
    This is the LangChain RecursiveCharacterTextSplitter approach,
    reimplemented without the LangChain dependency here for the core layer.
    """

    strategy_name = "recursive_character"

    # Ordered from coarse to fine: try each separator in sequence
    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: RawDocument) -> List[DocumentChunk]:
        text = document.content
        raw_chunks = self._split_text(text)

        segments: List[str] = []
        offsets: List[int] = []
        pos = 0
        for chunk_text in raw_chunks:
            idx = text.find(chunk_text, pos)
            offset = idx if idx != -1 else pos
            segments.append(chunk_text)
            offsets.append(offset)
            pos = offset + len(chunk_text) - self.chunk_overlap

        chunks = self._build_chunks(document, segments, offsets)
        logger.debug(
            f"Chunked '{document.metadata.filename}'",
            extra={"strategy": self.strategy_name, "chunks": len(chunks)},
        )
        return chunks

    def _split_text(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        if separators is None:
            separators = self.SEPARATORS

        separator = separators[-1]
        for sep in separators:
            if sep == "" or sep in text:
                separator = sep
                break

        splits = text.split(separator) if separator else list(text)
        good_splits: List[str] = []
        current: List[str] = []
        current_len = 0

        for s in splits:
            s_len = len(s)
            if current_len + s_len + (len(separator) if current else 0) > self.chunk_size:
                if current:
                    merged = self._merge_splits(current, separator)
                    good_splits.extend(merged)
                    # Keep overlap: retain trailing splits that fit in overlap window
                    while current and current_len > self.chunk_overlap:
                        removed = current.pop(0)
                        current_len -= len(removed) + len(separator)
                current.append(s)
                current_len = sum(len(x) for x in current) + len(separator) * (len(current) - 1)
            else:
                current.append(s)
                current_len += s_len + (len(separator) if len(current) > 1 else 0)

        if current:
            good_splits.extend(self._merge_splits(current, separator))

        return good_splits

    @staticmethod
    def _merge_splits(splits: List[str], separator: str) -> List[str]:
        """Join splits that are smaller than the chunk size."""
        return [separator.join(splits)]


# ─────────────────────────────────────────────────────────────
# Strategy 2: Fixed-Size Chunker
# ─────────────────────────────────────────────────────────────

class FixedSizeChunker(ChunkerBase):
    """
    Deterministic sliding window chunker.
    Used for ablation studies and baseline comparisons in evaluation pipelines.
    """

    strategy_name = "fixed_size"

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: RawDocument) -> List[DocumentChunk]:
        text = document.content
        step = self.chunk_size - self.chunk_overlap
        segments: List[str] = []
        offsets: List[int] = []

        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            segments.append(text[start:end])
            offsets.append(start)
            start += step

        return self._build_chunks(document, segments, offsets)


# ─────────────────────────────────────────────────────────────
# Strategy 3: Sentence-Aware Chunker
# ─────────────────────────────────────────────────────────────

class SentenceChunker(ChunkerBase):
    """
    Splits on sentence boundaries, then groups sentences to fill chunk_size.
    Preserves semantic coherence better than character-based methods. 
    Most useful for narrative documents (e.g., annual reports).
    """

    strategy_name = "sentence_aware"

    # Regex: split on ., !, ? followed by whitespace/end
    SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: RawDocument) -> List[DocumentChunk]:
        sentences = self.SENTENCE_BOUNDARY.split(document.content)
        groups: List[str] = []
        offsets: List[int] = []
        current_group: List[str] = []
        current_len = 0
        current_offset = 0
        search_from = 0

        for sentence in sentences:
            sent_len = len(sentence)
            if current_len + sent_len > self.chunk_size and current_group:
                joined = " ".join(current_group)
                groups.append(joined)
                offsets.append(current_offset)
                # Overlap: keep last sentence(s) that fit in overlap window
                while current_group and current_len > self.chunk_overlap:
                    removed = current_group.pop(0)
                    current_len -= len(removed) + 1
                current_offset = document.content.find(
                    current_group[0] if current_group else sentence, search_from
                )

            if not current_group:
                current_offset = document.content.find(sentence, search_from)
                search_from = max(current_offset, search_from)

            current_group.append(sentence)
            current_len += sent_len + 1  # +1 for space separator
            search_from += sent_len

        if current_group:
            groups.append(" ".join(current_group))
            offsets.append(current_offset)

        return self._build_chunks(document, groups, offsets)


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

CHUNKER_REGISTRY = {
    "recursive_character": RecursiveCharacterChunker,
    "fixed_size": FixedSizeChunker,
    "sentence_aware": SentenceChunker,
}


def get_chunker(strategy: str = "recursive_character", **kwargs) -> ChunkerBase:
    """
    Return a chunker instance by strategy name.
    This allows pipeline config to drive strategy selection.
    """
    cls = CHUNKER_REGISTRY.get(strategy)
    if not cls:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Available: {list(CHUNKER_REGISTRY.keys())}"
        )
    return cls(**kwargs)
