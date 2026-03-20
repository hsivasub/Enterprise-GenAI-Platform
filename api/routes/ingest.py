"""
Ingest Routes — API Gateway
=============================
Handles document upload and ingestion into the knowledge base.
Supports both file upload and URL-based ingestion.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from api.dependencies import get_vector_store, verify_api_key
from config.settings import settings
from ingestion.pipeline import IngestionPipeline, IngestionResult
from observability.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_created: int
    chunks_added: int
    total_chars: int
    elapsed_seconds: float
    strategy: str
    success: bool
    error: str | None = None


class BatchIngestResponse(BaseModel):
    total_files: int
    successful: int
    failed: int
    results: List[IngestResponse]


@router.post(
    "/file",
    response_model=IngestResponse,
    summary="Ingest a single document",
    description="Upload a PDF, TXT, MD, or DOCX file to add to the knowledge base.",
)
async def ingest_file(
    file: UploadFile = File(..., description="Document to ingest"),
    chunking_strategy: str = "recursive_character",
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
    _api_key: str = Depends(verify_api_key),
):
    # Validate file extension
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in settings.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext}. Supported: {settings.SUPPORTED_EXTENSIONS}",
        )

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        if len(content) > settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max: {settings.MAX_DOCUMENT_SIZE_MB} MB",
            )
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pipeline = IngestionPipeline(
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        result: IngestionResult = pipeline.ingest_file(tmp_path)

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": result.error, "filename": file.filename},
            )

        return IngestResponse(
            doc_id=result.doc_id,
            filename=file.filename or "",
            chunks_created=result.chunks_created,
            chunks_added=result.chunks_added,
            total_chars=result.total_chars,
            elapsed_seconds=result.elapsed_seconds,
            strategy=result.strategy,
            success=result.success,
        )
    finally:
        os.unlink(tmp_path)   # Always clean up temp file


@router.delete(
    "/{doc_id}",
    summary="Remove a document from the knowledge base",
)
async def delete_document(
    doc_id: str,
    vector_store=Depends(get_vector_store),
    _api_key: str = Depends(verify_api_key),
):
    removed = vector_store.delete_by_doc_id(doc_id)
    if removed == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found in vector store",
        )
    vector_store.save()
    return {"doc_id": doc_id, "chunks_removed": removed, "status": "deleted"}


@router.get(
    "/stats",
    summary="Get knowledge base statistics",
)
async def get_stats(
    vector_store=Depends(get_vector_store),
    _api_key: str = Depends(verify_api_key),
):
    return {
        "total_vectors": vector_store.total_vectors,
        "index_path": vector_store.index_path,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
    }
