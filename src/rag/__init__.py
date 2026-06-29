"""RAG knowledge base support."""

from .chunk_doc.chunk_doc import ChunkPipeline, ChunkPipelineConfig
from .chunk_doc.chunk_schema import Chunk, Document
from .config import RagConfig
from src.model.embedding import EmbeddingConfig, OpenAICompatibleEmbeddingClient
from .service import RagService, get_rag_service

__all__ = [
    "Chunk",
    "ChunkPipeline",
    "ChunkPipelineConfig",
    "Document",
    "EmbeddingConfig",
    "OpenAICompatibleEmbeddingClient",
    "RagConfig",
    "RagService",
    "get_rag_service",
]
