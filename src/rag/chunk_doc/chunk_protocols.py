from __future__ import annotations

from typing import Iterable, Protocol

from rag.chunk_doc.chunk_schema import (
    Chunk,
    ChunkDraft,
    CleanedDocument,
    Document,
    ParsedDocument,
    Section,
    Sentence,
)


class Tokenizer(Protocol):
    def count(self, text: str) -> int:
        ...

    def split_by_tokens(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        ...


class DocumentCleaner(Protocol):
    def clean(self, document: Document) -> CleanedDocument:
        ...


class StructureParser(Protocol):
    def parse(self, document: CleanedDocument) -> ParsedDocument:
        ...


class SectionSplitter(Protocol):
    def split(self, document: ParsedDocument) -> tuple[Section, ...]:
        ...


class SentenceSplitter(Protocol):
    def split(self, section: Section) -> tuple[Sentence, ...]:
        ...


class Chunker(Protocol):
    def chunk(self, sentences: Iterable[Sentence]) -> tuple[ChunkDraft, ...]:
        ...


class MetadataBuilder(Protocol):
    def build_all(
        self,
        document: Document,
        drafts: tuple[ChunkDraft, ...],
    ) -> list[Chunk]:
        ...
