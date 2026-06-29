from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    source: str
    text: str
    page: int | None = None
    title: str = ""


@dataclass(frozen=True)
class CleanedDocument:
    id: str
    source: str
    text: str
    page: int | None = None
    title: str = ""


@dataclass(frozen=True)
class Heading:
    title: str
    level: int
    line_number: int
    kind: str


@dataclass(frozen=True)
class ParsedDocument:
    document: CleanedDocument
    headings: tuple[Heading, ...]


@dataclass(frozen=True)
class Section:
    text: str
    title: str
    level: int
    start_line: int
    end_line: int
    heading: Heading | None = None
    hierarchy: tuple[str, ...] = ()


@dataclass(frozen=True)
class Sentence:
    text: str
    section: Section


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    sentences: tuple[Sentence, ...]
    token_count: int


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, str | int | None]


SourceDocument = Document
DocumentChunk = Chunk
