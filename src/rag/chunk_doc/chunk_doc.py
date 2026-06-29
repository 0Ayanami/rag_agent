from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from rag.chunk_doc.chunk_protocols import (
    Chunker,
    DocumentCleaner,
    MetadataBuilder,
    SectionSplitter,
    SentenceSplitter,
    StructureParser,
    Tokenizer,
)
from rag.chunk_doc.chunk_schema import (
    Chunk,
    ChunkDraft,
    CleanedDocument,
    Document,
    DocumentChunk,
    Heading,
    ParsedDocument,
    Section,
    Sentence,
    SourceDocument,
)


SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".markdown"})
_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMERIC_HEADING = re.compile(r"^(\d+(?:\.\d+)*\.?)(?:\s+(.+))?$")
_ROMAN_HEADING = re.compile(
    r"^(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))\.\s*(.*)$",
    re.I,
)
_SENTENCE_END = frozenset(".!?。！？;；")
_ABBREVIATIONS = (
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "Sr.",
    "Jr.",
    "St.",
    "e.g.",
    "i.e.",
    "Fig.",
    "Eq.",
    "No.",
    "vs.",
    "etc.",
)


@dataclass(frozen=True)
class ChunkPipelineConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_tokens: int = 0

    @property
    def max_tokens(self) -> int:
        return self.chunk_size

    @property
    def overlap_tokens(self) -> int:
        return self.chunk_overlap

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")
        if self.min_chunk_tokens < 0 or self.min_chunk_tokens > self.chunk_size:
            raise ValueError("min_chunk_tokens 必须介于 0 与 chunk_size 之间")


class RegexTokenizer:
    """默认轻量 tokenizer；生产环境可替换为 tiktoken/HF/Qwen tokenizer。"""

    def count(self, text: str) -> int:
        return len(_TOKEN_PATTERN.findall(text))

    def split_by_tokens(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        tokens = list(_TOKEN_PATTERN.finditer(text))
        if not tokens:
            return []
        chunks: list[str] = []
        step = max(1, max_tokens - overlap_tokens)
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            char_start = tokens[start].start()
            char_end = tokens[end - 1].end()
            chunk = text[char_start:char_end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(tokens):
                break
            start += step
        return chunks


RegexTokenCounter = RegexTokenizer


class DefaultDocumentCleaner:
    """做格式规范化，不改写正文语义。"""

    def clean(self, document: Document) -> CleanedDocument:
        text = unicodedata.normalize("NFC", document.text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._remove_control_characters(text)
        text = self._remove_repeated_headers_and_footers(text)
        text = self._normalize_spaces(text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return CleanedDocument(
            id=document.id,
            source=document.source,
            text=text,
            page=document.page,
            title=document.title,
        )

    @staticmethod
    def _remove_control_characters(text: str) -> str:
        return "".join(
            char
            for char in text
            if char in "\n\t\f" or (ord(char) >= 32 and char != "\ufffd")
        )

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        lines = []
        for line in text.split("\n"):
            line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _remove_repeated_headers_and_footers(text: str) -> str:
        pages = text.split("\f")
        if len(pages) < 2:
            return text
        first_lines = [_edge_line(page, first=True) for page in pages]
        last_lines = [_edge_line(page, first=False) for page in pages]
        repeated = {
            line
            for line in first_lines + last_lines
            if line and (first_lines + last_lines).count(line) >= 2
        }
        if not repeated:
            return "\n".join(pages)
        cleaned_pages = []
        for page in pages:
            lines = page.splitlines()
            while lines and lines[0].strip() in repeated:
                lines.pop(0)
            while lines and lines[-1].strip() in repeated:
                lines.pop()
            cleaned_pages.append("\n".join(lines))
        return "\n".join(cleaned_pages)


class HeadingStructureParser:
    """识别 Markdown、数字标题与 Roman 标题，只记录结构，不切块。"""

    def parse(self, document: CleanedDocument) -> ParsedDocument:
        headings: list[Heading] = []
        for line_number, line in enumerate(document.text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            heading = self._parse_heading(stripped, line_number)
            if heading is not None:
                headings.append(heading)
        return ParsedDocument(document=document, headings=tuple(headings))

    @staticmethod
    def _parse_heading(line: str, line_number: int) -> Heading | None:
        markdown = _MARKDOWN_HEADING.match(line)
        if markdown:
            return Heading(
                title=markdown.group(2).strip(),
                level=len(markdown.group(1)),
                line_number=line_number,
                kind="markdown",
            )
        numeric = _NUMERIC_HEADING.match(line)
        if numeric and (numeric.group(2) or "." in numeric.group(1)):
            title = numeric.group(2) or numeric.group(1).rstrip(".")
            number = numeric.group(1).rstrip(".")
            return Heading(
                title=title.strip(),
                level=number.count(".") + 1,
                line_number=line_number,
                kind="numeric",
            )
        roman = _ROMAN_HEADING.match(line)
        if roman and roman.group(1):
            return Heading(
                title=(roman.group(5) or roman.group(1)).strip(),
                level=1,
                line_number=line_number,
                kind="roman",
            )
        return None


class HeadingSectionSplitter:
    """按标题边界切分 section，每个 section 保持完整。"""

    def split(self, document: ParsedDocument) -> tuple[Section, ...]:
        lines = document.document.text.splitlines()
        if not lines:
            return ()
        if not document.headings:
            return (
                Section(
                    text=document.document.text,
                    title=document.document.title,
                    level=0,
                    start_line=1,
                    end_line=len(lines),
                    hierarchy=(),
                ),
            )

        sections: list[Section] = []
        hierarchy: list[str] = []
        heading_by_line = {heading.line_number: heading for heading in document.headings}

        if document.headings[0].line_number > 1:
            preface = "\n".join(lines[:document.headings[0].line_number - 1]).strip()
            if preface:
                sections.append(
                    Section(
                        text=preface,
                        title=document.document.title,
                        level=0,
                        start_line=1,
                        end_line=document.headings[0].line_number - 1,
                        hierarchy=(),
                    )
                )

        for index, heading in enumerate(document.headings):
            next_line = (
                document.headings[index + 1].line_number
                if index + 1 < len(document.headings)
                else len(lines) + 1
            )
            section_lines = lines[heading.line_number - 1:next_line - 1]
            text = "\n".join(section_lines).strip()
            if not text:
                continue
            hierarchy = hierarchy[:heading.level - 1]
            hierarchy.append(heading.title)
            sections.append(
                Section(
                    text=text,
                    title=heading.title,
                    level=heading.level,
                    start_line=heading.line_number,
                    end_line=next_line - 1,
                    heading=heading_by_line.get(heading.line_number),
                    hierarchy=tuple(hierarchy),
                )
            )
        return tuple(sections)


class AbbreviationAwareSentenceSplitter:
    """保护 Dr./e.g./i.e./Fig./Eq. 等缩写的句子切分器。"""

    def split(self, section: Section) -> tuple[Sentence, ...]:
        sentences: list[Sentence] = []
        for paragraph in re.split(r"\n\s*\n", section.text):
            paragraph = " ".join(paragraph.split())
            if not paragraph:
                continue
            for sentence in self._split_paragraph(paragraph):
                if sentence:
                    sentences.append(Sentence(text=sentence, section=section))
        return tuple(sentences)

    @staticmethod
    def _split_paragraph(paragraph: str) -> list[str]:
        protected, replacements = _protect_abbreviations(paragraph)
        output: list[str] = []
        start = 0
        index = 0
        while index < len(protected):
            char = protected[index]
            if char in _SENTENCE_END:
                next_index = index + 1
                if next_index == len(protected) or protected[next_index].isspace():
                    sentence = protected[start:next_index].strip()
                    if sentence:
                        output.append(_restore_abbreviations(sentence, replacements))
                    start = next_index
            index += 1
        tail = protected[start:].strip()
        if tail:
            output.append(_restore_abbreviations(tail, replacements))
        return output


RegexSentenceSplitter = AbbreviationAwareSentenceSplitter


class TokenAwareChunker:
    """尽量保持句子完整；不跨 section；超长句按 token budget 切分。"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_tokens: int = 0,
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens

    def chunk(self, sentences: Iterable[Sentence]) -> tuple[ChunkDraft, ...]:
        drafts: list[ChunkDraft] = []
        buffer: list[Sentence] = []
        buffer_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.tokenizer.count(sentence.text)
            if sentence_tokens > self.chunk_size:
                if buffer:
                    drafts.append(self._make_chunk(buffer))
                    buffer = []
                    buffer_tokens = 0
                drafts.extend(self._split_long_sentence(sentence))
                continue

            if buffer and buffer_tokens + sentence_tokens > self.chunk_size:
                drafts.append(self._make_chunk(buffer))
                buffer, buffer_tokens = self._overlap_tail(buffer)
                if buffer and buffer_tokens + sentence_tokens > self.chunk_size:
                    buffer = []
                    buffer_tokens = 0

            buffer.append(sentence)
            buffer_tokens += sentence_tokens

        if buffer:
            if (
                drafts
                and self.min_chunk_tokens
                and buffer_tokens < self.min_chunk_tokens
                and drafts[-1].token_count + buffer_tokens <= self.chunk_size
            ):
                previous = drafts.pop()
                drafts.append(self._make_chunk(list(previous.sentences + tuple(buffer))))
            else:
                drafts.append(self._make_chunk(buffer))

        return tuple(draft for draft in drafts if draft.text.strip())

    def _make_chunk(self, sentences: list[Sentence]) -> ChunkDraft:
        text = "\n".join(sentence.text for sentence in sentences).strip()
        return ChunkDraft(
            text=text,
            sentences=tuple(sentences),
            token_count=self.tokenizer.count(text),
        )

    def _overlap_tail(self, sentences: list[Sentence]) -> tuple[list[Sentence], int]:
        if self.chunk_overlap <= 0:
            return [], 0
        tail: list[Sentence] = []
        tokens = 0
        for sentence in reversed(sentences):
            sentence_tokens = self.tokenizer.count(sentence.text)
            if tail and tokens + sentence_tokens > self.chunk_overlap:
                break
            tail.append(sentence)
            tokens += sentence_tokens
        tail.reverse()
        return tail, tokens

    def _split_long_sentence(self, sentence: Sentence) -> list[ChunkDraft]:
        parts = self.tokenizer.split_by_tokens(
            sentence.text,
            self.chunk_size,
            self.chunk_overlap,
        )
        return [
            ChunkDraft(
                text=part,
                sentences=(Sentence(text=part, section=sentence.section),),
                token_count=self.tokenizer.count(part),
            )
            for part in parts
            if part.strip()
        ]


class DefaultMetadataBuilder:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def build_all(
        self,
        document: Document,
        drafts: tuple[ChunkDraft, ...],
    ) -> list[Chunk]:
        total_chunks = len(drafts)
        chunks: list[Chunk] = []
        for index, draft in enumerate(drafts, start=1):
            section = draft.sentences[0].section if draft.sentences else None
            chunk_digest = hashlib.sha256(draft.text.encode("utf-8")).hexdigest()[:16]
            section_title = section.title if section else ""
            metadata: dict[str, str | int | None] = {
                "source": document.source,
                "page": document.page,
                "title": document.title,
                "section": section_title,
                "chunk_index": index,
                "total_chunks": total_chunks,
                "token_count": draft.token_count,
                "character_count": len(draft.text),
                "hierarchy": " > ".join(section.hierarchy) if section else "",
                "parent_chunk": None,
                "children_chunk": "",
                "start_line": section.start_line if section else None,
                "end_line": section.end_line if section else None,
                "source_name": Path(document.source).name,
                "document_id": document.id,
            }
            chunks.append(
                Chunk(
                    id=f"{document.id}:{index}:{chunk_digest}",
                    text=draft.text,
                    metadata=metadata,
                )
            )
        return chunks


@dataclass
class ChunkPipeline:
    config: ChunkPipelineConfig = field(default_factory=ChunkPipelineConfig)
    cleaner: DocumentCleaner = field(default_factory=DefaultDocumentCleaner)
    parser: StructureParser = field(default_factory=HeadingStructureParser)
    tokenizer: Tokenizer = field(default_factory=RegexTokenizer)
    section_splitter: SectionSplitter = field(default_factory=HeadingSectionSplitter)
    sentence_splitter: SentenceSplitter = field(
        default_factory=AbbreviationAwareSentenceSplitter
    )
    chunker: Chunker | None = None
    metadata_builder: MetadataBuilder | None = None

    def __post_init__(self) -> None:
        self.config.validate()
        if self.chunker is None:
            self.chunker = TokenAwareChunker(
                tokenizer=self.tokenizer,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                min_chunk_tokens=self.config.min_chunk_tokens,
            )
        if self.metadata_builder is None:
            self.metadata_builder = DefaultMetadataBuilder(self.tokenizer)

    def __call__(self, document: Document) -> list[Chunk]:
        return self.run(document)

    def run(self, document: Document) -> list[Chunk]:
        cleaned = self.cleaner.clean(document)
        if not cleaned.text:
            return []
        parsed = self.parser.parse(cleaned)
        sections = self.section_splitter.split(parsed)
        drafts: list[ChunkDraft] = []
        for section in sections:
            sentences = self.sentence_splitter.split(section)
            drafts.extend(self.chunker.chunk(sentences))
        return self.metadata_builder.build_all(document, tuple(drafts))


def iter_source_files(source_paths: Iterable[Path]) -> Iterable[Path]:
    for source_path in source_paths:
        if source_path.is_file() and source_path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield source_path
        elif source_path.is_dir():
            for path in sorted(source_path.rglob("*")):
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                    yield path


def load_documents(source_paths: Iterable[Path]) -> list[Document]:
    documents: list[Document] = []
    for path in iter_source_files(source_paths):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        source = str(path)
        document_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        documents.append(
            Document(
                id=document_id,
                source=source,
                text=text,
                page=None,
                title=path.stem,
            )
        )
    return documents


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    pipeline = ChunkPipeline(
        config=ChunkPipelineConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_tokens=min(80, chunk_size),
        )
    )
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(pipeline(document))
    return chunks


def _edge_line(page: str, *, first: bool) -> str:
    lines = [line.strip() for line in page.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[0] if first else lines[-1]


def _protect_abbreviations(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}
    protected = text
    for index, abbreviation in enumerate(_ABBREVIATIONS):
        key = f"__ABBR_{index}__"
        if abbreviation in protected:
            replacements[key] = abbreviation
            protected = protected.replace(abbreviation, key)
    return protected, replacements


def _restore_abbreviations(text: str, replacements: dict[str, str]) -> str:
    restored = text
    for key, abbreviation in replacements.items():
        restored = restored.replace(key, abbreviation)
    return restored
