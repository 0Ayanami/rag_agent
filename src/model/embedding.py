from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx
from dotenv import load_dotenv
from openai import OpenAI


class EmbeddingConfigurationError(ValueError):
    """Embedding 模型环境变量缺失或不合法。"""


class EmbeddingRequestError(RuntimeError):
    """Embedding API 调用失败。"""


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _is_local_endpoint(base_url: str) -> bool:
    """检查给定的 base_url 是否是本地端点。
    Args:
        base_url: 要检查的 URL 字符串。
    Returns:
        如果 URL 是本地端点，则返回 True，否则返回 False。
    """
    normalized = base_url.lower()
    return any(host in normalized for host in ("localhost", "127.0.0.1", "::1"))


@dataclass(frozen=True)
class EmbeddingConfig:
    """OpenAI-compatible embedding 模型连接配置"""

    model_name: str
    base_url: str
    api_key: str

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        load_dotenv()
        model_name = _first_env("EMBEDDING_MODEL")
        base_url = _first_env("EMBEDDING_BASE_URL")
        api_key = _first_env("EMBEDDING_API_KEY")
        if base_url and not api_key and _is_local_endpoint(base_url):
            api_key = "ollama"

        missing = []
        if not model_name:
            missing.append("EMBEDDING_MODEL")
        if not base_url:
            missing.append("EMBEDDING_BASE_URL")
        if not api_key:
            missing.append("EMBEDDING_API_KEY")
        if missing:
            raise EmbeddingConfigurationError(
                "缺少 Embedding 环境变量：" + "、".join(missing)
            )

        return cls(model_name=model_name, base_url=base_url, api_key=api_key)


class OpenAICompatibleEmbeddingClient:
    """通过 OpenAI-compatible /v1/embeddings 接口生成向量。"""

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig.from_env()
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        self._is_local = _is_local_endpoint(self.config.base_url)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned = [text for text in texts if text.strip()]
        if len(cleaned) != len(texts):
            raise ValueError("Embedding 输入不能为空字符串")
        if not cleaned:
            return []

        try:
            response = self.client.embeddings.create(
                model=self.config.model_name,
                input=cleaned,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            if not self._is_local:
                raise EmbeddingRequestError(
                    f"Embedding API 调用失败，model={self.config.model_name}"
                ) from exc
            return self._embed_with_ollama_native(cleaned, exc)

    def _embed_with_ollama_native(
        self,
        texts: list[str],
        original_error: Exception,
    ) -> list[list[float]]:
        base_url = _ollama_base_url(self.config.base_url)
        errors: list[str] = [f"openai-compatible: {original_error}"]

        try:
            response = httpx.post(
                f"{base_url}/api/embed",
                json={"model": self.config.model_name, "input": texts},
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            embeddings = payload.get("embeddings")
            if _is_embedding_batch(embeddings, len(texts)):
                return embeddings
            errors.append("/api/embed: 响应中缺少 embeddings")
        except Exception as exc:
            errors.append(f"/api/embed: {exc}")

        legacy_embeddings: list[list[float]] = []
        try:
            for text in texts:
                response = httpx.post(
                    f"{base_url}/api/embeddings",
                    json={"model": self.config.model_name, "prompt": text},
                    timeout=120,
                )
                response.raise_for_status()
                payload = response.json()
                embedding = payload.get("embedding")
                if not _is_embedding(embedding):
                    raise EmbeddingRequestError("/api/embeddings 响应中缺少 embedding")
                legacy_embeddings.append(embedding)
            return legacy_embeddings
        except Exception as exc:
            errors.append(f"/api/embeddings: {exc}")

        raise EmbeddingRequestError(
            "Ollama embedding 调用失败，"
            f"model={self.config.model_name}，base_url={base_url}；"
            + "；".join(errors)
        ) from original_error


def _ollama_base_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")


def _is_embedding(value) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, int | float) for item in value
    )


def _is_embedding_batch(value, expected_len: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == expected_len
        and all(_is_embedding(item) for item in value)
    )
