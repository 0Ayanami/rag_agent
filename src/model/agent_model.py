from __future__ import annotations

import os
from dataclasses import dataclass

from agents import OpenAIChatCompletionsModel as CompatibleChatCompletionsModel
from dotenv import load_dotenv
from openai import AsyncOpenAI as AsyncCompatibleClient


class ModelConfigurationError(ValueError):
    """模型环境变量缺失或不合法。"""


@dataclass(frozen=True)
class ModelConfig:
    """OpenAI-compatible 模型连接配置。

    这里的 OpenAI-compatible 仅指接口协议兼容，供应商可以是 DeepSeek、
    OpenAI、Ollama 或其他提供 Chat Completions 兼容接口的服务。
    """

    model_name: str
    base_url: str
    api_key: str

    @classmethod
    def from_env(cls) -> "ModelConfig":
        load_dotenv()
        values = {
            "model_name": _read_env("MODEL_NAME", "LM_MODEL"),
            "base_url": _read_env("MODEL_BASE_URL", "BASE_URL"),
            "api_key": _read_env("MODEL_API_KEY", "API_KEY"),
        }
        missing = [
            display_name
            for field_name, display_name in (
                ("model_name", "MODEL_NAME（或 LM_MODEL）"),
                ("base_url", "MODEL_BASE_URL（或 BASE_URL）"),
                ("api_key", "MODEL_API_KEY（或 API_KEY）"),
            )
            if not values[field_name]
        ]
        if missing:
            raise ModelConfigurationError(
                "缺少模型环境变量：" + "、".join(missing)
            )

        return cls(**values)


def _read_env(primary_name: str, legacy_name: str) -> str:
    """读取新变量名，并兼容项目已有的变量名。"""
    return (
        os.getenv(primary_name, "").strip()
        or os.getenv(legacy_name, "").strip()
    )


def create_agent_model() -> CompatibleChatCompletionsModel:
    """根据 .env 创建 Agents SDK 模型，不读取任何业务 YAML 配置。"""
    config = ModelConfig.from_env()
    client = AsyncCompatibleClient(
        api_key=config.api_key,
        base_url=config.base_url,
    )
    return CompatibleChatCompletionsModel(
        model=config.model_name,
        openai_client=client,
    )
