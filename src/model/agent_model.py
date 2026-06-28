from __future__ import annotations

import os
from dataclasses import dataclass

from agents import OpenAIChatCompletionsModel as CompatibleChatCompletionsModel
from dotenv import load_dotenv
from openai import AsyncOpenAI as AsyncCompatibleClient


class ModelConfigurationError(ValueError):
    """模型环境变量缺失或不合法。"""


def _first_env(*names: str) -> str:
    """Return the first non-empty environment value from a preference list."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


@dataclass(frozen=True)
class ModelConfig:
    """模型连接配置"""

    model_name: str
    base_url: str
    api_key: str

    @classmethod
    def from_env(cls) -> "ModelConfig":
        load_dotenv()
        values = {
            "model_name": _first_env("MODEL_NAME", "LM_MODEL"),
            "base_url": _first_env("MODEL_BASE_URL", "BASE_URL"),
            "api_key": _first_env("MODEL_API_KEY", "API_KEY"),
        }
        missing = [
            display_name
            for field_name, display_name in (
                ("model_name", "MODEL_NAME/LM_MODEL"),
                ("base_url", "MODEL_BASE_URL/BASE_URL"),
                ("api_key", "MODEL_API_KEY/API_KEY"),
            )
            if not values[field_name]
        ]
        if missing:
            raise ModelConfigurationError(
                "缺少模型环境变量：" + "、".join(missing)
            )

        return cls(**values)


def create_agent_model() -> CompatibleChatCompletionsModel:
    """根据 .env 创建 Agents SDK 模型。"""
    config = ModelConfig.from_env()
    client = AsyncCompatibleClient(
        api_key=config.api_key,
        base_url=config.base_url,
    )
    return CompatibleChatCompletionsModel(
        model=config.model_name,
        openai_client=client,
    )
