from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from agents import SQLiteSession


class MemoryValidationError(ValueError):
    """Memory 身份或配置不合法。"""


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _validate_identifier(field_name: str, value: str) -> str:
    normalized = value.strip()
    if not _IDENTIFIER_PATTERN.fullmatch(normalized):
        raise MemoryValidationError(
            f"{field_name} 必须为 1-64 位字母、数字、点、下划线或连字符，"
            "且必须以字母或数字开头"
        )
    return normalized


@dataclass(frozen=True)
class MemoryScope:
    """政企会话记忆的隔离边界。"""

    tenant_id: str
    user_id: str
    conversation_id: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "tenant_id",
            _validate_identifier("tenant_id", self.tenant_id),
        )
        object.__setattr__(
            self,
            "user_id",
            _validate_identifier("user_id", self.user_id),
        )
        object.__setattr__(
            self,
            "conversation_id",
            _validate_identifier("conversation_id", self.conversation_id),
        )

    def session_id(self) -> str:
        """生成不暴露原始身份信息的稳定会话键。"""
        canonical = (
            f"v1\0{self.tenant_id}\0{self.user_id}\0{self.conversation_id}"
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"enterprise:v1:{digest}"


class MemoryStore:
    """OpenAI Agents SDK SQLiteSession 的多租户工厂。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def open_session(self, scope: MemoryScope) -> SQLiteSession:
        return SQLiteSession(
            session_id=scope.session_id(),
            db_path=os.fspath(self.db_path),
        )

    async def clear(self, scope: MemoryScope) -> None:
        session = self.open_session(scope)
        try:
            await session.clear_session()
        finally:
            session.close()
