from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


class SkillValidationError(ValueError):
    """技能声明不合法。"""


@dataclass(frozen=True)
class SkillDefinition:
    """一个经过校验、不可变的声明式技能。"""

    name: str
    description: str
    version: str
    instructions: str
    required_tools: tuple[str, ...]
    source_path: Path

    def to_prompt_section(self) -> str:
        tools = "、".join(self.required_tools) if self.required_tools else "无"
        return (
            f"### 技能：{self.name}（v{self.version}）\n"
            f"用途：{self.description}\n"
            f"可使用工具：{tools}\n"
            f"{self.instructions.strip()}"
        )


class SkillRegistry:
    """从本地 SKILL.md 目录加载并校验声明式技能。"""

    def __init__(self, skills: Iterable[SkillDefinition] = ()):
        self._skills: dict[str, SkillDefinition] = {}
        for skill in skills:
            if skill.name in self._skills:
                raise SkillValidationError(f"技能名称重复：{skill.name}")
            self._skills[skill.name] = skill

    @classmethod
    def from_directory(cls, root: str | Path) -> "SkillRegistry":
        root_path = Path(root)
        if not root_path.exists():
            raise SkillValidationError(f"技能目录不存在：{root_path}")
        if not root_path.is_dir():
            raise SkillValidationError(f"技能路径不是目录：{root_path}")

        skills = [
            cls._load_skill_file(path)
            for path in sorted(root_path.glob("*/SKILL.md"))
        ]
        return cls(skills)

    @staticmethod
    def _load_skill_file(path: Path) -> SkillDefinition:
        text = path.read_text(encoding="utf-8-sig")
        if not text.startswith("---"):
            raise SkillValidationError(f"{path} 缺少 YAML front matter")

        parts = text.split("---", 2)
        if len(parts) != 3:
            raise SkillValidationError(f"{path} 的 YAML front matter 未闭合")

        metadata = yaml.safe_load(parts[1]) or {}
        if not isinstance(metadata, dict):
            raise SkillValidationError(f"{path} 的技能元数据必须是对象")

        name = metadata.get("name")
        description = metadata.get("description")
        version = metadata.get("version")
        tools = metadata.get("tools", [])
        instructions = parts[2].strip()

        if not isinstance(name, str) or not name.strip():
            raise SkillValidationError(f"{path} 缺少有效的 name")
        if not isinstance(description, str) or not description.strip():
            raise SkillValidationError(f"{path} 缺少有效的 description")
        if not isinstance(version, str) or not version.strip():
            raise SkillValidationError(f"{path} 缺少有效的 version")
        if not isinstance(tools, list) or not all(
            isinstance(tool, str) and tool.strip() for tool in tools
        ):
            raise SkillValidationError(f"{path} 的 tools 必须是非空字符串列表")
        if not instructions:
            raise SkillValidationError(f"{path} 缺少技能指令正文")

        normalized_tools = tuple(dict.fromkeys(tool.strip() for tool in tools))
        return SkillDefinition(
            name=name.strip(),
            description=description.strip(),
            version=version.strip(),
            instructions=instructions,
            required_tools=normalized_tools,
            source_path=path.resolve(),
        )

    def names(self) -> tuple[str, ...]:
        return tuple(self._skills)

    def get(self, name: str) -> SkillDefinition:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise SkillValidationError(f"未找到技能：{name}") from exc

    def resolve(
        self,
        enabled_names: Iterable[str],
        available_tool_names: Iterable[str],
    ) -> tuple[SkillDefinition, ...]:
        available_tools = set(available_tool_names)
        resolved: list[SkillDefinition] = []

        for name in dict.fromkeys(enabled_names):
            skill = self.get(name)
            missing_tools = set(skill.required_tools) - available_tools
            if missing_tools:
                missing = "、".join(sorted(missing_tools))
                raise SkillValidationError(
                    f"技能 {skill.name} 缺少所需工具：{missing}"
                )
            resolved.append(skill)

        return tuple(resolved)

    @staticmethod
    def build_prompt(skills: Iterable[SkillDefinition]) -> str:
        sections = [skill.to_prompt_section() for skill in skills]
        if not sections:
            return ""
        return "## 当前启用技能\n\n" + "\n\n".join(sections)
