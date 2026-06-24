from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from agents import (
    Agent,
    ModelSettings,
    RunContextWrapper,
    Runner,
    set_tracing_disabled,
)

from src.agent.context import AgentContext
from src.agent.hooks import AgentRunHooks
from src.agent.tools.agent_tools import tools
from src.memory import MemoryScope, MemoryStore
from src.model.agent_model import create_agent_model
from src.skills import SkillDefinition, SkillRegistry
from src.utils.config_handler import agent_conf
from src.utils.prompt_loader import load_system_prompt


def resolve_project_path(configured_path: str) -> Path:
    """将配置路径稳定解析到项目根目录。"""
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def resolve_session_db_path(configured_path: str) -> Path:
    return resolve_project_path(configured_path)


def dynamic_instructions(
    context: RunContextWrapper[AgentContext],
    _agent: Agent[AgentContext],
) -> str:
    """组合基础系统提示词与当前启用技能。"""
    base_prompt = load_system_prompt()

    if context.context.skill_prompt:
        return f"{base_prompt.rstrip()}\n\n{context.context.skill_prompt}"
    return base_prompt


class OpenAIAgent:
    """基于 OpenAI Agents SDK 的政企问答智能体。"""

    def __init__(self, memory_scope: MemoryScope | None = None):
        set_tracing_disabled(agent_conf.get("tracing_disabled", True))

        self.max_turns = agent_conf.get("max_turns", 5)
        self.hooks = AgentRunHooks()
        skill_root = resolve_project_path(
            agent_conf.get("skill_root", "src/skills")
        )
        self.skill_registry = SkillRegistry.from_directory(skill_root)
        configured_skills = agent_conf.get("enabled_skills", [])
        self.enabled_skills = self._resolve_skills(configured_skills)
        self.agent = Agent[AgentContext](
            name=agent_conf.get("name", "政企问答助手"),
            instructions=dynamic_instructions,
            model=create_agent_model(),
            model_settings=ModelSettings(
                parallel_tool_calls=agent_conf.get("parallel_tool_calls", False),
            ),
            tools=tools,
        )

        self.memory_store = MemoryStore(
            resolve_session_db_path(
                agent_conf.get("session_db_path", "chat_history/agent_sessions.db")
            )
        )
        self.memory_scope = memory_scope or MemoryScope(
            tenant_id=agent_conf.get("default_tenant_id", "default"),
            user_id=agent_conf.get("default_user_id", "anonymous"),
            conversation_id=uuid4().hex,
        )
        self.session = None
        if agent_conf.get("session_enabled", True):
            self.session = self.memory_store.open_session(self.memory_scope)

    def _resolve_skills(
        self,
        enabled_skill_names,
    ) -> tuple[SkillDefinition, ...]:
        return self.skill_registry.resolve(
            enabled_skill_names,
            available_tool_names=(tool.name for tool in tools),
        )

    async def execute_stream(
        self,
        query: str,
        enabled_skills: list[str] | tuple[str, ...] | None = None,
    ):
        """执行智能体并仅向前端转发最终回答的文本增量。"""
        selected_skills = (
            self.enabled_skills
            if enabled_skills is None
            else self._resolve_skills(enabled_skills)
        )
        context = AgentContext(
            enabled_skills=tuple(skill.name for skill in selected_skills),
            enabled_tools=frozenset(
                tool_name
                for skill in selected_skills
                for tool_name in skill.required_tools
            ),
            skill_prompt=self.skill_registry.build_prompt(selected_skills),
            tenant_id=self.memory_scope.tenant_id,
            user_id=self.memory_scope.user_id,
            conversation_id=self.memory_scope.conversation_id,
        )
        result = Runner.run_streamed(
            self.agent,
            input=query,
            context=context,
            hooks=self.hooks,
            max_turns=self.max_turns,
            session=self.session,
        )

        async for event in result.stream_events():
            if event.type != "raw_response_event":
                continue
            if getattr(event.data, "type", None) == "response.output_text.delta":
                delta = getattr(event.data, "delta", "")
                if delta:
                    yield delta

        if result.run_loop_exception:
            raise result.run_loop_exception

    def close(self) -> None:
        if self.session is not None:
            self.session.close()

    async def clear_memory(self) -> None:
        if self.session is None:
            return
        await self.session.clear_session()
