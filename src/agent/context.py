from dataclasses import dataclass


@dataclass
class AgentContext:
    """单次 Agent 运行期间可被工具修改的上下文。"""

    enabled_skills: tuple[str, ...] = ()
    enabled_tools: frozenset[str] = frozenset()
    skill_prompt: str = ""
    tenant_id: str = ""
    user_id: str = ""
    conversation_id: str = ""
