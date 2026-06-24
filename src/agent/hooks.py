from agents import Agent, RunContextWrapper, RunHooks

from src.agent.context import AgentContext
from src.utils.logger_handler import logger


class AgentRunHooks(RunHooks[AgentContext]):
    """记录模型与工具调用生命周期。"""

    async def on_llm_start(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        system_prompt: str | None,
        input_items: list,
    ) -> None:
        logger.info(
            f"[agent hook]:准备调用模型，agent={agent.name}，输入项数量={len(input_items)}"
        )

    async def on_tool_start(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool,
    ) -> None:
        logger.info(f"[agent hook]:执行工具：{tool.name}")
        tool_arguments = getattr(context, "tool_arguments", None)
        if tool_arguments:
            logger.info(f"[agent hook]:工具参数：{tool_arguments}")

    async def on_tool_end(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool,
        result: object,
    ) -> None:
        logger.info(f"[agent hook]:工具 {tool.name} 执行成功")
