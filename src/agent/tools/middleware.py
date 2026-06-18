from langchain.agents.middleware import ModelRequest, before_model, dynamic_prompt, wrap_tool_call
from langchain.agents import AgentState
from langgraph.runtime import Runtime
from langchain.tools.tool_node import ToolCallRequest
from typing import Callable
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from src.utils.logger_handler import logger
from src.utils.prompt_loader import load_system_prompt, load_report_prompt

@wrap_tool_call
def monitor_tools(request:ToolCallRequest, handler: Callable[[ToolCallRequest], Command | ToolMessage]):
    """
    工具执行的监控
    """
    logger.info(f"[tool monitor]:执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]:工具参数：{request.tool_call['args']}")
    try:
        result = handler(request)
        logger.info(f"[tool monitor]:工具{request.tool_call['name']}执行成功")
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.error(f"[tool monitor]:工具{request.tool_call['name']}执行失败，原因：{str(e)}")
        raise e

@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    logger.info(f"[log before model]:准备执行模型，带有{len(state['messages'])}条消息")
    logger.debug(f"[log before model]:{type(state['messages'][-1]).__name___} | {state['messages'][-1].content.strip()}")
    return None

@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    """
    根据上下文动态切换提示词
    """
    if request.runtime.context.get("report", False):
        logger.info("[dynamic prompt]:已切换为报告生成的提示词")
        return load_report_prompt()
    else:
        logger.info("[dynamic prompt]:已切换为默认的系统提示词")
        return load_system_prompt()
    
middlewares = [monitor_tools, log_before_model, report_prompt_switch]
    