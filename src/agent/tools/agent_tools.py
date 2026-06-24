from __future__ import annotations

import json
from typing import Literal

from agents import RunContextWrapper, function_tool

from src.agent.context import AgentContext
from src.utils.config_handler import enterprise_faq_conf


def tool_enabled(tool_name: str):
    """按当前请求启用的技能限制模型可见工具。"""

    def is_enabled(context: RunContextWrapper[AgentContext], _agent) -> bool:
        return tool_name in context.context.enabled_tools

    return is_enabled


def _faq_score(question: str, entry: dict) -> int:
    normalized = question.lower()
    score = 0
    for keyword in entry.get("keywords", []):
        keyword_text = str(keyword).lower()
        if keyword_text and keyword_text in normalized:
            score += max(len(keyword_text), 1)
    if entry.get("category", "").lower() in normalized:
        score += 3
    return score


@function_tool(
    description_override=(
        "查询内置政企办事与制度 FAQ。适用于账号权限、行政办公、采购、"
        "财务、数据安全、信息安全和服务支持等原型场景。"
    ),
    is_enabled=tool_enabled("search_enterprise_faq"),
)
def search_enterprise_faq(question: str) -> str:
    entries = enterprise_faq_conf.get("entries", [])
    ranked = sorted(
        (
            (_faq_score(question, entry), entry)
            for entry in entries
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    matches = [entry for score, entry in ranked if score > 0][:3]
    if not matches:
        return (
            "未找到直接匹配的制度条目。请补充事项类型、涉及系统或部门；"
            "也可以使用 draft_service_request 生成服务工单草稿。"
        )

    return json.dumps(
        [
            {
                "id": entry["id"],
                "category": entry["category"],
                "question": entry["question"],
                "answer": entry["answer"],
            }
            for entry in matches
        ],
        ensure_ascii=False,
    )


@function_tool(
    description_override=(
        "返回当前请求的租户、用户和会话标识，用于确认问答上下文边界。"
    ),
    is_enabled=tool_enabled("get_request_identity"),
)
def get_request_identity(
    context: RunContextWrapper[AgentContext],
) -> str:
    return json.dumps(
        {
            "tenant_id": context.context.tenant_id,
            "user_id": context.context.user_id,
            "conversation_id": context.context.conversation_id,
        },
        ensure_ascii=False,
    )


@function_tool(
    description_override=(
        "为无法直接解决的政企服务事项生成工单草稿。只生成草稿，"
        "不会发送、提交或变更任何外部系统。"
    ),
    is_enabled=tool_enabled("draft_service_request"),
)
def draft_service_request(
    context: RunContextWrapper[AgentContext],
    title: str,
    description: str,
    priority: Literal["低", "普通", "高", "紧急"] = "普通",
) -> str:
    return json.dumps(
        {
            "status": "draft",
            "tenant_id": context.context.tenant_id,
            "requester": context.context.user_id,
            "title": title.strip(),
            "description": description.strip(),
            "priority": priority,
            "notice": "该工单尚未提交，请人工确认内容后再进入正式流程。",
        },
        ensure_ascii=False,
    )


tools = [
    search_enterprise_faq,
    get_request_identity,
    draft_service_request,
]
