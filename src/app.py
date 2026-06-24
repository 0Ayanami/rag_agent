import asyncio
from uuid import uuid4

import streamlit as st

from src.agent.openai_agent import OpenAIAgent
from src.memory import MemoryScope, MemoryValidationError

# 标题
st.title("政企问答智能体")
st.caption("OpenAI Agents SDK 原型 · RAG 功能已暂停")
st.divider()

if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = uuid4().hex[:12]

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def start_new_conversation():
    old_agent = st.session_state.pop("agent", None)
    if old_agent is not None:
        old_agent.close()
    st.session_state.pop("agent_scope_key", None)
    st.session_state["conversation_id"] = uuid4().hex[:12]
    st.session_state["messages"] = []


tenant_id = st.sidebar.text_input("租户 ID", value="default")
user_id = st.sidebar.text_input("用户 ID", value="anonymous")
conversation_id = st.sidebar.text_input(
    "会话 ID",
    key="conversation_id",
)

st.sidebar.button(
    "新建会话",
    use_container_width=True,
    on_click=start_new_conversation,
)

try:
    memory_scope = MemoryScope(
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
    )
except MemoryValidationError as exc:
    st.error(f"会话身份配置错误：{exc}")
    st.stop()

scope_key = (
    memory_scope.tenant_id,
    memory_scope.user_id,
    memory_scope.conversation_id,
)
if st.session_state.get("agent_scope_key") != scope_key:
    old_agent = st.session_state.pop("agent", None)
    if old_agent is not None:
        old_agent.close()
    st.session_state["agent"] = OpenAIAgent(memory_scope=memory_scope)
    st.session_state["agent_scope_key"] = scope_key
    st.session_state["messages"] = []

agent = st.session_state["agent"]
available_skills = list(agent.skill_registry.names())
default_skills = [skill.name for skill in agent.enabled_skills]
selected_skills = st.sidebar.multiselect(
    "启用技能",
    options=available_skills,
    default=default_skills,
    help="技能会同时控制提示词能力和当前请求可使用的工具。",
)
st.sidebar.caption("当前原型不加载 LangChain、ChromaDB 或任何 RAG 组件。")

if st.sidebar.button("清空当前会话记忆", use_container_width=True):
    asyncio.run(agent.clear_memory())
    st.session_state["messages"] = []
    st.success("当前租户、用户和会话下的记忆已清空。")
    st.rerun()

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])  

prompt = st.chat_input("请输入办事、制度或服务支持问题：")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    cache_list = []
    with st.spinner("正在思考..."):
        response = agent.execute_stream(
            prompt,
            enabled_skills=selected_skills,
        )

        async def capture(generator, cache_list):
            async for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        try:
            st.chat_message("assistant").write_stream(capture(response, cache_list))
        except Exception as exc:
            st.error(f"Agent 执行失败：{exc}")
        else:
            st.session_state["messages"].append(
                {"role": "assistant", "content": "".join(cache_list)}
            )
            st.rerun()
