import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from agents import OpenAIChatCompletionsModel
from agents.tool_context import ToolContext
from openai import AsyncOpenAI

from src.agent.context import AgentContext
from src.agent.openai_agent import (
    OpenAIAgent,
    dynamic_instructions,
    resolve_session_db_path,
)
from src.agent.tools.agent_tools import (
    draft_service_request,
    get_request_identity,
    search_enterprise_faq,
    tools,
)
from src.memory import MemoryScope, MemoryStore, MemoryValidationError
from src.model.agent_model import (
    ModelConfig,
    ModelConfigurationError,
    create_agent_model,
)
from src.skills import SkillRegistry, SkillValidationError
from src.utils.prompt_loader import load_system_prompt


class FakeStreamingResult:
    run_loop_exception = None

    async def stream_events(self):
        yield SimpleNamespace(
            type="raw_response_event",
            data=SimpleNamespace(
                type="response.output_text.delta",
                delta="hello",
            ),
        )
        yield SimpleNamespace(
            type="run_item_stream_event",
            data=SimpleNamespace(type="tool_output"),
        )
        yield SimpleNamespace(
            type="raw_response_event",
            data=SimpleNamespace(
                type="response.output_text.delta",
                delta=" world",
            ),
        )


def create_tool_context(
    tool_name: str,
    arguments: str,
    *,
    enabled_tools: frozenset[str] | None = None,
) -> ToolContext:
    return ToolContext(
        context=AgentContext(
            enabled_tools=enabled_tools or frozenset({tool_name}),
            tenant_id="tenant-a",
            user_id="user-a",
            conversation_id="conversation-a",
        ),
        tool_name=tool_name,
        tool_call_id=f"{tool_name}-call",
        tool_arguments=arguments,
    )


def create_mock_openai_model() -> OpenAIChatCompletionsModel:
    def handler(_request: httpx.Request) -> httpx.Response:
        chunks = [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "SDK_OK"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            },
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        body += "data: [DONE]\n\n"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://mock.openai.local/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return OpenAIChatCompletionsModel(
        model="test-model",
        openai_client=client,
    )


class OpenAIAgentSdkTests(unittest.IsolatedAsyncioTestCase):
    def test_agent_startup_does_not_import_paused_rag_stacks(self):
        forbidden = [
            name
            for name in sys.modules
            if name.startswith(("langchain", "langgraph", "chromadb"))
        ]
        self.assertEqual(forbidden, [])

    def test_model_config_supports_existing_deepseek_env_names(self):
        with patch.dict(
            "os.environ",
            {
                "API_KEY": "deepseek-test-key",
                "BASE_URL": "https://api.deepseek.com/v1",
                "LM_MODEL": "deepseek-chat",
                "MODEL_API_KEY": "",
                "MODEL_BASE_URL": "",
                "MODEL_NAME": "",
            },
            clear=False,
        ):
            config = ModelConfig.from_env()

        self.assertEqual(config.api_key, "deepseek-test-key")
        self.assertEqual(config.base_url, "https://api.deepseek.com/v1")
        self.assertEqual(config.model_name, "deepseek-chat")

    def test_model_config_prefers_model_prefix_names(self):
        with patch.dict(
            "os.environ",
            {
                "MODEL_API_KEY": "provider-key",
                "MODEL_BASE_URL": "https://provider.example/v1",
                "MODEL_NAME": "provider-model",
                "API_KEY": "legacy-key",
                "BASE_URL": "https://legacy.example/v1",
                "LM_MODEL": "legacy-model",
            },
            clear=False,
        ):
            config = ModelConfig.from_env()

        self.assertEqual(config.api_key, "provider-key")
        self.assertEqual(config.base_url, "https://provider.example/v1")
        self.assertEqual(config.model_name, "provider-model")

    def test_model_config_does_not_read_agent_yaml(self):
        with (
            patch.dict(
                "os.environ",
                {
                    "MODEL_API_KEY": "deepseek-key",
                    "MODEL_BASE_URL": "https://api.deepseek.com/v1",
                    "MODEL_NAME": "deepseek-reasoner",
                },
                clear=False,
            ),
            patch(
                "src.utils.config_handler.agent_conf",
                {
                    "api_key": "must-not-be-used",
                    "base_url": "https://must-not-be-used.invalid",
                    "model": "must-not-be-used",
                },
            ),
        ):
            model = create_agent_model()

        self.assertEqual(model.model, "deepseek-reasoner")
        self.assertEqual(
            str(model._client.base_url),
            "https://api.deepseek.com/v1/",
        )

    def test_model_config_reports_missing_environment_variables(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("src.model.agent_model.load_dotenv"),
        ):
            with self.assertRaisesRegex(
                ModelConfigurationError,
                "MODEL_NAME",
            ):
                ModelConfig.from_env()

    def test_tools_are_agents_sdk_function_tools(self):
        self.assertEqual(
            [tool.name for tool in tools],
            [
                "search_enterprise_faq",
                "get_request_identity",
                "draft_service_request",
            ],
        )
        self.assertIn("question", tools[0].params_json_schema["properties"])
        self.assertNotIn("context", tools[1].params_json_schema["properties"])

    async def test_enterprise_faq_matches_account_question(self):
        result = await search_enterprise_faq.on_invoke_tool(
            create_tool_context(
                "search_enterprise_faq",
                '{"question":"账号密码输错被锁定，怎么解锁？"}',
            ),
            '{"question":"账号密码输错被锁定，怎么解锁？"}',
        )
        entries = json.loads(result)
        self.assertEqual(entries[0]["id"], "account_unlock")
        self.assertIn("自助解锁", entries[0]["answer"])

    async def test_enterprise_faq_returns_guidance_when_no_match(self):
        result = await search_enterprise_faq.on_invoke_tool(
            create_tool_context(
                "search_enterprise_faq",
                '{"question":"完全未知的事项 xyz"}',
            ),
            '{"question":"完全未知的事项 xyz"}',
        )
        self.assertIn("未找到直接匹配", result)

    async def test_identity_tool_returns_current_scope(self):
        result = await get_request_identity.on_invoke_tool(
            create_tool_context("get_request_identity", "{}"),
            "{}",
        )
        identity = json.loads(result)
        self.assertEqual(identity["tenant_id"], "tenant-a")
        self.assertEqual(identity["user_id"], "user-a")
        self.assertEqual(identity["conversation_id"], "conversation-a")

    async def test_service_request_tool_only_creates_draft(self):
        arguments = json.dumps(
            {
                "title": "申请业务系统权限",
                "description": "需要开通只读角色",
                "priority": "普通",
            },
            ensure_ascii=False,
        )
        result = await draft_service_request.on_invoke_tool(
            create_tool_context("draft_service_request", arguments),
            arguments,
        )
        draft = json.loads(result)
        self.assertEqual(draft["status"], "draft")
        self.assertEqual(draft["tenant_id"], "tenant-a")
        self.assertIn("尚未提交", draft["notice"])

    def test_default_instructions(self):
        context = SimpleNamespace(context=AgentContext())
        self.assertEqual(dynamic_instructions(context, None), load_system_prompt())

    def test_skill_registry_loads_enterprise_skills(self):
        registry = SkillRegistry.from_directory("src/skills")
        self.assertEqual(
            registry.names(),
            ("enterprise_qa", "service_request"),
        )
        skills = registry.resolve(
            ["enterprise_qa"],
            available_tool_names=[tool.name for tool in tools],
        )
        prompt = registry.build_prompt(skills)
        self.assertIn("enterprise_qa", prompt)
        self.assertIn("search_enterprise_faq", prompt)

    def test_skill_registry_rejects_unknown_or_missing_tool(self):
        registry = SkillRegistry.from_directory("src/skills")
        with self.assertRaises(SkillValidationError):
            registry.resolve(
                ["unknown_skill"],
                available_tool_names=[tool.name for tool in tools],
            )
        with self.assertRaisesRegex(SkillValidationError, "search_enterprise_faq"):
            registry.resolve(
                ["enterprise_qa"],
                available_tool_names=[],
            )

    def test_dynamic_instructions_append_enabled_skill_prompt(self):
        context = SimpleNamespace(
            context=AgentContext(
                enabled_skills=("enterprise_qa",),
                skill_prompt="## 当前启用技能\n\n政企问答技能",
            )
        )
        prompt = dynamic_instructions(context, None)
        self.assertTrue(prompt.startswith(load_system_prompt().rstrip()))
        self.assertIn("政企问答技能", prompt)

    async def test_tool_visibility_follows_enabled_skills(self):
        registry = SkillRegistry.from_directory("src/skills")
        qa_skill = registry.resolve(
            ["enterprise_qa"],
            available_tool_names=[tool.name for tool in tools],
        )
        enabled_tools = frozenset(
            tool_name
            for skill in qa_skill
            for tool_name in skill.required_tools
        )
        context = SimpleNamespace(
            context=AgentContext(enabled_tools=enabled_tools)
        )

        visible = []
        for tool in tools:
            is_enabled = tool.is_enabled(context, None)
            if asyncio.iscoroutine(is_enabled):
                is_enabled = await is_enabled
            if is_enabled:
                visible.append(tool.name)

        self.assertEqual(
            visible,
            ["search_enterprise_faq", "get_request_identity"],
        )

    def test_relative_session_path_is_resolved_from_workspace(self):
        path = resolve_session_db_path("chat_history/test.db")
        self.assertTrue(path.is_absolute())
        self.assertEqual(path.name, "test.db")
        self.assertEqual(path.parent.name, "chat_history")

    def test_memory_scope_is_stable_opaque_and_tenant_isolated(self):
        scope_a = MemoryScope("tenant-a", "user-1", "conversation-1")
        scope_a_copy = MemoryScope("tenant-a", "user-1", "conversation-1")
        scope_b = MemoryScope("tenant-b", "user-1", "conversation-1")

        self.assertEqual(scope_a.session_id(), scope_a_copy.session_id())
        self.assertNotEqual(scope_a.session_id(), scope_b.session_id())
        self.assertNotIn("tenant-a", scope_a.session_id())

    def test_memory_scope_rejects_unsafe_identifiers(self):
        for invalid in ["", "../tenant", "tenant name", "a" * 65]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(MemoryValidationError):
                    MemoryScope(invalid, "user-1", "conversation-1")

    async def test_stream_adapter_only_emits_text_deltas(self):
        agent = OpenAIAgent.__new__(OpenAIAgent)
        agent.agent = object()
        agent.hooks = object()
        agent.max_turns = 5
        agent.session = None
        agent.skill_registry = SkillRegistry()
        agent.enabled_skills = ()
        agent.memory_scope = MemoryScope(
            "tenant-test",
            "user-test",
            "conversation-test",
        )

        with patch(
            "src.agent.openai_agent.Runner.run_streamed",
            return_value=FakeStreamingResult(),
        ):
            chunks = [chunk async for chunk in agent.execute_stream("test")]

        self.assertEqual(chunks, ["hello", " world"])

    async def test_request_can_override_enabled_skills(self):
        agent = OpenAIAgent.__new__(OpenAIAgent)
        agent.agent = object()
        agent.hooks = object()
        agent.max_turns = 5
        agent.session = None
        agent.skill_registry = SkillRegistry.from_directory("src/skills")
        agent.enabled_skills = ()
        agent.memory_scope = MemoryScope(
            "tenant-test",
            "user-test",
            "conversation-test",
        )

        with patch(
            "src.agent.openai_agent.Runner.run_streamed",
            return_value=FakeStreamingResult(),
        ) as run_streamed:
            chunks = [
                chunk
                async for chunk in agent.execute_stream(
                    "test",
                    enabled_skills=["enterprise_qa"],
                )
            ]

        run_context = run_streamed.call_args.kwargs["context"]
        self.assertEqual(chunks, ["hello", " world"])
        self.assertEqual(run_context.enabled_skills, ("enterprise_qa",))
        self.assertEqual(
            run_context.enabled_tools,
            frozenset({"search_enterprise_faq", "get_request_identity"}),
        )
        self.assertEqual(run_context.tenant_id, "tenant-test")

    async def test_full_agents_sdk_runner_with_mock_openai_transport(self):
        mock_model = create_mock_openai_model()
        with (
            patch(
                "src.agent.openai_agent.create_agent_model",
                return_value=mock_model,
            ),
            patch.dict(
                "src.agent.openai_agent.agent_conf",
                {"session_enabled": False, "enabled_skills": []},
            ),
        ):
            agent = OpenAIAgent()
            chunks = [
                chunk
                async for chunk in agent.execute_stream(
                    "不要调用工具，只回复 SDK_OK"
                )
            ]

        await mock_model.close()
        self.assertEqual("".join(chunks), "SDK_OK")

    async def test_sqlite_session_persists_agent_conversation(self):
        mock_model = create_mock_openai_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir) / "sessions.db"
            with (
                patch(
                    "src.agent.openai_agent.create_agent_model",
                    return_value=mock_model,
                ),
                patch.dict(
                    "src.agent.openai_agent.agent_conf",
                    {
                        "session_enabled": True,
                        "session_db_path": str(session_path),
                        "enabled_skills": [],
                    },
                ),
            ):
                scope = MemoryScope("tenant-a", "user-a", "memory-test")
                agent = OpenAIAgent(memory_scope=scope)
                output = "".join(
                    [
                        chunk
                        async for chunk in agent.execute_stream(
                            "不要调用工具，只回复 SDK_OK"
                        )
                    ]
                )
                items = await agent.session.get_items()
                agent.close()

            self.assertEqual(output, "SDK_OK")
            self.assertTrue(session_path.exists())
            self.assertGreaterEqual(len(items), 2)

        await mock_model.close()

    async def test_memory_store_prevents_cross_tenant_history_leakage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(Path(temp_dir) / "tenant-memory.db")
            session_a = store.open_session(
                MemoryScope("tenant-a", "user-1", "conversation-1")
            )
            session_b = store.open_session(
                MemoryScope("tenant-b", "user-1", "conversation-1")
            )
            try:
                await session_a.add_items(
                    [{"role": "user", "content": "tenant-a-secret"}]
                )
                await session_b.add_items(
                    [{"role": "user", "content": "tenant-b-secret"}]
                )
                items_a = await session_a.get_items()
                items_b = await session_b.get_items()
            finally:
                session_a.close()
                session_b.close()

        self.assertIn("tenant-a-secret", json.dumps(items_a))
        self.assertNotIn("tenant-b-secret", json.dumps(items_a))
        self.assertIn("tenant-b-secret", json.dumps(items_b))
        self.assertNotIn("tenant-a-secret", json.dumps(items_b))


if __name__ == "__main__":
    unittest.main()
