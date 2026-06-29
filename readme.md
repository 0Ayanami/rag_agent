# 政企问答智能体原型

当前版本支持基础 RAG 知识库：通过 OpenAI-compatible Embedding API 生成向量，
使用 ChromaDB 持久化知识库，并将检索能力接入 Agent 工具与 Skill。
智能体基于 OpenAI Agents SDK，面向政企内部办事、制度咨询与服务支持场景。

## 已实现

- OpenAI Agents SDK `Agent + Runner.run_streamed`
- OpenAI-compatible 模型接入，可使用 DeepSeek、Qwen、Ollama/vLLM、OpenAI 等兼容服务
- OpenAI-compatible Embedding 接入，可使用云端 API 或 Ollama 本地 embedding 模型
- ChromaDB 持久化知识向量库
- 声明式 `SKILL.md` 技能注册与请求级工具白名单
- 知识库检索工具，可基于入库文档片段回答并返回来源
- 政企办事 FAQ 查询工具
- 当前租户、用户、会话身份查询工具
- 服务工单草稿工具，不执行真实外部提交
- SQLiteSession 会话记忆
- 租户、用户、会话三级 Memory 隔离
- Streamlit 流式问答界面

## 内置技能

- `enterprise_qa`：制度、办事流程和服务支持问答
- `knowledge_rag`：从 ChromaDB 知识库检索文档片段
- `service_request`：生成服务工单草稿

FAQ 内容位于 `src/configs/enterprise_faq.yml`，仅用于原型演示，不代表正式制度。
RAG 配置位于 `src/configs/rag.yml`，默认读取 `data/extracted_txt` 与
`data/extracted_md` 下的 `.txt`、`.md` 和 `.markdown` 文件。

## 安装与运行

```powershell
python -m pip install -r requirements.txt
python -m streamlit run src/app.py
```

模型连接只从 `.env` 或进程环境变量读取，不从 `agent.yml` 读取。
本项目使用 `openai` Python SDK 和 OpenAI Agents SDK 作为 OpenAI-compatible
协议客户端，不绑定 OpenAI 官方模型，也不引入厂商专用 SDK。

基础版要求所选模型或网关支持：

- Chat Completions 兼容接口
- streaming 响应
- tool/function calling

如果模型或网关不支持原生工具调用，当前版本可以普通对话，但无法稳定使用
FAQ、身份查询或工单草稿等技能工具。

推荐变量名：

- `MODEL_NAME`
- `MODEL_BASE_URL`
- `MODEL_API_KEY`

同时兼容项目已有变量名：

- `LM_MODEL`
- `BASE_URL`
- `API_KEY`

例如 DeepSeek：

```dotenv
MODEL_API_KEY=your_deepseek_api_key
MODEL_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
```

例如 Qwen / DashScope OpenAI-compatible：

```dotenv
MODEL_API_KEY=your_dashscope_api_key
MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-plus
```

例如本地 Ollama OpenAI-compatible：

```dotenv
MODEL_API_KEY=ollama
MODEL_BASE_URL=http://localhost:11434/v1
MODEL_NAME=qwen2.5:7b
```

例如本地或内网 vLLM OpenAI-compatible：

```dotenv
MODEL_API_KEY=your_vllm_api_key
MODEL_BASE_URL=http://localhost:8000/v1
MODEL_NAME=your-served-model-name
```

Embedding 模型同样只从 `.env` 或进程环境变量读取：

- `EMBEDDING_MODEL`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_API_KEY`

例如 OpenAI-compatible 云端 embedding：

```dotenv
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
```

例如 Ollama 本地 embedding：

```dotenv
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
```

本地 Ollama 的 `EMBEDDING_API_KEY` 只是 OpenAI SDK 所需占位值；如果
`EMBEDDING_BASE_URL` 是 localhost 且未配置 key，代码会自动使用 `ollama`。

`src/configs/agent.yml` 只保存 Agent 名称、轮次、Memory、Skills 等业务配置。

## 构建知识库

安装依赖并配置好 embedding 环境变量后，执行：

```powershell
python -m src.rag.build_index --check
```

该命令只调用 embedding API，不写入 ChromaDB。确认返回 `status: ok`
后再执行：

```powershell
python -m src.rag.build_index
```

该命令会读取 `src/configs/rag.yml` 中的 `source_paths`，将文本切块、
调用 embedding API，并写入 `chroma_db` 下的 ChromaDB collection。

建库后可以直接验证检索：

```powershell
python -m src.rag.build_index --query "采购审批流程是什么？" --top-k 3
```

## 测试

```powershell
python -m unittest tests/test_openai_agent_sdk.py
```

## 暂不覆盖

- PDF 原文解析入库
- 外部文档系统同步
- 知识库自动增量监听
