# 政企问答智能体原型

当前版本已暂停 RAG 开发，运行主链不加载 LangChain、ChromaDB 或向量知识库。
智能体基于 OpenAI Agents SDK，面向政企内部办事、制度咨询与服务支持场景。

## 已实现

- OpenAI Agents SDK `Agent + Runner.run_streamed`
- OpenAI-compatible 模型接入，可使用 DeepSeek、OpenAI、Ollama 等服务
- 声明式 `SKILL.md` 技能注册与请求级工具白名单
- 政企办事 FAQ 查询工具
- 当前租户、用户、会话身份查询工具
- 服务工单草稿工具，不执行真实外部提交
- SQLiteSession 会话记忆
- 租户、用户、会话三级 Memory 隔离
- Streamlit 流式问答界面

## 内置技能

- `enterprise_qa`：制度、办事流程和服务支持问答
- `service_request`：生成服务工单草稿

FAQ 内容位于 `src/configs/enterprise_faq.yml`，仅用于原型演示，不代表正式制度。

## 安装与运行

```powershell
python -m pip install -r requirements.txt
python -m streamlit run src/app.py
```

模型连接只从 `.env` 或进程环境变量读取，不从 `agent.yml` 读取。

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

`src/configs/agent.yml` 只保存 Agent 名称、轮次、Memory、Skills 等业务配置。

## 测试

```powershell
python -m unittest tests/test_openai_agent_sdk.py
```

## 暂停范围

- RAG 与 ChromaDB 知识库
- PDF 或文档入库
- 向量检索
- LangChain 运行时依赖

原有 `chroma_db` 和 `data` 目录不会被删除或修改，后续稳定后可单独恢复 RAG 迭代。
