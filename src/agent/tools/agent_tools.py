from src.rag.rag_service import RagSummarizeService
from langchain_core.tools import tool

rag = RagSummarizeService()

@tool(description="使用RAG模型对查询进行摘要")
def rag_summarize(query: str):
    return rag.rag_summarize(query)

@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"

tools = [rag_summarize, fill_context_for_report]