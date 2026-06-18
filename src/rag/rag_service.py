from .vector_stores import VectorStoreService
from src.model.factory import chat_model
from src.utils.prompt_loader import load_rag_prompt
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()
    
    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retrieve_docs(self, query: str):
        return self.retriever.invoke(query)
    
    def rag_summarize(self, query: str):    
        context_docs = self.retrieve_docs(query)
        context = ""
        for i, doc in enumerate(context_docs):
            context += f"[参考资料{i+1}] 内容：{doc.page_content}\n 元数据：{doc.metadata}\n\n"
        
        return self.chain.invoke({"input": query, "context": context})
