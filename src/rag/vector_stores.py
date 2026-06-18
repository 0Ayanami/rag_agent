from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.config_handler import chroma_conf
from src.model.factory import embed_model
import os
from src.utils.path_tool import get_abs_path
from src.utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from src.utils.logger_handler import logger

"""
向量库服务类：提供向量库的存储、检索、文本分割...
"""
class VectorStoreService(object):
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"], 
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_kwargs={"k": chroma_conf["k"]},
        )
    
    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        """
        def check_md5_hex(md5_str: str):
            # 检查传入的md5字符串是否已经被处理过了
            md5_path = get_abs_path(chroma_conf["md5_hex_store"])
            if not os.path.exists(md5_path):
                open(md5_path, "w", encoding="utf-8").close()
                return False
            else:
                with open(md5_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line == md5_str:
                            return True
                return False
            
        def save_md5(md5_str: str):
        # 将传入的md5字符串记录到文件内保存
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_str + "\n")
        
        def get_file_document(read_path: str):
            if read_path.endswith(".txt"):
                return txt_loader(read_path)
            elif read_path.endswith(".pdf"):
                return pdf_loader(read_path)
            else:
                logger.error(f"[get_file_document]不支持的文件类型: {read_path}")
                return []
            
        allowed_files_path:list[str] = listdir_with_allowed_type(get_abs_path(chroma_conf["data_path"]),
                                                                  tuple(chroma_conf["allow_knowledge_file_type"]))
        for file in allowed_files_path:
            md5_hex = get_file_md5_hex(file)
            if check_md5_hex(md5_hex):
                logger.info(f"[load_document]文件{file}已被处理，跳过")
                continue
            try:
                documents = get_file_document(file)
                if not documents:
                    logger.warning(f"[load_document]文件{file}读取为空，跳过")
                    continue
                
                split_documents = self.splitter.split_documents(documents)
                if not split_documents:
                    logger.warning(f"[load_document]文件{file}分割为空，跳过")
                    continue

                # 将内容存入向量数据库
                self.vector_store.add_documents(split_documents)
                save_md5(md5_hex)
            except Exception as e:
                # 设置exc_info=True，打印详细的异常信息
                logger.error(f"[load_document]文件{file}处理异常: {str(e)}", exc_info=True)
                continue

if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("主线2——资源&再生资源：上游开采、后端再生产业受益于金属价格上行。")
    for r in res:
        print(r.page_content)
        print(r.metadata)