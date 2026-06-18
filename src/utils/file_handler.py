import os, hashlib
from .logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def get_file_md5_hex(file_path):
    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件路径不存在: {file_path}")
        return None
    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]文件路径不是文件: {file_path}")
        return None
    
    md5_obj = hashlib.md5()

    chunk_size = 4096
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"[md5计算]文件{file_path}读取异常: {str(e)}")
        return None

def listdir_with_allowed_type(path:str, allowed_types:tuple[str]):
    files = []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]路径不是目录: {path}")
        return allowed_types
    
    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)

def pdf_loader(file_path:str, pwd=None)-> list[Document]:
    return PyPDFLoader(file_path, password=pwd).load()

def txt_loader(file_path:str)-> list[Document]:
    return TextLoader(file_path, encoding="utf-8").load()
