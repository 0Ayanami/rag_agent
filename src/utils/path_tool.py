"""
为整个工程提供统一的绝对路径
"""
import os

def get_project_root():
    """
    获取工程所在的根目录
    """
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    return os.path.dirname(current_dir)

def get_abs_path(relative_path: str):
    """
    获取工程根目录下的绝对路径
    """
    return os.path.join(get_project_root(), relative_path)
