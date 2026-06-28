"""为整个工程提供统一的路径解析。"""
from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """获取仓库根目录。"""
    return Path(__file__).resolve().parents[2]


def get_source_root() -> Path:
    """获取 src 目录。"""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(configured_path: str | Path) -> Path:
    """将配置路径稳定解析到仓库根目录。"""
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return get_project_root() / path


def resolve_session_db_path(configured_path: str | Path) -> Path:
    """将会话数据库配置路径稳定解析到仓库根目录。"""
    return resolve_project_path(configured_path)


def resolve_source_path(configured_path: str | Path) -> Path:
    """将配置路径稳定解析到 src 目录。"""
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return get_source_root() / path

