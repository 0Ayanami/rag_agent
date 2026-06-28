import yaml

from .path_tool import resolve_source_path


def _load_yaml(config_path: str, encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as config_file:
        return yaml.safe_load(config_file) or {}


def load_prompt_config(
    config_path: str = resolve_source_path("configs/prompt.yml"),
):
    return _load_yaml(config_path)


def load_agent_config(
    config_path: str = resolve_source_path("configs/agent.yml"),
):
    return _load_yaml(config_path)


def load_enterprise_faq_config(
    config_path: str = resolve_source_path("configs/enterprise_faq.yml"),
):
    return _load_yaml(config_path)


def load_rag_config(
    config_path: str = resolve_source_path("configs/rag.yml"),
):
    return _load_yaml(config_path)


prompt_conf = load_prompt_config()
agent_conf = load_agent_config()
enterprise_faq_conf = load_enterprise_faq_config()
rag_conf = load_rag_config()
