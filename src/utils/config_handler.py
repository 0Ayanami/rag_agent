import yaml
from .path_tool import get_abs_path

def load_rag_config(config_path: str=get_abs_path("configs/rag.yml"), encoding="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, yaml.FullLoader)

def load_chroma_config(config_path: str=get_abs_path("configs/chroma.yml"), encoding="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, yaml.FullLoader)
    
def load_prompt_config(config_path: str=get_abs_path("configs/prompt.yml"), encoding="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, yaml.FullLoader)
    
def load_agent_config(config_path: str=get_abs_path("configs/agent.yml"), encoding="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, yaml.FullLoader)
    
rag_conf = load_rag_config()
chroma_conf = load_chroma_config() 
prompt_conf = load_prompt_config()
agent_conf = load_agent_config()
