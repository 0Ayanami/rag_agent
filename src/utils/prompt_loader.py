from .config_handler import prompt_conf
from .logger_handler import logger
from .path_tool import get_abs_path


def load_system_prompt() -> str:
    try:
        prompt_path = get_abs_path(prompt_conf["main_prompt_path"])
        with open(prompt_path, "r", encoding="utf-8") as prompt_file:
            return prompt_file.read()
    except Exception:
        logger.exception("[load prompt error]系统提示词加载失败")
        raise
