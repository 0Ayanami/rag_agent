from .config_handler import prompt_conf
from .path_tool import get_abs_path
from .logger_handler import logger

def load_system_prompt():
    try:
        main_prompt_path = get_abs_path(prompt_conf['main_prompt_path'])
    except KeyError as e:
        logger.error(f"[load prompt error]在yaml配置项中未找到main_prompt_path配置项")
        raise e

    try:
        return open(main_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[load prompt error]未找到系统提示词文件{str(e)}")
        raise e

def load_rag_prompt():
    try:
        rag_prompt_path = get_abs_path(prompt_conf['rag_summarize_prompt_path'])
    except KeyError as e:
        logger.error(f"[load prompt error]在yaml配置项中未找到rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[load prompt error]未找到rag提示词文件{str(e)}")
        raise e
    
def load_report_prompt():
    try:
        report_prompt_path = get_abs_path(prompt_conf['report_prompt_path'])
    except KeyError as e:
        logger.error(f"[load prompt error]在yaml配置项中未找到report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[load prompt error]未找到report提示词文件{str(e)}")
        raise e
    
if __name__ == "__main__":
    print(load_system_prompt())