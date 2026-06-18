from .tools.agent_tools import tools
from src.model.factory import chat_model
from langchain.agents import create_agent
from src.utils.prompt_loader import load_system_prompt
from .tools.middleware import middlewares

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=tools,
            middleware=middlewares
        )
    
    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }

        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"
