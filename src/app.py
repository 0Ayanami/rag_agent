import time

import streamlit as st
from src.agent.react_agent import ReactAgent

# 标题
st.title("React Agent Demo")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])  

prompt = st.chat_input("请输入您的问题：")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    cache_list = []
    with st.spinner("正在思考..."):
        response = st.session_state["agent"].execute_stream(prompt)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    time.sleep(0.01)  # 模拟打字效果
                    yield char
                

        st.chat_message("assistant").write_stream(capture(response, cache_list))
        st.session_state["messages"].append({"role": "assistant", "content": cache_list[-1]})
        st.rerun()
