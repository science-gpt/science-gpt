
import streamlit as st
from langchain_core.messages import AIMessage

st.title("Science-GPT Prototype")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

#st.session_state.orchestrator = Orchestrator()

if prompt := st.chat_input("Text here..."):
    with st.chat_message("User"):
        st.markdown(prompt)

    with st.chat_message("AI"):
        message_placeholder = st.empty()
        full_response = st.session_state.bot.query(
            prompt,
            chat_history=st.session_state.messages
        )
        message_placeholder.markdown(full_response)

    st.session_state.messages.append(AIMessage(content=full_response))
