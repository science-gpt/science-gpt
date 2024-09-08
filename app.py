import sys
import time

sys.path.insert(0, "./src")

import streamlit as st
from langchain_core.messages import AIMessage

from orchestrator.chat_orchestrator import ChatOrchestrator

st.title("Science-GPT Prototype")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.cost = 0.0
    st.session_state.orchestrator = ChatOrchestrator()

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

with st.sidebar:

    local = st.toggle("Use Local Model", False)

    if local:
        model = st.selectbox(
            "Model", ["Llama 2", "Llama 3.1"], index=None, placeholder="Select a model"
        )
        st.markdown("*Local models are not yet supported.*")
    else:
        model = st.selectbox(
            "Model", ["GPT-3.5", "GPT-4.0"], index=None, placeholder="Select a model"
        )

    col1, col2 = st.columns(2, vertical_alignment="center")

    with col1:
        st.write("Current Model:", model)

    with col2:
        test_con = st.button("Connect")

    # TODO: add functionality to this button
    if test_con:
        with st.status("Testing connection...") as status:

            if local:
                status.update(
                    label="Local models are not yet supported.",
                    state="error",
                    expanded=False,
                )

            else:
                # update model secrets in orchestrator
                time.sleep(1)
                st.write("Found model credentials.")
                time.sleep(1)
                # send test prompt to model
                status.update(
                    label="Connection established!", state="complete", expanded=False
                )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    st.write(f"Total Cost: ${format(st.session_state.cost, '.5f')}")


if prompt := st.chat_input("Write your query here..."):
    with st.chat_message("User"):
        st.markdown(prompt)

    with st.chat_message("AI"):
        message_placeholder = st.empty()

        response, cost = st.session_state.orchestrator.triage_query(
            prompt, chat_history=st.session_state.messages
        )
        message_placeholder.markdown(response)
        st.session_state.cost += cost

    st.session_state.messages.append(AIMessage(content=response))
