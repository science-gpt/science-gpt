import sys
import time
from types import SimpleNamespace

sys.path.insert(0, "./src")

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from data_broker.data_broker import DataBroker
from orchestrator.chat_orchestrator import ChatOrchestrator

st.title("Science-GPT Prototype")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.cost = 0.0
    st.session_state.orchestrator = ChatOrchestrator()
    st.session_state.databroker = DataBroker()

if "show_textbox" not in st.session_state:
    st.session_state.show_textbox = False

with st.sidebar:

    st.write(f"Total Cost: ${format(st.session_state.cost, '.5f')}")

    update_prompt = st.button("Modify System Prompt")

    local = st.toggle("Use Local Model", False)

    if local:
        model = st.selectbox(
            "Model", ["Llama 2", "Llama 3.1"], index=None, placeholder="Select a model"
        )
        st.markdown("*Local models are not yet supported.*")
    else:
        model = st.selectbox(
            "Model", ["GPT-3.5", "GPT-4.0"], index=0, placeholder="Select a model"
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

                st.session_state.orchestrator.load_secrets(model)

                time.sleep(1)
                st.write("Found model credentials.")
                time.sleep(1)
                # send test prompt to model
                status.update(
                    label="Connection established!", state="complete", expanded=False
                )

    seed = st.text_input("Seed", value=42)
    temperature = st.select_slider(
        "Temperature", options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], value=0.2
    )
    top_k = st.slider("Top K", 0, 20, 1)

    moderationfilter = st.checkbox("Moderation Filter")
    onlyusecontext = st.checkbox("Only Use Knowledge Base")

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Logic to update system prompt
if update_prompt:
    st.session_state.show_textbox = True

if st.session_state.get("show_textbox", False):
    current_prompt = st.session_state.orchestrator.system_prompt
    new_prompt = st.text_area("Modify the system prompt:", value=current_prompt)
    if st.button("Submit New Prompt"):
        st.session_state.orchestrator.update_system_prompt(new_prompt)
        st.session_state.show_textbox = False
        st.session_state.messages.append(
            AIMessage(content="System prompt updated successfully!")
        )
        st.rerun()

if prompt := st.chat_input("Write your query here..."):
    with st.chat_message("User"):
        st.markdown(prompt)

    with st.chat_message("AI"):
        message_placeholder = st.empty()

        query_config = SimpleNamespace(
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            moderationfilter=moderationfilter,
            onlyusecontext=onlyusecontext,
        )
        response, cost = st.session_state.orchestrator.triage_query(
            prompt, query_config, chat_history=st.session_state.messages
        )
        message_placeholder.markdown(response)
        st.session_state.cost += cost

    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.messages.append(AIMessage(content=response))
