import sys
import time
from types import SimpleNamespace

sys.path.insert(0, "./src")

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from streamlit_feedback import streamlit_feedback

from data_broker.data_broker import DataBroker
from orchestrator.chat_orchestrator import ChatOrchestrator

st.title("Science-GPT Prototype")

st.session_state.orchestrator = ChatOrchestrator()
st.session_state.databroker = DataBroker()

if "question_state" not in st.session_state:
    st.session_state.question_state = False

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.cost = 0.0

if "fbk" not in st.session_state:
    st.session_state.fbk = str(uuid.uuid4())

if "show_textbox" not in st.session_state:
    st.session_state.show_textbox = False


def create_answer(prompt):
    if prompt is None:
        return

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

    st.session_state.messages.append(
        {
            "content": HumanMessage(content=prompt),
        }
    )
    st.session_state.messages.append(
        {
            "content": AIMessage(content=response),
        }
    )


def display_answer():
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["content"].type):
            st.markdown(message["content"].content)

        if "feedback" not in message:
            continue
        # If there is no feedback show N/A
        if "feedback" in message:
            st.markdown(f"Feedback: {message['feedback']}")
        else:
            st.markdown("Feedback: N/A")


def fbcb(response):
    """Update the history with feedback.

    The question and answer are already saved in history.
    Now we will add the feedback in that history entry.
    """
    last_entry = st.session_state.messages[-1]  # get the last entry
    last_entry.update({"feedback": response})  # update the last entry
    st.session_state.messages[-1] = last_entry  # replace the last entry

    st.markdown("✔️ Feedback received!")
    # st.markdown(f"Feedback: {response}")

    # Create a new feedback by changing the key of feedback component.
    st.session_state.fbk = str(uuid.uuid4())


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
            {"content": AIMessage(content="System prompt updated successfully!")}
        )
        st.rerun()

if prompt := st.chat_input("Write your query here..."):
    st.session_state.question_state = True

if st.session_state.question_state:
    display_answer()
    create_answer(prompt)

    streamlit_feedback(
        feedback_type="faces",
        optional_text_label="How was this response?",
        align="flex-start",
        key=st.session_state.fbk,
        on_submit=fbcb,
    )
