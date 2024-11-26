import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, "./src")

import json
import uuid

import pandas as pd
import streamlit as st
from databroker.databroker import DataBroker
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from logs.logger import logger
from orchestrator.chat_orchestrator import ChatOrchestrator
from orchestrator.utils import load_config
from streamlit_feedback import streamlit_feedback
from streamlit_float import float_css_helper, float_init, float_parent
from streamlit_survey import StreamlitSurvey
from streamlit_tags import st_tags


def file_upload_cb():
    models_dir = st.session_state.userpath
    for file in st.session_state["file_upload"]:
        bfile = file.read()
        with open(models_dir + file.name, "wb") as f:
            f.write(bfile)

    if len(st.session_state["file_upload"]) > 0:
        databasecb(st.session_state.database_config)


def file_edit_cb():
    files_dir = st.session_state.userpath
    edited_rows = st.session_state["file_editor"]["edited_rows"]
    rows_to_delete = []

    for idx, value in edited_rows.items():
        if value["x"] is True:
            rows_to_delete.append(idx)

    nfile_table = get_file_table().drop(rows_to_delete, axis=0).reset_index(drop=True)

    files_keep = list(nfile_table["File"].values)
    files_in_dir = os.listdir(files_dir)

    for file in files_in_dir:
        if file not in files_keep:
            os.remove(os.path.join(files_dir, file))
    databasecb(st.session_state.database_config)


def get_file_table():
    files_dir = st.session_state.userpath
    files_in_dir = os.listdir(files_dir)
    file_sizes = [
        format(os.path.getsize(os.path.join(files_dir, file)) / (1024 * 1024), f".{2}f")
        for file in files_in_dir
    ]

    return pd.DataFrame(data=[files_in_dir, file_sizes], index=["File", "Size (MB)"]).T


def init_streamlit():
    st.title("Science-GPT Prototype")

    if "userpath" not in st.session_state:
        st.session_state.username = st.session_state.get("username", "test_user")
        st.session_state.userpath = (
            f"{os.getcwd()}/data/" + st.session_state.username + "/"
        )
        if not os.path.exists(st.session_state.userpath):
            os.makedirs(st.session_state.userpath)

        st.session_state.file_table = get_file_table()

    if "config" not in st.session_state:

        logger.set_user(st.session_state.get("name", "unknown"))

        st.session_state.config = load_config(
            config_name="system_config", config_dir=f"{os.getcwd()}/src/configs"
        )
        st.session_state.query_config = None
        st.session_state.seed = st.session_state.config.model_params.seed
        st.session_state.temperature = st.session_state.config.model_params.temperature
        st.session_state.top_p = st.session_state.config.model_params.top_p

        st.session_state.embedding_model = st.session_state.config.embedding.model
        st.session_state.chunking_method = st.session_state.config.chunking.method
        st.session_state.top_k = st.session_state.config.rag_params.top_k_retrieval

        st.session_state.nprompt = None
        st.session_state.moderationfilter = False
        st.session_state.onlyusecontext = False
        st.session_state.useknowledgebase = False
        st.session_state.keywords = None

        st.session_state.orchestrator = ChatOrchestrator()

        st.session_state.database_config = SimpleNamespace(
            username=st.session_state.username,
            userpath=st.session_state.userpath,
            embedding_model=st.session_state.embedding_model,
            chunking_method=st.session_state.chunking_method,
            pdf_extractor=st.session_state.config.extraction,
            vector_store=st.session_state.config.vector_db,
        )
        st.session_state.databroker = DataBroker(st.session_state.database_config)

    if "question_state" not in st.session_state:
        st.session_state.question_state = False

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.cost = 0.0

    if "survey" not in st.session_state:
        st.session_state.survey = StreamlitSurvey()
        st.session_state.feedback = []

    if "automate" not in st.session_state:
        st.session_state.automate = StreamlitSurvey()

    if "fbk" not in st.session_state:
        st.session_state.fbk = str(uuid.uuid4())

    if "pk" not in st.session_state:
        st.session_state.pk = [str(uuid.uuid4())]

    if "show_textbox" not in st.session_state:
        st.session_state.show_textbox = False

    if "selected_embedding_model" not in st.session_state:
        st.session_state.selected_embedding_model = None


def get_pk(i):
    if i < len(st.session_state.pk):
        return st.session_state.pk[i]
    st.session_state.pk += [
        str(uuid.uuid4()) for _ in range(i - len(st.session_state.pk) + 1)
    ]
    return st.session_state.pk[i]


def send_prompt(prompt):
    llm_prompt, response, cost = st.session_state.orchestrator.query(
        prompt,
    )
    st.session_state.cost += float(cost)
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
    st.session_state.messages.append(
        {
            "content": ToolMessage(content=llm_prompt, tool_call_id=response),
        }
    )


def edit_prompt(prompt, key=0):
    with st.popover("See LLM Prompt", use_container_width=True):
        st.subheader("The LLM Prompt")
        nprompt = st.text_area(
            "Modify the LLM Prompt:", value=prompt, height=300, key="ta" + get_pk(key)
        )
        st.button(
            "Submit Prompt",
            on_click=(lambda: send_prompt(nprompt)),
            key="b" + get_pk(key),
        )


def create_answer(prompt):
    if prompt is None:
        return

    with st.chat_message("User"):
        st.markdown(prompt)

    with st.chat_message("AI"):
        message_placeholder = st.empty()

        st.session_state.query_config = SimpleNamespace(
            seed=st.session_state.seed,
            temperature=st.session_state.temperature,
            top_k=st.session_state.top_k,
            top_p=st.session_state.top_p,
            moderationfilter=st.session_state.moderationfilter,
            onlyusecontext=st.session_state.onlyusecontext,
            useknowledgebase=st.session_state.useknowledgebase,
            keywords=st.session_state.keywords,
        )

        llm_prompt, response, cost = st.session_state.orchestrator.triage_query(
            st.session_state.model,
            prompt,
            st.session_state.query_config,
            use_rag=st.session_state.use_rag,
        )

        logger.info(
            "Prompt: " + llm_prompt + " Response: " + response,
        )

        print(cost)
        st.session_state.cost += float(cost)

        # Display the extracted response content
        message_placeholder.markdown(response)

    # Append the messages to session state
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
    st.session_state.messages.append(
        {
            "content": ToolMessage(content=llm_prompt, tool_call_id=response),
        }
    )
    edit_prompt(llm_prompt)


def display_answer():
    for i, message in enumerate(st.session_state.messages):
        if message["content"].type in ["human", "ai"]:
            with st.chat_message(message["content"].type):
                st.markdown(message["content"].content)
        else:
            edit_prompt(message["content"].content, key=i + 1)

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

    st.markdown("✔️ Feedback Received!")
    # st.markdown(f"Feedback: {response}")

    # Create a new feedback by changing the key of feedback component.
    st.session_state.fbk = str(uuid.uuid4())


def surveycb():
    st.session_state.feedback.append(st.session_state.survey)
    logger.survey(
        f"Survey response received: {st.session_state.survey.to_json()}",
        xtra={"user": st.session_state["name"]},
    )
    st.toast("Your feedback has been recorded.  Thank you!", icon="🎉")


def databasecb(database_config):
    if "databroker" not in st.session_state:
        st.session_state.databroker = DataBroker(st.session_state.database_config)
    # not a best practice: accessing protected method. but oh well
    st.session_state.databroker._init_databroker_pipeline(database_config)
    st.sidebar.success(f"Database Generated!")


def sidebar():

    with st.sidebar:

        st.metric(label="Session Cost", value=f"${st.session_state.cost:.5f}")

        st.session_state.model = st.selectbox(
            label="Model",
            options=st.session_state.orchestrator.config.supported_models,
            index=0,
        )

        st.session_state.orchestrator.load_model(st.session_state.model)

        # here we verify that the model is online
        if not st.session_state.orchestrator.test_connection(
            model_name=st.session_state.model
        ):
            st.error(f"{st.session_state.model} is not online.")

        with st.sidebar.expander("Retrieval Settings", expanded=False):
            st.session_state.use_rag = st.toggle(
                label="Retrieval Augmented Generation",
                value=True,
                help="Retrieve content from the document database for question answering",
            )

            if st.session_state.use_rag:
                st.session_state.useknowledgebase = st.toggle(
                    label="Use Uploaded Documents",
                    value=False,
                    help="Retrieve content from documents uploaded via the Knowledge Base tab. Do not enable this if you have not uploaded any documents.",
                )

                st.session_state.top_k = st.slider(
                    label="Top K",
                    min_value=0,
                    max_value=20,
                    value=st.session_state.top_k,
                    help="Number of text chunks to retrieve from the document database",
                )

                # creates a tag section to enter keywords
                st.session_state.keywords = st_tags(
                    label="Keyword Filters",
                    text="Enter keywords and press enter",
                    value=st.session_state.get("keywords", []),
                    maxtags=3,  # max number of tags
                    key="keyword_tags",
                )

        with st.sidebar.expander("Database Options", expanded=False):
            with st.form("advanced", border=False):

                # make sure first option matches system config
                st.session_state.embedding_model = st.selectbox(
                    "Choose embedding model:",
                    ("mxbai-embed-large", "nomic-embed-text"),
                )
                # make sure first option matches system config
                st.session_state.chunking_method = st.selectbox(
                    "Choose chunking method:",
                    (
                        "recursive_character",
                        "recursive_character:large_chunks",
                        "recursive_character:small_chunks",
                    ),
                )
                st.session_state.database_config = SimpleNamespace(
                    username=st.session_state.username,
                    userpath=st.session_state.userpath,
                    embedding_model=st.session_state.embedding_model,
                    chunking_method=st.session_state.chunking_method,
                    # Load default values from config
                    pdf_extractor=st.session_state.config.extraction,
                    vector_store=st.session_state.config.vector_db,
                )
                submitted = st.form_submit_button(
                    "Regenerate Database",
                    on_click=(lambda: databasecb(st.session_state.database_config)),
                )

        with st.sidebar.expander("Prompt Modifiers", expanded=False):
            st.session_state.moderationfilter = st.checkbox(
                "Moderation Filter",
                value=True,
                help="Filter out offensive, racist, homophobic, sexist, and pornographic content",
            )
            st.session_state.onlyusecontext = st.checkbox(
                "Only Use Knowledge Base",
                value=True,
                help="Instruct the model to use only the context provided to it to answer the question",
            )

        with st.sidebar.expander("General Model Settings", expanded=False):

            st.session_state.seed = st.number_input(
                "Seed",
                value=st.session_state.seed,
                help="A value that affects the randomness of the model.",
            )

            st.session_state.temperature = st.select_slider(
                "Temperature",
                options=[round(0.1 * i, 1) for i in range(0, 11)],
                value=st.session_state.temperature,
                help="Controls randomness in text generation; lower values make outputs more predictable, while higher values increase creativity. Adjust when you need more or less variability in responses.",
            )

            st.session_state.top_p = st.select_slider(
                "Top P",
                options=[round(0.1 * i, 1) for i in range(0, 11)],
                value=st.session_state.top_p,
                help="Limits token selection to the most probable ones, ensuring coherence; lower values restrict diversity, while higher values allow for more variation. Adjust for balanced creativity and relevance.",
            )


def chat(tab):
    with tab:
        with st.container():
            if prompt := st.chat_input("Write your query here..."):
                st.session_state.question_state = True
            button_b_pos = "3rem"
            button_css = float_css_helper(
                width="2.2rem", bottom=button_b_pos, transition=0
            )
            float_parent(css=button_css)

        if st.session_state.question_state:
            with st.container(height=500, border=False):
                display_answer()
                create_answer(prompt)

                streamlit_feedback(
                    feedback_type="faces",
                    optional_text_label="How was this response?",
                    align="flex-start",
                    key=st.session_state.fbk,
                    on_submit=fbcb,
                )


def survey(tab):
    with tab:
        st.text(
            "Please complete this short survey sharing your experiences with the team!"
        )
        overall = st.session_state.survey.radio(
            "How was your overall experience?",
            options=["😞", "🙁", "😐", "🙂", "😀"],
            index=3,
            horizontal=True,
            id="overall",
        )

        if overall in ["😞", "🙁"]:
            overall_1 = st.session_state.survey.text_area(
                "Is there something we can do better?", id="overall_1"
            )

        responsequality = st.session_state.survey.radio(
            "Was the model able to answer your questions?",
            options=["Yes 👍", "No 👎"],
            index=0,
            horizontal=True,
            id="responsequality",
        )

        if responsequality in ["Kind of", "No 👎"]:
            responsequality_1 = st.session_state.survey.text_input(
                "What was the question that the model failed to answer?",
                id="responsequality_1",
            )
            responsequality_2 = st.session_state.survey.text_area(
                "What kind of response / format do you want to get back from the model?",
                id="responsequality_2",
            )

        timesaved = st.number_input(
            "How many minutes of work has your LLM chat saved?",
            min_value=0,
            max_value=120,
            value=0,
        )

        visuals = st.session_state.survey.text_area(
            "Is there anyway we can improve the design / Make the application easier to use?"
        )
        visuals = st.session_state.survey.text_area("Other comments and feedback:")

        st.button("Submit", on_click=surveycb)


def knowledgebase(tab):
    with tab:
        with st.form("my-form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                key="file_upload",
                accept_multiple_files=True,
            )
            submitted = st.form_submit_button(
                "Upload", on_click=lambda: file_upload_cb()
            )

        columns = get_file_table().columns
        column_config = {
            column: st.column_config.Column(disabled=True) for column in columns
        }

        modified_df = get_file_table().copy()
        modified_df["x"] = False
        modified_df = modified_df[["x"] + modified_df.columns[:-1].tolist()]

        if len(modified_df.values) == 0:
            st.subheader("No Files Uploaded")
        else:
            nfile_table = st.data_editor(
                modified_df,
                key="file_editor",
                on_change=file_edit_cb,
                hide_index=True,
                column_config=column_config,
            )


def sciencegpt():
    float_init(theme=True, include_unstable_primary=False)
    init_streamlit()
    sidebar()
    chat_tab, knowledge_base, survey_tab = st.tabs(["Chat", "Knowledge Base", "Survey"])
    chat(chat_tab)
    knowledgebase(knowledge_base)
    survey(survey_tab)


if __name__ == "__main__":
    sciencegpt()
