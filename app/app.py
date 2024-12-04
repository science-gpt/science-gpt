import os
import sys
import uuid
from types import SimpleNamespace

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from streamlit_feedback import streamlit_feedback
from streamlit_float import float_css_helper, float_init, float_parent
from streamlit_survey import StreamlitSurvey
from streamlit_tags import st_tags

sys.path.insert(0, "./src")
from databroker.databroker import DataBroker
from logs.logger import logger
from orchestrator.chat_orchestrator import ChatOrchestrator


def init_streamlit():
    """
    Initializes the streamlit app and sets up the session state, including
    the system config, orchestrator, and databroker.
    """
    st.title("Science-GPT Prototype")

    if "userpath" not in st.session_state:
        st.session_state.username = st.session_state.get("username", "test_user")

        logger.set_user(st.session_state.get("username", "unknown"))

        st.session_state.userpath = (
            f"{os.getcwd()}/data/" + st.session_state.username + "/"
        )
        if not os.path.exists(st.session_state.userpath):
            os.makedirs(st.session_state.userpath)

        st.session_state.file_table = get_file_table()

    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = ChatOrchestrator()

    # !!!! this is a direct reference to the System Config and changes to this
    # dictionary will directly modify the orchestrator's system config
    # please update the config directly when updating system parameters !!!!
    system_config = st.session_state.orchestrator.config
    globals()["system_config"] = system_config

    if "databroker" not in st.session_state:
        st.session_state.database_config = SimpleNamespace(
            username=st.session_state.username,
            userpath=st.session_state.userpath,
            embedding_model=system_config.embedding.embedding_model,
            chunking_method=system_config.chunking.chunking_method,
            pdf_extractor=system_config.extraction,
            vector_store=system_config.vector_db,
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

    if "feedback_key" not in st.session_state:
        st.session_state.feedback_key = str(uuid.uuid4())

    if "pk" not in st.session_state:
        st.session_state.pk = [str(uuid.uuid4())]

    if "show_textbox" not in st.session_state:
        st.session_state.show_textbox = False


def file_upload_callback() -> None:
    """
    Uploads files to the user database via the databroker.
    """
    models_dir = st.session_state.userpath
    for file in st.session_state["file_upload"]:
        bfile = file.read()
        with open(models_dir + file.name, "wb") as f:
            f.write(bfile)

    if len(st.session_state["file_upload"]) > 0:
        database_callback(st.session_state.database_config)


def file_edit_callback():
    """
    Deletes files from the user database via the databroker.

    Note that this callback currently has a bug where it automatically deletes
    a file when the checkbox is clocked. This is not the intended behavior.
    """
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
    database_callback(st.session_state.database_config)


def get_file_table() -> pd.DataFrame:
    """
    Constructs a dataframe of files and their sizes in the user database
    """
    files_dir = st.session_state.userpath
    files_in_dir = os.listdir(files_dir)
    file_sizes = [
        format(os.path.getsize(os.path.join(files_dir, file)) / (1024 * 1024), f".{2}f")
        for file in files_in_dir
    ]

    return pd.DataFrame(data=[files_in_dir, file_sizes], index=["File", "Size (MB)"]).T


def get_prompt_key(i) -> str:
    """
    Retrieves the prompt key at the specified index. If the index is out of range,
    generates new UUIDs to extend the list of prompt keys.

    """
    if i < len(st.session_state.pk):
        return st.session_state.pk[i]
    st.session_state.pk += [
        str(uuid.uuid4()) for _ in range(i - len(st.session_state.pk) + 1)
    ]
    return st.session_state.pk[i]


def send_prompt(prompt):
    """
    When the user directly modifies the prompt, this directly prompts the model
    and updates the chat interface.
    """
    llm_prompt, response, cost = st.session_state.orchestrator.direct_query(
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
    """
    This is the textbox that allows the user to view and modify the prompt
    """
    with st.popover("See LLM Prompt", use_container_width=True):
        st.subheader("The LLM Prompt")
        nprompt = st.text_area(
            "Modify the LLM Prompt:",
            value=prompt,
            height=300,
            key="ta" + get_prompt_key(key),
        )
        st.button(
            "Submit Prompt",
            on_click=(lambda: send_prompt(nprompt)),
            key="b" + get_prompt_key(key),
        )


def create_answer(prompt):
    if prompt is None:
        return

    with st.chat_message("User"):
        st.markdown(prompt)

    with st.chat_message("AI"):
        message_placeholder = st.empty()

        llm_prompt, response, cost = st.session_state.orchestrator.triage_query(
            query=prompt, model=st.session_state.model
        )

        st.session_state.cost += float(cost)

        message_placeholder.markdown(response)

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


def display_chat_history():
    """
    Renders previous interactions on the page.
    """
    for i, message in enumerate(st.session_state.messages):
        if message["content"].type in ["human", "ai"]:
            with st.chat_message(message["content"].type):
                st.markdown(message["content"].content)

        if "feedback" in message:
            st.markdown(f"Feedback: {message['feedback']}")


def feedback_callback(response):
    """
    When feedback is submitted, this callback appends it to the
    last message in the chat.
    """
    st.session_state.messages[-1].update({"feedback": response})
    st.markdown("✔️ Feedback Received!")

    # Create a new feedback by changing the key of feedback component.
    st.session_state.feedback_key = str(uuid.uuid4())


def survey_callback():
    """
    Callback for the survey form. Logs the survey responses.
    """
    st.session_state.feedback.append(st.session_state.survey)
    logger.survey(
        f"Survey response received: {st.session_state.survey.to_json()}",
        xtra={"user": st.session_state["name"]},
    )
    st.toast("Your feedback has been recorded.  Thank you!", icon="🎉")


def database_callback(database_config):
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
            options=system_config.model_params.supported_models,
            index=system_config.model_params.supported_models.index(
                system_config.model_params.model_name
            ),
        )

        st.session_state.orchestrator.load_model(st.session_state.model)

        # here we verify that the model is online
        if not st.session_state.orchestrator.test_connection(
            model_name=st.session_state.model
        ):
            st.error(f"{st.session_state.model} is not online.")

        with st.sidebar.expander("Retrieval Settings", expanded=False):
            system_config.rag_params.use_rag = st.toggle(
                label="Retrieval Augmented Generation",
                value=system_config.rag_params.use_rag,
                help="Retrieve content from the document database for question answering",
            )

            if system_config.rag_params.use_rag:
                system_config.rag_params.useknowledgebase = st.toggle(
                    label="Use Uploaded Documents",
                    value=False,
                    help="Retrieve content from documents uploaded via the Knowledge Base tab. Do not enable this if you have not uploaded any documents.",
                )

                system_config.rag_params.top_k = st.slider(
                    label="Top K",
                    min_value=0,
                    max_value=20,
                    value=system_config.rag_params.top_k,
                    help="Number of text chunks to retrieve from the document database",
                )

                system_config.rag_params.keywords = st_tags(
                    label="Keyword Filters",
                    text="Enter keywords and press enter",
                    value=system_config.rag_params.keywords,
                    maxtags=3,  # max number of tags
                    key="keyword_tags",
                )

        with st.sidebar.expander("Prompt Modifiers", expanded=False):
            system_config.rag_params.moderationfilter = st.checkbox(
                "Moderation Filter",
                value=system_config.rag_params.moderationfilter,
                help="Filter out offensive, racist, homophobic, sexist, and pornographic content",
            )
            system_config.rag_params.onlyusecontext = st.checkbox(
                "Only Use Knowledge Base",
                value=system_config.rag_params.onlyusecontext,
                help="Instruct the model to use only the context provided to it to answer the question",
            )

        with st.sidebar.expander("General Model Settings", expanded=False):

            system_config.model_params.seed = st.number_input(
                "Seed",
                value=system_config.model_params.seed,
                help="A value that affects the randomness of the model.",
            )

            system_config.model_params.temperature = st.select_slider(
                "Temperature",
                options=[round(0.1 * i, 1) for i in range(0, 11)],
                value=system_config.model_params.temperature,
                help="Controls randomness in text generation; lower values make outputs more predictable, while higher values increase creativity. Adjust when you need more or less variability in responses.",
            )

            system_config.model_params.top_p = st.select_slider(
                "Top P",
                options=[round(0.1 * i, 1) for i in range(0, 11)],
                value=system_config.model_params.top_p,
                help="Limits token selection to the most probable ones, ensuring coherence; lower values restrict diversity, while higher values allow for more variation. Adjust for balanced creativity and relevance.",
            )

        with st.sidebar.expander("Database Options", expanded=False):
            with st.form("advanced", border=False):

                system_config.embedding.embedding_model = st.selectbox(
                    label="Choose embedding model:",
                    options=system_config.embedding.supported_embedders,
                    index=system_config.embedding.supported_embedders.index(
                        system_config.embedding.embedding_model
                    ),
                )

                system_config.chunking.chunking_method = st.selectbox(
                    label="Choose chunking method:",
                    options=system_config.chunking.supported_chunkers,
                    index=system_config.chunking.supported_chunkers.index(
                        system_config.chunking.chunking_method
                    ),
                )
                st.session_state.database_config = SimpleNamespace(
                    username=st.session_state.username,
                    userpath=st.session_state.userpath,
                    embedding_model=system_config.embedding.embedding_model,
                    chunking_method=system_config.chunking.chunking_method,
                    pdf_extractor=system_config.extraction,
                    vector_store=system_config.vector_db,
                )
                submitted = st.form_submit_button(
                    "Regenerate Database",
                    on_click=(
                        lambda: database_callback(st.session_state.database_config)
                    ),
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
                display_chat_history()
                create_answer(prompt)

                streamlit_feedback(
                    feedback_type="faces",
                    optional_text_label="How was this response?",
                    align="flex-start",
                    key=st.session_state.feedback_key,
                    on_submit=feedback_callback,
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

        st.button("Submit", on_click=survey_callback)


def knowledgebase(tab):
    with tab:
        with st.form("my-form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                key="file_upload",
                accept_multiple_files=True,
            )
            submitted = st.form_submit_button("Upload", on_click=file_upload_callback)

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
                on_change=file_edit_callback,
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
