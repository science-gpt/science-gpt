import os
import re
import sys
import uuid
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st
from annotated_text import annotated_text
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from streamlit_agraph import Config, Edge, Node, agraph
from streamlit_card import card
from streamlit_feedback import streamlit_feedback
from streamlit_float import float_css_helper, float_init, float_parent
from streamlit_survey import StreamlitSurvey
from streamlit_tags import st_tags

sys.path.insert(0, "./src")
import torch
from databroker.databroker import DataBroker
from logs.logger import logger
from orchestrator.chat_orchestrator import ChatOrchestrator

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]


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

    st.session_state.setdefault("question_state", False)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("cost", 0.0)
    st.session_state.setdefault("survey", StreamlitSurvey())
    st.session_state.setdefault("feedback", [])
    st.session_state.setdefault("automate", StreamlitSurvey())
    st.session_state.setdefault("feedback_key", str(uuid.uuid4()))
    st.session_state.setdefault("pk", [str(uuid.uuid4())])
    st.session_state.setdefault("show_textbox", False)


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
        format(os.path.getsize(os.path.join(files_dir, file)) / (1024 * 1024), ".2f")
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
    st.session_state.messages.extend(
        [
            {"content": HumanMessage(content=prompt)},
            {"content": AIMessage(content=response)},
            {"content": ToolMessage(content=llm_prompt, tool_call_id=response)},
        ]
    )


def edit_prompt(prompt, chunks, rewrite_prompt, key=0):
    """
    This is the textbox that allows the user to view and modify the prompt
    """

    # TODO create a card for the base prompt +  create a card for each chunk

    with st.popover("See LLM Prompt", use_container_width=True):

        if rewrite_prompt:
            st.subheader("Prompt Information")
            st.text_area("Your query was rewritten to", rewrite_prompt)

        if chunks:
            pattern = (
                r"Context\s*Source:\s*"
                r"(?P<context_source>.+?)"
                r"(?:\s*-\s*Chunk\s*"
                r"(?P<chunk_number>[\d_]+))?"
                r"(?:\nDistance:\s*(?P<distance>.+?))?"
                r"(?:\s*|\n)Document:\s*"
                r"(?P<document>.+?)"
                r"(?:Distance:\s*(?P<doc_distance>[\d.]+))?\s*$"
            )

            st.subheader("Chunks")
            for chunk in chunks:
                match = re.search(pattern, chunk, re.DOTALL)
                if not match:
                    st.error(f"Failed to parse chunk: {chunk[:50]}...")
                    continue

                context_source = match.group("context_source")
                chunk_number = match.group("chunk_number")
                distance = match.group("doc_distance")
                document = match.group("document")

                if match.group("doc_distance") and document.endswith(
                    f"Distance: {match.group('doc_distance')}"
                ):
                    document = document.rsplit("Distance:", 1)[0].strip()

                st.markdown(f"##### Context Source: {context_source}")
                if chunk_number:
                    # Create an inline distance with info icon
                    distance_info = f"""
                    <style>
                    .distance-container {{
                      display: flex;
                      align-items: center;
                      margin-bottom: 10px;
                    }}
                    .distance-info {{
                      font-size: 1.1rem;
                      font-weight: 600;
                    }}
                    .tooltip {{
                      position: relative;
                      margin-left: 8px;
                      display: inline-block;
                    }}
                    .tooltip .icon {{
                      cursor: pointer;
                      color: #0066cc;
                    }}
                    .tooltip .tooltiptext {{
                      visibility: hidden;
                      width: 300px;
                      background-color: #f0f2f6;
                      color: #31333F;
                      text-align: left;
                      border-radius: 4px;
                      padding: 10px;
                      position: absolute;
                      z-index: 1;
                      top: -5px;
                      left: 125%;
                      opacity: 0;
                      transition: opacity 0.3s;
                      font-size: 0.85rem;
                      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                      border: 1px solid #ddd;
                    }}
                    .tooltip:hover .tooltiptext,
                    .tooltip.active .tooltiptext {{
                      visibility: visible;
                      opacity: 1;
                    }}
                    </style>
                    
                    <div class="distance-container">
                      <div class="distance-info">Chunk {chunk_number} | Distance: {distance}</div>
                      <div class="tooltip" onclick="this.classList.toggle('active')">
                        <div class="icon">ℹ️</div>
                        <div class="tooltiptext">Higher similarity scores indicate stronger alignment between the query and such chunks.</div>
                      </div>
                    </div>
                    """
                    st.markdown(distance_info, unsafe_allow_html=True)
                st.markdown(f":blue-background[{document}]")
                st.divider()

        with st.expander("View LLM Prompt", expanded=False):
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
    """
    Generates a response from the user query and renders it on the page.
    """
    if not prompt:
        return

    with st.chat_message("User"):
        st.markdown(prompt)

    with st.chat_message("AI"):
        message_placeholder = st.empty()

        llm_prompt, response, cost, chunks, rewrite_prompt = (
            st.session_state.orchestrator.triage_query(
                query=prompt, model=st.session_state.model
            )
        )

        st.session_state.cost += float(cost)
        message_placeholder.markdown(response)

    st.session_state.messages.extend(
        [
            {"content": HumanMessage(content=prompt)},
            {"content": AIMessage(content=response)},
            {"content": ToolMessage(content=llm_prompt, tool_call_id=response)},
        ]
    )

    edit_prompt(llm_prompt, chunks, rewrite_prompt)


def display_chat_history():
    """
    Renders previous interactions on the page.
    """
    for message in st.session_state.messages:
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
    st.session_state.feedback_key = str(uuid.uuid4())


def survey_callback():
    """
    Callback for the survey form. Logs the survey responses.
    """
    st.session_state.feedback.append(st.session_state.survey)
    logger.survey(f"Survey response received: {st.session_state.survey.to_json()}")
    st.toast("Your feedback has been recorded.  Thank you!", icon="🎉")


def database_callback(database_config):
    """
    Regenerates the database when the database settings are changed.
    """
    # Clear existing DataBroker instance from session state
    if "databroker" in st.session_state:
        del st.session_state.databroker

    # Create new DataBroker instance with updated config
    st.session_state.databroker = DataBroker(database_config)

    # Force reinitialization of the pipeline
    st.session_state.databroker._init_databroker_pipeline(database_config)

    st.sidebar.success(f"Database regenerated with {database_config.embedding_model}!")


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

                system_config.rag_params.hybrid_weight = st.slider(
                    label="Hybrid Search Weighting",
                    min_value=0.0,
                    max_value=1.0,
                    value=system_config.rag_params.hybrid_weight,
                    key="hybrid_weight_slider",
                    help="Weighting for Hybrid Search (0 only dense, 1 only sparse)",
                )

                system_config.rag_params.top_k = st.slider(
                    label="Top K",
                    min_value=0,
                    max_value=20,
                    value=system_config.rag_params.top_k,
                    help="Number of text chunks to retrieve from the document database",
                )

                # Initialize session state if needed
                if "use_reranker_sidebar" not in st.session_state:
                    st.session_state.use_reranker_sidebar = (
                        system_config.rag_params.use_reranker
                    )

                # First place the model selection UI
                system_config.rag_params.reranker_model = st.selectbox(
                    label="Reranker Model",
                    options=system_config.rag_params.supported_rerankers,
                    index=system_config.rag_params.supported_rerankers.index(
                        system_config.rag_params.reranker_model
                    ),
                    help="Select the reranker model used to improve search result quality",
                    key="sidebar_reranker_model",
                    disabled=not st.session_state.use_reranker_sidebar,
                )

                # Then add the toggle after the model selection
                system_config.rag_params.use_reranker = st.toggle(
                    label="Enable Reranker",
                    value=st.session_state.use_reranker_sidebar,
                    help="Toggle to enable or disable the reranker. When enabled, the system refines search results by re-evaluating how well each chunk matches your query. This typically improves relevance but takes slightly longer. When disabled, raw retrieval results are used without this extra refinement step.",
                    key="use_reranker_toggle",
                    on_change=lambda: setattr(
                        st.session_state,
                        "use_reranker_sidebar",
                        st.session_state.use_reranker_toggle,
                    ),
                )

                system_config.rag_params.keywords = st_tags(
                    label="Keyword Filters",
                    text="Enter keywords and press enter",
                    value=system_config.rag_params.keywords,
                    maxtags=15,
                    key="keyword_tags_chat",
                )

                system_config.rag_params.filenames = st.multiselect(
                    "Select files for filtering:",
                    options=(
                        [
                            os.path.splitext(f)[0]
                            for f in os.listdir("/usr/src/app/data")
                            if f.endswith(".pdf")
                        ]
                        if os.path.exists("/usr/src/app/data")
                        else []
                    ),
                    default=system_config.rag_params.filenames,
                    help="Choose one or more files from the list to filter results.",
                    key="sidebar_filenames",
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

            system_config.agent_params.enable = st.checkbox(
                "Experimental: Advanced Reasoning (with google and wikipedia)",
                value=system_config.agent_params.enable,
                help="Instruct the model to use advanced reasoning (ReWOO) (Experimental Feature)",
            )

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
                # Update system config directly from widgets
                new_embedding_model = st.selectbox(
                    label="Choose embedding model:",
                    options=system_config.embedding.supported_embedders,
                    index=system_config.embedding.supported_embedders.index(
                        system_config.embedding.embedding_model
                    ),
                )

                new_chunking_method = st.selectbox(
                    label="Choose chunking method:",
                    options=system_config.chunking.supported_chunkers,
                    index=system_config.chunking.supported_chunkers.index(
                        system_config.chunking.chunking_method
                    ),
                )

                # Create NEW database config object on form submission
                submitted = st.form_submit_button("Regenerate Database")

            if submitted:
                # Update config FIRST
                system_config.embedding.embedding_model = new_embedding_model
                system_config.chunking.chunking_method = new_chunking_method

                # THEN create new database config
                st.session_state.database_config = SimpleNamespace(
                    username=st.session_state.username,
                    userpath=st.session_state.userpath,
                    embedding_model=new_embedding_model,
                    chunking_method=new_chunking_method,
                    pdf_extractor=system_config.extraction,
                    vector_store=system_config.vector_db,
                )

                # FINALLY trigger callback
                database_callback(st.session_state.database_config)
            selected_file = st.selectbox(
                "Show files from the data folder:",
                options=(
                    os.listdir("/usr/src/app/data")
                    if os.path.exists("/usr/src/app/data")
                    else ["No files available"]
                ),
                help="This dropdown lists all files in the data folder. Selecting an item does nothing.",
            )


def chat(tab):
    """
    Main chat window for users to submit queries.
    """
    with tab:
        with st.container():
            if prompt := st.chat_input("Write your query here..."):
                st.session_state.question_state = True
            button_css = float_css_helper(
                width="33%",
                bottom="3rem",
                left="50%",
                transform="translateX(-50%)",
                transition=0,
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
    """
    Tab for users to answer survey questions.
    """
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
            # Carter: we should just log this directly, rather than ask the user to rewrite their question
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
    """
    User document upload and management tab.
    """
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


def search(search_tab):
    with search_tab:
        query = st.text_input("Search Query", "")
        # st.session_state.filenames = st_tags(
        #     label="File Filtered Retrieval",
        #     text="Enter filenames and press enter",
        #     value=st.session_state.get("filenames", []),
        #     suggestions=["Acephate.pdf", "Atrazine.pdf", "Paraquat.pdf"],
        #     maxtags=15,  # max number of tags
        #     key="filename_tags",
        # )
        st.session_state.filenames = st.multiselect(
            "Select files for filtering:",
            options=(
                [
                    os.path.splitext(f)[0]
                    for f in os.listdir("/usr/src/app/data")
                    if f.endswith(".pdf")
                ]
                if os.path.exists("/usr/src/app/data")
                else []
            ),
            default=st.session_state.get("filenames", []),
            help="Choose one or more files from the list to filter results.",
            key="searchtab_filenames",
        )
        st.session_state.keywords = st_tags(
            label="Keyword Filtered Retrieval",
            text="Enter keywords and press enter",
            value=st.session_state.get("keywords", []),
            suggestions=["Toxicology", "Regulation", "Environment"],
            maxtags=10,  # max number of tags
            key="keyword_tags",
        )

        st.session_state.hybrid_weight = st.slider(
            label="Hybrid Search Weighting",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            help="Weighting for Hybrid Search (0 only dense, 1 only sparse)",
        )

        # Initialize session state if needed
        if "use_reranker_search" not in st.session_state:
            st.session_state.use_reranker_search = system_config.rag_params.use_reranker

        # Show model selection first
        st.session_state.reranker_model = st.selectbox(
            label="Reranker Model",
            options=system_config.rag_params.supported_rerankers,
            index=system_config.rag_params.supported_rerankers.index(
                system_config.rag_params.reranker_model
            ),
            help="Select the reranker model used to improve search result quality",
            key="search_tab_reranker_model",
            disabled=not st.session_state.use_reranker_search,
        )

        # Then add toggle after the model selection
        use_reranker = st.toggle(
            label="Enable Reranker",
            value=st.session_state.use_reranker_search,
            help="Toggle to enable or disable the reranker. When enabled, the system refines search results by re-evaluating how well each chunk matches your query. This typically improves relevance but takes slightly longer. When disabled, raw retrieval results are used without this extra refinement step.",
            key="search_tab_use_reranker_toggle",
            on_change=lambda: setattr(
                st.session_state,
                "use_reranker_search",
                st.session_state.search_tab_use_reranker_toggle,
            ),
        )

        if len(query) > 0:
            # If reranker is disabled, use default model (won't be used anyway)
            reranker_model = (
                st.session_state.reranker_model
                if use_reranker
                else system_config.rag_params.reranker_model
            )

            search_results = st.session_state.databroker.search(
                [query],
                system_config.rag_params.top_k + 10,
                collection="base",
                keywords=st.session_state.keywords,
                filenames=st.session_state.filenames,
                hybrid_weighting=st.session_state.hybrid_weight,
                reranker_model=reranker_model,
                use_reranker=use_reranker,
            )

            if len(search_results[0]) == 0:
                st.header("No Results")
                return

            if "selected_chunk" not in st.session_state:
                st.session_state.selected_chunk = None

            if "edge_thresh" not in st.session_state:
                st.session_state.edge_thresh = 0.5

            nodes = [
                Node(
                    id=r.id,
                    title=r.document,
                    label=r.id,
                    # link= Use this to open up a PDF view?
                    color=(
                        "#ff80ed" if i < system_config.rag_params.top_k else "#065535"
                    ),
                )
                for i, r in enumerate(search_results[0])
            ]

            dist = [
                np.linalg.norm(np.array(j.embedding) - np.array(k.embedding))
                for i, k in enumerate(search_results[0])
                for j in search_results[0][i + 1 :]
            ]
            thresh = min(dist) + st.session_state.edge_thresh * (max(dist) - min(dist))
            edges = [
                Edge(source=k.id, target=j.id, type="CURVE_SMOOTH")
                for i, k in enumerate(search_results[0])
                for h, j in enumerate(search_results[0][i + 1 :])
                if dist[int(i * (i + 1) / 2) + h] < thresh
            ]

            config = Config(
                width=700,
                height=700,
                directed=False,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",  # or "blue"
                collapsible=False,
                node={"labelProperty": "label"},
                # **kwargs e.g. node_size=1000 or node_color="blue"
            )

            results = [
                {
                    "Chunk Source": r.id,
                    "Distance": r.distance,
                    "Chunk": r.document,
                }
                for r in search_results[0]
            ]

            df = pd.DataFrame(results)

            st.session_state.edge_thresh = st.slider("Edge Threshold", 0.0, 1.0, 0.5)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Graph Representation")
                return_value = agraph(nodes=nodes, edges=edges, config=config)

                if return_value and st.session_state.selected_chunk != return_value:
                    st.session_state.selected_chunk = return_value

            with col2:
                st.subheader("Table Representation")

                def highlight_selected(row):
                    color = (
                        "background-color: yellow"
                        if row["Chunk Source"] == st.session_state.selected_chunk
                        else ""
                    )
                    return [color] * len(row)

                styled_df = df.style.apply(highlight_selected, axis=1)

                st.dataframe(styled_df, use_container_width=True, hide_index=True)


def sciencegpt():
    float_init(theme=True, include_unstable_primary=False)
    init_streamlit()
    sidebar()
    chat_tab, retrieval_tab, knowledge_base, survey_tab = st.tabs(
        ["Chat", "Retrieval", "Knowledge Base", "Survey"]
    )
    search(retrieval_tab)
    chat(chat_tab)
    knowledgebase(knowledge_base)
    survey(survey_tab)


if __name__ == "__main__":
    sciencegpt()
