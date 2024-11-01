import os

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from app import sciencegpt

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.set_page_config(layout="wide")
st.markdown(
    """
        <style>
            .appview-container .main .block-container {{
                padding-top: {padding_top}rem;
                padding-bottom: {padding_bottom}rem;
                }}

        </style>""".format(
        padding_top=0, padding_bottom=1
    ),
    unsafe_allow_html=True,
)

with open("./src/configs/user_config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)


def signup():
    try:
        (
            email_of_registered_user,
            username_of_registered_user,
            name_of_registered_user,
        ) = authenticator.register_user(
            pre_authorized=config["pre-authorized"]["emails"]
        )
        if email_of_registered_user:
            st.success("User registered successfully")
        with open("./src/configs/user_config.yaml", "w") as file:
            yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        st.error(e)


col1, col2 = st.columns(2)

with col1:
    try:
        authenticator.login()
        if st.session_state["authentication_status"] is False:
            st.error("Username/password is incorrect")
            st.session_state.logged_in = False
        elif st.session_state["authentication_status"] is None:
            st.warning("Please enter your username and password")
            st.session_state.logged_in = False
    except Exception as e:
        st.error(e)

if st.session_state["authentication_status"]:
    authenticator.logout(location="sidebar")
    if not st.session_state.logged_in:
        st.toast(f'Welcome *{st.session_state["name"]}*', icon="👋")
        st.session_state.logged_in = True
    sciencegpt()

else:
    with col2:
        signup()
