import os
from typing import Tuple, Callable

import streamlit as st

from kidney.reports.session import SessionState

ENV_DASHBOARD_PASSWORD = "DASHBOARD_PASSWORD"

print(os.environ[ENV_DASHBOARD_PASSWORD])


def is_authenticated(pwd: str) -> bool:
    return pwd == os.environ[ENV_DASHBOARD_PASSWORD]


def login(blocks: Tuple) -> str:
    style, element = blocks
    style.markdown("""
    <style>
        input { -webkit-text-security: disc; }
    </style>
    """, unsafe_allow_html=True)
    return element.text_input("Password")


def clean_blocks(*blocks):
    for block in blocks:
        block.empty()


def with_password(session_state: SessionState):
    assert ENV_DASHBOARD_PASSWORD in os.environ, "Password is not defined!"

    if session_state["password"]:
        login_blocks = None
        password = session_state["password"]
    else:
        login_blocks = st.empty(), st.empty()
        password = login(login_blocks)
        session_state["password"] = password

    def wrapper(entry_point: Callable):

        def wrapped():
            if is_authenticated(password):
                if login_blocks is not None:
                    clean_blocks(*login_blocks)
                entry_point()
            elif password:
                st.info("Please enter a valid password")

        return wrapped

    return wrapper
