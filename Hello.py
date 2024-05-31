import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Channel 1 Demo")

if __name__ == "__main__":
    run()