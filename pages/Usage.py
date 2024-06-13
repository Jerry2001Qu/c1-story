import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from src.authentication import check_password

import shutil
import psutil

def run():
    if not check_password():
        st.stop()
        
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    total, used, free = shutil.disk_usage("/")

    st.metric("Storage", f"{used / 2**30:.1f} / {total / 2**30:.1f} GB")
    st.metric("RAM", f"{psutil.virtual_memory()[3] / 1000000000:.1f} / {psutil.virtual_memory()[0] / 1000000000:.1f} GB")

    st.button("Refresh")

if __name__ == "__main__":
    run()