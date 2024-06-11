import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from src.transcription import WhisperResults

from pathlib import Path

def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Channel 1 Demo")

    def save_uploaded(file_data):
        with open("video.mp4" ,"wb") as f:
            f.write(file_data.getbuffer())

    video = st.file_uploader("Video", type=["mp4"], accept_multiple_files=False)
    if video:
        save_uploaded(video)
        with st.expander("Uploaded video"):
            st.video("video.mp4")

        whisper = WhisperResults.from_file(Path("video.mp4"))

        st.write(f"{whisper.has_speech=}, {whisper.language=}, {whisper.no_speech_prob=}")

        st.write(whisper.english_text)
        st.write(whisper.text)
        st.write(whisper.timestamps)

if __name__ == "__main__":
    run()