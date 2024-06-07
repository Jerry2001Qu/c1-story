import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from src.tts import TTS

from pathlib import Path

def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Channel 1 Demo")

    text = st.text_area("Text")
    start_padding = st.slider("Padding before", 0.0, 5.0)
    end_padding = st.slider("Padding after", 0.0, 5.0)

    if st.button("Generate"):
        audio_file = Path("/tmp/audio.mp3")
        TTS(text, str(audio_file), start_padding=start_padding, end_padding=end_padding)
        st.audio(str(audio_file), format="audio/mpeg")

if __name__ == "__main__":
    run()