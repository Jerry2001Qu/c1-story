import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from src.tts import TTS_voice, get_voice_ids
from src.heygen import animate_anchor, get_heygen_avatars
from src.error_handler import StreamlitErrorHandler

from pathlib import Path

def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )
    
    with st.sidebar:
        error_bar = st.container()
    
    error_handler = StreamlitErrorHandler(error_bar, True)

    st.write("# Channel 1 Demo")

    text = st.text_area("Text")
    voice = st.selectbox("Voice", get_voice_ids(), format_func=lambda x: x.name)
    avatar = st.selectbox("Avatar", get_heygen_avatars(), format_func=lambda x: x["avatar_name"])
    test = st.toggle("Test Mode", True)
    
    if st.button("Generate"):
        audio_file = Path("/tmp/audio.mp3")
        TTS_voice(text, audio_file, voice)
        st.audio(str(audio_file), format="audio/mpeg")

        video_file = Path("/tmp/video.mp4")
        animate_anchor(audio_file, text, avatar["avatar_id"], video_file, error_handler=error_handler, test=test)
        st.video(str(video_file))

if __name__ == "__main__":
    run()