import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from src.tts import TTS_voice, get_voice_ids
from src.heygen import generate_heygen_video, get_heygen_avatars

from pathlib import Path

def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Channel 1 Demo")

    text = st.text_area("Text")
    voice = st.selectbox("Voice", get_voice_ids(), format_func=lambda x: x.name)
    avatar = st.selectbox("Avatar", get_heygen_avatars(), format_func=lambda x: x["avatar_name"])
    
    if st.button("Generate"):
        audio_file = Path("/tmp/audio.mp3")
        TTS_voice(text, audio_file, voice)
        st.audio(str(audio_file), format="audio/mpeg")

        video_file = Path("/tmp/video.mp4")
        generate_heygen_video(audio_file, text, avatar["avatar_id"], video_file)
        st.video(str(video_file))

if __name__ == "__main__":
    run()