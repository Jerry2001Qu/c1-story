import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from src.tts import TTS_voice, get_voice_ids
from src.authentication import check_password
from elevenlabs import VoiceSettings, Voice

from pathlib import Path

def run():
    if not check_password():
        st.stop()

    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Channel 1 Demo")

    text = st.text_area("Text")

    voice = st.selectbox("Voice", get_voice_ids(), format_func=lambda x: x.name)
    stability = st.slider("Stability", 0.0, 1.0, value=0.8)
    style = st.slider("Style", 0.0, 1.0, value=0.75)
    similarity_boost = st.slider("Similarity Boost", 0.0, 1.0, value=0.0)
    use_speaker_boost = st.toggle("Use Speaker Boost", False)

    voice_id = voice.voice_id
    voice_settings = VoiceSettings(stability=stability, style=style, similarity_boost=similarity_boost, use_speaker_boost=use_speaker_boost)

    voice = Voice(voice_id=voice_id, settings=voice_settings)

    start_padding = st.slider("Padding Before", 0.0, 5.0)
    end_padding = st.slider("Padding After", 0.0, 5.0)

    if st.button("Generate"):
        audio_file = Path("/tmp/audio.mp3")
        TTS_voice(text, str(audio_file), voice, start_padding=start_padding, end_padding=end_padding)
        st.audio(str(audio_file), format="audio/mpeg")

if __name__ == "__main__":
    run()