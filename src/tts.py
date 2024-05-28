# TTS

# STREAMLIT
from src.constants import ELEVENLABS_API_KEY
import streamlit as st
# /STREAMLIT

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY
)

@st.cache_data(show_spinner=False)
def TTS(text, filename, previous_text="", next_text=""):
    additional_body_parameters = {}
    if previous_text:
        additional_body_parameters['previous_text'] = previous_text
    if next_text:
        additional_body_parameters['next_text'] = next_text
    audio = client.generate(
        text=text,
        voice=Voice(
            voice_id="LHgN09QqKzsRsniiMpww", # 9f8o652aaiVK5HavyCf1 daniel
            settings=VoiceSettings(stability=0.3, similarity_boost=0.5, style=0.5, use_speaker_boost=True)
        ),
        request_options={
            "additional_body_parameters": additional_body_parameters
        }
    )

    with open(filename, 'wb') as file:
        for chunk in audio:
            if chunk:
                file.write(chunk)