# TTS

# STREAMLIT
from src.constants import ELEVENLABS_API_KEY
import streamlit as st
# /STREAMLIT

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

from pathlib import Path, PosixPath

client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY
)

@st.cache_data(show_spinner=False)
def get_voice_ids():
    voices = client.voices.get_all()
    return voices.voices

@st.cache_data(show_spinner=False, hash_funcs={PosixPath: lambda x: str(x.resolve())})
def TTS(text, filename, voice_id="LHgN09QqKzsRsniiMpww", previous_text="", next_text=""):
    additional_body_parameters = {}
    # if previous_text:
    #     additional_body_parameters['previous_text'] = previous_text
    # if next_text:
    #     additional_body_parameters['next_text'] = next_text
    audio = client.generate(
        text=text,
        voice=Voice(
            voice_id=voice_id, # 9f8o652aaiVK5HavyCf1 daniel
            settings=VoiceSettings(stability=1.0, similarity_boost=0.0, style=0.3, use_speaker_boost=False)
        ),
        model="eleven_multilingual_v2",
        request_options={
            "additional_body_parameters": additional_body_parameters
        }
    )

    with open(filename, 'wb') as file:
        for chunk in audio:
            if chunk:
                file.write(chunk)

@st.cache_data(show_spinner=False, hash_funcs={Voice: lambda x: x.dict(), PosixPath: lambda x: str(x.resolve())})
def TTS_voice(text, filename, voice: Voice, previous_text="", next_text=""):
    additional_body_parameters = {}
    # if previous_text:
    #     additional_body_parameters['previous_text'] = previous_text
    # if next_text:
    #     additional_body_parameters['next_text'] = next_text
    audio = client.generate(
        text=text,
        voice=voice,
        request_options={
            "additional_body_parameters": additional_body_parameters
        }
    )

    with open(filename, 'wb') as file:
        for chunk in audio:
            if chunk:
                file.write(chunk)