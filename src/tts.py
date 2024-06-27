# TTS

# STREAMLIT
from src.constants import ELEVENLABS_API_KEY
import streamlit as st
# /STREAMLIT

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

from pathlib import Path, PosixPath
import uuid

import moviepy.editor as mp

client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY
)

@st.cache_data(show_spinner=False)
def get_voice_ids():
    voices = client.voices.get_all()
    return voices.voices

def create_silent_audio_clip(duration: float) -> mp.ColorClip:
    return mp.AudioClip(lambda t: 0, duration=duration)

@st.cache_data(show_spinner=False, hash_funcs={PosixPath: lambda x: str(x.resolve())})
def TTS(text, filename, voice_id="LHgN09QqKzsRsniiMpww", previous_text="", next_text="", start_padding=0, end_padding=0):
    additional_body_parameters = {}
    # if previous_text:
    #     additional_body_parameters['previous_text'] = previous_text
    # if next_text:
    #     additional_body_parameters['next_text'] = next_text
    audio = client.generate(
        text=text,
        voice=Voice(
            voice_id=voice_id, # 9f8o652aaiVK5HavyCf1 daniel
            settings=VoiceSettings(stability=0.4, similarity_boost=0.75, style=0.0, use_speaker_boost=True)
        ),
        model="eleven_multilingual_v2",
        request_options={
            "additional_body_parameters": additional_body_parameters
        }
    )

    filename = Path(filename)
    tmp_file = filename.with_name(f"{filename.stem}_tmp{filename.suffix}")
    with open(tmp_file, 'wb') as file:
        for chunk in audio:
            if chunk:
                file.write(chunk)
    
    audio_clip = mp.AudioFileClip(tmp_file)
    audio_clip = audio_clip.subclip(0, audio_clip.duration - 0.1)

    if start_padding:
        start_pad_clip = create_silent_audio_clip(start_padding)
        audio_clips = [start_pad_clip, audio_clip]
        if end_padding:
            end_pad_clip = create_silent_audio_clip(end_padding)
            audio_clips += [end_pad_clip]
        
        audio_clip = mp.concatenate_audioclips(audio_clips)
    audio_clip.write_audiofile(str(filename), logger=None)
    Path(tmp_file).unlink()

@st.cache_data(show_spinner=False, hash_funcs={Voice: lambda x: x.dict(), PosixPath: lambda x: str(x.resolve())})
def TTS_voice(text, filename, voice: Voice, previous_text="", next_text="", start_padding=0, end_padding=0):
    additional_body_parameters = {}
    # if previous_text:
    #     additional_body_parameters['previous_text'] = previous_text
    # if next_text:
    #     additional_body_parameters['next_text'] = next_text
    audio = client.generate(
        text=text,
        voice=voice,
        model="eleven_multilingual_v2",
        request_options={
            "additional_body_parameters": additional_body_parameters
        }
    )

    tmp_file = f"/tmp/{str(uuid.uuid4())}.mp3"
    with open(tmp_file, 'wb') as file:
        for chunk in audio:
            if chunk:
                file.write(chunk)
    
    audio_clip = mp.AudioFileClip(tmp_file)
    audio_clip = audio_clip.subclip(0, audio_clip.duration - 0.1)

    if start_padding:
        start_pad_clip = create_silent_audio_clip(start_padding)
        audio_clips = [start_pad_clip, audio_clip]
        if end_padding:
            end_pad_clip = create_silent_audio_clip(end_padding)
            audio_clips += [end_pad_clip]
        
        audio_clip = mp.concatenate_audioclips(audio_clips)
    audio_clip.write_audiofile(str(filename), logger=None)
    Path(tmp_file).unlink()
