# Transcription

# STREAMLIT
from src.constants import OPENAI_API_KEY, DEEPGRAM_API_KEY
from src.language import Language
from src.hashing import sha256sum, hash_audio_file
import streamlit as st
# /STREAMLIT

from dataclasses import dataclass
from typing import List
from pathlib import Path, PosixPath

from openai import OpenAI

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
)

import httpx

openai_client = OpenAI(api_key=OPENAI_API_KEY)
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

@dataclass
class Word:
    word: str
    start: float
    end: float

    def get_adjusted_start(self, threshold=2.0):
        if self.end - self.start < threshold:
            return self.start
        else:
            return self.end - 2.0

@dataclass
class WhisperResults:
    """Represents the results of Whisper speech recognition on an audio file."""

    text: str
    timestamps: List[Word]
    no_speech_prob: float
    has_speech: bool
    language: Language
    english_text: str

    @classmethod
    def from_file(cls, file: Path):
        """
        Performs speech recognition on an audio file using OpenAI's Whisper API.

        Args:
            file: The audio file to transcribe.

        Returns:
            A WhisperResults object containing the transcription data.
        """
        transcription, language = deepgram_transcribe(file)

        timestamps = [Word(word.punctuated_word, word.start, word.end) for word in transcription.words]
        text = transcription.transcript
        confidence = transcription.confidence
        min_no_speech_prob = 1.0 - confidence
        language = Language.from_str(language) if language else None

        has_speech = bool(text)

        if not language:
            english_text = text
        elif language == Language.from_str("english"):
            english_text = text
        else:
            english_text = openai_translate(file.resolve())

        return cls(text, timestamps, min_no_speech_prob, has_speech, language, english_text)

@st.cache_data(show_spinner=False, hash_funcs={PosixPath: hash_audio_file})
def openai_translate(abs_file_path: Path):
    translation = openai_client.audio.translations.create(
        file=abs_file_path,
        model="whisper-1",
        response_format="json",
    )
    return translation.text

@st.cache_data(show_spinner=False, hash_funcs={PosixPath: hash_audio_file})
def openai_transcribe(abs_file_path: Path):
    return openai_client.audio.transcriptions.create(
            file=abs_file_path,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"]
        )

@st.cache_data(show_spinner=False, hash_funcs={PosixPath: hash_audio_file})
def deepgram_transcribe(abs_file_path: Path):
    with open(abs_file_path, "rb") as file:
        buffer_data = file.read()

    payload = {
        "buffer": buffer_data,
    }

    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        utterances=True,
        punctuate=True,
        detect_language=True,
    )

    response = deepgram.listen.prerecorded.v("1").transcribe_file(
        payload, options, timeout=httpx.Timeout(300.0, connect=10.0)
    )

    return response.results.channels[0].alternatives[0], response.results.channels[0].detected_language

def get_adjusted_timestamps(timestamps, start_timestamp, end_timestamp, max_duration):
    exact_start = timestamps[0].start
    exact_end = timestamps[-1].end
    
    before_timestamp_idx = timestamps.index(start_timestamp) - 1
    after_timestamp_idx = timestamps.index(end_timestamp) + 1

    if before_timestamp_idx >= 0:
        before_timestamp = timestamps[before_timestamp_idx]
        min_start = exact_start - 0.5
        max_start = exact_start - 0.01
        between_start = exact_start - ((2/3) * (exact_start - before_timestamp.end))
        start = max(min_start, min(max_start, between_start))
    else:
        start = max(exact_start - 0.5, 0.0)
    
    if after_timestamp_idx < len(timestamps):
        after_timestamp = timestamps[after_timestamp_idx]
        min_end = exact_end + 0.01
        max_end = exact_end + 0.5
        between_end = exact_end + ((2/3) * (after_timestamp.start - exact_end))
        end = min(max_end, max(min_end, between_end))
    else:
        end = min(exact_end + 0.5, max_duration)
    
    return start, end
