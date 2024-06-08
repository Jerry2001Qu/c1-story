# Transcription

# STREAMLIT
from src.constants import OPENAI_API_KEY
from src.language import Language
from src.hashing import sha256sum, hash_audio_file
import streamlit as st
# /STREAMLIT

from dataclasses import dataclass
from typing import List
from pathlib import Path, PosixPath

from openai import OpenAI

openai_client = OpenAI(api_key=OPENAI_API_KEY)

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
        transcript = openai_transcribe(file.resolve())

        segments = transcript.segments
        language = Language.from_str(transcript.language) if transcript.language else None
        timestamps = [Word(word['word'], word['start'], word['end']) for word in transcript.words]

        text = "".join([seg["text"] for seg in segments])

        if segments:
            min_no_speech_prob = min([seg["no_speech_prob"] for seg in segments])
        else:
            min_no_speech_prob = 1.0
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