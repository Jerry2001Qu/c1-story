from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import streamlit as st
from openai import OpenAI

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


@dataclass
class WhisperResults:
    text: str
    timestamps: List = field(repr=False)
    no_speech_prob: float
    has_speech: bool
    language: str

    @classmethod
    def from_file(cls, file):
        audio_file = open(file, "rb")
        transcript = openai_client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )

        text = transcript.text
        timestamps = transcript.words
        language = transcript.language

        min_no_speech_prob = 1
        has_speech = True

        # if segments:
        #     min_no_speech_prob = min([seg.no_speech_prob for seg in segments])
        # else:
        #     min_no_speech_prob = 1.0
        # has_speech = min_no_speech_prob < 0.2

        return cls(text, timestamps, min_no_speech_prob, has_speech, language)

@dataclass
class Clip:
    id: int
    path: str
    description: str = None
    duration: float = None
    is_sot: bool = None

    vad: bool = None
    whisper: WhisperResults = None

    def __init__(self, id, path):
        self.id = id
        self.path = path

    def scores_summary(self):
        if self.whisper:
            return f"CLIP {self.id} (vad: {self.vad}, whisper: {self.whisper.has_speech} {self.whisper.no_speech_prob}): {self.whisper.text}"
        else:
            return f"CLIP {self.id} (vad: {self.vad})"