# STREAMLIT
import streamlit as st
# /STREAMLIT

from abc import ABC, abstractmethod
import time
from pathlib import Path

class ErrorHandler(ABC):

    @abstractmethod
    def error(self, msg: str) -> None:
        pass

    @abstractmethod
    def warning(self, msg: str) -> None:
        pass

    @abstractmethod
    def info(self, msg: str) -> None:
        pass

class StreamlitErrorHandler(ErrorHandler):

    def __init__(self, error_bar: st.container, verbosity: bool):
        self.error_bar = error_bar.empty()
        self.verbosity = verbosity
        self.previous_msgs = []
        self.latest = error_bar
    
    def reset(self) -> None:
        self.previous_msgs = []
        self.error_bar.empty()
    
    def get_container(self) -> None:
        first = self.latest.container()
        second = self.latest.empty()

        self.latest = first
        return second
    
    def error(self, msg: str) -> None:
        if msg in self.previous_msgs:
            return
        self.previous_msgs += [msg]
        self.get_container().error(msg, icon="🚨")
    
    def warning(self, msg: str) -> None:
        if msg in self.previous_msgs:
            return
        self.previous_msgs += [msg]
        self.get_container().warning(msg, icon="⚠️")

    def info(self, msg: str) -> None:
        if msg in self.previous_msgs:
            return
        self.previous_msgs += [msg]
        self.get_container().info(msg, icon="ℹ️")
    
    def stream_status(self, msg: str, title: str = None, video: Path = None, audio: Path = None) -> None:
        if not self.verbosity:
            return
        if msg in self.previous_msgs:
            return
        self.previous_msgs += [msg]
        
        def stream(msg: str):
            for word in msg.split(" "):
                yield word + " "
                time.sleep(0.02)

        container = self.get_container().container(border=True)
        if title:
            container.header(title)
        if video:
            container.video(str(video))
        if audio:
            container.audio(str(audio), format="audio/mpeg")
        container.write_stream(stream(msg))