# STREAMLIT
import streamlit as st
# /STREAMLIT

from abc import ABC, abstractmethod
import time
from pathlib import Path
import random

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
        if "flip" not in st.session_state:
            st.session_state["flip"] = 1
        for _ in range(st.session_state["flip"]):
            self.error_bar = error_bar.empty()
        if st.session_state["flip"] == 1:
            st.session_state["flip"] = 2
        else:
            st.session_state["flip"] = 1
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
        if msg is None:
            msg = ""
        _hash = msg + (title if title else "")
        if _hash in self.previous_msgs:
            return
        self.previous_msgs += [_hash]
        
        def stream(msg: str):
            if self.verbosity:
                for word in msg.split(" "):
                    yield word + " "
                    time.sleep(0.02)
            else:
                yield msg

        container = self.get_container().container(border=True)
        if title:
            container.header(title)
        if video:
            container.video(str(video))
        if audio:
            container.audio(str(audio), format="audio/mpeg")
        container.write_stream(stream(msg))
