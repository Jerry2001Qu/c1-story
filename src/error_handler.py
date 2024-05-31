# STREAMLIT
import streamlit as st
# /STREAMLIT

from abc import ABC, abstractmethod

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

    def __init__(self, error_bar: st.container):
        self.error_bar = error_bar
    
    def error(self, msg: str) -> None:
        self.error_bar.error(msg, icon="🚨")
    
    def warning(self, msg: str) -> None:
        self.error_bar.warning(msg, icon="⚠️")

    def info(self, msg: str) -> None:
        self.error_bar.info(msg, icon="ℹ️")