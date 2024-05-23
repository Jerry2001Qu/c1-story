# DataLoader

# STREAMLIT
from src.reuters import get_item, get_assets, download_asset
# /STREAMLIT

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import requests

import html

class DataLoader(ABC):
    """Abstract base class for loading story data."""

    @abstractmethod
    def load_storyline(self) -> str:
        """Loads the storyline text."""
        pass 

    @abstractmethod
    def load_shotlist(self) -> str:
        """Loads the shotlist text."""
        pass
    
    @abstractmethod
    def get_story_title(self) -> str:
        """Loads the story title."""
        pass

    @abstractmethod
    def get_video_file_path(self) -> Path:
        """Returns the path to the video file."""
        pass

class LocalDataLoader(DataLoader):
    """Loads story data from local files."""

    def __init__(self, folder_path: str):
        """
        Initializes the LocalDataLoader.

        Args:
            folder_path: Path to the folder containing the data files.
        """
        self.folder_path = Path(folder_path)

    def load_storyline(self) -> str:
        """Loads the storyline from 'storyline.txt'."""
        storyline_file = self.folder_path / "storyline.txt"
        with open(storyline_file, "r") as f:
            return f.read()

    def load_shotlist(self) -> str:
        """Loads the shotlist from 'shotlist.txt'."""
        shotlist_file = self.folder_path / "shotlist.txt"
        with open(shotlist_file, "r", encoding="utf-8") as f:
            return f.read()
    
    def get_story_title(self) -> str:
        """Loads the story title."""
        return self.folder_path.stem

    def get_video_file_path(self) -> Path:
        """Returns the path to the video file 'video.mp4'."""
        return self.folder_path / "video.mp4"

class ReutersAPIDataLoader(DataLoader):
    """Loads story data from Reuters' API"""

    def __init__(self, reuters_id: str, storage_path: Path):
        self.reuters_id = reuters_id
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.pulled_reuters_api: bool = False

        self.shotlist: Optional[str] = None
        self.storyline: Optional[str] = None
        self.language: Optional[str] = None
        self.location: Optional[str] = None
    
    def pull_reuters_api(self) -> None:
        if self.pulled_reuters_api:
            return
        
        bodyXhtml, headline, language, located = get_item(self.reuters_id)

        bodyhtml = extract_str_between(html.unescape(bodyXhtml), "<body>", "</body>")
        shotlist = extract_str_between(bodyhtml, "</p><p>1.", "</p><p>STORY:")[7:-13]
        shotlist = "\n".join(shotlist.split("</p><p>"))

        storyline = extract_str_between(bodyhtml, "<p>STORY:", "</body>")[9:-11]
        storyline = "\n".join([line.strip() for line in storyline.split("</p><p>")[:-1]])

        self.shotlist = shotlist
        self.storyline = storyline
        self.story_title = headline
        self.language = language
        self.location = located

        video_asset = get_assets(self.reuters_id)[0]
        video_url, asset_type = download_asset(self.reuters_id, video_asset["uri"])
        res = requests.get(video_url)
        video_file_path = self.storage_path / "video.mp4"
        with open(video_file_path, "wb") as file:
            file.write(res.content)
        
        self.video_file_path = video_file_path

        self.pulled_reuters_api = True
        
    def load_storyline(self) -> str:
        self.pull_reuters_api()
        return self.storyline

    def load_shotlist(self) -> str:
        self.pull_reuters_api()
        return self.shotlist
    
    def get_story_title(self) -> str:
        self.pull_reuters_api()
        return self.story_title

    def get_video_file_path(self) -> Path:
        self.pull_reuters_api()
        return self.video_file_path

def extract_str_between(text: str, left_tag: str, right_tag: str) -> str:
    return text[text.find(left_tag):text.rfind(right_tag)+len(right_tag)]
