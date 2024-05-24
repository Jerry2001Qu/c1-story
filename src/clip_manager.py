# ClipManager

# STREAMLIT
from src.transcription import WhisperResults
from src.gemini import full_description, describe_clips
import streamlit as st
# /STREAMLIT

from pathlib import Path
from typing import List, Dict, Optional
import moviepy.editor as mp

from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg

def is_folder_empty(folder_path: Path) -> bool:
    return not any(folder_path.iterdir())

class Clip:
    """Represents a single video clip."""

    def __init__(self, clip_id: str, clip_file: Path, shot_id: int, shotlist_description: str, has_quote: bool, clips_folder: Path):
        self.id = clip_id
        self.file_path = clip_file
        self.shot_id = shot_id
        self.shotlist_description = shotlist_description
        self.has_quote = has_quote
        self.clips_folder = clips_folder

        self.whisper_results: Optional[WhisperResults] = None
        self.full_description: Optional[str] = None

    def load_video(self) -> mp.VideoFileClip:
        """Loads the video clip using moviepy."""
        return mp.VideoFileClip(str(self.file_path))

    def transcribe_clip(self):
        """Performs speech recognition on the clip's audio."""
        self.whisper_results = WhisperResults.from_file(self.file_path)

    def generate_full_description(self, story_title: str):
        """Generates a detailed description of the clip."""
        description_file = self.clips_folder / "descriptions" / f"{self.id}.txt"
        if description_file.exists():
            with open(description_file, "r") as f:
                self.full_description = f.read()
        else:
            description_file.parent.mkdir(parents=True, exist_ok=True)
            self.full_description = self._get_full_description_from_gemini(story_title)
            with open(description_file, "w") as f:
                f.write(self.full_description)

    def _get_full_description_from_gemini(self, story_title: str) -> str:
        """Calls the Gemini API to generate a full description for the clip."""
        return full_description(self.file_path, self.shotlist_description, story_title)

class ClipManager:
    """Manages video clips, including splitting, description, and speech recognition."""

    def __init__(self, video_file_path: Path, clips_folder: Path, shotlist: str):
        self.video_file_path = video_file_path
        self.clips_folder = clips_folder
        self.shotlist = shotlist
        self.clips: List[Clip] = []

    def split_video_into_clips(self):
        """Splits the main video into clips based on scene detection."""
        self.clips_folder.mkdir(parents=True, exist_ok=True)
        if is_folder_empty(self.clips_folder):
            scene_list = detect(str(self.video_file_path), AdaptiveDetector(adaptive_threshold=4, min_scene_len=1))
            st.write(scene_list)
            from contextlib import contextmanager, redirect_stdout
            from io import StringIO
            import os

            @contextmanager
            def st_capture():
                with StringIO() as stdout, redirect_stdout(stdout):
                    old_write = stdout.write

                    def new_write(string):
                        ret = old_write(string)
                        st.write(f"{stdout.getvalue()}\n".encode()) 
                        return ret
                    
                    stdout.write = new_write
                    yield
            
            with st_capture():
                split_video_ffmpeg(str(self.video_file_path), scene_list,
                                output_file_template=f"{str(self.clips_folder)}/$SCENE_NUMBER.mp4", show_output=True)
                print("hi")
            st.write(list(self.clips_folder.glob("*.mp4")))

    def load_and_match_clips(self):
        """Loads clips, creating Clip objects."""
        clips_xml = self.describe_clips()
        for clip_data in clips_xml["response"]:
            clip_dict = {}
            for part in clip_data["clip"]:
                for key, val in part.items():
                    if not val:
                        clip_dict[key] = val
                    if isinstance(val, str):
                        clip_dict[key] = val.strip()
                    else:
                        clip_dict[key] = val
            clip_file = self.clips_folder / f"{clip_dict['id']}.mp4"
            x = clip_file
            st.write(f"{x.exists()}, {x}")
            clip = Clip(clip_dict['id'], clip_file, clip_dict['shot'], clip_dict['description'], clip_dict['quote'], self.clips_folder)
            self.clips.append(clip)
    
    def transcribe_clips(self):
        for clip in self.clips:
            clip.transcribe_clip()
    
    def generate_full_descriptions(self, story_title: str):
        for clip in self.clips:
            clip.generate_full_description(story_title)

    def describe_clips(self) -> Dict:
        """Uses Gemini to match clips to shot descriptions."""
        return describe_clips(self.clips_folder, self.shotlist)

    def get_clip(self, clip_id):
        for clip in self.clips:
            if clip.id == clip_id:
                return clip
        return None