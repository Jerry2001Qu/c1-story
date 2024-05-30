# ClipManager

# STREAMLIT
from src.transcription import WhisperResults
from src.prompts import run_chain, run_chain_json, match_clip_to_sots_chain, get_sot_chain
import streamlit as st
# /STREAMLIT

from pathlib import Path
from typing import List, Dict, Optional
import moviepy.editor as mp
import time

def is_folder_empty(folder_path: Path) -> bool:
    return not any(folder_path.iterdir())

class Clip:
    """Represents a single video clip."""

    def __init__(self, clip_id: str, clip_file: Path, clips_folder: Path):
        self.id = clip_id
        self.file_path = clip_file
        self.clips_folder = clips_folder

        self.shot_id: Optional[int] = None
        self.shotlist_description: Optional[str] = None
        self.has_quote: Optional[bool] = None
        self.whisper_results: Optional[WhisperResults] = None
        self.full_description: Optional[str] = None

    def __repr__(self):
        return f"""{self.id} ({self.shot_id}, quote: {self.has_quote}): {self.shotlist_description}"""

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
        from src.gemini import full_description
        return full_description(self.file_path, self.shotlist_description, story_title)

class ClipManager:
    """Manages video clips, including splitting, description, and speech recognition."""

    def __init__(self, video_file_path: Path, clips_folder: Path, shotlist: str, anchor_image_path: Path, has_splash_screen: bool = False, error_handler = None):
        self.video_file_path = video_file_path
        self.clips_folder = clips_folder
        self.shotlist = shotlist
        self.anchor_image_path = anchor_image_path
        self.has_splash_screen = has_splash_screen
        self.error_handler = error_handler
        self.clips: List[Clip] = []

    def split_video_into_clips(self):
        """Splits the main video into clips based on scene detection."""
        self.clips_folder.mkdir(parents=True, exist_ok=True)
        if is_folder_empty(self.clips_folder):
            from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
            scene_list = detect(str(self.video_file_path), AdaptiveDetector(adaptive_threshold=4, min_scene_len=1))
            status = split_video_ffmpeg(str(self.video_file_path), scene_list, show_progress=True,
                            output_file_template=str(self.clips_folder / "$SCENE_NUMBER.mp4"))
            if status != 0:
                st.error(f"Splitting video into clips failed with code: {status}")

    def load_clips(self):
        self.clips = [Clip(file.stem, file, self.clips_folder) for file in sorted(self.clips_folder.glob("*.mp4"))]
        if self.has_splash_screen:
            self.clips = self.clips[1:]

    def match_clips(self):
        sot_matches = run_chain_json(match_clip_to_sots_chain, {"SOTS": self._extract_sots(), "CLIPS_WITH_TRANSCRIPTS": self.get_quotes_str()})
        used_sot_ids = set()
        for sot_match in sot_matches["matches"]:
            clip_id = sot_match["clip_id"]
            sot_id = sot_match["sot_id"]
            shotlist_description = sot_match["shotlist_description"]
            if sot_id is None:
                continue
            if sot_id in used_sot_ids:
                print(f"WARNING: Two clips were matched to the same sot {sot_id}")
                continue

            clip = self.get_clip(clip_id)
            clip.shot_id = int(sot_id)
            clip.shotlist_description = shotlist_description
            clip.has_quote = 1

            used_sot_ids.add(sot_id)

        # Find groups of clips where has_quote is None
        groups = []
        current_group = []
        for clip in self.clips:
            if clip.has_quote is None:
                current_group.append(clip)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
        if current_group:
            groups.append(current_group)

        # Describe each group with shot_id of previous and next clip
        for group in groups:
            # Get the shot_id of the previous and next clip
            previous_shot_id = None
            next_shot_id = None
            if group:  # Check if group is not empty
                group_start_index = self.clips.index(group[0])
                if group_start_index > 0:
                    previous_shot_id = self.clips[group_start_index - 1].shot_id
                group_end_index = self.clips.index(group[-1])
                if group_end_index < len(self.clips) - 1:
                    next_shot_id = self.clips[group_end_index + 1].shot_id

            shotlist_start_idx = 0
            if previous_shot_id is not None:
                shotlist_start_idx = self.shotlist.find(f"{previous_shot_id+1}. ")
                if shotlist_start_idx == -1:
                    shotlist_start_idx = 0
            shotlist_end_idx = len(self.shotlist)
            if next_shot_id is not None:
                shotlist_end_idx = self.shotlist.find(f"{next_shot_id}. ")
                if shotlist_end_idx == -1:
                    shotlist_end_idx = len(self.shotlist)
            shotlist = self.shotlist[shotlist_start_idx:shotlist_end_idx]

            # Describe the group
            clips_xml = self.describe_clips(group, shotlist, previous_shot_id=previous_shot_id, next_shot_id=next_shot_id)
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
                try:

                    clip = self.get_clip(str(clip_dict['id']))
                except StopIteration:
                    print(f"Clip not found with id, {clip_dict['id']}")
                    continue
                clip.shot_id = clip_dict['shot']
                clip.shotlist_description = clip_dict["description"]
                clip.has_quote = clip_dict['quote']

    def _extract_sots(self) -> str:
        """Extracts and parses soundbites (SOTs) from the shotlist."""
        sots = run_chain(get_sot_chain, {"SHOTLIST": self.shotlist})
        return sots

    def transcribe_clips(self):
        for clip in self.clips:
            clip.transcribe_clip()

    def get_quotes_str(self):
        output = ""
        for clip in self.clips:
            if clip.whisper_results.has_speech:
                output += f"""<clip>
ID {clip.id}: {clip.whisper_results.english_text}
</clip>
"""
        return output

    def generate_full_descriptions(self, story_title: str):
        # STREAMLIT
        progress_bar = st.progress(0.0)
        for i, clip in enumerate(self.clips):
            try:
                clip.generate_full_description(story_title)
            except ValueError:
                if self.error_handler:
                    self.error_handler.error(f"Error in generating full description for clip {clip.id}, likely content blocked by Gemini.")
            progress_bar.progress(i / (len(self.clips)-1))
        # /STREAMLIT

    def describe_clips(self, clips, shotlist, previous_shot_id, next_shot_id) -> Dict:
        """Uses Gemini to match clips to shot descriptions."""
        from src.gemini import describe_clips
        return describe_clips(clips, shotlist, previous_shot_id, next_shot_id)

    def get_clip(self, clip_id):
        for clip in self.clips:
            if clip.id == clip_id:
                return clip
        return None
    
    def get_anchor_image_clip(self):
        return mp.ImageClip(str(self.anchor_image_path))
