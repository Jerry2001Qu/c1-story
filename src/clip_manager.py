# ClipManager

# STREAMLIT
from src.transcription import WhisperResults
from src.prompts import run_chain, run_chain_json, match_clip_to_sots_chain, get_sot_chain
import streamlit as st
# /STREAMLIT

from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import moviepy.editor as mp
import traceback
import copy
import os

def folder_has_no_videos(folder_path: Path) -> bool:
    return not list(folder_path.glob("*.mp4"))

class Clip:
    """Represents a single video clip."""

    def __init__(self, clip_id: str, clip_file: Path, clips_folder: Path, error_handler = None):
        self.id = clip_id
        self.file_path = clip_file
        self.clips_folder = clips_folder
        self.error_handler = error_handler

        self.duration = self.load_video().duration

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
        try:
            audio_file_path = self.file_path.with_suffix('.mp3')
            video = self.load_video()
            video.audio.write_audiofile(str(audio_file_path))
            
            self.whisper_results = WhisperResults.from_file(audio_file_path)
        except Exception as e:
            self.whisper_results = WhisperResults("", [], 1.0, False, "Unknown", "")
            raise e

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

    def __init__(self, video_file_path: Path, clips_folder: Path, shotlist: str, anchor_image_path: Path, anchor_voice_id: str, voiceover_voice_id: str, anchor_avatar_id: str, has_splash_screen: bool = False, error_handler = None):
        self.video_file_path = video_file_path
        self.clips_folder = clips_folder
        self.shotlist = shotlist
        self.anchor_image_path = anchor_image_path
        self.anchor_voice_id = anchor_voice_id
        self.voiceover_voice_id = voiceover_voice_id
        self.anchor_avatar_id = anchor_avatar_id
        self.has_splash_screen = has_splash_screen
        self.error_handler = error_handler
        self.clips: List[Clip] = []

    def split_video_into_clips(self):
        """Splits the main video into clips based on scene detection."""
        self.clips_folder.mkdir(parents=True, exist_ok=True)
        if folder_has_no_videos(self.clips_folder):
            from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg

            clip = mp.VideoFileClip(str(self.video_file_path))
            fps = clip.fps
            clip.close()

            scene_list = detect(str(self.video_file_path), AdaptiveDetector(adaptive_threshold=4, min_scene_len=fps))
            status = split_video_ffmpeg(str(self.video_file_path), scene_list, show_progress=False,
                            output_file_template=str(self.clips_folder / "$SCENE_NUMBER.mp4"))
            if status != 0:
                if self.error_handler:
                    self.error_handler.error(f"ERROR: Splitting video into clips failed with code: {status}")
        if self.error_handler:
            self.error_handler.info(f"Detected {len(list(self.clips_folder.glob('*.mp4')))} clips")

    def load_clips(self):
        self.clips = [Clip(file.stem, file, self.clips_folder, error_handler=self.error_handler) for file in sorted(self.clips_folder.glob("*.mp4"))]
        if self.has_splash_screen:
            self.clips = self.clips[1:]

    def match_clips(self):
        sot_matches = run_chain_json(match_clip_to_sots_chain, {"SOTS": self._extract_sots(), "CLIPS_WITH_TRANSCRIPTS": self.get_quotes_str()})

        for sot_match in sot_matches["matches"]:
            try:
                clip_id = sot_match["clip_id"]
                sot_id = sot_match["sot_id"]
                # shotlist_description = sot_match["shotlist_description"]
                if sot_id is None:
                    continue
                
                clip = self.get_clip(clip_id)
                clip.shot_id = int(sot_id)
                # clip.shotlist_description = shotlist_description
                clip.has_quote = 1
            except Exception as e:
                if self.error_handler:
                    self.error_handler.error(f"ERROR: {traceback.format_exc()}")
        
        if self.error_handler:
            sot_matches_str = ""
            for clip in self.clips:
                sot_matches_str += f"{clip.id}: {clip.shot_id}\n"
            self.error_handler.stream_status(sot_matches_str, "SOT Matches")

        # Combine clips that match the same sot and are either next to each other or have one clip in between
        i = 0
        combined_clips = []

        while i < len(self.clips):
            current_clip = self.clips[i]
            group = [current_clip]
            
            while i + 1 < len(self.clips):
                next_clip = self.clips[i + 1]
                next_next_clip = self.clips[i + 2] if i + 2 < len(self.clips) else None
                
                if current_clip.shot_id is not None:
                    if next_clip.shot_id == current_clip.shot_id:
                        group.append(next_clip)
                        i += 1
                    elif next_next_clip and next_next_clip.shot_id == current_clip.shot_id:
                        group.extend([next_clip, next_next_clip])
                        i += 2
                    else:
                        break
                else:
                    break
                
                current_clip = self.clips[i]
            
            if len(group) > 1:
                combined_clip = self.combine_clips(group)
                combined_clips.append(combined_clip)
                if self.error_handler:
                    self.error_handler.stream_status(combined_clip.whisper_results.english_text, f"Combined clips ({combined_clip.id}) with same sot ({current_clip.shot_id})", video=combined_clip.file_path)
            else:
                combined_clips.append(current_clip)
            
            i += 1

        self.clips = combined_clips
        
        used_sot_ids = set()
        for clip in self.clips:
            if clip.shot_id is None:
                continue
            if clip.shot_id in used_sot_ids:
                if self.error_handler:
                    self.error_handler.warning(f"WARNING: Two clips ({clip.id}) were assigned the same SOT ({clip.shot_id}). Removing SOT from the second clip.")
                clip.shot_id = None
                clip.shotlist_description = None
                clip.has_quote = None
            used_sot_ids.add(clip.shot_id)

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
        
        print(groups)

        # Describe each group with shot_id of previous and next clip
        for group in groups:
            try:
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
                try:
                    print(group, shotlist)
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
                        clip.has_quote = int(clip_dict['quote'])
                except ValueError:
                    if self.error_handler:
                        self.error_handler.warning(f"WARNING: Could not match clips, likely content blocked by Gemini. ({group})")
            except Exception as e:
                if self.error_handler:
                    self.error_handler.error(f"ERROR: {traceback.format_exc()}")
        
        if self.error_handler:
            clip_matches_str = ""
            for clip in self.clips:
                clip_matches_str += f"{clip.id}: {clip.shot_id}\n"
            self.error_handler.stream_status(clip_matches_str, "Clip Matches")

    
    def combine_clips(self, clips: List[Clip]) -> Clip:
        """Combines multiple video clips into a new clip."""
        if len(clips) == 0:
            return None
        if len(clips) == 1:
            return clips[0]

        # Load all video clips
        video_clips = [clip.load_video() for clip in clips]

        print([clip.duration for clip in video_clips])

        # Concatenate the video clips
        combined_video = mp.concatenate_videoclips(video_clips, method="compose")

        # Generate the new file name
        new_id = "_".join([clip.file_path.stem for clip in clips])
        new_file_name = new_id + ".mp4"
        new_file_path = self.clips_folder / new_file_name

        # Write the combined video to the new file
        combined_video.write_videofile(str(new_file_path), logger=None)

        # Delete the original clip files
        for clip in clips:
            clip.file_path.unlink()

        # Update the first clip with the new file path and transcribe
        clips[0].id = new_id
        clips[0].file_path = new_file_path
        clips[0].duration = combined_video.duration
        clips[0].transcribe_clip()
        return clips[0]

    def _extract_sots(self) -> str:
        """Extracts and parses soundbites (SOTs) from the shotlist."""
        sots = run_chain(get_sot_chain, {"SHOTLIST": self.shotlist})
        return sots

    def transcribe_clips(self, multi: bool = True):
        os.environ['GRPC_POLL_STRATEGY'] = 'poll'

        if multi:
            def transcribe_and_handle_errors(clip):
                try:
                    clip.transcribe_clip()
                    return True, None, clip
                except Exception as e:
                    return False, traceback.format_exc(), clip
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(transcribe_and_handle_errors, clip) for clip in self.clips]

                results = []
                for future in as_completed(futures):
                    success, error, clip = future.result()
                    results.append(success)
                    if not success and self.error_handler:
                        self.error_handler.error(f"ERROR: {error}")
                    else:
                        if self.error_handler:
                            if clip.whisper_results.no_speech_prob < 0.3:
                                self.error_handler.stream_status(clip.whisper_results.english_text, f"Identified Speech ({clip.id})", clip.file_path)

                successful_transcriptions = sum(results)
                failed_transcriptions = len(self.clips) - successful_transcriptions
                
                if self.error_handler:
                    self.error_handler.stream_status(f"Transcription complete. Successful: {successful_transcriptions}, Failed: {failed_transcriptions}")
        else:
            for clip in self.clips:
                try:
                    clip.transcribe_clip()
                except Exception as e:
                    if self.error_handler:
                        self.error_handler.error(f"ERROR: {traceback.format_exc()}")

    def break_up_clips(self, max_duration=8.0):
        num_clips_before = len(self.clips)
        for clip in self.clips:
            if clip.has_quote:
                continue
            if clip.duration > max_duration:
                num_clips = int(clip.duration / max_duration) + 1
                clip_duration = clip.duration / num_clips
                clip_file = clip.file_path
                clip_file_name = clip_file.stem
                for i in range(num_clips):
                    start = i * clip_duration
                    end = (i + 1) * clip_duration
                    if end > clip.duration:
                        end = clip.duration
                    new_clip_file = self.clips_folder / f"{clip_file_name}_{i}.mp4"
                    video_clip = mp.VideoFileClip(str(clip_file)).subclip(start, end)
                    if not new_clip_file.exists():
                        video_clip.write_videofile(str(new_clip_file), logger=None)

                    new_clip = copy.deepcopy(clip)
                    new_clip.file_path = new_clip_file
                    new_clip.duration = video_clip.duration
                    new_clip.id = f"{clip.id}_{i}"
                    self.clips.append(new_clip)

                    if self.error_handler:
                        self.error_handler.stream_status(f"Split clip {clip.id} into {new_clip.id}", video=new_clip_file)
                self.clips.remove(clip)
                clip_file.unlink()
        num_clips_after = len(self.clips)

        if self.error_handler:
            self.error_handler.info(f"Split {num_clips_before} clips into {num_clips_after} clips")

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
        os.environ['GRPC_POLL_STRATEGY'] = 'poll'

        def generate_description(clip):
            try:
                clip.generate_full_description(story_title)
                return (clip, None)
            except ValueError as e:
                return (clip, e)
        
        # STREAMLIT
        progress_bar = st.progress(0.0)
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_clip = {executor.submit(generate_description, clip): clip for clip in self.clips}
            for i, future in enumerate(as_completed(future_to_clip)):
                clip, exception = future.result()
                if exception:
                    if self.error_handler:
                        self.error_handler.warning(f"WARNING: Could not generate full description for clip {clip.id}, likely content blocked by Gemini.")
                if self.error_handler:
                    self.error_handler.stream_status(clip.full_description, f"Analyzing clip {clip.id}", clip.file_path)
                progress_bar.progress((i + 1) / len(self.clips))
        progress_bar.progress(1.0)
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
    
    def get_anchor_voice_id(self):
        return self.anchor_voice_id
    
    def get_voiceover_voice_id(self):
        return self.voiceover_voice_id
    
    def get_anchor_avatar_id(self):
        return self.anchor_avatar_id
