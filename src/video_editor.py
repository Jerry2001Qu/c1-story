# VideoEditor

# STREAMLIT
from src.clip_manager import ClipManager
from src.news_script import NewsScript, AnchorScriptSection, SOTScriptSection, is_type
# /STREAMLIT

from pathlib import Path
from typing import Dict, Tuple

import moviepy.editor as mp

class VideoEditor:
    """Handles video editing, including assembling clips and B-roll."""

    def __init__(self, news_script: NewsScript, clip_manager: ClipManager, 
                 output_resolution: Tuple[int, int] = (640, 480)):
        self.news_script = news_script
        self.clip_manager = clip_manager
        self.output_resolution = output_resolution

    def assemble_video(self, output_file: Path = Path("output.mp4")):
        """Assembles the final video from script sections and B-roll."""
        video_clips = []
        for section in self.news_script.sections:
            if is_type(section, SOTScriptSection):
                if section.clip is not None:
                    video_clips.append(self._process_sot_section(section))
            elif is_type(section, AnchorScriptSection):
                video_clips.append(self._process_anchor_section(section))
            else:
                print(f"ERROR: Unknown section type: {type(section)}")

        final_video = mp.concatenate_videoclips(video_clips, method="compose")
        final_video.write_videofile(str(output_file), fps=24, threads=8, 
                                    verbose=False, logger=None)

    def _process_sot_section(self, section: SOTScriptSection) -> mp.VideoFileClip:
        """Processes a SOTScriptSection, extracting and resizing the clip."""
        clip = section.clip.load_video()
        clip = resize_image_clip(clip, self.output_resolution)
        clip = clip.subclip(section.start, min(section.end, clip.duration))
        return clip

    def _process_anchor_section(self, section: AnchorScriptSection) -> mp.VideoFileClip:
        """Processes an AnchorScriptSection, assembling B-roll and audio."""
        broll_clips = []
        for broll_info in section.brolls:
            broll_clip = self._load_and_process_broll(broll_info)
            broll_clips.append(broll_clip)

        combined_broll = mp.concatenate_videoclips(broll_clips, method="compose")
        voiceover_audio = mp.AudioFileClip(str(section.anchor_audio_file))

        combined_broll = combined_broll.set_audio(voiceover_audio)
        combined_broll = combined_broll.set_duration(voiceover_audio.duration)
        return combined_broll

    def _load_and_process_broll(self, broll_info: Dict) -> mp.VideoFileClip:
        """Loads, processes (resizing, speed adjustment), and returns a B-roll clip."""
        clip = self.clip_manager.get_clip(broll_info['id'])
        broll_file = clip.file_path
        broll_clip = mp.VideoFileClip(str(broll_file))
        broll_clip = cap_loudness(broll_clip)

        broll_start = broll_info['start']
        broll_end = broll_info['end']
        broll_duration = broll_end - broll_start

        if broll_clip.duration < broll_duration:
            speed_factor = broll_clip.duration / broll_duration
            if speed_factor > 0.7:  # Adjust the threshold as needed
                print(f"Broll {broll_info['id']}: B-ROLL too short, adjusting speed ({speed_factor:.2f})")
                broll_clip = broll_clip.fx(mp.vfx.speedx, speed_factor)
            else:
                print(f"Broll {broll_info['id']}: B-ROLL too short, using full clip")

        broll_clip = broll_clip.subclip(0, min(broll_duration, broll_clip.duration))
        broll_clip = broll_clip.set_duration(broll_duration)
        broll_clip = resize_image_clip(broll_clip, self.output_resolution)
        return broll_clip

import moviepy.editor as mp
import pyloudnorm as pyln
import numpy as np

def resize_image_clip(image_clip, target_resolution):
    target_width, target_height = target_resolution
    target_aspect = target_width / target_height
    original_width, original_height = image_clip.size
    original_aspect = original_width / original_height

    if original_aspect > target_aspect:
        new_width = target_width
        new_height = target_width / original_aspect
    else:
        new_height = target_height
        new_width = target_height * original_aspect

    return image_clip.resize(newsize=(int(new_width), int(new_height)))

def cap_loudness(clip, max_lufs=-50):
    audio_data = np.array(list(clip.audio.iter_frames(fps=48000)))

    meter = pyln.Meter(rate=48000)  # The sample rate should match the rate of audio_data
    current_loudness = meter.integrated_loudness(audio_data)
    if current_loudness < max_lufs:
        return clip

    adjustment_factor = 10 ** ((max_lufs - current_loudness) / 20)
    adjusted_audio = clip.audio.volumex(adjustment_factor)
    return clip.set_audio(adjusted_audio)