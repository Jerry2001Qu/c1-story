# VideoEditor

# STREAMLIT
from src.clip_manager import ClipManager
from src.news_script import NewsScript, AnchorScriptSection, SOTScriptSection, is_type
# /STREAMLIT

from pathlib import Path
from typing import Dict, Tuple

import moviepy.editor as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class VideoEditor:
    """Handles video editing, including assembling clips and B-roll."""

    def __init__(self, news_script: NewsScript, clip_manager: ClipManager,
                 output_resolution: Tuple[int, int] = (640, 480),
                 font: Path = None, font_size=24, logo_path: Path = None,
                 logline_padding=40):
        self.news_script = news_script
        self.clip_manager = clip_manager
        self.output_resolution = output_resolution
        self.font = font
        self.font_size = font_size
        self.logo_path = logo_path
        self.logline_padding = logline_padding

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

        logline = self.news_script.get_anchor_sections()[0].logline
        concat_video = mp.concatenate_videoclips(video_clips, method="compose")
        final_video = self._add_logline(concat_video, logline)
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

    def _add_logline(self, clip: mp.VideoFileClip, logline_text: str) -> mp.VideoFileClip:
        """Adds a lower-third logline to the given clip."""

        output_width, output_height = clip.w, clip.h

        # Calculate logline dimensions based on video resolution and padding
        logline_height = int(output_height * 0.1)  # 10% of video height
        logline_width = output_width - self.logline_padding * 2
        if self.logo_path:
            logline_width -= logline_height
        
        logline_x = self.logline_padding
        logline_y = output_height - self.logline_padding - logline_height

        # 1. Create a white background ImageClip
        bg_clip = (
            mp.ColorClip(size=(logline_width, logline_height), color=(255, 255, 255))
            .set_position(
                (logline_x, logline_y)
            )
            .set_duration(clip.duration)
        )

        # 2. Create text ImageClip
        text_clip = self._create_text_clip(logline_text, logline_width, logline_height)
        text_clip = text_clip.set_position(
            (logline_x, logline_y)
        ).set_duration(clip.duration)

        # 3. Load and position logo (if provided)
        if self.logo_path:
            logo = (
                mp.ImageClip(str(self.logo_path))
                .set_opacity(0.8)
                .resize(height=logline_height)
                .set_position(
                    (logline_x + logline_width, logline_y)
                )
                .set_duration(clip.duration)
            )
            final_elements = [clip, bg_clip, text_clip, logo]
        else:
            final_elements = [clip, bg_clip, text_clip]

        # 4. Compose final clip
        return mp.CompositeVideoClip(final_elements)
    
    def _create_text_clip(self, text: str, width: int, height: int) -> mp.ImageClip:
        """Creates a transparent ImageClip with the given text."""
        text_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_image)

        # Load the font
        if not self.font:
            font = ImageFont.load_default()
        else:
            font = ImageFont.truetype(str(self.font), self.font_size)

        # Get text size, adjust height, calculate position
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = int((text_bbox[3] - text_bbox[1]) * 3 / 2)  # Adjust height
        text_x = self.logline_padding // 3
        text_y = (height - text_height) // 2

        # Draw the text
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0, 255))

        return mp.ImageClip(np.array(text_image))

import moviepy.editor as mp
import pyloudnorm as pyln
import numpy as np

def resize_image_clip(image_clip, target_resolution):
    target_width, target_height = target_resolution
    original_width, original_height = image_clip.size

    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    if width_ratio > height_ratio:
        resized_clip = image_clip.resize(width=target_width)
    else:
        resized_clip = image_clip.resize(height=target_height)

    cropped_clip = mp.vfx.crop(
        resized_clip,
        x1=(resized_clip.w - target_width) / 2,
        y1=(resized_clip.h - target_height) / 2,
        x2=(resized_clip.w + target_width) / 2,
        y2=(resized_clip.h + target_height) / 2
    )

    return cropped_clip

def cap_loudness(clip, max_lufs=-30):
    audio_data = np.array(list(clip.audio.iter_frames(fps=48000)))

    meter = pyln.Meter(rate=48000, block_size=min(0.4, clip.duration)) # block_size must not exceed clip duration
    current_loudness = meter.integrated_loudness(audio_data)
    if current_loudness < max_lufs:
        return clip

    adjustment_factor = 10 ** ((max_lufs - current_loudness) / 20)
    adjusted_audio = clip.audio.volumex(adjustment_factor)
    return clip.set_audio(adjusted_audio)