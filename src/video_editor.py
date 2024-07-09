# VideoEditor

# STREAMLIT
from src.clip_manager import ClipManager
from src.news_script import NewsScript, AnchorScriptSection, SOTScriptSection, is_type
from src.movie_utils import resize_image_clip, cap_loudness, cap_loudness_audio_clip, set_loudness, set_loudness_audio_clip
from src.heygen import animate_anchor
import streamlit as st
# /STREAMLIT

from pathlib import Path
from typing import Dict, Tuple
import traceback

import moviepy.editor as mp
from moviepy.audio.fx.audio_loop import audio_loop
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class VideoEditor:
    """Handles video editing, including assembling clips and B-roll."""

    def __init__(self, news_script: NewsScript, clip_manager: ClipManager,
                 live_anchor: bool, test_mode: bool, music: bool,
                 music_file: Path,
                 output_resolution: Tuple[int, int] = (1920, 1080),
                 bitrate: str = "10M",
                 font: Path = None, logo_path: Path = None,
                 logline_padding_ratio=1.0909, dub_volume_lufs=-40,
                 lower_volume_duration=1.5, dub_delay=0.5, error_handler=None):
        self.news_script = news_script
        self.clip_manager = clip_manager
        self.live_anchor = live_anchor
        self.test_mode = test_mode
        self.music = music
        self.music_file = music_file
        self.output_resolution = output_resolution
        self.bitrate = bitrate
        self.font = font
        self.logo_path = logo_path
        self.logline_padding = int((self.output_resolution[0] - (self.output_resolution[0] // logline_padding_ratio)) // 2)
        self.dub_volume_lufs = dub_volume_lufs
        self.lower_volume_duration = lower_volume_duration
        self.dub_delay = dub_delay
        self.error_handler = error_handler
        self.fps = 29.97

    def assemble_video(self, output_file: Path = Path("output.mp4")):
        """Assembles the final video from script sections and B-roll."""
        video_clips = []
        # STREAMLIT
        progress_bar = st.progress(0.0)
        for i, section in enumerate(self.news_script.sections):
            try:
                if is_type(section, SOTScriptSection):
                    if section.clip is not None:
                        video_clips.append(self._process_sot_section(section))
                elif is_type(section, AnchorScriptSection):
                    if section.text:
                        video_clips.append(self._process_anchor_section(section))
                else:
                    print(f"ERROR: Unknown section type: {type(section)}")
                progress_bar.progress((i+1) / len(self.news_script.sections))
            except Exception as e:
                if self.error_handler:
                    self.error_handler.warning(f"Error when assembling section {section.id}: {traceback.format_exc()}")

        logline = self.news_script.headline
        concat_video = mp.concatenate_videoclips(video_clips, method="compose")
        final_video = self._add_logline(concat_video, logline)
        if self.music:
            final_video = self._add_background_music(final_video)
        final_video = final_video.fadein((1.0/self.fps)*10).fadeout((1.0/self.fps)*10)

        st.write("Rendering final video file")
        if self.error_handler:
            self.error_handler.info("Rendering final video")
        final_video.write_videofile(str(output_file), fps=self.fps, threads=8,
                                    codec='libx264', audio_codec='aac',
                                    bitrate=self.bitrate, logger="bar")
        # /STREAMLIT

    def _process_sot_section(self, section: SOTScriptSection) -> mp.VideoFileClip:
        """Processes a SOTScriptSection, extracting and resizing the clip."""
        clip = section.clip.load_video()
        clip = resize_image_clip(clip, self.output_resolution)
        clip = set_loudness(clip)

        if section.dub_audio_file is None:
            clip = clip.subclip(section.start, min(section.end, clip.duration))
        else:
            clip = clip.subclip(section.start)
            dub_audio = mp.AudioFileClip(str(section.dub_audio_file))
            dub_audio = set_loudness_audio_clip(dub_audio)

            # 1. Calculate the time the dub starts
            dub_start_time = self.lower_volume_duration + self.dub_delay
            dub_end_time = dub_audio.duration + dub_start_time

            if dub_start_time > clip.duration:
                return clip

            # 2. Original audio with fadeout and lower volume
            original_audio = clip.audio
            original_audio = mp.concatenate_audioclips([
                original_audio.subclip(0, self.lower_volume_duration*1.1).audio_fadeout(self.lower_volume_duration*1.1).subclip(0, self.lower_volume_duration),
                cap_loudness_audio_clip(original_audio.subclip(self.lower_volume_duration, clip.duration), self.dub_volume_lufs).set_start(self.lower_volume_duration)
            ])

            # 3. Delayed dubbed audio
            delayed_dub_audio = dub_audio.set_start(dub_start_time)

            # 4. Combine original and dubbed audio
            new_audio = mp.CompositeAudioClip([original_audio, delayed_dub_audio])

            # 5. Adjust video speed if needed
            if clip.duration > dub_end_time:
                clip = clip.subclip(0, dub_end_time)
            elif clip.duration < dub_end_time:
                speed_factor = clip.duration / dub_end_time
                clip = clip.fx(mp.vfx.speedx, speed_factor)

                if self.error_handler:
                    self.error_handler.info(f"INFO: Section {section.id}, dubed SOT is too long. Slowing down SOT with factor {speed_factor}")

            # 6. Set new audio
            clip = clip.set_audio(new_audio)
            clip = clip.set_duration(dub_end_time)
        
        if section.is_interview():
            clip = self._add_byline(clip, section.name, section.title)
        clip = clip.subclip(0, clip.duration - 0.1)
        clip = clip.audio_fadein((1.0/self.fps)*2).audio_fadeout((1.0/self.fps)*2)
        return clip

    def _process_anchor_section(self, section: AnchorScriptSection) -> mp.VideoFileClip:
        """Processes an AnchorScriptSection, assembling B-roll and audio."""
        broll_clips = []
        for broll_info in section.brolls:
            if broll_info["id"] == "Anchor":
                broll_clip = self._load_and_process_anchor(broll_info, section)
            else:
                broll_clip = self._load_and_process_broll(broll_info)
            broll_clips.append(broll_clip)

        if not broll_clips:
            if self.error_handler:
                self.error_handler.warning(f"Anchor section {section.id} had no broll. Skipping section in final video.")
            return None
        combined_broll = mp.concatenate_videoclips(broll_clips, method="compose")
        voiceover_audio = mp.AudioFileClip(str(section.anchor_audio_file))

        if combined_broll.duration < voiceover_audio.duration:
            speed_factor = combined_broll.duration / voiceover_audio.duration
            if speed_factor > 0.7:
                if speed_factor < 0.99:
                    if self.error_handler:
                        self.error_handler.info(f"INFO: Brolls in section {section.id} are too short, adjusting speed ({speed_factor:.2f})")
                combined_broll = combined_broll.fx(mp.vfx.speedx, speed_factor)
            else:
                if self.error_handler:
                    self.error_handler.info(f"INFO: Brolls in section {section.id} are too short, adding Anchor shot")
                anchor_clip = self._load_and_process_anchor({"start": combined_broll.duration, "end": voiceover_audio.duration}, section)
                combined_broll = mp.concatenate_videoclips([combined_broll, anchor_clip], method="compose")

        combined_broll = combined_broll.set_audio(voiceover_audio)
        combined_broll = combined_broll.set_duration(voiceover_audio.duration)

        return combined_broll.subclip(0, combined_broll.duration - 0.1)
    
    def _load_and_process_anchor(self, broll_info: Dict, section: AnchorScriptSection) -> mp.VideoFileClip:
        if self.live_anchor:
            anchor_video_file = self.news_script.folder / f"{section.id}_anchor.mp4"
            animate_anchor(section.anchor_audio_file, section.text, self.clip_manager.get_anchor_avatar_id(), anchor_video_file, test=self.test_mode)
            if self.error_handler:
                self.error_handler.stream_status(section.text, title="Generated anchor video", video=anchor_video_file)
            anchor_clip = mp.VideoFileClip(str(anchor_video_file))

            anchor_start = broll_info['start']
            anchor_end = broll_info['end']

            anchor_clip = anchor_clip.subclip(anchor_start, anchor_end)
            anchor_clip = resize_image_clip(anchor_clip, self.output_resolution)
        else:
            anchor_clip = self.clip_manager.get_anchor_image_clip()
            duration = broll_info['end'] - broll_info['start']
            anchor_clip = anchor_clip.set_duration(duration)
            anchor_clip = resize_image_clip(anchor_clip, self.output_resolution)
        return anchor_clip

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
            if speed_factor > 0.7:
                if speed_factor < 0.99:
                    if self.error_handler:
                        self.error_handler.info(f"INFO: Broll {broll_info['id']} is too short, adjusting speed ({speed_factor:.2f})")
                broll_clip = broll_clip.fx(mp.vfx.speedx, speed_factor)
            else:
                if self.error_handler:
                    self.error_handler.info(f"INFO: Broll {broll_info['id']} is too short, video will be slow ({speed_factor:.2f}).")
                broll_clip = broll_clip.fx(mp.vfx.speedx, speed_factor)

        broll_clip = broll_clip.subclip(0, min(broll_duration, broll_clip.duration))
        broll_clip = broll_clip.set_duration(broll_duration)
        broll_clip = resize_image_clip(broll_clip, self.output_resolution)
        return broll_clip
    
    def _add_byline(self, clip: mp.VideoFileClip, name: str, title: str) -> mp.VideoFileClip:
        """Adds a lower-third byline above the logline to the given clip."""

        output_width, output_height = clip.w, clip.h
        bottom_margin = 2

        byline_height = int(output_height * (162/1080))
        inner_content_height = int(byline_height * (82/162))
        byline_padding = (byline_height - inner_content_height) // 2

        top_inner_content_height = int(inner_content_height * (33/82))
        bottom_inner_content_height = int(inner_content_height * (25/82))
        middle_inner_content_spacing = inner_content_height - top_inner_content_height - bottom_inner_content_height

        name_text_clip = self._create_raw_text_clip(name.upper(), top_inner_content_height, (255, 255, 255, 255))
        title_text_clip = self._create_raw_text_clip(title.upper(), bottom_inner_content_height, (255, 255, 255, 255))

        inner_content_width = max(name_text_clip.w, title_text_clip.w)
        byline_width = inner_content_width + byline_padding * 2

        byline_x = self.logline_padding
        logline_height = int(output_height * (150/1080))
        byline_y = output_height - self.logline_padding - byline_height - logline_height - bottom_margin

        bg_clip = (
            mp.ColorClip(size=(byline_width, byline_height), color=(0, 5, 52, 255 * 0.9))
            .set_position(
                (byline_x, byline_y)
            )
            .set_duration(clip.duration)
        )

        name_x = byline_x + byline_padding
        name_y = byline_y + byline_padding

        name_text_clip = name_text_clip.set_position(
            (name_x, name_y)
        ).set_duration(clip.duration)

        title_x = byline_x + byline_padding
        title_y = byline_y + byline_padding + top_inner_content_height + middle_inner_content_spacing

        title_text_clip = title_text_clip.set_position(
            (title_x, title_y)
        ).set_duration(clip.duration)

        final_elements = [clip, bg_clip, name_text_clip, title_text_clip]
        return mp.CompositeVideoClip(final_elements)

    def _add_logline(self, clip: mp.VideoFileClip, logline_text: str) -> mp.VideoFileClip:
        """Adds a lower-third logline to the given clip."""

        output_width, output_height = clip.w, clip.h
        logo_logline_padding = 2

        # Calculate logline dimensions based on video resolution and padding
        logline_height = int(output_height * (150/1080))  # 10% of video height
        logline_width = output_width - self.logline_padding * 2
        if self.logo_path:
            logline_width -= logline_height
            logline_width -= logo_logline_padding

        logline_x = self.logline_padding
        logline_y = output_height - self.logline_padding - logline_height

        # 1. Create a white background ImageClip
        bg_clip = (
            mp.ColorClip(size=(logline_width, logline_height), color=(255, 255, 255, 255 * 0.9))
            .set_position(
                (logline_x, logline_y)
            )
            .set_duration(clip.duration)
        )

        # 2. Create text ImageClip
        text_clip = self._create_text_clip(logline_text.upper(), logline_width, logline_height)
        text_clip = text_clip.set_position(
            (logline_x, logline_y)
        ).set_duration(clip.duration)

        # 3. Load and position logo (if provided)
        if self.logo_path:
            logo = (
                mp.ImageClip(str(self.logo_path))
                .set_opacity(1.0)
                .resize(height=logline_height)
                .set_position(
                    (logline_x + logline_width + logo_logline_padding, logline_y)
                )
                .set_duration(clip.duration)
            )
            final_elements = [clip, bg_clip, text_clip, logo]
        else:
            final_elements = [clip, bg_clip, text_clip]

        # 4. Compose final clip
        return mp.CompositeVideoClip(final_elements)
    
    def _create_raw_text_clip(self, text: str, height: int, text_color = (0, 0, 0, 255)) -> mp.ImageClip:
        """Creates a transparent ImageClip with the given text."""
        ref_text_image = Image.new("RGBA", (1, height), (0, 0, 0, 0))
        ref_draw = ImageDraw.Draw(ref_text_image)

        def get_font_size_for_height(font_path, desired_height):
            font_size = 1
            font = ImageFont.truetype(font_path, font_size)
            text_bbox = ref_draw.textbbox((0, 0), "H", font=font)
            while text_bbox[3] - text_bbox[1] < desired_height:
                font_size += 1
                font = ImageFont.truetype(font_path, font_size)
                text_bbox = ref_draw.textbbox((0, 0), "H", font=font)
            return font_size - 1

        # Load the font
        if not self.font:
            font = ImageFont.load_default()
        else:
            font_path = str(self.font)
            font_size = get_font_size_for_height(font_path, height)
            font = ImageFont.truetype(font_path, font_size)

        ref_text_bbox = ref_draw.textbbox((0, 0), text, font=font)
        ref_text_width = ref_text_bbox[2] - ref_text_bbox[0]
        text_x = 0
        text_y = -ref_text_bbox[1]

        text_image = Image.new("RGBA", (ref_text_width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_image)
        draw.text((text_x, text_y), text, font=font, fill=text_color)
        return mp.ImageClip(np.array(text_image))

    def _create_text_clip(self, text: str, width: int, height: int) -> mp.ImageClip:
        """Creates a transparent ImageClip with the given text."""
        text_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_image)

        def get_font_size_for_height(font_path, desired_height):
            font_size = 1
            font = ImageFont.truetype(font_path, font_size)
            text_bbox = draw.textbbox((0, 0), "H", font=font)
            while text_bbox[3] - text_bbox[1] < desired_height:
                font_size += 1
                font = ImageFont.truetype(font_path, font_size)
                text_bbox = draw.textbbox((0, 0), "H", font=font)
            return font_size - 1

        # Load the font
        if not self.font:
            font = ImageFont.load_default()
        else:
            font_path = str(self.font)
            font_size = get_font_size_for_height(font_path, int(height * (59/150)))
            font = ImageFont.truetype(font_path, font_size)
        
        reference_char = "H"
        ref_text_bbox = draw.textbbox((0, 0), reference_char, font=font)
        ref_text_height = ref_text_bbox[3] - ref_text_bbox[1]

        # Get text size, adjust height, calculate position
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = self.logline_padding // 2
        text_y = -ref_text_bbox[1] + (height - ref_text_height) // 2

        # Draw the text
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0, 255))

        return mp.ImageClip(np.array(text_image))
    
    def _add_background_music(self, video: mp.VideoClip) -> mp.VideoClip:
        background_music = mp.AudioFileClip(str(self.music_file))
        background_music = cap_loudness_audio_clip(background_music, -40)
        background_music = audio_loop(background_music, duration=video.duration)
        return video.set_audio(mp.CompositeAudioClip([video.audio, background_music]))
