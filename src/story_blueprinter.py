# VideoEditor

# STREAMLIT
from src.clip_manager import ClipManager
from src.news_script import NewsScript, AnchorScriptSection, SOTScriptSection, is_type
from src.story_blueprint_schema import (
    VideoSchema, Timeline, Output, Section, Track, Clip, Asset,
    VideoAsset, ImageAsset, AudioAsset, HtmlAsset, PrebuiltAsset,
    Soundtrack, SoundEffect, SoundEffectType, Font, FitOption, Transition,
    OutputFormat, OutputResolution, OutputQuality
)
import streamlit as st
# /STREAMLIT

from pathlib import Path
from typing import Dict, Tuple
import traceback
import json

from pydantic import HttpUrl

class StoryBlueprinter:
    def __init__(self, news_script: NewsScript, clip_manager: ClipManager,
                 live_anchor: bool, test_mode: bool, music: bool,
                 music_file: Path,
                 output_resolution: Tuple[int, int] = (1920, 1080),
                 bitrate: str = "10M",
                 font: Path = None, logo_path: Path = None,
                 add_logline: bool = True,
                 add_courtesy: bool = True,
                 logline_padding_ratio=1.0909, dub_volume_lufs=-40,
                 lower_volume_duration=1.5, dub_delay=0.5, error_handler=None):
        self.news_script = news_script
        self.clip_manager = clip_manager
        self.live_anchor = live_anchor
        self.test_mode = test_mode
        self.music = music
        self.music_file = music_file
        self.add_logline = add_logline
        self.add_courtesy = add_courtesy
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
        timeline = self._create_timeline()
        output = self._create_output()

        video_schema = VideoSchema(timeline=timeline, output=output)
        with open("video_schema.json", "w") as file:
            json.dump(video_schema.dict(), file, indent=2)
    
    def _create_timeline(self) -> Timeline:
        sections = []
        for section in self.news_script.sections:
            if isinstance(section, SOTScriptSection):
                sections.append(self._process_sot_section(section))
            elif isinstance(section, AnchorScriptSection):
                sections.append(self._process_anchor_section(section))
            else:
                if self.error_handler:
                    self.error_handler.warning(f"Unknown section type: {type(section)}")
        
        soundtrack = self._create_soundtrack() if self.music else None

        return Timeline(
            soundtrack=soundtrack,
            background=(0, 0, 0),
            sections=sections
        )
    
    def _create_soundtrack(self) -> Soundtrack:
        return Soundtrack(
            src=HttpUrl(str(self.music_file)),
            volume=1.0
        )
    
    def _process_sot_section(self, section: SOTScriptSection) -> Section:
        clip = self._create_sot_clip(section)
        track = Track(clips=[clip])
        return Section(
            tracks=[track]
        )

    def _create_sot_clip(self, section: SOTScriptSection) -> Clip:
        asset = VideoAsset(
            type="video",
            src=HttpUrl(str(section.clip.file_path)),
            trim=section.start,
            volume=1.0,
            volumeEffects=[
                SoundEffect(type=SoundEffectType.FADE_IN_FADE_OUT, duration=2)
            ]
        )
        return Clip(
            asset=asset,
            start=0,
            length=section.end - section.start,
            fit=FitOption.CROP
        )
    
    def _process_anchor_section(self, section: AnchorScriptSection) -> Section:
        clips = []
        for broll_info in section.brolls:
            if broll_info["id"] == "Anchor":
                clips.append(self._create_anchor_clip(broll_info, section))
            else:
                clips.append(self._create_broll_clip(broll_info))
        
        track = Track(clips=clips)
        return Section(
            tracks=[track]
        )

    def _create_anchor_clip(self, broll_info: Dict, section: AnchorScriptSection) -> Clip:
        asset = ImageAsset(
            type="image",
            src=HttpUrl(str(self.clip_manager.get_anchor_image_path()))
        )
        return Clip(
            asset=asset,
            start=broll_info['start'],
            length=broll_info['end'] - broll_info['start'],
            fit=FitOption.CROP
        )

    def _create_broll_clip(self, broll_info: Dict) -> Clip:
        clip = self.clip_manager.get_clip(broll_info['id'])
        asset = VideoAsset(
            type="video",
            src=HttpUrl(str(clip.file_path)),
            trim=0,
            volume=1.0
        )
        return Clip(
            asset=asset,
            start=broll_info['start'],
            length=broll_info['end'] - broll_info['start'],
            fit=FitOption.CROP
        )
    
    def _create_output(self) -> Output:
        return Output(
            format=OutputFormat.MP4,
            resolution=OutputResolution.FHD,
            fps=self.fps,
            quality=OutputQuality.HIGH,
            bitrate=self.bitrate
        )
