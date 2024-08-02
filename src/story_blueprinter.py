# STREAMLIT
from src.clip_manager import ClipManager
from src.news_script import NewsScript, AnchorScriptSection, SOTScriptSection
from src.story_blueprint_schema import *
from src.gcp import GCSManager
import streamlit as st
# /STREAMLIT

from pathlib import Path
from typing import Dict, Tuple
import traceback
import json
import moviepy.editor as mp

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

        self.gcs = GCSManager()

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

        if self.add_logline:
            overlay = self._create_overlay()
        else:
            overlay = None

        soundtrack = self._create_soundtrack() if self.music else None

        return Timeline(
            soundtrack=soundtrack,
            overlay=overlay,
            background=(0, 0, 0),
            sections=sections
        )

    def _create_overlay(self) -> Overlay:
        clip = self._create_logline(self.news_script.headline)
        track = Track(clips=[clip])
        return Overlay(
            tracks=[track]
        )

    def _create_logline(self, text: str) -> Clip:
        asset = PrebuiltAsset(
            type=AssetType.PREBUILT,
            key=PrebuiltTypes.LOGLINE,
            data=PrebuiltLogline(text=text.upper())
        )
        return Clip(
            asset=asset,
            start=0,
            length=self._get_total_duration()
        )

    def _get_total_duration(self) -> float:
        total_duration = 0
        for section in self.news_script.sections:
            if isinstance(section, SOTScriptSection):
                total_duration += section.end - section.start
            elif isinstance(section, AnchorScriptSection):
                total_duration += section.anchor_audio_clip.duration
        return total_duration

    def _create_soundtrack(self) -> Soundtrack:
        url = self.gcs.upload_to_gcs_url(self.music_file, bucket_name="public-heygen-assets")
        return Soundtrack(
            src=str(url),
            volume=1.0
        )

    def _process_sot_section(self, section: SOTScriptSection) -> Section:
        tracks = []
        if section.dub_audio_file is None:
            sot_tracks = self._create_sot_tracks(section)
            tracks += sot_tracks
            section_length = section.end - section.start
        else:
            sot_tracks = self._create_dubbed_sot_tracks(section)
            tracks += sot_tracks
            section_length = sot_tracks[0].clips[0].length

        if self.add_courtesy and section.clip.courtesy:
            credit_clip = self._create_courtesy(section.clip.courtesy, section_length)
            credit_track = Track(clips=[credit_clip])
            tracks += [credit_track]

        if section.is_interview():
            byline_clip = self._create_byline(section.name, section.title, section_length)
            byline_track = Track(clips=[byline_clip])
            tracks += [byline_track]

        return Section(
            tracks=tracks,
            length=section_length - 0.1,
            volumeEffects=[
                SoundEffect(type=SoundEffectType.FADE_IN_FADE_OUT, duration=(1.0/self.fps)*2)
            ]
        )

    def _create_courtesy(self, courtesy: str, duration: float) -> Clip:
        asset = PrebuiltAsset(
            type=AssetType.PREBUILT,
            key=PrebuiltTypes.CREDITS,
            data=PrebuiltCredits(text=courtesy.upper())
        )
        return Clip(
            asset=asset,
            start=0,
            length=duration
        )

    def _create_byline(self, name: str, title: str, duration: float) -> Clip:
        asset = PrebuiltAsset(
            type=AssetType.PREBUILT,
            key=PrebuiltTypes.BYLINE,
            data=PrebuiltByline(name=name.upper(), title=title.upper())
        )
        return Clip(
            asset=asset,
            start=0,
            length=duration
        )
    
    def _create_dubbed_sot_tracks(self, section: SOTScriptSection) -> List[Track]:
        dub_audio = mp.AudioFileClip(str(section.dub_audio_file))

        dub_start_time = self.lower_volume_duration + self.dub_delay
        dub_end_time = dub_audio.duration + dub_start_time
        if dub_start_time > (dub_audio.duration - section.start):
            return self._create_sot_tracks(section)
        else:
            original_audio_url = self.gcs.upload_to_gcs_url(section.clip.file_path, bucket_name="public-heygen-assets")
            fade_out_audio_asset = AudioAsset(
                type="audio",
                src=str(original_audio_url),
                trim=section.start,
                volume=1.0,
                volumeEffects=[
                    SoundEffect(type=SoundEffectType.FADE_OUT, duration=self.lower_volume_duration*1.1)
                ]
            )
            fade_out_audio_clip = Clip(
                asset=fade_out_audio_asset,
                start=0,
                length=self.lower_volume_duration*1.1
            )

            capped_audio_asset = AudioAsset(
                type="audio",
                src=str(original_audio_url),
                trim=self.lower_volume_duration,
                volume=0.2
            )
            capped_audio_clip = Clip(
                asset=capped_audio_asset,
                start=self.lower_volume_duration,
                length=dub_end_time - dub_audio.duration - self.dub_delay
            )

            dubed_audio_url = self.gcs.upload_to_gcs_url(section.dub_audio_file, bucket_name="public-heygen-assets")
            dubed_audio_asset = AudioAsset(
                type="audio",
                src=str(dubed_audio_url),
                volume=1.0
            )
            dubed_audio_clip = Clip(
                asset=dubed_audio_asset,
                start=dub_start_time,
                length=dub_audio.duration
            )

            video_url = self.gcs.upload_to_gcs_url(section.clip.file_path, bucket_name="public-heygen-assets")
            trimmed_duration = section.clip.duration - section.start
            if trimmed_duration > dub_end_time:
                video_asset = VideoAsset(
                    type="video",
                    src=str(video_url),
                    trim=section.start,
                    volume=0.0
                )
                video_clip = Clip(
                    asset=video_asset,
                    start=0,
                    length=dub_end_time
                )
            else:
                speed_factor = trimmed_duration / dub_end_time
                video_asset = VideoAsset(
                    type="video",
                    src=str(video_url),
                    trim=section.start,
                    volume=1.0,
                    speed=speed_factor
                )
                video_clip = Clip(
                    asset=video_asset,
                    start=0,
                    length=dub_end_time
                )
            
            return [
                Track(clips=[video_clip]),
                Track(clips=[fade_out_audio_clip]),
                Track(clips=[capped_audio_clip]),
                Track(clips=[dubed_audio_clip])
            ]

    def _create_sot_tracks(self, section: SOTScriptSection) -> List[Track]:
        url = self.gcs.upload_to_gcs_url(section.clip.file_path, bucket_name="public-heygen-assets")
        asset = VideoAsset(
            type="video",
            src=str(url),
            trim=section.start,
            volume=1.0,
            volumeEffects=[
                SoundEffect(type=SoundEffectType.FADE_IN_FADE_OUT, duration=2)
            ]
        )
        clip = Clip(
            asset=asset,
            start=0,
            length=section.end - section.start,
            fit=FitOption.CROP
        )
        video_track = Track(clips=[clip])
        return [video_track]

    def _process_anchor_section(self, section: AnchorScriptSection) -> Section:
        clips = []
        for broll_info in section.brolls:
            if broll_info["id"] == "Anchor":
                clips.append(self._create_anchor_clip(broll_info, section))
            else:
                clips.append(self._create_broll_clip(broll_info))
        video_track = Track(clips=clips)

        clips = []
        for broll_info in section.brolls:
            if broll_info["id"] != "Anchor":
                clip = self.clip_manager.get_clip(broll_info["id"])
                if clip.courtesy:
                    clips.append(self._create_courtesy(clip.courtesy, clip.duration))
        credits_track = Track(clips=clips)

        voiceover_track = self._create_voiceover_track(section)

        section_length = section.anchor_audio_clip.duration
        return Section(
            tracks=[video_track, credits_track, voiceover_track],
            length=section_length - 0.1
        )

    def _create_voiceover_track(self, section: AnchorScriptSection) -> Track:
        url = self.gcs.upload_to_gcs_url(section.anchor_audio_file, bucket_name="public-heygen-assets")
        asset = AudioAsset(
            type="audio",
            src=str(url),
            volume=1.0
        )
        clip = Clip(
            asset=asset,
            start=0,
            length=section.anchor_audio_clip.duration
        )
        return Track(clips=[clip])

    def _create_anchor_clip(self, broll_info: Dict, section: AnchorScriptSection) -> Clip:
        url = self.gcs.upload_to_gcs_url(section.anchor_video_file, bucket_name="public-heygen-assets")
        asset = VideoAsset(
            type="video",
            src=str(url),
            trim=broll_info['start']
        )
        return Clip(
            asset=asset,
            start=broll_info['start'],
            length=broll_info['end'] - broll_info['start'],
            fit=FitOption.CROP
        )

    def _create_broll_clip(self, broll_info: Dict) -> Clip:
        clip = self.clip_manager.get_clip(broll_info['id'])
        url = self.gcs.upload_to_gcs_url(clip.file_path, bucket_name="public-heygen-assets")
        asset = VideoAsset(
            type="video",
            src=str(url),
            trim=0,
            volume=0.0,
            speed=broll_info['speed_factor'] if 'speed_factor' in broll_info else None
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
