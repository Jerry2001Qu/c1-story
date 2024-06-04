# STREAMLIT
from src.clip_manager import ClipManager
from src.news_script import NewsScript, AnchorScriptSection, SOTScriptSection, is_type
from src.language import Language
import streamlit as st
# /STREAMLIT

from pathlib import Path
from typing import Dict, Tuple, List

import moviepy.editor as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class EDLExporter:
    """Handles EDL generation."""

    def __init__(self, news_script: NewsScript, clip_manager: ClipManager,
                 output_resolution: Tuple[int, int] = (768, 432),
                 font: Path = None, font_size=None, logo_path: Path = None,
                 logline_padding=40, dub_volume_lufs=-40,
                 lower_volume_duration=3.0, dub_delay=0.5, error_handler=None):
        self.news_script = news_script
        self.clip_manager = clip_manager
        self.output_resolution = output_resolution
        self.font = font
        self.font_size = font_size
        self.logo_path = logo_path
        self.logline_padding = logline_padding
        self.dub_volume_lufs = dub_volume_lufs
        self.lower_volume_duration = lower_volume_duration
        self.dub_delay = dub_delay
        self.error_handler = error_handler

    def generate_edl(self, output_file: Path = Path("output.edl")):
        """Generates an EDL file describing the video structure."""
        edl_entries: List[Dict] = []
        total_duration = 0.0
        for i, section in enumerate(self.news_script.sections):
            try:
                if is_type(section, SOTScriptSection):
                    if section.clip is not None:
                        entry = self._generate_sot_edl_entry(section, total_duration)
                        edl_entries.append(entry)
                        total_duration += entry["duration"]
                elif is_type(section, AnchorScriptSection):
                    entries = self._generate_anchor_edl_entries(section, total_duration)
                    if entries: # Only add if there's any b-roll
                        edl_entries.extend(entries)
                        total_duration += entries[-1]["out"] - entries[0]["in"]
                else:
                    print(f"ERROR: Unknown section type: {type(section)}")
                # STREAMLIT
                st.write(f"Generated EDL entry for section {i+1}/{len(self.news_script.sections)}")
                # /STREAMLIT
            except Exception as e:
                if self.error_handler:
                    self.error_handler.warning(f"Error when generating EDL entry for section {section.id}: {e}")

        with open(output_file, "w") as f:
            for entry in edl_entries:
                f.write(
                    f"{entry['source']}\t{entry['in']:.2f}\t{entry['out']:.2f}\t{entry['track']}\n"
                )

    def _generate_sot_edl_entry(self, section: SOTScriptSection, start_time: float) -> Dict:
        """Generates an EDL entry for a SOTScriptSection."""
        clip_duration = section.clip.load_video().duration

        if section.language == Language.from_str("English") or section.dub_audio_file is None:
            end_time = min(section.end, start_time + clip_duration)
        else:
            dub_audio = mp.AudioFileClip(str(section.dub_audio_file))
            dub_end_time = self.lower_volume_duration + self.dub_delay + dub_audio.duration
            end_time = min(dub_end_time, start_time + clip_duration)

        return {
            "source": section.clip.file_path.name,
            "in": start_time,
            "out": end_time,
            "track": 1,  # Assuming SOTs are on track 1
        }

    def _generate_anchor_edl_entries(self, section: AnchorScriptSection, start_time: float) -> List[Dict]:
        """Generates EDL entries for an AnchorScriptSection's B-roll."""
        entries = []
        current_time = start_time
        for broll_info in section.brolls:
            if broll_info["id"] == "Anchor":
                source = "anchor.jpg" # Placeholder, adjust as needed
            else:
                source = self.clip_manager.get_clip(broll_info['id']).file_path.name
            broll_duration = broll_info['end'] - broll_info['start']
            entries.append({
                "source": source,
                "in": current_time,
                "out": current_time + broll_duration,
                "track": 2,  # Assuming B-roll is on track 2
            })
            current_time += broll_duration
        return entries