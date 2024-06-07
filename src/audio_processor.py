# AudioProcessor

# STREAMLIT
from src.clip_manager import ClipManager
from src.prompts import run_chain_json, parse_broll_chain
from src.news_script import NewsScript, AnchorScriptSection, is_type
from src.tts import TTS
from src.gemini import add_broll
from src.language import Language

import streamlit as st
# /STREAMLIT

from pathlib import Path
import pprint

import moviepy.editor as mp

class AudioProcessor:
    """Handles audio processing, clip matching, and B-roll placement."""

    def __init__(self, news_script: NewsScript, clip_manager: ClipManager, folder: Path = Path("./"), error_handler = None):
        self.news_script = news_script
        self.clip_manager = clip_manager
        self.folder = folder
        self.error_handler = error_handler
        
        self.anchor_audio_file = folder / "anchor_audio.mp3"
        self.anchor_audio_folder = folder / "audio"
        self.anchor_audio_folder.mkdir(parents=True, exist_ok=True)

    def process_audio_and_broll(self):
        """Processes audio for anchor sections, and adds B-roll."""
        st.write("Generating anchor audio")
        if self.error_handler:
            self.error_handler.info("Generating anchor audio")
        self._process_anchor_audio()
        st.write("Generating SOT translations")
        if self.error_handler:
            self.error_handler.info("Generating SOT translations")
        self._generate_sot_translations()
        st.write("Adding broll placements")
        if self.error_handler:
            self.error_handler.info("Adding broll placements")
        self._add_broll_placements()

    def _process_anchor_audio(self):
        """Generates audio for anchor sections using TTS."""        
        audio_clips = []
        for i, section in enumerate(self.news_script.sections):
            if is_type(section, AnchorScriptSection):
                audio_file = self.anchor_audio_folder / f"{section.id}.mp3"
                previous_text = self.news_script.sections[i - 1].text if i > 0 else ""
                next_text = self.news_script.sections[i + 1].text if i < len(self.news_script.sections) - 1 else ""
                start_padding = 0.5 if i == 0 else 0.3
                end_padding = 1.0 if i == len(self.news_script.sections)-1 else 0.3
                TTS(section.text, str(audio_file), voice_id=self.clip_manager.anchor_voice_id, previous_text=previous_text, next_text=next_text, start_padding=start_padding, end_padding=end_padding)

                audio_clip = mp.AudioFileClip(str(audio_file))
                section.anchor_audio_file = audio_file
                section.anchor_audio_clip = audio_clip
                audio_clips.append(audio_clip)

                if self.error_handler:
                    self.error_handler.stream_status(section.text, "Generating anchor audio", audio=audio_file)

        anchor_audio = mp.concatenate_audioclips(audio_clips)
        anchor_audio.write_audiofile(str(self.anchor_audio_file), logger=None)
    
    def _generate_sot_translations(self):
        """Generates dubbed translations for non-English SOT"""
        for section in self.news_script.get_sot_sections():
            if section.language == Language.from_str("english"):
                continue
            audio_file = self.anchor_audio_folder / f"{section.id}_dub.mp3"
            section.generate_dub(audio_file, voice_id=self.clip_manager.get_voiceover_voice_id())

            if self.error_handler:
                self.error_handler.stream_status(f"Dubbing section {section.id} ({section.language.name})", audio=audio_file)

    def _add_broll_placements(self):
        """Generates and adds B-roll placement instructions to AnchorScriptSections."""
        full_descriptions_str = ""
        for clip in self.clip_manager.clips:
            if not any(section.clip == clip for section in self.news_script.get_sot_sections() if section.clip is not None):
                duration = clip.load_video().duration
                full_descriptions_str += f"<clip{clip.id}>\n{clip.full_description}\n\nMax duration: {duration} seconds\n</clip{clip.id}>\n"

        sections_str = ""
        section_start = 0
        for i, section in enumerate(self.news_script.sections):
            if is_type(section, AnchorScriptSection):
                section_id = section.id
                section_duration = section.anchor_audio_clip.duration
                section_end = section_start + section_duration
                sections_str += f"Section {section_id}: {section_start} - {section_end}\n"
                if i == 0:
                    sections_str += f"Anchor must be shown till atleast 5s.\n"
                if i == len(self.news_script.sections)-1:
                    sections_str += f"Anchor must be shown at or before {section_end-5}s till end.\n"
                sections_str += f"{section.text}\n"
                section_start = section_end

        broll_placements = add_broll(self.anchor_audio_file, full_descriptions_str, sections_str)
        parsed_broll_json = run_chain_json(parse_broll_chain, {"SECTIONS": sections_str, "BROLL_PLACEMENTS": broll_placements})

        if self.error_handler:
            self.error_handler.stream_status(broll_placements, "Placing BROLL")

        if len(self.news_script.get_anchor_sections()) != len(parsed_broll_json["sections"]):
            print(f"WARNING: SECTION LENGTHS DONT MATCH script: {len(self.news_script.get_anchor_sections())}, loglines: {len(parsed_broll_json['sections'])}")
        for section, broll_data in zip(self.news_script.get_anchor_sections(), parsed_broll_json["sections"]):
            if section.id != broll_data["id"]:
                print(f"WARNING: IDS DON'T MATCH script: {section.id}, broll: {broll_data['id']}")
            for broll in broll_data["brolls"]:
                if broll["id"] != "Anchor" and self.clip_manager.get_clip(broll["id"]) is None:
                    if self.error_handler:
                        self.error_handler.warning(f"WARNING: B-roll {broll['id']} in Section {section.id} does not exist. Replacing with anchor shot.")
                    broll["id"] = "Anchor"
            section.brolls = broll_data["brolls"]
        
        for section in self.news_script.get_anchor_sections():
            duration = section.anchor_audio_clip.duration
            for i, broll in enumerate(section.brolls[:]):
                if broll["start"] > duration:
                    del section.brolls[i]
            for broll in section.brolls:
                if broll["end"] > duration:
                    broll["end"] = duration
        
        for section in self.news_script.get_anchor_sections():
            for i, broll in enumerate(section.brolls):
                broll_duration = broll["end"] - broll["start"]
                if broll["id"] == "Anchor":
                    if broll_duration < 2.0:
                        if self.error_handler:
                            self.error_handler.warning(f"Anchor placement in section {section.id} is too short ({broll_duration}). Removing clip.")
                        if i-1 >= 0:
                            last_broll = section.brolls[i-1]
                            if last_broll["id"] == "Anchor":
                                last_broll["end"] = broll["end"]
                                broll["start"] = broll["end"]
                            else:
                                last_broll_clip = self.clip_manager.get_clip(last_broll["id"])
                                if last_broll_clip is not None:
                                    last_broll_duration = last_broll["end"] - last_broll["start"]
                                    if last_broll_duration < last_broll_clip.duration:
                                        available_time = last_broll_clip.duration - last_broll_duration
                                        needed_time = broll["end"] - broll["start"]
                                        added_time = min(available_time, needed_time)
                                        last_broll["end"] += added_time
                                        broll["start"] = last_broll["end"]
                        if len(section.brolls) > i+1:
                            next_broll = section.brolls[i+1]
                            if next_broll["id"] == "Anchor":
                                next_broll["start"] = broll["start"]
                                broll["end"] = broll["start"]
                            else:
                                next_broll_clip = self.clip_manager.get_clip(next_broll["id"])
                                if next_broll_clip is not None:
                                    next_broll_duration = next_broll["end"] - next_broll["start"]
                                    if next_broll_duration < next_broll_clip.duration:
                                        available_time = next_broll_clip.duration - next_broll_duration
                                        needed_time = broll["end"] - broll["start"]
                                        added_time = min(available_time, needed_time)
                                        next_broll["start"] -= added_time
                                        broll["end"] = next_broll["start"]
                        if broll["end"] != broll["start"]:
                            if i-1 >= 0:
                                section.brolls[i-1]["end"] = broll["end"]
                            else:
                                section.brolls[i+1]["start"] = broll["start"]
                            if self.error_handler:
                                self.error_handler.warning(f"Broll {broll['id']} in section {section.id} could not be filled. Surrounding clips will be slowed")
                                del section.brolls[i]
                        else:
                            if self.error_handler:
                                self.error_handler.stream_status(pprint.pformat(section.brolls), f"Broll {broll['id']} in section {section.id} filled")
                            del section.brolls[i]
                else:
                    if broll_duration < 1.0:
                        if self.error_handler:
                            self.error_handler.warning(f"Broll placement in section {section.id} is too short ({broll_duration}). Removing clip.")
                        if i-1 >= 0:
                            last_broll = section.brolls[i-1]
                            if last_broll["id"] == "Anchor":
                                last_broll["end"] = broll["end"]
                                broll["start"] = broll["end"]
                            else:
                                last_broll_clip = self.clip_manager.get_clip(last_broll["id"])
                                if last_broll_clip is not None:
                                    last_broll_duration = last_broll["end"] - last_broll["start"]
                                    if last_broll_duration < last_broll_clip.duration:
                                        available_time = last_broll_clip.duration - last_broll_duration
                                        needed_time = broll["end"] - broll["start"]
                                        added_time = min(available_time, needed_time)
                                        last_broll["end"] += added_time
                                        broll["start"] = last_broll["end"]
                        if len(section.brolls) > i+1:
                            next_broll = section.brolls[i+1]
                            if next_broll["id"] == "Anchor":
                                next_broll["start"] = broll["start"]
                                broll["end"] = broll["start"]
                            else:
                                next_broll_clip = self.clip_manager.get_clip(next_broll["id"])
                                if next_broll_clip is not None:
                                    next_broll_duration = next_broll["end"] - next_broll["start"]
                                    if next_broll_duration < next_broll_clip.duration:
                                        available_time = next_broll_clip.duration - next_broll_duration
                                        needed_time = broll["end"] - broll["start"]
                                        added_time = min(available_time, needed_time)
                                        next_broll["start"] -= added_time
                                        broll["end"] = next_broll["start"]
                        if broll["end"] != broll["start"]:
                            if i-1 >= 0:
                                section.brolls[i-1]["end"] = broll["end"]
                            else:
                                section.brolls[i+1]["start"] = broll["start"]
                            if self.error_handler:
                                self.error_handler.warning(f"Broll {broll['id']} in section {section.id} could not be filled. Surrounding clips will be slowed")
                                del section.brolls[i]
                        else:
                            if self.error_handler:
                                self.error_handler.stream_status(pprint.pformat(section.brolls), f"Broll {broll['id']} in section {section.id} filled")
                            del section.brolls[i]
