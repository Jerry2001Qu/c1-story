# AudioProcessor

# STREAMLIT
from src.clip_manager import ClipManager
from src.prompts import run_chain_json, parse_broll_chain
from src.news_script import NewsScript, AnchorScriptSection, is_type
from src.tts import TTS
from src.gemini import add_broll
from src.language import Language
# /STREAMLIT

from pathlib import Path

import moviepy.editor as mp

class AudioProcessor:
    """Handles audio processing, clip matching, and B-roll placement."""

    def __init__(self, news_script: NewsScript, clip_manager: ClipManager, folder: Path = Path("./")):
        self.news_script = news_script
        self.clip_manager = clip_manager

        self.folder = folder
        self.anchor_audio_file = folder / "anchor_audio.mp3"
        self.anchor_audio_folder = folder / "audio"
        self.anchor_audio_folder.mkdir(parents=True, exist_ok=True)

    def process_audio_and_broll(self):
        """Processes audio for anchor sections, and adds B-roll."""
        self._process_anchor_audio()
        self._generate_sot_translations()
        self._add_broll_placements()

    def _process_anchor_audio(self):
        """Generates audio for anchor sections using TTS."""
        audio_clips = []
        for i, section in enumerate(self.news_script.sections):
            if is_type(section, AnchorScriptSection):
                audio_file = self.anchor_audio_folder / f"{section.id}.mp3"
                previous_text = self.news_script.sections[i - 1].text if i > 0 else ""
                next_text = self.news_script.sections[i + 1].text if i < len(self.news_script.sections) - 1 else ""
                TTS(section.text, str(audio_file), previous_text=previous_text, next_text=next_text)
                audio_clip = mp.AudioFileClip(str(audio_file))
                section.anchor_audio_file = audio_file
                section.anchor_audio_clip = audio_clip
                audio_clips.append(audio_clip)

        anchor_audio = mp.concatenate_audioclips(audio_clips)
        anchor_audio.write_audiofile(str(self.anchor_audio_file), logger=None)
    
    def _generate_sot_translations(self):
        """Generates dubbed translations for non-English SOT"""
        for section in self.news_script.get_sot_sections():
            if section.language == Language.from_str("english"):
                continue
            audio_file = self.anchor_audio_folder / f"{section.id}_dub.mp3"
            section.generate_dub(audio_file)

    def _add_broll_placements(self):
        """Generates and adds B-roll placement instructions to AnchorScriptSections."""
        full_descriptions_str = ""
        for clip in self.clip_manager.clips:
            if not any(section.clip == clip for section in self.news_script.get_sot_sections() if section.clip is not None):
                duration = clip.load_video().duration
                full_descriptions_str += f"<clip{clip.id}>\n{clip.full_description}\n\nMax duration: {duration} seconds\n</clip{clip.id}>\n"

        sections_str = ""
        section_start = 0
        for section in self.news_script.sections:
            if is_type(section, AnchorScriptSection):
                section_id = section.id
                section_duration = section.anchor_audio_clip.duration
                section_end = section_start + section_duration
                sections_str += f"Section {section_id}: {section_start} - {section_end}\n"
                sections_str += f"{section.text}\n"
                section_start = section_end

        broll_placements = add_broll(self.anchor_audio_file, full_descriptions_str, sections_str)
        parsed_broll_json = run_chain_json(parse_broll_chain, {"SECTIONS": sections_str, "BROLL_PLACEMENTS": broll_placements})

        if len(self.news_script.get_anchor_sections()) != len(parsed_broll_json["sections"]):
            print(f"WARNING: SECTION LENGTHS DONT MATCH script: {len(self.news_script.get_anchor_sections())}, loglines: {len(parsed_broll_json['sections'])}")
        for section, broll_data in zip(self.news_script.get_anchor_sections(), parsed_broll_json["sections"]):
            if section.id != broll_data["id"]:
                print(f"WARNING: IDS DON'T MATCH script: {section.id}, broll: {broll_data['id']}")
            section.brolls = broll_data["brolls"]
