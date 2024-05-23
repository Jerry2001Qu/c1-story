# AudioProcessor

# STREAMLIT
from src.clip_manager import ClipManager
from src.prompts import run_chain_json, parse_broll_chain
from src.news_script import NewsScript, AnchorScriptSection, is_type
from src.tts import TTS
from src.gemini import add_broll
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
        """Processes audio for anchor sections, matches SOT clips, and adds B-roll."""
        self._process_anchor_audio()
        self._match_sot_clips()
        self._add_broll_placements()

    def _process_anchor_audio(self):
        """Generates audio for anchor sections using TTS."""
        audio_clips = []
        for i, section in enumerate(self.news_script.sections):
            if is_type(section, AnchorScriptSection):
                audio_file = self.anchor_audio_folder / f"{section.id}.mp3"
                previous_text = self.news_script.sections[i - 1].text if i > 0 else ""
                next_text = self.news_script.sections[i + 1].text if i < len(self.news_script.sections) - 1 else ""
                TTS(section.text, str(audio_file), previous_text, next_text)
                audio_clip = mp.AudioFileClip(str(audio_file))
                section.anchor_audio_file = audio_file
                section.anchor_audio_clip = audio_clip
                audio_clips.append(audio_clip)

        anchor_audio = mp.concatenate_audioclips(audio_clips)
        anchor_audio.write_audiofile(str(self.anchor_audio_file))

    def _match_sot_clips(self):
        """Matches SOTScriptSections with corresponding clips from ClipManager."""
        for section in self.news_script.get_sot_sections():
            try:
                clip = next(clip for clip in self.clip_manager.clips if str(clip.shot_id) == str(section.shot_id))
            except StopIteration:
                print(f"No clip found for shot ID: {section.shot_id}")
                section.clip = None
                continue
            section.clip = clip

            quote = section.quote
            timestamps = fuzzy_match(quote, clip.whisper_results)
            if timestamps:
                section.start = timestamps[0].start
                section.end = timestamps[-1].end + 0.5
            else:
                if clip.whisper_results.has_speech:
                    section.start = clip.whisper_results.timestamps[0].start
                    section.end = clip.whisper_results.timestamps[-1].end
                    print(f"SOT not found, adding all speech, section: {section.id}, clip: {clip.id}, language: {clip.whisper_results.language}, quote: {section.quote}, whisper: {clip.whisper_results}")
                else:
                    section.start = 0.0
                    section.end = clip.load_video().duration
                    print(f"SOT not found, adding full clip, section: {section.id}, clip: {clip.id}, quote: {section.quote}, whisper: {clip.whisper_results}")

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

from difflib import SequenceMatcher
from fuzzysearch import find_near_matches

def prep_text(text):
    lower = text.lower().strip().replace("-", " ")
    return ''.join(filter(lambda x: x.isalpha() or x == ' ', lower))

def prep_word(word):
    lower = word.lower()
    return ''.join(filter(lambda x: x.isalpha(), lower))

def fuzzy_match(quote, whisper_results):
    fuzzy = find_near_matches(prep_text(quote), prep_text(whisper_results.text), max_l_dist=int(len(quote) / 5))
    if not fuzzy:
        return None
    fuzzy_match = fuzzy[0].matched

    quote_words = [prep_word(word) for word in fuzzy_match.split()]
    whisper_words = [prep_word(word.word) for word in whisper_results.timestamps]
    matcher = SequenceMatcher(None, quote_words, whisper_words)
    seq_match = matcher.find_longest_match(0, len(quote_words), 0, len(whisper_words))

    if seq_match.size > 0:
        return whisper_results.timestamps[seq_match.b:seq_match.b + seq_match.size]
    else:
        return None