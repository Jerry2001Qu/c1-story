# NewsScript

# STREAMLIT
from src.clip_manager import ClipManager, Clip
from src.prompts import run_chain, run_chain_json, \
                        get_sot_chain, reformat_chain, sot_chain, parse_chain, logline_chain, parse_sot_chain, headline_chain, match_sot_chain, match_hard_sot_chain, facts_chain
from src.language import Language
from src.tts import TTS
# /STREAMLIT

from typing import List, Optional
from abc import ABC
from pathlib import Path
import moviepy.editor as mp
import pandas as pd
import pprint

class ScriptSection(ABC):
    """Represents a single section of the news script."""

    def __init__(self, id: int, text: str):
        self.id = id
        self.text = text

class AnchorScriptSection(ScriptSection):
    """Represents an anchor-spoken section of the script."""

    def __init__(self, id: int, text: str):
        super().__init__(id, text)
        self.logline: Optional[str] = None

        self.anchor_audio_file: Optional[Path] = None
        self.anchor_audio_clip: Optional[mp.AudioFileClip] = None
        self.brolls: Optional[List] = None
    
    def __repr__(self):
        return f"""{self.text}

{pprint.pformat(self.brolls)}"""


class SOTScriptSection(ScriptSection):
    """Represents a soundbite (SOT) section of the script."""

    def __init__(self, id: int, text: str, shot_id: int, quote: str):
        super().__init__(id, text)
        self.shot_id = shot_id
        self.quote = quote

        self.name: Optional[str] = None
        self.title: Optional[str] = None
        self.language: Optional[Language] = None

        self.clip: Optional[Clip] = None
        self.start: Optional[float] = None
        self.end: Optional[float] = None
        self.match_type: Optional[str] = None

        self.dub_audio_file: Optional[Path] = None
    
    def __repr__(self):
        return f"""SOT ({self.get_byline() if self.is_interview() else "No byline"}): {self.quote}

CLIP {self.clip.id}: {self.start} - {self.end}"""

    def is_interview(self):
        return self.name != "Unknown"

    def get_byline(self):
        if self.is_interview():
            return f"{self.name}, {self.title}"
        else:
            return ""
    
    def generate_dub(self, audio_file: Path, voice_id: str = "9f8o652aaiVK5HavyCf1"):
        self.dub_audio_file = audio_file
        TTS(self.quote, str(audio_file), voice_id=voice_id)

class NewsScript:
    """Represents the entire news script."""
    
    def __init__(self, storyline: str, shotlist: str, clip_manager: ClipManager, dataloader, folder: Path = Path("./"), error_handler = None):
        self.storyline = storyline
        self.shotlist = shotlist
        self.clip_manager = clip_manager
        self.dataloader = dataloader
        self.folder = folder
        self.error_handler = error_handler

        self.headline: Optional[str] = None
        self.sections: List[ScriptSection] = []
        self.sots: Optional[str] = None
        self.text_script: Optional[str] = None
    
    def __repr__(self):
        result = f"# {self.headline}"
        for section in self.sections:
            result += f"\n\n{section.__repr__()}"
        return result
    
    def generate_facts(self):
        self.facts_list = run_chain(facts_chain, {"SCRIPT": self.storyline, "SHOTLIST": self.shotlist})
        if self.error_handler:
            self.error_handler.stream_status(self.facts_list, "Generating list of facts")
    
    def generate_script(self):
        sots = self._extract_sots()
        reformated_story = self._reformat_story()
        if self.error_handler:
            self.error_handler.stream_status(reformated_story, "Writing story")
        story_with_sots = self._insert_sots_into_story(reformated_story, sots)
        if self.error_handler:
            self.error_handler.stream_status(story_with_sots, "Inserted SOTs")
        self._parse_script(story_with_sots, sots)

        self.sots = sots
        self.text_script = story_with_sots
    
    def generate_lower_thirds(self):
        self.headline = self._generate_headline()
        if self.error_handler:
            self.error_handler.stream_status(self.headline, "Generating headline")
        self._generate_loglines()
        if self.error_handler:
            self.error_handler.stream_status(self.get_loglines(), "Generating loglines")
        self._generate_bylines()
        if self.error_handler:
            self.error_handler.stream_status(self.get_bylines(), "Generating bylines")
    
    def get_loglines(self):
        sections = [section for section in self.sections if is_type(section, AnchorScriptSection) and section.logline]
        loglines = [f"{section.id}: {section.logline}" for section in sections]
        return "\n\n".join(loglines)
    
    def get_bylines(self):
        sections = [section for section in self.sections if is_type(section, SOTScriptSection)]
        bylines = [f"{section.id}: {section.get_byline()}" for section in sections]
        return "\n\n".join(bylines)
    
    def match_sot_clips(self):
        """Matches SOTScriptSections with corresponding clips from ClipManager."""
        for section in self.get_sot_sections():
            try:
                try:
                    clip = next(clip for clip in self.clip_manager.clips if str(clip.shot_id) == str(section.shot_id))
                except StopIteration:
                    if self.error_handler:
                        self.error_handler.warning(f"WARNING: No clip found for shot ID: {section.shot_id}, this section {section.id} will be omited.")
                    section.clip = None
                    continue
                section.clip = clip

                if section.language != clip.whisper_results.language:
                    if self.error_handler:
                        self.error_handler.warning(f"WARNING: Language does not match {section.language} (section {section.id}) != {clip.whisper_results.language} (clip {clip.id})")

                if clip.whisper_results.language == Language.from_str("English"):
                    self._match_sot_clips_same_language(section, clip, section.quote)
                else:
                    self._match_sot_clips_different_language(section, clip)
            except Exception as e:
                if self.error_handler:
                    self.error_handler.warning(f"WARNING: Problem when matching section {section.id}. {e}")

    def _match_sot_clips_same_language(self, section, clip, quote):
        timestamps = fuzzy_match(quote, clip.whisper_results)
        if timestamps:
            section.start = timestamps[0].get_adjusted_start()
            section.end = timestamps[-1].end + 0.5
            section.match_type = "SUCCESS"
            if self.error_handler:
                self.error_handler.stream_status(f"Found quote in clip {clip.id}. From {int(section.start)}s to {int(section.end)}s. {section.quote}", "Matched SOT", clip.file_path)
        else:
            matched_quote = run_chain(match_hard_sot_chain, {"QUOTE": quote, "TRANSCRIPT": clip.whisper_results.text})
            timestamps = fuzzy_match(matched_quote, clip.whisper_results)
            if timestamps:
                section.start = timestamps[0].get_adjusted_start()
                section.end = timestamps[-1].end + 0.5
                section.match_type = "SUCCESS"
                if self.error_handler:
                    self.error_handler.stream_status(f"Found quote in clip {clip.id}. From {int(section.start)}s to {int(section.end)}s. {section.quote}", "Matched SOT", clip.file_path)
            else:
                if clip.whisper_results.has_speech:
                    section.start = clip.whisper_results.timestamps[0].get_adjusted_start()
                    section.end = clip.whisper_results.timestamps[-1].end
                    section.match_type = "SPEECH"
                    if self.error_handler:
                        self.error_handler.warning(f"SOT not found, adding all speech, section: {section.id}, clip: {clip.id}, language: {clip.whisper_results.language}, quote: {quote}, whisper: {clip.whisper_results.text}")
                    print(f"SOT not found, adding all speech, section: {section.id}, clip: {clip.id}, language: {clip.whisper_results.language}, quote: {quote}, whisper: {clip.whisper_results.text}")
                else:
                    section.start = 0.0
                    section.end = clip.load_video().duration
                    section.match_type = "CLIP"
                    if self.error_handler:
                        self.error_handler.warning(f"SOT not found, adding full clip, section: {section.id}, clip: {clip.id}, language: {clip.whisper_results.language}, quote: {quote}, whisper: {clip.whisper_results.text}")
                    print(f"SOT not found, adding full clip, section: {section.id}, clip: {clip.id}, quote: {quote}, whisper: {clip.whisper_results.text}")

    def _match_sot_clips_different_language(self, section, clip):
        english_str = section.quote
        other_language_str = clip.whisper_results.text

        other_language_substring = run_chain(match_sot_chain, {"ENGLISH_STRING": english_str, "OTHER_LANGUAGE_STRING": other_language_str})
        self._match_sot_clips_same_language(section, clip, other_language_substring)

    def _extract_sots(self) -> str:
        """Extracts and parses soundbites (SOTs) from the shotlist."""
        sots = run_chain(get_sot_chain, {"SHOTLIST": self.shotlist})
        return sots
    
    def _reformat_story(self) -> str:
        reformated_story = run_chain(reformat_chain, {"STORY": self.storyline})
        return reformated_story

    def _insert_sots_into_story(self, reformated_story: str, sots: str):
        """Inserts the SOTs into the script."""
        if "NO SOT" in sots:
            story_with_sots = reformated_story
        else:
            story_with_sots = run_chain(sot_chain, {"QUOTATIONS": sots, "SCRIPT": reformated_story})
        return story_with_sots

    def _parse_script(self, story_with_sots: str, sots: str):
        """Parses the storyline into individual ScriptSection objects."""
        parsed_script_json = run_chain_json(parse_chain, {"QUOTATIONS": sots, "SCRIPT": story_with_sots})

        for section_json in parsed_script_json["sections"]:
            section_type = section_json["type"]
            if section_type == "SOT":
                section = SOTScriptSection(section_json["id"], section_json["text"], section_json["shot_id"], section_json["quote"])
            elif section_type == "ANCHOR":
                section = AnchorScriptSection(section_json["id"], section_json["text"])
            self.sections.append(section)

    def _generate_loglines(self):
        """Generates loglines for each section in the script."""
        sections_text = ""
        for section in self.get_anchor_sections():
            sections_text += f"Section {section.id}:\n"
            sections_text += f"{section.text}\n"

        loglines_json = run_chain_json(logline_chain, {"SECTIONS": sections_text})

        anchor_script_sections = self.get_anchor_sections()
        loglines_sections = loglines_json["sections"]

        if len(anchor_script_sections) != len(loglines_sections):
            print(f"WARNING: SECTION LENGTHS DONT MATCH script: {len(anchor_script_sections)}, loglines: {len(loglines_json['sections'])}")
        for section, logline_data in zip(anchor_script_sections, loglines_sections):
            if section.id != logline_data["id"]:
                print(f"WARNING: IDS DON'T MATCH script: {section.id}, loglines: {logline_data['id']}")
            section.logline = logline_data["logline"]
    
    def _generate_bylines(self):
        """Generates info for bylines for each section in the script."""
        parsed_sots = run_chain_json(parse_sot_chain, {"QUOTES": self.sots})

        for section in self.get_sot_sections():
            shot_id_str = str(section.shot_id)
            section.name = parsed_sots[shot_id_str]["name"]
            section.title = parsed_sots[shot_id_str]["title"]
            section.language = Language.from_str(parsed_sots[shot_id_str]["language"])

    def get_sot_sections(self) -> List[SOTScriptSection]:
        """Returns a list of SOTScriptSections."""
        return [section for section in self.sections if is_type(section, SOTScriptSection)]
    
    def get_anchor_sections(self) -> List[AnchorScriptSection]:
        """Returns a list of AnchorScriptSections."""
        return [section for section in self.sections if is_type(section, AnchorScriptSection)]

    def _generate_headline(self):
        """Generates a headline for the news script."""
        headline = run_chain(headline_chain, {"SCRIPT": self.text_script, "ORIGINAL_HEADLINE": self.dataloader.get_story_title()})
        while len(headline) > 45:
            if self.error_handler:
                self.error_handler.info(f"Headline was too long, shortening: {headline}")
            headline = run_chain(headline_chain, {"SCRIPT": self.text_script + f"\n\nOld headline was '{headline}'. This is too long.", "ORIGINAL_HEADLINE": self.dataloader.get_story_title()})
        return headline
    
    def get_total_read_time_seconds(self):
        """Returns the total read time of the text script."""
        import readtime
        return readtime.of_text(self.text_script).seconds
    
    def to_dataframe(self):
        data = {
            'type': [],
            'shot_id': [],
            'name': [],
            'text': [],
        }
        for section in self.sections:
            if is_type(section, AnchorScriptSection):
                data['type'] += ["ANCHOR"]
                data['shot_id'] += [None]
                data['name'] += [None]
                data['text'] += [section.text]
            elif is_type(section, SOTScriptSection):
                data['type'] += ["SOT"]
                data['shot_id'] += [section.shot_id]
                if section.name and section.title:
                    data['name'] += [f"{section.name}, {section.title}"]
                elif section.name:
                    data['name'] += [section.name]
                elif section.title:
                    data['name'] += [section.title]
                else:
                    data['name'] += ["No Identity"]
                data['text'] += [section.quote]
            else:
                print(f"ERROR: Unrecognized section of type: {type(section)}")
        df = pd.DataFrame(data)
        return df
    
    def from_dataframe(self, df):
        self.sections = []
        for id, row in df.iterrows():
            id = int(id)
            if not row["text"]:
                continue
            if row["type"] == "ANCHOR":
                self.sections += [AnchorScriptSection(id+1, row["text"])]
            elif row["type"] == "SOT":
                self.sections += [SOTScriptSection(id+1, row["text"], int(row["shot_id"]), row["text"])]
            else:
                print(f"ERROR: Unrecognized section of type: {row['type']}")
        self.generate_lower_thirds()
        self.match_sot_clips()
    
def is_type(obj, type_):
    return str(type(obj)) == str(type_)

from difflib import SequenceMatcher
from fuzzysearch import find_near_matches

def prep_text(text):
    lower = text.lower().strip().replace("-", " ")
    return ''.join(filter(lambda x: x.isalpha() or x == ' ', lower))

def prep_word(word):
    lower = word.lower()
    return ''.join(filter(lambda x: x.isalpha(), lower))

def split_into_words_by_language(text, language):
    if language in [Language.from_str("chinese"), Language.from_str("lao"), Language.from_str("burmese")]:
        return list(text)
    return text.split()

def fuzzy_match(quote, whisper_results):
    fuzzy = find_near_matches(prep_text(quote), prep_text(whisper_results.text), max_l_dist=int(len(quote) / 5))
    if not fuzzy:
        return None
    fuzzy_match = fuzzy[0].matched

    quote_words = [prep_word(word) for word in split_into_words_by_language(fuzzy_match, whisper_results.language)]
    whisper_words = [prep_word(word.word) for word in whisper_results.timestamps]
    matcher = SequenceMatcher(None, quote_words, whisper_words)
    seq_match = matcher.find_longest_match(0, len(quote_words), 0, len(whisper_words))

    if seq_match.size > 0:
        return whisper_results.timestamps[seq_match.b:seq_match.b + seq_match.size]
    else:
        return None
