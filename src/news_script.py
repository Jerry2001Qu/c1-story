# NewsScript

# STREAMLIT
from src.clip_manager import ClipManager, Clip
from src.prompts import run_chain, run_chain_json, \
                        get_sot_chain, reformat_chain, sot_chain, parse_chain, logline_chain, parse_sot_chain, headline_chain
# /STREAMLIT

from typing import List, Optional
from abc import ABC
from pathlib import Path
import moviepy.editor as mp

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
        return f"""{self.id}: {self.logline}

{self.text}"""


class SOTScriptSection(ScriptSection):
    """Represents a soundbite (SOT) section of the script."""

    def __init__(self, id: int, text: str, shot_id: int, quote: str):
        super().__init__(id, text)
        self.shot_id = shot_id
        self.quote = quote

        self.name: Optional[str] = None
        self.title: Optional[str] = None
        self.language: Optional[str] = None

        self.clip: Optional[Clip] = None
        self.start: Optional[float] = None
        self.end: Optional[float] = None
    
    def __repr__(self):
        return f"""{self.id}: {self.get_byline()}

{self.quote}"""

    def is_interview(self):
        return self.name != "Unknown"

    def get_byline(self):
        if self.is_interview():
            return f"{self.name}, {self.title}"
        else:
            return ""

class NewsScript:
    """Represents the entire news script."""
    
    def __init__(self, storyline: str, shotlist: str, clip_manager: ClipManager, folder: Path = Path("./")):
        self.storyline = storyline
        self.shotlist = shotlist
        self.clip_manager = clip_manager
        self.folder = folder

        self.headline: Optional[str] = None
        self.sections: List[ScriptSection] = []
        self.sots: Optional[str] = None
        self.text_script: Optional[str] = None
    
    def __repr__(self):
        result = f"# {self.headline}"
        for section in self.sections:
            result += f"\n\n{section.__repr__()}"
        return result
    
    def generate_script(self):
        sots = self._extract_sots()
        reformated_story = self._reformat_story()
        story_with_sots = self._insert_sots_into_story(reformated_story, sots)
        self._parse_script(story_with_sots, sots)

        self.sots = sots
        self.text_script = story_with_sots
    
    def generate_lower_thirds(self):
        self.headline = self._generate_headline()
        self._generate_loglines()
        self._generate_bylines()
    
    def generate_audio_and_broll(self):
        """Generates audio, matches clips, and adds B-roll placements."""
        from src.audio_processor import AudioProcessor
        audio_processor = AudioProcessor(self, self.clip_manager, self.folder)
        audio_processor.process_audio_and_broll()

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
            section.language = parsed_sots[shot_id_str]["language"]

    def get_sot_sections(self) -> List[SOTScriptSection]:
        """Returns a list of SOTScriptSections."""
        return [section for section in self.sections if is_type(section, SOTScriptSection)]
    
    def get_anchor_sections(self) -> List[AnchorScriptSection]:
        """Returns a list of AnchorScriptSections."""
        return [section for section in self.sections if is_type(section, AnchorScriptSection)]

    def _generate_headline(self):
        """Generates a headline for the news script."""
        headline = run_chain(headline_chain, {"SCRIPT": self.text_script})
        return headline
    
def is_type(obj, type_):
    return str(type(obj)) == str(type_)