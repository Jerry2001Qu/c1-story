# AudioProcessor

# STREAMLIT
from src.clip_manager import ClipManager
from src.prompts import run_chain_json, run_chain, broll_chain, parse_broll_chain, fix_broll_chain, broll_request_chain
from src.news_script import NewsScript, AnchorScriptSection, is_type
from src.tts import TTS
from src.gemini import add_broll, add_broll_clips
from src.language import Language
from src.heygen import animate_anchor
from src.transcription import WhisperResults

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
                TTS(section.text, str(audio_file), voice_id=self.clip_manager.anchor_voice_id, start_padding=start_padding, end_padding=end_padding)

                transcript_file = self.anchor_audio_folder / f"{section.id}.txt"
                with open(transcript_file, 'w') as f:
                    f.write(section.text)

                audio_clip = mp.AudioFileClip(str(audio_file))
                section.anchor_audio_file = audio_file
                section.anchor_audio_clip = audio_clip
                section.whisper_results = WhisperResults.from_file(audio_file)
                audio_clips.append(audio_clip)

                if self.error_handler:
                    self.error_handler.stream_status(section.text, "Generating anchor audio", audio=audio_file)

        if audio_clips:
            anchor_audio = mp.concatenate_audioclips(audio_clips)
            anchor_audio.write_audiofile(str(self.anchor_audio_file), logger=None)
    
    def _generate_sot_translations(self):
        """Generates dubbed translations for non-English SOT"""
        for section in self.news_script.get_sot_sections():
            if not section.clip:
                continue
            if section.language == Language.from_str("english"):
                continue
            # if section.clip.whisper_results.language == Language.from_str("english"):
            #     continue
            audio_file = self.anchor_audio_folder / f"{section.id}_dub.mp3"
            section.generate_dub(audio_file, voice_id=self.clip_manager.get_voiceover_voice_id())

            if self.error_handler:
                self.error_handler.stream_status(f"Dubbing section {section.id} ({section.clip.whisper_results.language.name})", audio=audio_file)

    def _add_broll_placements(self):
        """Generates and adds B-roll placement instructions to AnchorScriptSections."""
        if not self.news_script.get_anchor_sections():
            return

        full_descriptions_str = ""
        for clip in self.clip_manager.clips:
            duration = clip.duration
            if not clip.has_quote:
                full_descriptions_str += f"<clip {clip.id}>\n{clip.full_description}\n\nMax duration: {duration} seconds\n</clip {clip.id}>\n"
            else:
                if clip.full_description:
                    lines = clip.full_description.split("\n")
                    full_description = '\n'.join(line for line in lines if not line.startswith("Minimum Timing:"))
                else:
                    full_description = clip.shotlist_description
                full_descriptions_str += f"<clip {clip.id}>\nThis clip is a SOT/Interview. \n{full_description}\n\nMax duration: {min(duration, 3)} seconds\n</clip {clip.id}>\n"

        sections_str = ""
        section_start = 0
        for i, section in enumerate(self.news_script.sections):
            if is_type(section, AnchorScriptSection):
                section_id = section.id
                section_duration = section.anchor_audio_clip.duration
                section_end = section_start + section_duration
                sections_str += f"Section {section_id}: {section_start:.2f} - {section_end:.2f}\n"
                if i == 0:
                    sections_str += f"Anchor must be shown till atleast 5s.\n"
                if i == len(self.news_script.sections)-1:
                    sections_str += f"Anchor must be shown at or before {max(section_end-5, section_start):.2f}s till end.\n"
                sections_str += f"{section.text}\n"
                sections_str += f"Timestamps:\n"
                for word in section.whisper_results.timestamps:
                    sections_str += f"{word.word}: {section_start + word.start:.2f}-{section_start + word.end:.2f}\n"
                section_start = section_end
        
        if self.error_handler:
            self.error_handler.stream_status(sections_str, "Generating BROLL requests")

        # broll_placements = add_broll(self.anchor_audio_file, full_descriptions_str, sections_str)
        broll_placements = add_broll_clips(self.anchor_audio_file, self.clip_manager.clips, self.news_script.get_sot_clip_ids(), sections_str)
        # broll_placements = run_chain(broll_chain, {"BROLL_DESCRIPTIONS": full_descriptions_str, "SECTION_TIMINGS": sections_str})
        
        parsed_broll_json = run_chain_json(parse_broll_chain, {"SECTIONS": sections_str, "BROLL_PLACEMENTS": broll_placements})

        if self.error_handler:
            self.error_handler.stream_status(broll_placements, "Placing BROLL")

        # sections_str = ""
        # for i, section in enumerate(self.news_script.sections):
        #     if is_type(section, AnchorScriptSection):
        #         section_id = section.id
        #         section_duration = section.anchor_audio_clip.duration
        #         sections_str += f"Section {section_id}: 0.00 - {section_duration}\n"
        #         if i == 0:
        #             sections_str += f"Anchor must be shown till atleast {min(5, section_duration)}s.\n"
        #         if i == len(self.news_script.sections)-1:
        #             sections_str += f"Anchor must be shown at or before {max(section_end-5, 0)}s till end.\n"
        #         sections_str += f"{section.text}\n"
        
        # broll_timings = ""
        # for clip in self.clip_manager.clips:
        #     if not clip.has_quote:
        #         duration = clip.load_video().duration
        #         broll_timings += f"<clip{clip.id}>\nMax duration: {duration} seconds\n</clip{clip.id}>\n"

        # fixed_broll_json = run_chain_json(fix_broll_chain, {"BROLL_PLACEMENTS": parsed_broll_json, "SECTION_TIMINGS": sections_str, "BROLL_TIMINGS": broll_timings})

        # if self.error_handler:
        #     self.error_handler.stream_status(pprint.pformat(fixed_broll_json), "Fixing BROLL")

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
        
        # Remove brolls after the end of the anchor audio
        for section in self.news_script.get_anchor_sections():
            duration = section.anchor_audio_clip.duration
            section.brolls = [broll for broll in section.brolls if broll["start"] <= duration]
            for broll in section.brolls:
                if broll["end"] > duration:
                    broll["end"] = duration

        # Ensure last anchor is at least 5 seconds
        # last_section = self.news_script.sections[-1]
        # if is_type(last_section, AnchorScriptSection):
        #     last_broll = last_section.brolls[-1]
        #     if last_broll["id"] == "Anchor":
        #         last_broll_duration = last_broll["end"] - last_broll["start"]
        #         if last_broll_duration < 5:
        #             needed_duration = 5 - last_broll_duration
        #             if len(last_section.brolls) > 1:
        #                 second_last_broll = last_section.brolls[-2]
        #                 second_last_broll_duration = second_last_broll["end"] - second_last_broll["start"]
        #                 if second_last_broll_duration - needed_duration > 1:
        #                     last_broll["start"] -= needed_duration
        #                     second_last_broll["end"] = last_broll["start"]
        #                 else:
        #                     second_last_broll["end"] = min(second_last_broll["start"]+1, second_last_broll["end"])
        #                     last_broll["start"] = second_last_broll["end"]
        #                 if self.error_handler:
        #                     self.error_handler.warning(f"Final anchor placement was too short, extended into previous broll.")
        #                     self.error_handler.stream_status(pprint.pformat(last_section.brolls), f"Final anchor placement was too short, extended into previous broll.")
        #     else:
        #         last_broll["id"] = "Anchor"
        #         if self.error_handler:
        #             self.error_handler.warning(f"Final video was not Anchor, swapping broll with anchor")
        #             self.error_handler.stream_status(pprint.pformat(last_section.brolls), f"Final video was not Anchor, swapping broll with anchor")

        # Remove brolls or anchor placements that are too short
        for section in self.news_script.get_anchor_sections():
            new_brolls = []

            for i, broll in enumerate(section.brolls):
                keep_broll = True
                broll_duration = broll["end"] - broll["start"]
                if broll["id"] == "Anchor":
                    if broll_duration < 1.0:
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
                            elif i+1 < len(section.brolls):
                                section.brolls[i+1]["start"] = broll["start"]
                            else:
                                if self.error_handler:
                                    self.error_handler.warning(f"Anchor placement in section {section.id} is only clip. Keeping short.")
                            if self.error_handler:
                                self.error_handler.warning(f"Broll {broll['id']} in section {section.id} could not be filled. Surrounding clips will be slowed")
                            keep_broll = False
                        else:
                            if self.error_handler:
                                self.error_handler.stream_status(pprint.pformat(section.brolls), f"Broll {broll['id']} in section {section.id} filled")
                            keep_broll = False
                else:
                    if broll_duration < 0.8:
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
                            elif i+1 < len(section.brolls):
                                section.brolls[i+1]["start"] = broll["start"]
                            else:
                                if self.error_handler:
                                    self.error_handler.warning(f"Broll {broll['id']} in section {section.id} is only clip. Keeping short.")
                            if self.error_handler:
                                self.error_handler.warning(f"Broll {broll['id']} in section {section.id} could not be filled. Surrounding clips will be slowed")
                            keep_broll = False
                        else:
                            if self.error_handler:
                                self.error_handler.stream_status(pprint.pformat(section.brolls), f"Broll {broll['id']} in section {section.id} filled")
                            keep_broll = False
                if keep_broll:
                    new_brolls.append(broll)
            section.brolls = new_brolls
        
        # Remove short sequences of non-anchor brolls between anchor brolls (less than 5 seconds), and extend the surrounding anchor brolls
        for section in self.news_script.get_anchor_sections():
            new_brolls = []
            for i, broll in enumerate(section.brolls):
                if broll["id"] == "Anchor":
                    new_brolls.append(broll)
                    continue
                broll_duration = broll["end"] - broll["start"]
                if broll_duration < 5:
                    if i-1 >= 0 and i+1 < len(section.brolls):
                        previous_broll = section.brolls[i-1]
                        next_broll = section.brolls[i+1]
                        if previous_broll["id"] == "Anchor" and next_broll["id"] == "Anchor":
                            next_broll["start"] = broll["start"]
                            if self.error_handler:
                                self.error_handler.warning(f"Short broll {broll['id']} in section {section.id} between anchors. Merged with surrounding anchors.")
                            continue
                new_brolls.append(broll)
            section.brolls = new_brolls
        
        # Remove short brolls after anchor at the end of sections
        for section in self.news_script.get_anchor_sections():
            if len(section.brolls) >= 2:
                last_broll = section.brolls[-1]
                last_broll_duration = last_broll["end"] - last_broll["start"]
                second_last_broll = section.brolls[-2]
                if last_broll_duration < 2:
                    if last_broll["id"] != "Anchor" and second_last_broll["id"] == "Anchor":
                        second_last_broll["end"] = last_broll["end"]
                        section.brolls = section.brolls[:-1]
                        if self.error_handler:
                            self.error_handler.warning(f"Last broll in section {section.id} was too short, merged with previous anchor")
                            self.error_handler.stream_status(pprint.pformat(section.brolls), f"Last broll in section {section.id} was too short, merged with previous anchor")


    def _generate_anchor(self, live_anchor: bool, test_mode: bool):
        for section in self.news_script.get_anchor_sections():
            if section.has_anchor_on_screen():
                anchor_video_file = self.news_script.folder / f"{section.id}_anchor.mp4"
                if live_anchor:
                    animate_anchor(section.anchor_audio_file, section.text, self.clip_manager.get_anchor_avatar_id(), anchor_video_file, test=test_mode)
                    if self.error_handler:
                        self.error_handler.stream_status(section.text, title="Generated anchor video", video=anchor_video_file)
                    section.anchor_video_file = anchor_video_file
                else:
                    anchor_clip = self.clip_manager.get_anchor_image_clip()
                    anchor_clip = anchor_clip.set_duration(section.anchor_audio_clip.duration)
                    anchor_clip.write_videofile(str(anchor_video_file), fps=29.97, threads=8,
                                    bitrate="10M", logger=None)
                    section.anchor_video_file = anchor_video_file
