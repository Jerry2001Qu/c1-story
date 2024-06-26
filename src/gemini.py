# Gemini

# STREAMLIT
from src.constants import GOOGLE_JSON
import os
import json
import streamlit as st

google_json = json.loads(GOOGLE_JSON, strict=False)
temp_file_path = "/tmp/service_account.json"
with open(temp_file_path, "w") as f:
    json.dump(google_json, f)

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the temporary file path
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path

from src.prompts import extract_xml
from src.clip_manager import Clip
from src.gcp import GCSManager
from src.hashing import sha256sum, hash_audio_file
# /STREAMLIT

from typing import Tuple, Dict, List
from pathlib import Path, PosixPath

import vertexai
from vertexai.generative_models import (GenerationConfig, GenerativeModel,
                                         HarmBlockThreshold, HarmCategory,
                                         SafetySetting)

import moviepy.editor as mp
from PIL import Image

# Gemini Initialization 
vertexai.init(project="stg-transcription", location="us-central1")

GENERATION_CONFIG = GenerationConfig(
    max_output_tokens=8192, temperature=0
)

# Safety config
SAFETY_CONFIG = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    )
]

GEMINI = GenerativeModel(model_name="gemini-1.5-pro-preview-0409", 
                         generation_config=GENERATION_CONFIG, 
                         safety_settings=SAFETY_CONFIG)

def extract_frame(input_file: Path, time: float, output_file: Path) -> None:
    """Extracts a frame from a video at a specific time and saves it as an image.

    Args:
        input_file: Path to the input video file.
        time: Time in seconds to extract the frame from.
        output_file: Path to save the extracted frame image.
    """
    with mp.VideoFileClip(str(input_file)) as video:
        frame = video.get_frame(time)
        image = Image.fromarray(frame)
        
        if not output_file.exists():
            image.save(output_file)

def extract_middle_frame_and_audio(input_file: Path) -> Tuple[Path, Path]:
    """Extracts the middle frame and audio from a video.

    Args:
        input_file: Path to the input video file.

    Returns:
        A tuple containing:
            - Path to the extracted frame image.
            - Path to the extracted audio file. 
    """
    frame_output = input_file.with_suffix(".jpg")
    audio_output = input_file.with_suffix(".mp3")

    with mp.VideoFileClip(str(input_file)) as video:
        # Extract middle frame
        extract_frame(input_file, video.duration / 2, frame_output)

        # Extract audio
        if not audio_output.exists():
            video.audio.write_audiofile(str(audio_output), logger=None)

    return frame_output, audio_output 

@st.cache_data(show_spinner=False, hash_funcs={Clip: lambda x: x.__repr__()})
def describe_clips(clips: List[Clip], shotlist: str, previous_shot_id, next_shot_id) -> Dict:
    """
    Uses the Gemini model to match video clips to shot descriptions from a shotlist. Also determines if each video clip has a quote.
    """
    gcs = GCSManager()
    content: List = []

    # Instructions and example for Gemini
    content += [
        "You are a video editor who is matching video clips with shots from a shotlist.\n",
        "Here is an example:",
        """<example>
Shot1 has a quote saying, "GET OUT OF THE WAY, IT'S SPREADING!", and likely matches with the audio in clip002.

<response>
<clip>
    <id>002</id>
    <shot>1</shot>
    <description>FIREFIGHTERS FORMING A LINE AND CLOSING IN ON A RAGING BLAZE, RESIDENTS SHOUTING (English): "GET OUT OF THE WAY, IT'S SPREADING!"</description>
    <quote>1</quote>
</clip>
<clip>
    <id>003</id>
    <shot>2</shot>
    <description>RESIDENTS HUDDLED TOGETHER BEHIND TEMPORARY SHELTERS</description>
    <quote>0</quote>
</clip>
<clip>
    <id>004</id>
    <shot>2</shot>
    <description>RESIDENTS HUDDLED TOGETHER BEHIND TEMPORARY SHELTERS</description>
    <quote>0</quote>
</clip>
</response>
</example>

Here are the clips:
<clips>"""
    ]

    for clip in clips:
        name = clip.id
        frame_file, audio_file = extract_middle_frame_and_audio(clip.file_path)

        content += ["<clip>\n"]
        content += [f"ID {name}:", gcs.upload_to_gcs_part(frame_file), gcs.upload_to_gcs_part(audio_file)]
        content += [f"Clip has speech with transcript: {clip.whisper_results.english_text}" if clip.whisper_results.has_speech else "Clip may not have speech"]
        content += ["\n</clip>"]
        content += ["\n\n"]
    content += ["</clips>"]

    content += [f"""
These clips are between shot {previous_shot_id} and shot {next_shot_id}
"""]

    if shotlist:
        content += ["\nShotlist:\n", shotlist]
    else:
        print("ERROR: No shotlist")

    prompt = f"""Please match each clip with its shot in the shotlist.
Each clip should be in the same order, but shots may be matched to multiple adjacent clips. You can skip to the right shot if need be.
Include <id></id>, <shot></shot>, <description></description>, & <quote></quote> for each clip. Quote is 1 if the clip contains a quote, otherwise 0.
Try to match shots with quotes to the correct clip. The audio might be hard to hear but try your best.
Generally just fill it with shots based on order, but if a clip really doesn't match anything, say so and set <shot></shot> to -1, and <description></description> to Unknown.
Only label descriptions with those in the shotlist exactly. <shot></shot> should just be the number. <id></id> should be what I gave you, in a form like 001. Output XML in <response></response> tags"""

    content += ["\n\n", prompt]

    content = [part for part in content if part is not None]

    response = GEMINI.generate_content(content)

    gcs.clear_uploaded_blobs()

    clips_xml = extract_xml(response.text)
    return clips_xml

@st.cache_data(show_spinner=False, hash_funcs={PosixPath: hash_audio_file})
def full_description(clip_file, description, title):
    gcs = GCSManager()
    content = []
    content += ["Video clip:"]

    content += [gcs.upload_to_gcs_part(clip_file)]

    if title:
        content += ["This clip is from a video about: ", title]
    else:
        print("ERROR: Title is None")
    if description:
        content += ["This clip should specifically contain: ", description]
    else:
        print("ERROR: Description is None")
    content += ["""You are a news video editor. Please describe this video with as much detail as possible."""]

    content += ["""<example>
Shot: Wide, handheld, shaky, likely captured by a news crew on the scene. The footage suggests a sense of immediacy and chaos.
Location: An urban environment, possibly outside a government building or public space. The architecture visible in the background could provide clues to the specific location.
Time: Nighttime, illuminated by artificial lighting and possibly police floodlights. This creates a high-contrast scene with deep shadows, adding to the drama of the footage.
0:00-0:03
The camera pans right, following a line of police officers clad in full riot gear. They wear helmets with transparent visors and carry batons. The officers advance purposefully towards a makeshift barricade.
The barricade is constructed from a jumble of materials: wooden pallets, sheets of plywood, fabric. It's decorated with protest signs and colorful artwork, including a prominent image of a pineapple with a face. This suggests a degree of entrenched occupation, with protesters personalizing the space.
0:03-0:06
The officers reach the barricade and begin physically dismantling it, pushing against the structure with their combined weight.
Behind the barricade, glimpses of the protesters can be seen. They are packed tightly, forming a human wall behind the makeshift defenses. While individual faces are difficult to make out, the crowd appears agitated, their movements frantic.
Shouts and yells from both the officers and the protesters create a cacophony of sound. The specific words are difficult to discern, but the tone is urgent and aggressive.
Comments:
The video captures a pivotal moment of confrontation: the police initiating the dismantling of the protest encampment.
The protesters' makeshift barricade symbolizes their attempt to establish a physical presence and claim space.
The artwork and signs on the barricade hint at the protesters' motivations and demands. Identifying these details could be crucial to understanding the context of the protest.
The chaotic camera work, combined with the intense audio, immerses the viewer in the tension of the moment. It conveys the raw energy of both the police advance and the protesters' resistance.
This footage is valuable for illustrating the escalation of a protest. It could be used in a news report to provide viewers with a visceral understanding of the events unfolding on the ground. In a documentary context, this scene could represent a turning point in the protest timeline, signaling a shift towards more direct confrontation.
Minimum Timing: 0:00-0:05, this completes the pan and establishes the chaotic scene.
</example>"""]

    content += ["""
Describe this video with as much detail as possible. Include timestamps sections of the video. If possible, please give comments on people, location, timing,
shot (wide, professional, etc.), & anything else you feel could be relevant to fully understand what is happening in this video & making video editing decisions.
Also include the minimum amount of time this shot should be on screen for, before cutting away. Generally panning shots need to be on screen longer to not seem abrupt,
while a still shot can be cut off sooner. Include the key informational part of the clip."""]

    content = [part for part in content if part is not None]

    response = GEMINI.generate_content(content)

    # print(gemini.count_tokens(content))
    gcs.clear_uploaded_blobs()

    return response.text

@st.cache_data(show_spinner=False, hash_funcs={PosixPath: hash_audio_file})
def add_broll(audio_file, full_descriptions_str, section_timings_str):
    gcs = GCSManager()
    content = []
    content += ["You are a news video editor tasked with editing together an audio story with relevant B-roll video clips to make it compelling for a TV audience."]
    content += ["Here are the broll descriptions: ", f"<broll>{full_descriptions_str}</broll>\n\n"]
    content += ["Here is the audio clip: ", gcs.upload_to_gcs_part(audio_file)]
    content += ["Here are section timings: ", f"<section_timings>{section_timings_str}</section_timings>\n\n"]
    content += ["Here is an example: ", """<example>
**Section 1: 0 - 11.96**
Transcript: Activists in Canada...

* **Anchor (max 10 seconds):** 0.00 - 6.24 - We start on an Anchor to introduce the story and set the scene.
* **Clip 008 (max 10 seconds):** 6.24 - 7.56 - The image of a national flag acts as a transition.
* **Clip 020 (max 12 seconds):** 7.56 - 9.82 - This clip shows fireworks being launched, visually illustrating the audio description of fireworks.
* **Clip 002 (max 4 seconds):** 9.82 - 11.96 - End the section with...
</example>\n\n"""]
    content += ["""Please listen to the audio clip carefully, its script should match the section timings. I have given you <broll></broll> clips and want you to place them in the Audio clip.
Each broll clip has a Max duration which you should copy into the list, ex (max 10 seconds).
Give me timestamps for when you want a broll clip to start and end. Always fill each section with brolls till the end. But don't go beyond each section. You must end clip timings at the section end time!
Broll clips have a length, so you can't use more than that & have to switch. You don't have to use the entire Broll clip. Aim to switch around 6 seconds or sooner. Switching creates a more intense experience.
You may also insert Anchor blocks. These should be max 10 seconds. Place Anchor blocks for at least 5 seconds at the beginning of the whole story to set the scene, and at least 5 seconds at the end of the whole story to conclude. (probably the whole start and end sections. If the section is at the start or end of the whole story, you will be told so in <section_timings> with a message like "Anchor must be shown...")
This isn't necessary in other sections, just make your best judgement on when the Anchor should be on screen. For example, if there isn't any related broll, or not enough broll, you may choose to switch to the Anchor.
An Anchor block must be at least 3 seconds long, so don't place it at the very end of a section. Make sure everything flows nicely!
Show clips for at least 1 second before switching. Make sure your section numbers are correct, they may skip numbers. Broll show always be referenced as Clip ###, like in the example. Never make up broll clips. If not enough, use the Anchor.
This will be aired on TV, so select & place clips informatively and dramatically. Remember to write your thoughts as you insert clips (ex. We start on an Anchor to introduce the story and set the scene.)"""]

    content = [part for part in content if part is not None]

    response = GEMINI.generate_content(content)
    gcs.clear_uploaded_blobs()

    try:
        return response.text
    except ValueError:
        print(response.prompt_feedback)
        print(response.candidates[0].finish_reason)
        print(response.candidates[0].safety_ratings)
        return None
