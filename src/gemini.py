# Gemini

# STREAMLIT
from src.constants import GOOGLE_JSON
import os
import json

google_json = json.loads(GOOGLE_JSON, strict=False)
temp_file_path = "/tmp/service_account.json"
with open(temp_file_path, "w") as f:
    json.dump(google_json, f)

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the temporary file path
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path

from src.prompts import extract_xml
import streamlit as st
# /STREAMLIT

import random
import string
from typing import Union, Tuple, Dict, List
from pathlib import Path
import functools

from google.cloud import storage
import vertexai
from vertexai.generative_models import (GenerationConfig, GenerativeModel,
                                         HarmBlockThreshold, HarmCategory,
                                         Part, SafetySetting)

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

# Google Cloud Storage Setup
storage_client = storage.Client()
bucket = storage_client.bucket("gemini-colab")
blobs = []  # Store uploaded blobs for later cleanup

def upload_to_gcs(local_file_path: Union[str, Path]) -> Part:
    """Uploads a file to Google Cloud Storage and returns a Vertex AI Part.

    Args:
        local_file_path: Path to the local file to upload.

    Returns:
        A Vertex AI Part representing the uploaded file.
    """
    local_file_path = Path(local_file_path)
    random_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
    extension = local_file_path.suffix
    blob_name = random_string + extension
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_file_path)

    global blobs
    blobs.append(blob)

    uri = f"gs://{bucket.name}/{blob_name}"
    mime_types = {
        ".mp4": "video/mp4",
        ".mp3": "audio/mpeg",
        ".jpg": "image/jpeg",
    }
    mime_type = mime_types.get(extension) 
    if not mime_type:
        raise ValueError(f"Unsupported file extension: {extension}")

    return Part.from_uri(uri, mime_type=mime_type)

def clear_uploaded_blobs():
    """Deletes blobs that were uploaded to Google Cloud Storage."""
    global blobs
    for blob in blobs:
        blob.delete()
    blobs.clear()

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
            video.audio.write_audiofile(str(audio_output))

    return frame_output, audio_output 


@st.cache_data(show_spinner=False)
def describe_clips(clips_folder: Path, shotlist: str) -> Dict:
    """
    Uses the Gemini model to match video clips to shot descriptions from a shotlist. Also determines if each video clip has a quote.
    """
    content: List = []

    # Instructions and example for Gemini
    content += [
        "You are a video editor who is matching video clips with shots from a shotlist.\n",
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

Clips:"""
    ]

    files = list(sorted(clips_folder.glob("*.mp4")))
    for file in files:
        name = file.stem
        frame_file, audio_file = extract_middle_frame_and_audio(file)

        content += [f"{name}:", upload_to_gcs(frame_file), upload_to_gcs(audio_file)]

    content += ["\nShotlist:\n", shotlist]

    prompt = """Please match each clip with its shot in the shotlist. It should be in the same order, but shots may be matched to multiple adjacent clips.
You can skip to the right shot if need be. Start by going through shots with quotes in them and assign them to matching clips.
The quote should match with what is said in the clip. If the quote doesn't match any shot, make your best guess (perhaps find a shot that has unclear speech), but say you're not sure. Write these down.
Ex: Shot1 has a quote saying, "GET OUT OF THE WAY, IT'S SPREADING!", and likely matches with the audio in clip002.
Then assemble the final list using these quote clips as guide posts. Include <id></id>, <shot></shot>, <description></description>, & <quote></quote> for each clip.
Quote is 1 if the clip contains a quote, otherwise 0. If a clip doesn't doesn't match anything, say so and set <shot></shot> to -1.
Only label descriptions with those in the shotlist exactly. <shot></shot> should just be the number. Output XML in <response></response> tags"""

    content += ["\n\n", prompt]

    response = GEMINI.generate_content(content)

    clear_uploaded_blobs()

    clips_xml = extract_xml(response.text)
    return clips_xml

@st.cache_data(show_spinner=False)
def full_description(clip_file, description, title):
    content = []
    content += ["Video clip:"]

    content += [upload_to_gcs(clip_file)]

    content += ["This clip is from a video about: ", title]
    content += ["This clip should specifically contain: ", description]
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
</example>"""]

    content += ["""
Describe this video with as much detail as possible. Include timestamps sections of the video. If possible, please give comments on people, location, timing,
shot (wide, professional, etc.), & anything else you feel could be relevant to fully understand what is
happening in this video & making video editing decisions."""]

    response = GEMINI.generate_content(content)

    # print(gemini.count_tokens(content))
    clear_uploaded_blobs()

    return response.text

@st.cache_data(show_spinner=False)
def add_broll(audio_file, full_descriptions_str, section_timings_str):
    content = []
    content += ["You are a news video editor tasked with editing together an audio story with relevant B-roll video clips to make it compelling for a TV audience."]
    content += ["Here are the broll descriptions: ", f"<broll>{full_descriptions_str}</broll>\n\n"]
    content += ["Here is the audio clip: ", upload_to_gcs(audio_file)]
    content += ["Here are section timings: ", f"<section_timings>{section_timings_str}</section_timings>\n\n"]
    content += ["Here is an example: ", """<example>
**Section 1: 0 - 11.96**
Transcript: Activists in Canada...

* **Clip 008 (max 10 seconds):** 0.00 - 5.54 - The image of a national flag sets the scene and introduces the story.
* **Clip 020 (max 12 seconds):** 5.54 - 9.82 - This clip shows fireworks being launched, visually illustrating the audio description of fireworks.
* **Clip 002 (max 4 seconds):** 9.82 - 11.96 - End the section with...
</example>\n\n"""]
    content += ["""Please listen to the audio clip carefully & transcribe every section. I have given you <broll></broll> clips and want you to place them in the Audio clip.
Each broll clip has a Max duration which you should copy into the list, ex (max 10 seconds).
Give me timestamps for when you want a broll clip to start and end. Always fill each section with brolls till the end.
Broll clips have a length, so you can't use more than that & have to switch. You don't have to use the entire Broll clip. Aim to switch around 6 seconds or sooner. Switching creates a more intense experience.
Show clips for at least 1 second before switching. Make sure your section numbers are correct, they may skip numbers. This will be aired on TV, so select & place clips informatively and dramatically."""]

    response = GEMINI.generate_content(content)
    clear_uploaded_blobs()

    return response.text