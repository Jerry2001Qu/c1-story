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

from src.prompts import extract_xml, extract_response
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
Rating: 8/10, the footage is high-quality and captures a key moment in the protest narrative.
</example>"""]

    content += ["""
Describe this video with as much detail as possible. Include timestamps sections of the video. If possible, please give comments on people, location, timing,
shot (wide, professional, etc.), & anything else you feel could be relevant to fully understand what is happening in this video & making video editing decisions.
Additional details could include if the person is speaking, if you can see their face, logos, landmarks, all details to describe the scene.
Also include the minimum amount of time this shot should be on screen for, before cutting away. Generally panning shots need to be on screen longer to not seem abrupt,
while a still shot can be cut off sooner. Include the key informational part of the clip. Give a rating out of 10 for how useful this clip is for a news report, and a reason why.
A 10 would be pivotal to the story, and high quality. A 1 would not be useful, and be of low quality. A still shot of a picture would get a lower rating. Very short clips (< 2s) would get a lower rating."""]

    content = [part for part in content if part is not None]

    response = GEMINI.generate_content(content)

    # print(gemini.count_tokens(content))
    gcs.clear_uploaded_blobs()

    return response.text

@st.cache_data(show_spinner=False, hash_funcs={PosixPath: hash_audio_file})
def add_broll(audio_file, full_descriptions_str, section_timings_str):
    gcs = GCSManager()
    content = []

    content += [
f"""You are a news video editor tasked with editing together an audio story with relevant B-roll video clips to make it compelling 
for a TV audience. Your goal is to create a visually engaging and informative news segment by matching appropriate video clips to the audio content.

First, review the available B-roll clips:
<broll_descriptions>
{full_descriptions_str}
</broll_descriptions>

Next, listen carefully to the audio clip:
""",
gcs.upload_to_gcs_part(audio_file),
f"""
Here are the section timings for the audio clip:
<section_timings>
{section_timings_str}
</section_timings>

Your task is to select and place B-roll clips and Anchor segments to accompany the audio story. Follow these guidelines:

1. Match B-roll clips to the content of each section in the audio.
2. Always fill each section with B-roll clips until the end, but don't exceed the section's end time.
3. Aim to switch clips every 6 seconds or sooner for a more intense experience.
4. Use Anchor blocks (max 10 seconds) at the beginning and end of the story, and as needed throughout.
5. Ensure smooth transitions between clips and sections.

When selecting and placing clips:
- Copy the max duration for each B-roll clip (e.g., "max 10 seconds"). Never go over this limit.
- Use at least 5 seconds of Anchor at the beginning and end of the whole story.
- Place Anchor blocks for at least 1.5 seconds, so never at the very end of a section.
- Don't use too many Anchor blocks. One at the beginning, one at the end, and some for transitioning.
- An Anchor block should last for an entire thought, or be used as a transition.
- Show clips for at least 1 second before switching.
- Don't reuse B-roll if you can avoid it. Repetition is not good.
- If there isn't enough relevant B-roll, use Anchor segments.
- Prefer B-roll over Anchor segments. If there's no related B-roll, perhaps there's background B-roll that could be used as filler?
- Anchor blocks can be used as transitions.
  - For example, "People are looking forward to this weeks events..." could have Anchor on screen
  - That can be followed by "including a jazz concert, carnival games, and a prize to be won." with all 3 being shown as B-roll.
- When selecting clips, try to show & tell. The b-roll should match what is being said in the script.
- If placing a b-roll that is an interview or a person speaking, cut quickly (within 3 seconds) to another b-roll. It's weird since the audio is different.

Format your output as follows for each section:
1. Section number and time range
2. Brief transcript or description of the audio content
3. Brief thoughts on the section and what visuals would enhance it
3. List of clips with timestamps, max duration, and brief explanation for each choice

Here's an example of the expected output format:

<example>
<thoughts>
This story has less b-roll available, we should not transition as frequently... (continued thoughts)
</thoughts>

**Section 1: 0:00 - 11.96**
Transcript: Activists in Canada are protesting against the government's decision to expand a pipeline. The protests have been ongoing for weeks, with activists demanding action on climate change by using fireworks to make their point.
Thoughts: This section is about the protests in Canada. We should show footage of the protests and the activists involved. We should cut at 7.56 to transition to the fireworks...

* Anchor (max 10 seconds): 0.00 - 7.24 - We start on an Anchor to introduce the story and set the scene when saying ...
* Clip 008 (max 10 seconds):  7.24 - 10.11 - The image of a national flag acts as a transition when saying ...
* Clip 020 (max 12 seconds): 10.11 - 16.68 - This clip shows fireworks being launched, visually illustrating the audio description of fireworks when saying ...
* Clip 002 (max 4 seconds): 16.68 - 18.82 - End the section with...
</example>

<example>
<thoughts>
This story had more B-roll available, allowing us to transition more frequently... (continued thoughts)
</thoughts>

**Section 1: 0.00 - 12.98**
Transcript: US President Joe Biden at 81 faces a critical moment in his reelection campaign as he prepares for a high-stakes NATO summit in Washington amid mounting pressure from fellow Democrats. Despite calls from within his own party to end his bid, Biden remains resolute.
Thoughts: This section introduces the main topic of the video: Biden's reelection campaign and the pressure he faces from Democrats. We should show footage of Biden speaking, the NATO summit, and Democrats expressing concern.

* Clip 004 (max 3 seconds): 0.00 - 2.00 - We start with a still image of Biden speaking to set the scene when saying "US President Joe Biden at 81"
* Anchor (max 10 seconds): 2.00 - 3.65 - Swap to the Anchor to transition to the NATO summit when saying "faces a critical moment in his reelection campaign"
* Clip 002 (max 1.22 seconds): 3.65 - 4.77 - This clip shows Biden shaking hands at NATO, visually illustrating the audio description of the summit when saying "as he prepares for a high-stakes"
* Clip 003 (max 1.32 seconds): 4.77 - 6.09 - The last clip reached it's max. This clip is a continuation of the last clip with Biden still shaking hands when saying "NATO summit in Washington"
* Clip 017 (max 4.56 seconds): 6.09 - 8.56 - Footage of democrats to visually show mouinting pressure, when saying "amid mounting pressure from fellow Democrats"
* Anchor (max 10 seconds): 8.56 - 11.23 - Back to the Anchor to transition to Biden remaining resolute, when saying "Despite calls from within his own party"
* Clip 015 (max 8.23 seconds): 11.23 - 12.98 - This clip shows Biden speaking, visually illustrating the audio description of Biden's reelection campaign when saying "Biden remains resolute"

**Section 5: 12.98 - 24.54**
Transcript: As Biden navigates these challenges, he faces a pivotal moment in his bid for reelection.
Thoughts: This section concludes the video by emphasizing the importance of the coming days for Biden's campaign. We should show footage of Biden at the White House, the Capitol building, Biden boarding Air Force One, and Biden speaking at a campaign event.

* Anchor (max 10 seconds): 6.45 - 8.56 - No more footage of Biden so we go back to the Anchor when saying "As Biden navigates these challenges"
* Clip 008 (max 1 second): 8.56 - 9.59 - The footage of the white house acts as a transition to reelection when saying "he faces"
* Clip 010 (max 4.32 seconds): 9.59 - 10.85 - A different angle of the white house allows for a faster paced feeling, when saying "a pivotal moment"
* Clip 018 (max 3.21 seconds): 10.85 - 12.03 - This video of Biden speaking ends the transition to Biden and reelection, when saying "in his bid for reelection"
</example>

Remember:
- Always use the correct section numbers, even if they skip.
- Write the whole transcript out for each section. Don't do ...
- Refer to B-roll as "Clip ###" (e.g., Clip 008).
- Never invent B-roll clips that aren't provided.
- Select and place clips informatively and dramatically for TV audiences.
- Explain your thought process for each clip selection.
- Reference the transcript for each clip selection.

Start by writing a first draft of the edited video sequence inside <draft> tags. Follow the format shown in the example, for all sections of the audio clip.

Critique your draft inside <critique> tags.

KEEP WRITING DRAFTS AND CRITIQUES UNTIL THE CRITIQUE IS SATISFACTORY. Usually 1-2 times. Do this now, don't ask to continue.
When writing drafts ensure all requirements are met (min timings, etc.), then suggest improvements.

Put your final edited video sequence inside <response> tags. YOU MUST ALWAYS END WITH THIS.
"""
]

    content = [part for part in content if part is not None]

    response = GEMINI.generate_content(content)
    gcs.clear_uploaded_blobs()

    print("## BROLL GEMINI RESPONSE\n\n", response.text)

    try:
        return extract_response(response.text)
    except ValueError:
        print(response.prompt_feedback)
        print(response.candidates[0].finish_reason)
        print(response.candidates[0].safety_ratings)
        return None

@st.cache_data(show_spinner=False, hash_funcs={PosixPath: hash_audio_file, Clip: lambda x: x.__repr__()})
def add_broll_clips(audio_file, clips, sot_clip_ids, section_timings_str):
    gcs = GCSManager()
    content = []

    content += [
f"""You are a news video editor tasked with editing together an audio story with relevant B-roll video clips to make it compelling 
for a TV audience. Your goal is to create a visually engaging and informative news segment by matching appropriate video clips to the audio content.

First, review the available B-roll clips:
<broll_descriptions>
"""]
    
    for clip in clips:
        duration = clip.duration
        frame_file = clip.file_path.with_suffix(".jpg")
        extract_frame(clip.file_path, min(duration, 1.0), frame_file)
        if not clip.has_quote:
            clip_section = [f"<clip {clip.id}>\n", gcs.upload_to_gcs_part(frame_file), f"\n{clip.full_description}\n\nMax duration: {duration} seconds\n</clip {clip.id}>\n"]
        elif clip.id not in sot_clip_ids:
            if clip.full_description:
                lines = clip.full_description.split("\n")
                full_description = '\n'.join(line for line in lines if not line.startswith("Minimum Timing:"))
            else:
                full_description = clip.shotlist_description
            clip_section = [f"<clip {clip.id}>\n", gcs.upload_to_gcs_part(frame_file), f"\nThis clip is a SOT/Interview. \n{full_description}\n\nMax duration: {min(duration, 3)} seconds\n</clip {clip.id}>\n"]
        else:
            clip_section = []
        content += clip_section

    content += [
f"""
Next, listen carefully to the audio clip:
""",
gcs.upload_to_gcs_part(audio_file),
f"""
Here are the section timings for the audio clip:
<section_timings>
{section_timings_str}
</section_timings>

Your task is to select and place B-roll clips and Anchor segments to accompany the audio story. Follow these guidelines:

1. Match B-roll clips to the content of each section in the audio.
2. Always fill each section with B-roll clips until the end, but don't exceed the section's end time.
3. Aim to switch clips every 6 seconds or sooner for a more intense experience.
4. Use Anchor blocks (max 10 seconds) at the beginning and end of the story, and as needed throughout.
5. Ensure smooth transitions between clips and sections.

When selecting and placing clips:
- Copy the max duration for each B-roll clip (e.g., "max 10 seconds"). Never go over this limit.
- Use at least 5 seconds of Anchor at the beginning and end of the whole story.
- Place Anchor blocks for at least 2 seconds, so never at the very end of a section.
- Don't use too many Anchor blocks. One at the beginning, one at the end, and some for transitioning.
- An Anchor block should last for an entire thought, or be used as a transition.
- Show clips for at least 1 second before switching.
- Don't reuse B-roll if you can avoid it. Repetition is not good.
- If there isn't enough relevant B-roll, use Anchor segments.
- Prefer B-roll over Anchor segments. If there's no related B-roll, perhaps there's background B-roll that could be used as filler?
- Anchor blocks can be used as transitions.
  - For example, "People are looking forward to this weeks events..." could have Anchor on screen
  - That can be followed by "including a jazz concert, carnival games, and a prize to be won." with all 3 being shown as B-roll.
- When selecting clips, try to show & tell. The b-roll should match what is being said in the script.
- If placing a b-roll that is an interview or a person speaking, cut quickly (within 3 seconds) to another b-roll. It's weird since the audio is different.

Format your output as follows for each section:
1. Section number and time range
2. Brief transcript or description of the audio content
3. Brief thoughts on the section and what visuals would enhance it
3. List of clips with max duration, timestamps, duration, brief explanation for each choice, and the part of the transcript it corresponds to

Here's an example:
<example>
<response>
<thoughts>
This story had more B-roll available, allowing us to transition more frequently... (continued thoughts)
</thoughts>

**Section 1: 0.00 - 12.98**
Transcript: US President Joe Biden at 81 faces a critical moment in his reelection campaign as he prepares for a high-stakes NATO summit in Washington amid mounting pressure from fellow Democrats. Despite calls from within his own party to end his bid, Biden remains resolute.
Thoughts: This section introduces the main topic of the video: Biden's reelection campaign and the pressure he faces from Democrats. We should show footage of Biden speaking, the NATO summit, and Democrats expressing concern.

* Clip 004 (max 3 seconds): 0.00 - 2.00 (2.00s) - We start with a still image of Biden speaking to set the scene. ("US President Joe Biden at 81")
* Anchor (max 10 seconds): 2.00 - 3.65 (1.65s) - Swap to the Anchor to transition to the NATO summit. ("faces a critical moment in his reelection campaign")
* Clip 002 (max 1.22 seconds): 3.65 - 4.77 (1.12s) - This clip shows Biden shaking hands at NATO, visually illustrating the audio description of the summit. ("as he prepares for a high-stakes")
* Clip 003 (max 1.32 seconds): 4.77 - 6.09 (1.32s) - The last clip reached it's max. This clip is a continuation of the last clip with Biden still shaking hands. ("NATO summit in Washington")
* Clip 017 (max 4.56 seconds): 6.09 - 8.56 (2.47s) - Footage of democrats to visually show mouinting pressure. ("amid mounting pressure from fellow Democrats")
* Anchor (max 10 seconds): 8.56 - 11.23 (2.67s) - Back to the Anchor to transition to Biden remaining resolute. ("Despite calls from within his own party")
* Clip 015 (max 8.23 seconds): 11.23 - 12.98 (1.75s) - This clip shows Biden speaking, visually illustrating the audio description of Biden's reelection campaign. ("Biden remains resolute")

**Section 5: 12.98 - 24.54**
Transcript: As Biden navigates these challenges, he faces a pivotal moment in his bid for reelection.
Thoughts: This section concludes the video by emphasizing the importance of the coming days for Biden's campaign. We should show footage of Biden at the White House, the Capitol building, Biden boarding Air Force One, and Biden speaking at a campaign event.

* Anchor (max 10 seconds): 6.45 - 8.56 (2.11s) - No more footage of Biden so we go back to the Anchor ("As Biden navigates these challenges")
* Clip 008 (max 1 second): 8.56 - 9.59 (1.03s) - The footage of the white house acts as a transition to reelection ("he faces")
* Clip 010 (max 4.32 seconds): 9.59 - 10.85 (1.26s) - A different angle of the white house allows for a faster paced feeling ("a pivotal moment")
* Clip 018 (max 3.21 seconds): 10.85 - 12.03 (1.18s) - This video of Biden speaking ends the transition to Biden and reelection ("in his bid for reelection")

**Section 6: 24.54 - 30.00**
Transcript: The fate of his presidency hangs in the balance as he confronts the challenges of a divided nation.
Thoughts: This is the last section. We must show the Anchor starting at 25.00 to the end. Since it's short, we might as well fill the whole section with the Anchor.

* Anchor (max 10 seconds): 24.54 - 30.00 - This is the final section, so we should end with the Anchor. ("The fate of his presidency hangs in the balance as he confronts the challenges of a divided nation")
</response>
</example>
<example>
<response>
<thoughts>
This draft incorporates the feedback from the previous critique, aiming for more visual variety and a stronger narrative flow. We've replaced repetitive B-roll, added ground-level shots to break up the aerial perspective, and incorporated clips that better reflect the human impact of the hurricane.
</thoughts>

**Section 1: 0.00 - 8.99**
Transcript: "Go home Beryl" read a spray-painted message on a storefront protected with wooden planks, as Hurricane Beryl tore through the Caribbean.
Thoughts: This section sets the scene and introduces Hurricane Beryl. We should start with the Anchor to introduce the story and then transition to B-roll showing the aftermath of the hurricane in the Caribbean.

* Anchor (max 10 seconds): 0.00 - 5.00 (5.00s) - We start with the Anchor to introduce the story. ("Go home Beryl" read a spray-painted message on a storefront protected with wooden planks, as Hurricane Beryl tore through the Caribbean.)
* Clip 018 (max 4.61 seconds): 5.00 - 8.99 (3.99s) - This clip shows a boarded-up storefront with the message "GO HOME BERYL" spray-painted on it, perfectly matching the audio. ("Go home Beryl")

**Section 2: 8.99 - 20.64**
Transcript: Residents of Union Island could be seen navigating piles of debris to assess the damage to their homes after the hurricane destroyed more than ninety percent of buildings, according to government officials.
Thoughts: This section focuses on the destruction caused by the hurricane in Union Island. We should show B-roll of the damage and people assessing the situation.

* Clip 001_0 (max 7.47 seconds): 8.99 - 16.46 (7.47s) - This clip shows the aftermath of the hurricane in a Caribbean town, with people navigating debris. ("Residents of Union Island could be seen navigating piles of debris to assess the damage to their homes")
* Clip 003 (max 5.54 seconds): 16.46 - 20.64 (4.18s) - This clip shows destroyed boats on a beach, providing a different visual of the destruction. ("after the hurricane destroyed more than ninety percent of buildings, according to government officials.")

**Section 3: 20.64 - 37.91**
Transcript: Drone footage revealed the aftermath of Beryl's destruction across the region. In Petite Martinique, Grenada, video showed destroyed buildings, boats and debris along the shore. In Carriacou, Grenada, footage showed destroyed houses, uprooted trees and debris strewn on roads.
Thoughts: This section provides a broader view of the hurricane's impact across the region. We should use drone footage to show the destruction in different locations, but also incorporate a ground-level shot for visual variety.

* Clip 002 (max 17.52 seconds): 20.64 - 28.16 (7.52s) - This drone footage shows widespread destruction in a coastal town, matching the audio description. ("Drone footage revealed the aftermath of Beryl's destruction across the region.")
* Clip 004_0 (max 7.26 seconds): 28.16 - 35.42 (7.26s) - This ground-level shot shows a wrecked sailboat and a damaged beach shack, providing a different perspective on the destruction. ("In Petite Martinique, Grenada, video showed destroyed buildings, boats and debris along the shore.")
* Clip 009_0 (max 7.25 seconds): 35.42 - 37.91 (2.49s) - This drone footage shows damage to buildings and uprooted trees, matching the audio description of Carriacou. ("In Carriacou, Grenada, footage showed destroyed houses, uprooted trees and debris strewn on roads.")

**Section 4: 37.91 - 45.69**
Transcript: The hurricane has left at least ten people dead, but that number was expected to rise as communications are restored on devastated islands.
Thoughts: This section highlights the human cost of the hurricane. While there's no B-roll showing casualties, we can use existing B-roll of destruction to visually reinforce the impact.

* Clip 007_1 (max 5.24 seconds): 37.91 - 43.15 (5.24s) - This clip shows a devastated residential area, emphasizing the severity of the hurricane's impact. ("The hurricane has left at least ten people dead, but that number was expected to rise as communications are restored on devastated islands.")
* Anchor (max 10 seconds): 43.15 - 45.69 (2.54s) - We transition back to the Anchor to finish the section. ("but that number was expected to rise as communications are restored on devastated islands.")

**Section 5: 45.69 - 52.09**
Transcript: As Beryl rumbled towards the Cayman Islands and Mexico, residents in Tulum prepared for its arrival.
Thoughts: This section shifts the focus to Mexico and the preparations being made for the hurricane's arrival. We should show B-roll of Tulum and people preparing for the storm.

* Clip 025 (max 5.01 seconds): 45.69 - 50.70 (5.01s) - This clip shows strong winds and preparations at a resort in Tulum, setting the scene for the hurricane's arrival. ("As Beryl rumbled towards the Cayman Islands and Mexico, residents in Tulum prepared for its arrival.")
* Clip 027_1 (max 4.35 seconds): 50.70 - 52.09 (1.39s) - This clip shows workers taping up windows at a resort, further illustrating the preparations. ("residents in Tulum prepared for its arrival.")

**Section 7: 52.09 - 65.99**
Transcript: Authorities closed beaches, urging people to remain indoors, secure their windows, clear their drains, and stock up on food and supplies. While some tourists adhered to the government's directives, others seemed less concerned.
Thoughts: This section details the precautions being taken in Mexico. We should show B-roll of closed beaches, people preparing their homes, and tourists reacting to the warnings.

* Clip 030 (max 4.67 seconds): 52.09 - 56.76 (4.67s) - This clip shows a closed beach with red warning tape, matching the audio description. ("Authorities closed beaches, urging people to remain indoors,")
* Clip 038 (max 4.47 seconds): 56.76 - 61.23 (4.47s) - This clip shows workers securing a beach club, illustrating preparations for the hurricane. ("secure their windows, clear their drains, and stock up on food and supplies.")
* Clip 044 (max 4.27 seconds): 61.23 - 65.50 (4.27s) - This clip shows a crowded airport, suggesting tourists leaving the area due to the hurricane. ("While some tourists adhered to the government's directives,")
* Anchor (max 10 seconds): 65.50 - 65.99 (0.49s) - We transition back to the Anchor to bridge the gap to the next section. ("others seemed less concerned.")

**Section 9: 65.99 - 84.54**
Transcript: In Cancun, businesses were preparing for Beryl's arrival. Workers could be seen filling sandbags and boarding up doors and windows of shops and hotels. The international airport was packed with tourists hoping to catch the last flights out, with around one hundred flights cancelled according to local authorities.
Thoughts: This section focuses on Cancun and the preparations being made there. We should show B-roll of businesses boarding up, sandbags being filled, and the crowded airport.

* Clip 039 (max 3.97 seconds): 65.99 - 69.96 (3.97s) - This clip shows sandbags being used to protect a beach club in Cancun. ("In Cancun, businesses were preparing for Beryl's arrival.")
* Clip 041 (max 4.31 seconds): 69.96 - 74.27 (4.31s) - This clip shows workers securing a sign at a nightclub, illustrating preparations for the hurricane. ("Workers could be seen filling sandbags and boarding up doors and windows of shops and hotels.")
* Clip 045 (max 4.97 seconds): 74.27 - 79.24 (4.97s) - This clip shows a crowded airport, matching the audio description of tourists trying to leave Cancun. ("The international airport was packed with tourists hoping to catch the last flights out,")
* Anchor (max 10 seconds): 79.24 - 84.54 (5.30s) - We end with the Anchor to conclude the story. ("with around one hundred flights cancelled according to local authorities.")
</response>
</example>

Remember:
- Always use the correct section numbers, even if they skip.
- Write the whole transcript out for each section Transcript:. Don't do ...
- Refer to B-roll as "Clip ###" (e.g., Clip 008).
- Never invent B-roll clips that aren't provided.
- Select and place clips informatively and dramatically for TV audiences.
- Explain your thought process for each clip selection.
- Quote the part of transcript for each clip selection. ONLY the part during the timestamps, not the whole quote!
- B-roll must be used for at least 1 second before switching to another clip.
- Anchor blocks must be used for at least 2 seconds before switching to another clip.

Start by writing a first draft of the edited video sequence inside <draft> tags. Follow the format shown in the example, for all sections of the audio clip.

Critique your draft inside <critique> tags.

KEEP WRITING DRAFTS AND CRITIQUES UNTIL THE CRITIQUE IS SATISFACTORY. Usually 1-2 times. Do this now, don't ask to continue.
When writing drafts ensure all requirements are met (min timings, etc.), then suggest improvements.

Put your final edited video sequence inside <response></response> tags. YOU MUST ALWAYS END WITH THIS. Do not leave this section blank, fill it with your final draft.
"""
]

    content = [part for part in content if part is not None]

    response = GEMINI.generate_content(content)

    print("## BROLL GEMINI RESPONSE\n\n", response.text)

    try:
        placements = extract_response(response.text)
        if len(placements) > 50:
            gcs.clear_uploaded_blobs()
            return placements
        else:
            print("ERROR: GEMINI BROLL RESPONSE NOT FILLED. CONTINUING")
            continue_content = [
                f"""You gave me this: {response.text}

My original instructions were this: {content}

Please continue where you left off and generate a final <response> for me."""
            ]
            response = GEMINI.generate_content(continue_content)
            print("## CONTINUED BROLL GEMINI RESPONSE\n\n", response.text)

            placements = extract_response(response.text)
            if len(placements) > 50:
                gcs.clear_uploaded_blobs()
                return placements
            else:
                placements = extract_str_between(response.text[response.text.rfind("<draft>"):], "<draft>", "</draft>")
                if len(placements) > 50:
                    gcs.clear_uploaded_blobs()
                    return placements
                else:
                    gcs.clear_uploaded_blobs()
                    raise Exception("GEMINI BROLL RESPONSE NOT FILLED")
    except ValueError:
        print(response.prompt_feedback)
        print(response.candidates[0].finish_reason)
        print(response.candidates[0].safety_ratings)
        return None

def extract_str_between(text: str, left_tag: str, right_tag: str) -> str:
    return text[text.find(left_tag):text.rfind(right_tag)+len(right_tag)]
