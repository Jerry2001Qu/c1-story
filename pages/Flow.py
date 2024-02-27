import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

import os
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import shutil
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI

DIR = "/tmp/"

gpt4 = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview")

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def call_gpt(text):
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": text},
        ],
        n=1
    )
    return response.choices[0].message.content


def save_uploaded(file_data):
    with open(os.path.join(DIR, file_data.name),"wb") as f:
        f.write(file_data.getbuffer())


def source_material():
    video = st.file_uploader("Choose a video", type=["mp4"], accept_multiple_files=False)
    if video:
        save_uploaded(video)
        with st.expander("Uploaded video"):
            st.video(os.path.join(DIR, video.name))
    
    shotlist = st.text_area("Shotlist")
    story = st.text_area("Story")

    return os.path.join(DIR, video.name) if video else None, shotlist, story


from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg

def split_clips(video_filename):
    scene_list = detect(video_filename, AdaptiveDetector(adaptive_threshold=5, min_scene_len=1))
    os.makedirs(os.path.join(DIR, "clips/"), exist_ok=True)
    if scene_list:
        st.write(f"Found {len(scene_list)} clips")
        split_video_ffmpeg(video_filename, scene_list, show_progress=True, output_file_template=os.path.join(DIR, "clips/$SCENE_NUMBER.mp4"), show_output=True)
    else:
        st.write("Only 1 clip")
        shutil.copy(video_filename, os.path.join(DIR, "clips/1.mp4"))


from src.clip import Clip, WhisperResults

import torch
import moviepy.editor as mp

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=True,
                              trust_repo=True)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

def transcribe_clips():
    clips = {int(path.stem): Clip(int(path.stem), path) for path in sorted(Path(DIR + "clips/").glob("*.mp4"))}
    
    vad_p_bar = st.progress(0, text="VAD Progress")
    current = 0
    for clip_num, clip in clips.items():
        path = clip.path
        video = mp.VideoFileClip(str(path))
        video.audio.write_audiofile(path.with_suffix(".wav"), verbose=False)

        wav = read_audio(str(path.with_suffix(".wav")))
        speech_timestamps = get_speech_timestamps(wav, vad_model, min_speech_duration_ms=2000)
        clip.vad = True if speech_timestamps else False

        current += 1
        vad_p_bar.progress(current / len(clips), text="VAD Progress")

    whisper_p_bar = st.progress(0, text="Whisper Progress")
    current = 0
    for clip_num, clip in list(filter(lambda x: x[1].vad, clips.items())):
        clip.whisper = WhisperResults.from_file(clip.path)

        current += 1
        whisper_p_bar.progress(current / len(list(filter(lambda x: x[1].vad, clips.items()))), text="Whisper Progress")

    st.session_state.clips = clips


from src.describe import describe_clips

def label_clips():
    descriptions_json = describe_clips(st.session_state.clips, st.session_state.shotlist)

    for item in descriptions_json["clips"]:
        st.session_state.clips[int(item["clip_id"])].description = item["description"]
        st.session_state.clips[int(item["clip_id"])].duration = mp.VideoFileClip(str(st.session_state.clips[int(item["clip_id"])].path)).duration


def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Channel 1 Demo")

    video, shotlist, story = source_material()
    if not video:
        return
    st.session_state.video = video
    st.session_state.shotlist = shotlist
    st.session_state.story = story

    if st.button("Split clips"):
        split_clips(video)
    
    clips = Path(os.path.join(DIR, "clips/")).glob("*.mp4")
    if not clips:
        return

    with st.expander("Clips"):
        for clip in clips:
            st.video(str(clip))
    
    if st.button("Transcribe clips"):
        transcribe_clips()

    if "clips" not in st.session_state:
        return
    
    with st.expander("Transcripts"):
        for clip_num, clip in st.session_state.clips.items():
            if clip.vad:
                if clip.whisper.has_speech:
                    clip.is_sot = True
                    st.write(clip.scores_summary())
                else:
                    clip.is_sot = False
                    st.write(clip.scores_summary())
            else:
                clip.is_sot = False

    if st.button("Label clips"):
        label_clips()

    if not st.session_state.clips or not next(iter(st.session_state.clips.values())).description:
        return
    
    with st.expander("Labels"):
        for clip_num, clip in st.session_state.clips.items():
            st.write(clip.description)
            st.video(str(clip.path))
    
    

#     if "data" not in st.session_state:
#         st.session_state.data = pd.read_csv("cleaned.csv").drop(columns=["Unnamed: 0"])

#     instructions = st.text_area(
#         "Instructions",
#         value="Given this info, write a news script that's about 300 words. Include quotes where possible with SOT: at the beginning of the paragraph. Never make up information.",
#     )

#     prompt_template = PromptTemplate.from_template(
# """{story}

# \###

# {instructions}""")

#     submitted = st.button("Submit")

#     row = st.columns(3)
#     if "cols" not in st.session_state:
#         st.session_state.cols = [[col, i] for i, col in enumerate(row)]
#     else:
#         for col, col_state in zip(row, st.session_state.cols):
#             col_state[0] = col
    
#     if "responses" not in st.session_state:
#         st.session_state.responses = {}

#     if submitted:
#         for _, i in st.session_state.cols:
#             st.session_state.responses[i] = call_gpt(prompt_template.format(story=st.session_state.data["Main Content"][i], instructions=instructions))

#     for i, (col, story_id) in enumerate(st.session_state.cols):
#         key = f"{i}-{story_id}"
#         with col:
#             with st.container(border=True):
#                 story_id = st.selectbox(
#                     "Prefill Story",
#                     st.session_state.data.index,
#                     index=story_id,
#                     format_func=lambda x: st.session_state.data.Slug[x],
#                     key=key+"story_selector",
#                 )
#                 st.session_state.cols[i][1] = story_id
#                 with st.expander(st.session_state.data.Title[story_id]):
#                     st.write(st.session_state.data["Main Content"][story_id])
#                 if story_id in st.session_state.responses:
#                     st.write(st.session_state.responses[story_id])
        
    

if __name__ == "__main__":
    run()
