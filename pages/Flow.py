import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI

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
    with open(file_data.name,"wb") as f:
        f.write(file_data.getbuffer())
    return st.success(f"Saved File: {file_data.name}")


def source_material():
    video = st.file_uploader("Choose a video", type=["mp4"], accept_multiple_files=False)
    if video:
        save_uploaded(video)
        with st.expander("Uploaded video"):
            st.video(video.name)
    
    shotlist = st.text_area("Shotlist")
    story = st.text_area("Story")

    return video.name if video else None, shotlist, story


from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg

def split_clips(video_filename):
    scene_list = detect(video_filename, AdaptiveDetector(adaptive_threshold=5, min_scene_len=1))
    st.write(f"Found {len(scene_list)} clips")
    split_video_ffmpeg(video_filename, scene_list, show_progress=True, output_file_template="clips/$SCENE_NUMBER.mp4")


def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Channel 1 Demo")

    video, shotlist, story = source_material()
    if video:
        if st.button("Split clips"):
            split_clips(video)


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
