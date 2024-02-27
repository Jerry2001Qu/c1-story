import os
import streamlit as st

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Channel 1"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

vision_model = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-vision-preview", max_tokens=4096)

from PIL import Image
import base64
from io import BytesIO
import moviepy.editor as mp


def encode_first_frame(clip):
    path = clip.path
    frame = Image.fromarray(mp.VideoFileClip(str(path)).get_frame(0))
    buffered = BytesIO()
    frame.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def describe_clips(clips, b_roll_str):
    chain = vision_model | JsonOutputParser()

    result = chain.invoke([
        HumanMessage(
            content=[
                {"type": "text", "text": f"# B-rolls list\n{b_roll_str}"},
                {"type": "text", "text": "# Clip frames\n"},
            ] +
            [y for x in [
                [{"type": "text", "text": f"clip_id: {clip_id}"},
                {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_first_frame(clip)}",
                        "detail": "low"
                    }
                }] for clip_id, clip in list(clips.items())
            ] for y in x] +
            [
                {"type": "text", "text": "For each clip frame, identify which B-roll it belongs to, copy the description exactly & add additional context to a max of 15 words. It fits roughly in order of the B-rolls list. Don't mention it's a video frame. Output JSON."},
                {"type": "text", "text": """
# EXAMPLE #
B-rolls list
BROLL1: People walking down the street in California.

JSON
```json
{{
  "clips": [
    {{
        "clip_id": 1,
        "broll": "BROLL1",
        "description": "People walking down the street in California. A sign reads, 'humanity'.",
    }},
    ...
  ]
}}"""},
            ]
        )
    ])

    return result