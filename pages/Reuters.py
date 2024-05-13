import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Channel 1"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, XMLOutputParser

openai_api_key = st.secrets["OPENAI_API_KEY"]
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]
elevenlabs_api_key = st.secrets["ELEVENLABS_API_KEY"]

gpt4 = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=openai_api_key)

opus = ChatAnthropic(model="claude-3-opus-20240229", temperature=0, max_tokens=4096, anthropic_api_key=anthropic_api_key)
sonnet = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, max_tokens=4096, anthropic_api_key=anthropic_api_key)
haiku = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0, max_tokens=4096, anthropic_api_key=anthropic_api_key)

set_llm_cache(SQLiteCache(database_path="langchain.db"))

import google.generativeai as genai
genai.configure(api_key=google_api_key)

generation_config = {
  "temperature": 0,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]

gemini = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config, safety_settings=safety_settings)

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice

client = ElevenLabs(
    api_key=elevenlabs_api_key
)

@st.cache_data
def TTS(text, filename):
    audio = client.generate(
        text=text,
        voice=Voice(
            voice_id="H2LXjnBS1droRODepT50"
        )
    )

    with open(filename, 'wb') as file:
        for chunk in audio:
            if chunk:
                file.write(chunk)

from src.prompts import get_sot_prompt, reformat_prompt, sot_prompt, parse_prompt, section_summary_prompt

get_sot_chain = get_sot_prompt | opus
reformat_chain = reformat_prompt | opus
sot_chain = sot_prompt | opus
parse_chain = parse_prompt | opus
section_summary_chain = section_summary_prompt | opus

from PIL import Image
import base64
from io import BytesIO
import moviepy.editor as mp

def extract_xml(text):
    return XMLOutputParser().invoke(text[text.find("<"):text.rfind(">")+1].replace("&", "and"))

def extract_first_frame_and_audio(input_file):
    video = mp.VideoFileClip(str(input_file))
    first_frame = video.get_frame(0)
    frame_output = input_file.with_suffix(".jpg")
    image = Image.fromarray(first_frame)
    if not frame_output.exists():
        image.save(frame_output)

    audio = video.audio
    audio_output = input_file.with_suffix(".mp3")
    if not audio_output.exists():
        audio.write_audiofile(str(audio_output))

    video.close()
    audio.close()
    
    return frame_output, audio_output

@st.cache_data
def describe_clips(files, shotlist):
    content = []

    content += ["You are a video editor who is matching video clips with shots from a shotlist.\n"]
    content += ["""<example>
<clip>
    <id>1</id>
    <shot>1</shot>
    <description>FIREFIGHTERS FORMING A LINE AND CLOSING IN ON A RAGING BLAZE, RESIDENTS SHOUTING (English): "GET OUT OF THE WAY, IT'S SPREADING!"</description>
</clip>
<clip>
    <id>2</id>
    <shot>2</shot>
    <description>RESIDENTS HUDDLED TOGETHER BEHIND TEMPORARY SHELTERS</description>
</clip>
<clip>
    <id>3</id>
    <shot>2</shot>
    <description>RESIDENTS HUDDLED TOGETHER BEHIND TEMPORARY SHELTERS</description>
</clip>
</example>

Clips:"""]

    for file in files:
        name = file.stem
        frame_file, audio_file = extract_first_frame_and_audio(file)
        
        content += [f"{name}:", genai.upload_file(frame_file), genai.upload_file(audio_file)]
    
    content += ["\nShotlist:\n", shotlist]

    prompt = """Please match each clip with its shot in the shotlist. I've given you the first frame & audio from each clip. It should be in the same order,
but shots may be matched to multiple adjacent clips. Shots with quotes can only be matched to one clip. Include <id></id>, <shot></shot>, & <description>
for each clip. Only label descriptions with those in the shotlist exactly. <shot></shot> should just be the number. Output XML in <response></response> tags"""

    content += ["\n\n", prompt]

    convo = gemini.start_chat()
    convo.send_message(content)

    print(gemini.count_tokens(content))

    clips_xml = extract_xml(convo.last.text)
    return clips_xml

DIR = "/tmp/"

def save_uploaded(file_data):
    with open(os.path.join(DIR, file_data.name),"wb") as f:
        f.write(file_data.getbuffer())

import readtime
from annotated_text import annotated_text
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
from pathlib import Path
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pandas as pd

def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Channel 1 Demo")

    if "run" not in st.session_state:
        st.session_state["run"] = False
    if "ran" not in st.session_state:
        st.session_state["ran"] = False
    
    def reset():
        st.session_state["run"] = False
        st.session_state["ran"] = False

    # story_id = st.text_input("ID", value="tag:reuters.com,2024:newsml_RW956402052024RP1:6")

    # if st.button("Get assets"):
    #     video_asset, shotlist_asset = get_assets(story_id)
    #     video_url, asset_type = download_asset(story_id, video_asset["uri"])
    #     st.write(video_url)
    #     res = requests.get(video_url)
    #     st.write(len(res.content))
    #     video_file = Path(os.path.join(DIR, "video.mp4"))
    #     with open(video_file, "wb") as file:
    #         file.write(res.content)

    #     shotlist_url, asset_type = download_asset(story_id, shotlist_asset["uri"])
    #     st.write(shotlist_url)
    #     res = requests.get(shotlist_url)
    #     st.write(res.content)
    #     # shotlist = res.json()
        
    #     st.video(str(video_file))
    #     # st.write(shotlist)

    input_col_1, input_col_2, input_col_3 = st.columns(3)
    with input_col_1:
        shotlist = st.text_area(
            "Shotlist",
            height=300,
            on_change=reset,
        )
    with input_col_2:
        story = st.text_area(
            "Story",
            height=300,
            on_change=reset,
        )
    with input_col_3:
        video = st.file_uploader("Upload video", type=["mp4"], accept_multiple_files=False)
        if video:
            video_file = Path(os.path.join(DIR, video.name))
            save_uploaded(video)
            with st.expander("Uploaded video"):
                st.video(str(video_file))
    
    if st.button("Run"):
        st.session_state["run"] = True

    if st.session_state["run"]:
        with st.status("Running"):
            if not st.session_state["ran"]:
                st.write("Splitting video")
                scene_list = detect(str(video_file), AdaptiveDetector(adaptive_threshold=4, min_scene_len=1))
                video_folder = Path(DIR) / video_file.stem
                video_folder.mkdir(parents=True, exist_ok=True)
                split_video_ffmpeg(str(video_file), scene_list, output_file_template=f"{str(video_folder)}/$SCENE_NUMBER.mp4")

                st.write("Labelling clips")
                clips = list(sorted(video_folder.glob("*.mp4")))
                clips_xml = describe_clips(clips[1:], shotlist)
                clips = []
                for clip in clips_xml["response"]:
                    new_section = {}
                    for part in clip["clip"]:
                        for key, val in part.items():
                            new_section[key] = val.strip() if type(val) is str else val
                    clips.append(new_section)
                print(clips)
                st.session_state["clips"] = clips

                st.write("Extracting SOT")
                sots_raw = get_sot_chain.invoke({"SHOTLIST": shotlist}).content
                sots_xml = extract_xml(sots_raw)
                sots = sots_xml['response']
                st.session_state["sots"] = sots
                print(sots)

                st.write("Reformatting story")
                reformated_story_raw = reformat_chain.invoke({"STORY": story}).content
                reformated_story_xml = extract_xml(reformated_story_raw)
                reformated_story = reformated_story_xml['response']
                st.session_state["reformated_story"] = reformated_story

                st.write("Adding SOT to story")
                if "NO SOT" in sots:
                    sot_script = reformated_story
                else:
                    sot_script_raw = sot_chain.invoke({"QUOTATIONS": sots, "SCRIPT": reformated_story}).content
                    sot_script_xml = extract_xml(sot_script_raw)
                    sot_script = sot_script_xml['response']
                st.session_state["sot_script"] = sot_script
                
                st.write("Parsing story")
                parsed_script_raw = parse_chain.invoke({"QUOTATIONS": sots, "SCRIPT": sot_script}).content
                parsed_script_xml = extract_xml(parsed_script_raw)
                parsed_script_json = JsonOutputParser().invoke(parsed_script_xml['response'])
                st.session_state["parsed_script_json"] = parsed_script_json

                st.write("Writing summaries")
                print(str(parsed_script_json))
                section_summary_raw = section_summary_chain.invoke({"SCRIPT", str(parsed_script_json)}).content
                section_summary_xml = extract_xml(section_summary_raw)
                section_summary_json = JsonOutputParser().invoke(section_summary_xml['response'])
                st.session_state["section_summary_json"] = section_summary_json
            else:
                video_folder = Path(DIR) / video_file.stem
                clips = st.session_state["clips"]
                sots = st.session_state["sots"]
                reformated_story = st.session_state["reformated_story"]
                sot_script = st.session_state["sot_script"]
                parsed_script_json = st.session_state["parsed_script_json"]
                section_summary_json = st.session_state["section_summary_json"]
        
        st.session_state["ran"] = True

        for s1, s2 in zip(parsed_script_json["sections"], section_summary_json["sections"]):
            if s1["id"] != s2["id"]:
                print("ERROR: IDs didn't match", s1, s2)
            s1["summary"] = s2["summary"]

        with st.expander("See details"):
            st.subheader("Labelled clips")
            st.write(clips)
            st.divider()

            st.subheader("SOTs")
            st.write(sots)
            st.divider()

            st.subheader("Reformatted story")
            st.write(reformated_story)
            st.divider()

            st.subheader("SOT story")
            st.write(sot_script)

            st.subheader("Parsed story")
            st.write(parsed_script_json)

            st.subheader("Section summaries")
            st.write(section_summary_json)

        trt = readtime.of_text(sot_script).seconds
        st.write(f"Estimated TRT: {trt}s")

        output_col_1, output_col_2 = st.columns([1, 2])
        with output_col_1:
            st.subheader("Original Story")
            st.write(story)
        with output_col_2:
            st.subheader("Final Story")
            data = {
                'type': [],
                'summary': [],
                'shot_id': [],
                'text': [],
            }
            for section in parsed_script_json["sections"]:
                data['type'].append(section['type'])
                data['summary'].append(section['summary'])
                data['shot_id'].append(section['shot_id'] if section['type'] == 'SOT' else None)
                data['text'].append(section['text'])
            df = pd.DataFrame(data)

            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_column("type", width=40, rowDrag=True, rowDragManaged=True, rowDragEntireRow = True, editable=True)
            gb.configure_column("summary", width=100, wrapText=True, autoHeight=True, editable=True)
            gb.configure_column("text", wrapText=True, autoHeight=True, editable=True)
            gb.configure_column("shot_id", width=40, editable=True)

            gb.configure_grid_options(
                rowDragManaged=True,
                animateRows=True,
                rowHeight=60,
                domLayout='autoHeight'
            )
            df = AgGrid(df, gridOptions=gb.build(), allow_unsafe_jscode=True, update_mode=GridUpdateMode.MANUAL, fit_columns_on_grid_load=True, theme="alpine", height=1000)["data"]
            # for section in parsed_script_json["sections"]:
            #     with st.container(border=True):
            #         if section["type"] == "SOT":
            #             annotated_text(("SOT", "", "#8ef"))
            #         elif section["type"] == "ANCHOR":
            #             annotated_text(("ANCHOR", "", "#faa"))
            #         st.write(section["text"])
        
        if st.button("Generate audio"):
            audio_clips = []
            for i, section in df.iterrows():
                try:
                    if section["type"] == "SOT":
                        options = [clip for clip in clips if str(clip["shot"]) == str(int(section["shot_id"]))]
                        if options:
                            clip = options[0]
                            audio_clips.append(mp.AudioFileClip(str(video_folder / f"{clip['id']}.mp4")))
                        else:
                            st.write(f"Clip not found for shot {section['shot_id']}, skipping")
                    elif section["type"] == "ANCHOR":
                        filename = str(video_folder / f"{i}.mp3")
                        TTS(section["text"], filename)
                        audio_clips.append(mp.AudioFileClip(filename))
                except Exception as e:
                    st.write(f"Error with section {i}, {e}")
            
            final_audio = mp.concatenate_audioclips(audio_clips)

            output_audio_file = video_folder / "final_audio.mp3"
            final_audio.write_audiofile(output_audio_file)

            for audio in audio_clips:
                audio.close()
            
            st.audio(str(output_audio_file), format="audio/mpeg")


if __name__ == "__main__":
    run()