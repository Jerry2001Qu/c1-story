import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from pathlib import Path

from src.dataloader import ReutersAPIDataLoader
from src.clip_manager import ClipManager
from src.news_script import NewsScript
from src.video_editor import VideoEditor

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
    if "download_run" not in st.session_state:
        st.session_state["download_run"] = False
    
    def reset():
        st.session_state["run"] = False
        st.session_state["ran"] = False
        st.session_state["download_run"] = False

    reuters_id = st.text_input("Reuters ID", value="tag:reuters.com,2024:newsml_RW956402052024RP1:6", on_change=reset)
    clean_reuters_id = "".join(filter(lambda x: x.isalnum() or x.isspace(), reuters_id))
    if st.button("Download from Reuters"):
        st.session_state["download_run"] = True
    
    if st.session_state["download_run"]:
        story_folder = Path("/tmp") / clean_reuters_id
        dataloader = ReutersAPIDataLoader(reuters_id, story_folder)
        storyline = dataloader.load_storyline()
        shotlist = dataloader.load_shotlist()
        story_title = dataloader.get_story_title()
        video_file_path = dataloader.get_video_file_path()

        st.title(story_title)
        st.video(str(video_file_path))
        st.write(storyline)
        st.write(shotlist)
    
        if st.button("Generate script"):
            st.session_state["run"] = True
    
    x = story_folder
    st.write(f"{x.exists()}, {x}")

    if st.session_state["run"]:
        with st.status("Running"):
            if not st.session_state["ran"]:
                clips_folder = story_folder / "clips"
                clip_manager = ClipManager(video_file_path, clips_folder, shotlist)
                st.write("Splitting video")
                clip_manager.split_video_into_clips()
                st.write("Load & matching clips")
                clip_manager.load_and_match_clips()
                st.write("Transcribing clips")
                clip_manager.transcribe_clips()
                st.write("Generating full descriptions")
                clip_manager.generate_full_descriptions(story_title)
                
                script = NewsScript(storyline, shotlist, clip_manager, folder=story_folder)
                script.generate_script()
                script.generate_lower_thirds()

                st.session_state["clip_manager"] = clip_manager
                st.session_state["script"] = script
            else:
                clip_manager = st.session_state["clip_manager"]
                script = st.session_state["script"]
        
        st.session_state["ran"] = True

        trt = script.get_total_read_time_seconds()
        st.write(f"Estimated TRT: {trt}s")

        output_col_1, output_col_2 = st.columns([1, 2])
        with output_col_1:
            st.subheader("Original Story")
            st.write(storyline)
        with output_col_2:
            st.subheader("Final Story")
            st.write(script.__repr__())
            # data = {
            #     'type': [],
            #     'shot_id': [],
            #     'text': [],
            # }
            # for section in parsed_script_json["sections"]:
            #     data['type'].append(section['type'])
            #     data['shot_id'].append(section['shot_id'] if section['type'] == 'SOT' else None)
            #     data['text'].append(section['text'])
            # df = pd.DataFrame(data)

            # gb = GridOptionsBuilder.from_dataframe(df)
            # gb.configure_column("type", width=60, rowDrag=True, rowDragManaged=True, rowDragEntireRow = True, editable=True)
            # gb.configure_column("text", wrapText=True, autoHeight=True, editable=True)
            # gb.configure_column("shot_id", width=50, editable=True)

            # gb.configure_grid_options(
            #     rowDragManaged=True,
            #     animateRows=True,
            #     rowHeight=60,
            #     scrollbar=True,
            #     domLayout='autoHeight'
            # )
            # df = AgGrid(df, gridOptions=gb.build(), allow_unsafe_jscode=True, update_mode=GridUpdateMode.MANUAL, fit_columns_on_grid_load=True, theme="alpine")["data"]
            # for section in script.sections:
            #     with st.container(border=True):
            #         if section["type"] == "SOT":
            #             annotated_text(("SOT", "", "#8ef"))
            #         elif section["type"] == "ANCHOR":
            #             annotated_text(("ANCHOR", "", "#faa"))
            #         st.write(section["text"])
        
        if st.button("Generate video"):
            with st.status("Running"):
                st.write("Generating audio & broll")
                script.generate_audio_and_broll()

                st.write("Assembling video")
                video_output_file = story_folder / "output.mp4"
                video_editor = VideoEditor(script, clip_manager, logo_path=Path("./assets/lower_thirds_logo.png"), font=Path("./assets/Khand-Regular.ttf"))
                video_editor.assemble_video(output_file=video_output_file)

            st.video(str(video_output_file), autoplay=True)


if __name__ == "__main__":
    run()