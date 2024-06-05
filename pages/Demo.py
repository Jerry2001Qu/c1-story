import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from pathlib import Path
import shutil

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from src.dataloader import ReutersAPIDataLoader
from src.clip_manager import ClipManager
from src.news_script import NewsScript
from src.video_editor import VideoEditor
from src.error_handler import StreamlitErrorHandler

from src.prompts import cache

def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    with st.sidebar:
        if st.button("Clear caches"):
            st.cache_data.clear()
            cache.clear()
            for folder in Path("/tmp").glob("tagreuters*"):
                shutil.rmtree(folder)
        
        error_bar = st.container()

    st.write("# Channel 1 Demo")
    
    anchor_name = st.selectbox("Anchor", ("Eddie", "Daniel"))
    anchor_map = {
        "Eddie": ("yELTnbNFhESclGsoYVTM", "LHgN09QqKzsRsniiMpww", "4c10e2d75ddd4169819b61c0289af17d"),
        "Daniel": ("yELTnbNFhESclGsoYVTM", "LHgN09QqKzsRsniiMpww", "4c10e2d75ddd4169819b61c0289af17d"),
    }
    anchor_voice_id, voiceover_voice_id, anchor_avatar_id = anchor_map[anchor_name]

    verbosity = st.toggle("Show output", value=True)
    live_anchor = st.toggle("Live anchor", value=True)
    test_mode = st.toggle("Test mode", value=True)

    st.divider()

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
        error_handler.reset()

    error_handler = StreamlitErrorHandler(error_bar, verbosity)

    reuters_id = st.text_input("Reuters ID", value="tag:reuters.com,2024:newsml_RW635429052024RP1:5", on_change=reset)
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

    if st.session_state["run"]:
        print(reuters_id)
        with st.status("Running"):
            if not st.session_state["ran"]:
                clips_folder = story_folder / "clips"
                clip_manager = ClipManager(video_file_path, clips_folder, shotlist, Path("./assets/anchor-default.png"), anchor_voice_id, voiceover_voice_id, anchor_avatar_id, has_splash_screen=False, error_handler=error_handler)
                script = NewsScript(storyline, shotlist, clip_manager, folder=story_folder, error_handler=error_handler)
                st.write("Splitting video")
                error_handler.info("Splitting video")
                clip_manager.split_video_into_clips()
                st.write("Loading clips")
                error_handler.info("Loading clips")
                clip_manager.load_clips()
                st.write("Transcribing clips")
                error_handler.info("Transcribing clips")
                clip_manager.transcribe_clips()
                st.write("Matching clips to shotlist")
                error_handler.info("Matching clips to shotlist")
                clip_manager.match_clips()
                st.write("Generating full descriptions")
                error_handler.info("Generating full descriptions")
                clip_manager.generate_full_descriptions(story_title)

                st.write("Extracting facts")
                error_handler.info("Extracting facts")
                script.generate_facts()
                st.write("Generating script")
                error_handler.info("Generating script")
                script.generate_script()
                st.write("Generating lower thirds")
                error_handler.info("Generating lower thirds")
                script.generate_lower_thirds()
                
                st.write("Matching sots to clips")
                error_handler.info("Matching sots to clips")
                script.match_sot_clips()

                st.session_state["clip_manager"] = clip_manager
                st.session_state["script"] = script
            else:
                clip_manager = st.session_state["clip_manager"]
                script = st.session_state["script"]

                clip_manager.error_handler = error_handler
                script.error_handler = error_handler
        
        st.session_state["ran"] = True

        with st.expander("Details"):
            st.write(clip_manager.clips)

        trt = script.get_total_read_time_seconds()
        st.write(f"Estimated TRT: {trt}s")

        st.subheader("Extracted facts")
        st.write(script.facts_list)

        st.subheader("Final Story")

        with st.container(height=800):
            df = script.to_dataframe()
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_column("type", width=60, rowDrag=True, rowDragManaged=True, rowDragEntireRow = True, autoHeight=True, editable=True)
            gb.configure_column("text", wrapText=True, autoHeight=True, editable=True)
            gb.configure_column("shot_id", width=50, autoHeight=True, editable=True)

            gb.configure_grid_options(
                rowDragManaged=True,
                animateRows=True,
                scrollbar=True,
            )
            df = AgGrid(df, gridOptions=gb.build(), height=2000, update_mode=GridUpdateMode.MANUAL, fit_columns_on_grid_load=True, theme="alpine")["data"]
            script.from_dataframe(df)
        if st.button("Generate video"):
            with st.status("Running"):
                st.write("Generating audio & broll")
                error_handler.info("Generating audio & broll")
                script.generate_audio_and_broll()

                st.write("Assembling video")
                error_handler.info("Assembling video")
                video_output_file = story_folder / "output.mp4"
                video_editor = VideoEditor(script, clip_manager, live_anchor=live_anchor, test_mode=test_mode, logo_path=Path("./assets/lower_thirds_logo.png"), font=Path("./assets/Khand-SemiBold.ttf"), error_handler=error_handler)
                video_editor.assemble_video(output_file=video_output_file)

            st.video(str(video_output_file), autoplay=True)


if __name__ == "__main__":
    run()