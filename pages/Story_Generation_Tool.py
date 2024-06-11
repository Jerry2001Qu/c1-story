import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from pathlib import Path
import shutil

# Download and setup FFmpeg
from src.ffmpeg import download_ffmpeg

if "switched_ffmpeg" not in st.session_state:
    download_ffmpeg()
    st.session_state["switched_ffmpeg"] = True

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit_image_select import image_select

from src.dataloader import ReutersAPIDataLoader
from src.clip_manager import ClipManager
from src.news_script import NewsScript
from src.video_editor import VideoEditor
from src.error_handler import StreamlitErrorHandler
from src.audio_processor import AudioProcessor

from src.prompts import cache

from streamlit_profiler import Profiler

class DummyContextManager:
    def __enter__(self):
        # Do nothing when entering the context
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        # Do nothing when exiting the context
        pass

def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="./assets/favicon.png",
        layout="wide"
    )

    st.logo("./assets/logo.png")
    # st.image("./assets/story-gen-header.png", width=600)
    # st.title("Story Generation Tool (Beta)")

    with st.sidebar:
        if st.button("Clear caches"):
            st.cache_data.clear()
            cache.clear()
            for folder in Path("/tmp").glob("tagreuters*"):
                shutil.rmtree(folder)
        
        error_bar = st.container()

    anchor_imgs = ["assets/EDDIE-square.png", "assets/Kristen-square.png", "assets/DANIEL-square.png", "assets/MIRANDA-square.png"]
    anchor_names = ["Eddie", "Kristen", "Daniel", "Miranda"]
    
    anchor_idx = image_select("Choose an Anchor", anchor_imgs, anchor_names, return_value="index", use_container_width=False)

    anchor_map = [
        ("yELTnbNFhESclGsoYVTM", "LHgN09QqKzsRsniiMpww", "6afc5b115c6f440aa92f43a32f50616f"),
        ("yELTnbNFhESclGsoYVTM", "LHgN09QqKzsRsniiMpww", "6afc5b115c6f440aa92f43a32f50616f"),
        ("yELTnbNFhESclGsoYVTM", "LHgN09QqKzsRsniiMpww", "6afc5b115c6f440aa92f43a32f50616f"),
        ("yELTnbNFhESclGsoYVTM", "LHgN09QqKzsRsniiMpww", "6afc5b115c6f440aa92f43a32f50616f"),
    ]
    anchor_voice_id, voiceover_voice_id, anchor_avatar_id = anchor_map[anchor_idx]

    verbosity = st.toggle("Display Generation Data", value=True)
    live_anchor = st.toggle("Motion Anchor", value=True)
    high_res_anchor = st.toggle("High Res Anchor", value=True)
    music = st.toggle("Add Music", value=True)
    profile = st.toggle("Enable Profiler", value=False)
    test_mode = not high_res_anchor

    languages = ["English", "Spanish", "French", "German", "Polish", "Italian", "Portuguese", "Russian", "Arabic", "Dutch", "Swedish", "Norwegian", "Turkish", "Japanese", "Korean", "Filipino", "Tamil", "Indonesian", "Greek", "Chinese"]
    language = st.selectbox("Generate Story In:", languages)

    st.divider()

    if "run" not in st.session_state:
        st.session_state["run"] = False
    if "ran" not in st.session_state:
        st.session_state["ran"] = False
    if "download_run" not in st.session_state:
        st.session_state["download_run"] = False
    if "video_run" not in st.session_state:
        st.session_state["video_run"] = False
    if "video_ran" not in st.session_state:
        st.session_state["video_ran"] = False
    
    def reset():
        st.session_state["run"] = False
        st.session_state["ran"] = False
        st.session_state["download_run"] = False
        st.session_state["video_run"] = False
        st.session_state["video_ran"] = False
        error_handler.reset()
    
    error_handler = StreamlitErrorHandler(error_bar, verbosity)

    reuters_id = st.text_input("Story Asset ID (Paste Here)", value="tag:reuters.com,2024:newsml_RW635429052024RP1:5", on_change=reset)
    clean_reuters_id = "".join(filter(lambda x: x.isalnum() or x.isspace(), reuters_id))
    if st.button("Download Story Assets"):
        st.session_state["download_run"] = True
    
    with Profiler() if profile else DummyContextManager():
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
        
            if st.button("Generate Script"):
                st.session_state["run"] = True

        if st.session_state["run"]:
            print(reuters_id)
            with st.status("Running"):
                if not st.session_state["ran"]:
                    clips_folder = story_folder / "clips"
                    clip_manager = ClipManager(video_file_path, clips_folder, shotlist, Path("./assets/anchor-default.png"), anchor_voice_id, voiceover_voice_id, anchor_avatar_id, has_splash_screen=False, error_handler=error_handler)
                    script = NewsScript(storyline, shotlist, clip_manager, dataloader, folder=story_folder, error_handler=error_handler)
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

            st.subheader("Extracted Facts")
            st.write(script.facts_list)

            st.subheader("Final Story")

            with st.container(height=800):
                df = script.to_dataframe()
                df = df[["type", "text", "shot_id"]]
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

            for section in script.get_sot_sections():
                if not section.clip:
                    st.warning(f"SOT Section {section.id} could not be matched to a clip. Skipping section.")
                if section.match_type == "SPEECH":
                    st.warning(f"SOT Section {section.id}'s transcript did not contain the given quote. Adding all detected speech.")
                if section.match_type == "CLIP":
                    st.warning(f"SOT Section {section.id}'s had no detected speech. Adding entire clip.")
            

            if not st.session_state["video_run"]:
                if st.button("Generate Video"):
                    st.session_state["video_run"] = True
            else:
                if st.button("Regenerate Video"):
                    st.session_state["video_ran"] = False
            
            if st.session_state["video_run"]:
                with st.status("Running"):
                    if not st.session_state["video_ran"]:
                        audio_processor = AudioProcessor(script, clip_manager, story_folder, error_handler)
                        st.write("Generating anchor audio")
                        if error_handler:
                            error_handler.info("Generating anchor audio")
                        audio_processor._process_anchor_audio()
                        st.write("Generating SOT translations")
                        if error_handler:
                            error_handler.info("Generating SOT translations")
                        audio_processor._generate_sot_translations()
                        st.write("Adding broll placements")
                        if error_handler:
                            error_handler.info("Adding broll placements")
                        audio_processor._add_broll_placements()

                        st.write("Assembling video")
                        error_handler.info("Assembling video")
                        video_output_file = story_folder / "output.mp4"
                        video_editor = VideoEditor(script, clip_manager, live_anchor, test_mode, music, Path("./assets/music-1.mp3"), logo_path=Path("./assets/lower_thirds_logo.png"), font=Path("./assets/Khand-SemiBold.ttf"), error_handler=error_handler)
                        video_editor.assemble_video(output_file=video_output_file)
                    else:
                        video_output_file = story_folder / "output.mp4"
                
                st.session_state["video_ran"] = True
                
                with st.expander("Details"):
                    st.write(script.sections)

                st.video(str(video_output_file), autoplay=True)


if __name__ == "__main__":
    run()