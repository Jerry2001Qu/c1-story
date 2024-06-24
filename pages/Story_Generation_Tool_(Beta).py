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
from src.authentication import check_password

from src.prompts import cache

def run():
    if not check_password():
        st.stop()
        
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

    anchor_imgs = ["assets/EDDIE-square.png", "assets/DANIEL-square.png", "assets/MIRANDA-square.png", "assets/UNKNOWN-square.png", "assets/UNKNOWN-square.png"]
    anchor_names = ["Eddie", "Daniel", "Miranda", "Sarah", "Kara"]
    
    anchor_idx = image_select("Choose an Anchor", anchor_imgs, anchor_names, return_value="index", use_container_width=False)

    anchor_map = [
        ("yELTnbNFhESclGsoYVTM", "l6Qo5Atx1JTwyCLkMKQm", "6afc5b115c6f440aa92f43a32f50616f"),
        ("9f8o652aaiVK5HavyCf1", "l6Qo5Atx1JTwyCLkMKQm", "20251eb0e4504ddbb913f1b09e2bbb8e"),
        ("dyhDlCWGL3pDsrANTLru", "l6Qo5Atx1JTwyCLkMKQm", "395e53f63eba4ce9a9e3dddc9a0263ed"),
        ("H2LXjnBS1droRODepT50", "l6Qo5Atx1JTwyCLkMKQm", "e1dfdd60549940159aa4eb529e6f78a7"),
        ("GxOlMeAhhAqPmZNfxxUm", "l6Qo5Atx1JTwyCLkMKQm", "2c67e653633c4894834585e3a9d5b2be"),
    ]
    anchor_voice_id, voiceover_voice_id, anchor_avatar_id = anchor_map[anchor_idx]

    if "verbosity" not in st.session_state:
        st.session_state["verbosity"] = False
    if "live_anchor" not in st.session_state:
        st.session_state["live_anchor"] = False
    if "high_res_anchor" not in st.session_state:
        st.session_state["high_res_anchor"] = False
    if "music" not in st.session_state:
        st.session_state["music"] = True
    if "high_res" not in st.session_state:
        st.session_state["high_res"] = False
    
    def on_change(var_name):
        def _toggle():
            st.session_state[var_name] = not st.session_state[var_name]
        return _toggle

    verbosity = st.toggle("Display Generation Data", value=st.session_state["verbosity"], on_change=on_change("verbosity"))
    live_anchor = st.toggle("Motion Anchor", value=st.session_state["live_anchor"], on_change=on_change("live_anchor"))
    high_res_anchor = st.toggle("High Res Anchor", value=st.session_state["high_res_anchor"], on_change=on_change("high_res_anchor"))
    music = st.toggle("Add Music", value=st.session_state["music"], on_change=on_change("music"))
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
    
    if "reuters_id" not in st.session_state:
        st.session_state["reuters_id"] = "tag:reuters.com,2024:newsml_RW635429052024RP1:5"
    
    def reset_and_update():
        st.session_state["run"] = False
        st.session_state["ran"] = False
        st.session_state["download_run"] = False
        st.session_state["video_run"] = False
        st.session_state["video_ran"] = False
        error_handler.reset()

        st.session_state["reuters_id"] = st.session_state["reuters_id_input"]
    
    error_handler = StreamlitErrorHandler(error_bar, verbosity)

    reuters_id = st.text_input("Story Asset ID (Paste Here)", value=st.session_state["reuters_id"], on_change=reset_and_update, key="reuters_id_input",)
    clean_reuters_id = "".join(filter(lambda x: x.isalnum() or x.isspace(), reuters_id))
    if st.button("Download Story Assets"):
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
        st.write(storyline.replace("$", "\$"))
        st.write(shotlist.replace("$", "\$"))
    
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
            df = df[["type", "shot_id", "name", "text"]]
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_column("type", width=40, rowDrag=True, rowDragManaged=True, rowDragEntireRow = True, autoHeight=True, editable=True)
            gb.configure_column("shot_id", width=30, autoHeight=True, editable=True)
            gb.configure_column("name", width=50, wrapText=True, autoHeight=True, editable=True)
            gb.configure_column("text", wrapText=True, autoHeight=True, editable=True)

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
        
        def reset_video_and_toggle():
            st.session_state["video_run"] = False
            st.session_state["video_ran"] = False

            st.session_state["high_res"] = not st.session_state["high_res"]
        
        high_res = st.toggle("High Resolution", value=st.session_state["high_res"], on_change=reset_video_and_toggle)
        output_resolution = (1920, 1080) if high_res else (640, 360)
        bitrate = "10M" if high_res else "1M"

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
                    video_editor = VideoEditor(script, clip_manager, live_anchor, test_mode, music, Path("./assets/music-1.mp3"), output_resolution=output_resolution, bitrate=bitrate, logo_path=Path("./assets/lower_thirds_logo.png"), font=Path("./assets/Khand-SemiBold.ttf"), error_handler=error_handler)
                    video_editor.assemble_video(output_file=video_output_file)
                else:
                    video_output_file = story_folder / "output.mp4"
            
            st.session_state["video_ran"] = True
            
            with st.expander("Details"):
                st.write(script)

            st.video(str(video_output_file), autoplay=True)


if __name__ == "__main__":
    run()
