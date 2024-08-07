import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from src.video_editor import VideoEditor
from src.authentication import check_password

from pathlib import Path
import moviepy.editor as mp

def run():
    if not check_password():
        st.stop()
        
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Channel 1 Demo")

    clip = mp.VideoFileClip("./assets/sample_video.mp4")
    final_clip_file = "/tmp/tmp.mp4"

    logline = st.text_input("Logline", value="N. Korea's Balloon Attack Sparks War Fears")

    if st.button("Run"):
        video_editor = VideoEditor(None, None, None, None, True, Path("./assets/music-1.mp3"), font=Path("./assets/Khand-SemiBold.ttf"))
        clip = video_editor._add_logline(clip, logline)
        clip = video_editor._add_background_music(clip)
        final_clip = clip
        final_clip.write_videofile(final_clip_file, logger=None)

        st.video(final_clip_file)

if __name__ == "__main__":
    run()