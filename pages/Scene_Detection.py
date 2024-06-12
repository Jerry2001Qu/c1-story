import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg, ContentDetector
from pathlib import Path
import moviepy.editor as mp

def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Channel 1 Demo")

    def save_uploaded(file_data):
        with open("/tmp/video.mp4" ,"wb") as f:
            f.write(file_data.getbuffer())

    video = st.file_uploader("Video", type=["mp4"], accept_multiple_files=False)
    if video:
        save_uploaded(video)
        with st.expander("Uploaded video"):
            st.video("/tmp/video.mp4")
        
        clip = mp.VideoFileClip("/tmp/video.mp4")
        fps = clip.fps
        clip.close()
        
        detector_name = st.selectbox("Detection method", ["Adaptive", "Content"])

        if detector_name == "Adaptive":
            threshold = st.slider("Threshold", 0, 10, 4)
            detector = AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=fps)
        elif detector_name == "Content":
            threshold = st.slider("Threshold", 0, 100, 27)
            detector = ContentDetector(threshold=threshold, min_scene_len=fps)

        if st.button("Run"):
            clips_folder = Path("/tmp/clips")
            video_file_path = Path("/tmp/video.mp4")
            clips_folder.mkdir(parents=True, exist_ok=True)
            scene_list = detect(str(video_file_path), detector)
            status = split_video_ffmpeg(str(video_file_path), scene_list, show_progress=False,
                            output_file_template=str(clips_folder / "$SCENE_NUMBER.mp4"))
            
            for video_path in sorted(clips_folder.glob("*.mp4")):
                st.video(str(video_path))

if __name__ == "__main__":
    run()