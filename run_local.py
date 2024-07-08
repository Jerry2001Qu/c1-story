from pathlib import Path
import os
import random

from src.dataloader import LocalDataLoader
from src.clip_manager import ClipManager
from src.news_script import NewsScript
from src.video_editor import VideoEditor
from src.error_handler import StdOutErrorHandler 
from src.audio_processor import AudioProcessor
from src.gcp import GCSManager

def main():
    anchor_idx = random.randint(0, 3)
    live_anchor = os.environ.get("LIVE_ANCHOR", "false").lower() == "true"
    test_mode = os.environ.get("TEST_MODE", "true") == "true"

    anchor_map = [
        ("yELTnbNFhESclGsoYVTM", "l6Qo5Atx1JTwyCLkMKQm", "6afc5b115c6f440aa92f43a32f50616f", "assets/EDDIE-square.png"),
        ("9f8o652aaiVK5HavyCf1", "l6Qo5Atx1JTwyCLkMKQm", "20251eb0e4504ddbb913f1b09e2bbb8e", "assets/DANIEL-square.png"),
        ("dyhDlCWGL3pDsrANTLru", "l6Qo5Atx1JTwyCLkMKQm", "395e53f63eba4ce9a9e3dddc9a0263ed", "assets/MIRANDA-square.png"),
        # ("H2LXjnBS1droRODepT50", "l6Qo5Atx1JTwyCLkMKQm", "e1dfdd60549940159aa4eb529e6f78a7", "assets/SARAH-square.png"),
        ("GxOlMeAhhAqPmZNfxxUm", "l6Qo5Atx1JTwyCLkMKQm", "2c67e653633c4894834585e3a9d5b2be", "assets/KARA-square.png"),
    ]
    anchor_voice_id, voiceover_voice_id, anchor_avatar_id, anchor_image_path = anchor_map[anchor_idx]

    music = False
    high_res = True
    output_resolution = (1920, 1080) if high_res else (640, 360)
    bitrate = "10M" if high_res else "1M"

    error_handler = StdOutErrorHandler()

    story_folder = Path("assets/Biden Defiant Amid Democratic Pressure, Vows to Stay in Race as NATO Summit Looms")
    dataloader = LocalDataLoader(story_folder)
    storyline = dataloader.load_storyline()
    shotlist = dataloader.load_shotlist()
    story_title = dataloader.get_story_title()
    video_file_path = dataloader.get_video_file_path()

    clips_folder = story_folder / "clips"
    clip_manager = ClipManager(video_file_path, clips_folder, shotlist, anchor_image_path, anchor_voice_id, voiceover_voice_id, anchor_avatar_id, has_splash_screen=False, error_handler=error_handler)
    script = NewsScript(storyline, shotlist, clip_manager, dataloader, folder=story_folder, error_handler=error_handler)
    clip_manager.split_video_into_clips()
    clip_manager.load_clips()
    clip_manager.transcribe_clips(multi=True)
    clip_manager.match_clips()
    clip_manager.break_up_clips()
    clip_manager.generate_full_descriptions(story_title)
    
    script.spell_check()
    script.generate_facts()
    script.generate_script()
    script.generate_lower_thirds()
    script.match_sot_clips()

    combined_script = script.with_combined_script()

    audio_processor = AudioProcessor(combined_script, clip_manager, story_folder, error_handler)
    audio_processor._process_anchor_audio()
    audio_processor._generate_sot_translations()
    audio_processor._add_broll_placements()

    video_output_file = story_folder / "output.mp4"
    video_editor = VideoEditor(combined_script, clip_manager, live_anchor, test_mode, music, Path("./assets/music-1.mp3"), output_resolution=output_resolution, bitrate=bitrate, logo_path=Path("./assets/lower_thirds_logo.png"), font=Path("./assets/Khand-SemiBold.ttf"), error_handler=error_handler)
    video_editor.assemble_video(output_file=video_output_file)

    gcs = GCSManager()
    gcs.upload_to_gcs_url(video_output_file, filename=script.headline, bucket_name="c1-videos")

if __name__ == "__main__":
    main()
