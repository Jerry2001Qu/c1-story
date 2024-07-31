from pathlib import Path
import os
import random

from src.dataloader import ReutersAPIDataLoader
from src.clip_manager import ClipManager
from src.news_script import NewsScript
from src.video_editor import VideoEditor
from src.error_handler import StdOutErrorHandler 
from src.audio_processor import AudioProcessor
from src.gcp import GCSManager

def main():
    anchor_idx = int(os.environ.get("ANCHOR_INDEX", random.randint(0, 2)))
    live_anchor = os.environ.get("LIVE_ANCHOR", "false").lower() == "true"
    test_mode = os.environ.get("TEST_MODE", "true") == "true"
    reuters_id = os.environ.get("REUTERS_ID", "tag:reuters.com,2024:newsml_RW327824062024RP1:6")
    add_logline = os.environ.get("ADD_LOGLINE", "false").lower() == "true"
    add_courtesy = os.environ.get("ADD_COURTESY", "false").lower() == "true"
    edit = os.environ.get("EDIT", "false").lower() == "true"

    anchor_map = [
        ("yELTnbNFhESclGsoYVTM", "l6Qo5Atx1JTwyCLkMKQm", "6afc5b115c6f440aa92f43a32f50616f", "assets/EDDIE-square.png"),
        ("9f8o652aaiVK5HavyCf1", "l6Qo5Atx1JTwyCLkMKQm", "20251eb0e4504ddbb913f1b09e2bbb8e", "assets/DANIEL-square.png"),
        ("dyhDlCWGL3pDsrANTLru", "l6Qo5Atx1JTwyCLkMKQm", "395e53f63eba4ce9a9e3dddc9a0263ed", "assets/MIRANDA-square.png"),
        # ("H2LXjnBS1droRODepT50", "l6Qo5Atx1JTwyCLkMKQm", "e1dfdd60549940159aa4eb529e6f78a7", "assets/SARAH-square.png"),
        # ("GxOlMeAhhAqPmZNfxxUm", "l6Qo5Atx1JTwyCLkMKQm", "2c67e653633c4894834585e3a9d5b2be", "assets/KARA-square.png"),
    ]
    anchor_voice_id, voiceover_voice_id, anchor_avatar_id, anchor_image_path = anchor_map[anchor_idx]

    music = False
    high_res = True
    output_resolution = (1920, 1080) if high_res else (640, 360)
    bitrate = "10M" if high_res else "1M"

    error_handler = StdOutErrorHandler()

    clean_reuters_id = "".join(filter(lambda x: x.isalnum() or x.isspace(), reuters_id))
    story_folder = Path("/tmp") / clean_reuters_id
    dataloader = ReutersAPIDataLoader(reuters_id, story_folder)
    storyline = dataloader.load_storyline()
    shotlist = dataloader.load_shotlist()
    story_title = dataloader.get_story_title()
    video_file_path = dataloader.get_video_file_path()
    body = dataloader.get_body()

    print(f"Story title: {story_title}")
    print(f"Storyline: {storyline}")
    print(f"Shotlist: {shotlist}")
    print(f"Video file path: {video_file_path}")
    print(f"Body: {body}")

    clips_folder = story_folder / "clips"
    clip_manager = ClipManager(video_file_path, clips_folder, shotlist, anchor_image_path, anchor_voice_id, voiceover_voice_id, anchor_avatar_id, has_splash_screen=False, error_handler=error_handler)
    script = NewsScript(storyline, shotlist, clip_manager, dataloader, folder=story_folder, error_handler=error_handler)
    print("Splitting video into clips")
    clip_manager.split_video_into_clips()
    print("Loading clips")
    clip_manager.load_clips()
    print("Transcribing clips")
    clip_manager.transcribe_clips(multi=True)
    print("Matching clips")
    clip_manager.match_clips()
    print("Breaking up clips")
    clip_manager.break_up_clips()
    print("Applying courtesy to clips")
    clip_manager.courtesy_clips(body)
    print("Generating full descriptions")
    clip_manager.generate_full_descriptions(story_title)
    
    print("Spell checking")
    script.spell_check()
    print("Generating facts")
    script.generate_facts()
    print("Generating script")
    script.generate_script(edit=edit)
    print("Generating lower thirds")
    script.generate_lower_thirds()
    print("Matching SOT clips")
    script.match_sot_clips()

    audio_processor = AudioProcessor(script, clip_manager, story_folder, error_handler)
    print("Processing anchor audio")
    audio_processor._process_anchor_audio()
    print("Generating SOT translations")
    audio_processor._generate_sot_translations()
    print("Adding B-roll placements")
    audio_processor._add_broll_placements()
    print("Validating graphic placements")
    audio_processor._validate_and_adjust_graphics_placements()
    print("Generating anchor videos")
    audio_processor._generate_anchor(live_anchor, test_mode)

    video_output_file = story_folder / "output.mp4"
    video_editor = VideoEditor(script, clip_manager, live_anchor, test_mode, music, Path("./assets/music-1.mp3"), output_resolution=output_resolution, bitrate=bitrate, logo_path=Path("./assets/lower_thirds_logo.png"), font=Path("./assets/Khand-SemiBold.ttf"), add_logline=add_logline, add_courtesy=add_courtesy, error_handler=error_handler)
    print("Assembling video")
    video_editor.assemble_video(output_file=video_output_file)

    gcs = GCSManager()
    print("Uploading video to GCS")
    gcs.upload_to_gcs_url(video_output_file, filename=script.headline, bucket_name="c1-videos")

if __name__ == "__main__":
    main()
