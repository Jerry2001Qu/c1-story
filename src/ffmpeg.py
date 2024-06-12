import os
from pathlib import Path
import tarfile

# Function to download and extract FFmpeg
def download_ffmpeg():
    ffmpeg_tar = "./assets/ffmpeg-release-amd64-static.tar.xz"
    ffmpeg_dir = "/tmp/ffmpeg"
    ffmpeg_bin = os.path.join(ffmpeg_dir, "ffmpeg-7.0.1-amd64-static", "ffmpeg")

    if Path(ffmpeg_bin).exists():
        return

    # Extract the tar file
    with tarfile.open(ffmpeg_tar, "r:xz") as tar:
        tar.extractall(path=ffmpeg_dir)

    # Set the FFmpeg binary path
    os.chmod(ffmpeg_bin, 0o755)

    # Set the FFMPEG_BINARY environment variable
    os.environ['FFMPEG_BINARY'] = ffmpeg_bin