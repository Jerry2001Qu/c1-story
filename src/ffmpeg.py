import subprocess
import os
import requests
import tarfile

# Function to download and extract FFmpeg
def download_ffmpeg():
    url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    ffmpeg_tar = "/tmp/ffmpeg-release-amd64-static.tar.xz"
    ffmpeg_dir = "/tmp/ffmpeg"

    # Download the FFmpeg tar file
    response = requests.get(url, stream=True)
    with open(ffmpeg_tar, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    # Extract the tar file
    with tarfile.open(ffmpeg_tar, "r:xz") as tar:
        tar.extractall(path=ffmpeg_dir)

    # Set the FFmpeg binary path
    ffmpeg_bin = os.path.join(ffmpeg_dir, "ffmpeg-7.0.1-amd64-static", "ffmpeg")
    os.chmod(ffmpeg_bin, 0o755)

    # Move FFmpeg binary to /usr/bin/ffmpeg
    subprocess.run(['sudo', 'mv', ffmpeg_bin, '/usr/bin/ffmpeg'])

    # Set the FFMPEG_BINARY environment variable
    os.environ['FFMPEG_BINARY'] = '/usr/bin/ffmpeg'