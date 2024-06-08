import hashlib
from pathlib import Path, PosixPath
import moviepy.editor as mp

def sha256sum(file: PosixPath):
    if file.exists():
        with open(file, 'rb', buffering=0) as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    else:
        return str(file.resolve())

def hash_audio_file(path: PosixPath):
    return str(path.resolve()) + str(mp.AudioFileClip(str(path)).duration)

def hash_ignore(_):
    return 0

def hash_absolute_path(path: PosixPath):
    return str(path.resolve())