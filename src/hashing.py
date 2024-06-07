import hashlib
from pathlib import Path

def sha256sum(file: Path):
    if file.exists():
        with open(file, 'rb', buffering=0) as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    else:
        return "A"