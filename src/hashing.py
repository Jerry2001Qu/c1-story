import hashlib

def sha256sum(file):
    with open(file, 'rb', buffering=0) as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()