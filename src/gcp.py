# GCP

# STREAMLIT
from src.constants import GOOGLE_JSON
import os
import json

google_json = json.loads(GOOGLE_JSON, strict=False)
temp_file_path = "/tmp/service_account.json"
with open(temp_file_path, "w") as f:
    json.dump(google_json, f)

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the temporary file path
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path
# /STREAMLIT

from google.cloud import storage
from google.cloud.storage import Blob
from vertexai.generative_models import Part

from typing import Union
from pathlib import Path
import random
import string

# Google Cloud Storage Setup
storage_client = storage.Client()
blobs = []

def upload_to_gcs_blob(local_file_path: Union[str, Path], bucket_name="gemini-colab") -> Blob:
    bucket = storage_client.bucket(bucket_name)
    local_file_path = Path(local_file_path)
    random_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
    extension = local_file_path.suffix
    blob_name = random_string + extension
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_file_path)

    global blobs
    blobs.append(blob)

    return blob

def upload_to_gcs_part(local_file_path: Union[str, Path], bucket_name="gemini-colab") -> Part:
    local_file_path = Path(local_file_path)
    extension = local_file_path.suffix
    blob = upload_to_gcs_blob(local_file_path, bucket_name)

    uri = f"gs://{bucket_name}/{blob.name}"
    mime_types = {
        ".mp4": "video/mp4",
        ".mp3": "audio/mpeg",
        ".jpg": "image/jpeg",
    }
    mime_type = mime_types.get(extension) 
    if not mime_type:
        raise ValueError(f"Unsupported file extension: {extension}")

    return Part.from_uri(uri, mime_type=mime_type)

def upload_to_gcs_url(local_file_path: Union[str, Path], bucket_name="gemini-colab") -> str:
    blob = upload_to_gcs_blob(local_file_path, bucket_name)
    return blob.public_url

def clear_uploaded_blobs():
    """Deletes blobs that were uploaded to Google Cloud Storage."""
    global blobs
    for blob in blobs:
        blob.delete()
    blobs.clear()