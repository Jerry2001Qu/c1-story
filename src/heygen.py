# STREAMLIT
from src.gcp import upload_to_gcs_url, clear_uploaded_blobs
from src.constants import HEYGEN_API_KEY

import streamlit as st
# /STREAMLIT

import requests
import time
from pathlib import Path, PosixPath

@st.cache_data(show_spinner=False)
def get_heygen_avatars():
    url = "https://api.heygen.com/v2/avatars"
    headers = {
        "X-Api-Key": HEYGEN_API_KEY,
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data["data"]["avatars"]
    else:
        raise Exception(f"Error fetching avatars: {response.status_code} - {response.text}")

@st.cache_data(show_spinner=False, hash_funcs={PosixPath: lambda x: str(x.resolve())})
def generate_heygen_video(local_audio_file_path: Path, transcript: str, avatar_id: str, output_path: Path, avatar_style: str = 'normal', test: bool = True):
    # Upload the local audio file to GCP and get the URL
    audio_url = upload_to_gcs_url(local_audio_file_path, "public-heygen-assets")
    
    # Define the request payload
    payload = {
        "video_inputs": [
            {
                "character": {
                    "type": "avatar",
                    "avatar_id": avatar_id,
                    "avatar_style": avatar_style
                },
                "voice": {
                    "type": "audio",
                    "audio_url": audio_url
                },
                "background": {
                    "type": "color",
                    "value": "#FFFFFF"
                }
            }
        ],
        "test": test,
        "aspect_ratio": "16:9"
    }
    
    # Define the headers
    headers = {
        "X-Api-Key": HEYGEN_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Send the request to generate the video
    response = requests.post("https://api.heygen.com/v2/video/generate", json=payload, headers=headers)
    response_data = response.json()
    
    if response.status_code != 200 or response_data.get("error"):
        raise Exception(f"Error generating video: {response_data.get('error')}")
    
    video_id = response_data["data"]["video_id"]
    
    # Poll the status of the video until it is complete
    video_status_url = f"https://api.heygen.com/v1/video_status.get?video_id={video_id}"
    while True:
        status_response = requests.get(video_status_url, headers=headers)
        status_data = status_response.json()["data"]
        
        if status_data["status"] == "completed":
            clear_uploaded_blobs()
            video_url = status_data["video_url"]
            # Download the video
            video_response = requests.get(video_url)
            with open(output_path, 'wb') as video_file:
                video_file.write(video_response.content)
            return
        elif status_data["status"] == "failed":
            clear_uploaded_blobs()
            raise Exception(f"Video generation failed: {status_data.get('error')}")
        elif status_data["status"] in ["processing", "pending"]:
            time.sleep(10)  # Wait before polling again