# Constants
# GCP VERSION

import os
from google.cloud import secretmanager

def get_secret(secret_id, version_id="latest"):
    """
    Access the payload for the given secret version if one exists.
    """
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version.
    name = f"projects/stg-transcription/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version.
    response = client.access_secret_version(request={"name": name})

    # Return the decoded payload.
    return response.payload.data.decode("UTF-8")

ELEVENLABS_API_KEY = get_secret("ELEVENLABS_API_KEY")
LANGCHAIN_API_KEY = get_secret("LANGCHAIN_API_KEY")
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
HEYGEN_API_KEY = get_secret("HEYGEN_API_KEY")
DEEPGRAM_API_KEY = get_secret("DEEPGRAM_API_KEY")

GOOGLE_JSON = get_secret("GOOGLE_JSON")

REUTERS_CLIENT_ID = get_secret("REUTERS_CLIENT_ID")
REUTERS_CLIENT_SECRET = get_secret("REUTERS_CLIENT_SECRET")

password = get_secret("password")
