import requests
import streamlit as st

from src.constants import REUTERS_CLIENT_ID, REUTERS_CLIENT_SECRET

@st.cache_data(show_spinner=False, ttl=86400)
def get_oauth_token():
    url = "https://auth.thomsonreuters.com/oauth/token"
    headers = {"Content-Type": "application/json"}
    payload = {
        "client_id": REUTERS_CLIENT_ID,
        "client_secret": REUTERS_CLIENT_SECRET,
        "grant_type": "client_credentials",
        "audience": "7a14b6a2-73b8-4ab2-a610-80fb9f40f769",
        "scope": "https://api.thomsonreuters.com/auth/reutersconnect.contentapi.read https://api.thomsonreuters.com/auth/reutersconnect.contentapi.write"
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["access_token"]

def graphql_query(query, variables, token=get_oauth_token()):
    url = "https://api.reutersconnect.com/content/graphql"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json={"query": query, "variables": variables})
    response.raise_for_status()
    return response.json()

@st.cache_data(show_spinner=False)
def get_item(item_id, token=get_oauth_token()):
    query = """
    query GetItem($itemId: ID!) {
        item(id: $itemId) {
            associations {
                bodyXhtml
                headLine
                language
                located
                type
            }
        }
    }
    """
    variables = {
        "itemId": item_id,
    }
    data = graphql_query(query, variables, token=get_oauth_token())

    associations = data["data"]["item"]["associations"]
    text_association = [association for association in associations if association["type"] == "text"][0]
    return text_association["bodyXhtml"], text_association["headLine"], text_association["language"], text_association["located"]

@st.cache_data(show_spinner=False, ttl=60*5)
def download_asset(item_id, rendition_id, token=get_oauth_token()):
    query = """
    mutation DownloadAsset($itemId: ID!, $renditionId: ID!) {
        download(itemId: $itemId, renditionId: $renditionId) {
            ... on GenericItem {
                url
                type
            }
        }
    }
    """
    variables = {
        "itemId": item_id,
        "renditionId": rendition_id
    }
    data = graphql_query(query, variables, token=get_oauth_token())
    return data["data"]["download"]["url"], data["data"]["download"]["type"]

@st.cache_data(show_spinner=False)
def get_assets(item_id, token=get_oauth_token(), desired_codes=["stream:8256:16x9:mp4"]):
    query = """
    query GetAssets($itemId: ID!) {
        item(id: $itemId) {
            associations {
                renditions {
                    code
                    mimeType
                    type
                    uri
                    version
                    ... on VideoRendition {
                        mimeType
                        fileName
                        audioBitRate
                        audioSampleRate
                        colourIndicator
                        duration
                        code
                        format
                        height
                        sizeInBytes
                        type
                        uri
                        version
                        videoAspectRatio
                        videoAvgBitrate
                        width
                    }
                }
            }
        }
    }
    """
    variables = {"itemId": item_id}
    data = graphql_query(query, variables, token)
    associations = data["data"]["item"]["associations"]

    filtered_renditions = [
        rendition for association in associations for rendition in association["renditions"]
        if rendition["code"] in desired_codes
    ]
    return filtered_renditions