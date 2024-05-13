import requests
import streamlit as st

reuters_client_id = st.secrets["REUTERS_CLIENT_ID"]
reuters_client_secret = st.secrets["REUTERS_CLIENT_SECRET"]

@st.cache_data(show_spinner=False, ttl=86400)
def get_oauth_token():
    url = "https://auth.thomsonreuters.com/oauth/token"
    headers = {"Content-Type": "application/json"}
    payload = {
        "client_id": reuters_client_id,
        "client_secret": reuters_client_secret,
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

def get_assets(item_id, token=get_oauth_token(), desired_codes=["stream:shotlist:json", "stream:6756:16x9:mpg"]):
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