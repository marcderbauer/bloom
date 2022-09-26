# -*- coding: utf-8 -*-

# Sample Python code for youtube.playlistItems.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/code-samples#python

import os

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import argparse

#----------------------------------------------------------------------------
#                               ARGPARSE
#----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate inference for a given prompt')
parser.add_argument("playlist_ids", type=str, nargs="?", default="UUn8zNIfYAQNdrFRrr8oibKw PLw613M86o5o7q1cjb26MfCgdxJtshvRZ-", help="The ids of the playlists to download")
parser.add_argument("--verbose", default=False, action='store_true')
args = parser.parse_args()

scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
RESULTS_PER_PAGE = 50


#----------------------------------------------------------------------------
#                               MAIN
#----------------------------------------------------------------------------
def main():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "0"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "client_secret_900137197244-2trfkrsu9h0a57oi51dcgkg30mbetlha.apps.googleusercontent.com.json"

    # Get credentials and create an API client
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file, scopes)
    credentials = flow.run_console()
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, credentials=credentials)

    all_titles = []

    for playlist_id in args.playlist_ids.split():
        channel_titles = []
        nextPageToken = None
        channel= None

        while True:
            request = youtube.playlistItems().list(
                part="snippet",
                maxResults=50,
                pageToken = nextPageToken,
                playlistId=playlist_id
            )
            response = request.execute()
            try:
                nextPageToken = response['nextPageToken']
            except:
                nextPageToken = None

            if not channel:
                channel = response['items'][0]['snippet']['channelTitle']
            for item in response['items']:
                title = item['snippet']['title']
                channel_titles.append(title)
                if args.verbose:
                    print(title)

            if not nextPageToken:
                all_titles.extend(channel_titles)
                break

        if not channel:
            channel = "titles"
            
        with open(f"data/{channel}.txt", "w") as f:
                for title in channel_titles:
                    f.write(f"{title}\n")
        print(f"\n{'-'*100}\nFinished downloading {channel}. Total items: {len(channel_titles)}\n")
    
    # Save all titles if multiple playlists were given
    if len(args.playlist_ids.split()) > 1:
        with open(f"data/combined.txt", "w") as f:
                for title in all_titles:
                    f.write(f"{title}\n")

if __name__ == "__main__":
    main()

