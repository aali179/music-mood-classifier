# Generating dataset using Spotify's API

import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
cid = '5410e774c83040bb85936efc4672828a'
secret = 'e893db8c750a439489d6a8c11560e470'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
token = 'BQB6ZOzsxi3YQVa7LJrB1w9LFdJC06vHT2Et_syuCoHFnBR9XvZjcxsA-lnkItppbxQB5B94AjBtckbkumwSV99RXukc3-4hlJh_y9Nb0JkhE1PxzL38JsbmhLRqVXS1EiY3tcDsgwukNiHPkwB5OM76CtHZDJC_DQAhEV4NCdZRYvot8cE'
spotify = spotipy.Spotify(token)

def get_happy_playlist_tracks():
    results = spotify.user_playlist_tracks("afrah", "1VxPsBO0n1dho7hI71U8ne")
    for i in range (0,65):
        dataset.append({"mood": "happy", "trackId": results['items'][i]['track']['id']})    
    
def get_sad_playlist_tracks():
    results = spotify.user_playlist_tracks("afrah", "6IVsBRyqA6GbzzSsLkXyHH")
    for i in range (0,65):
        dataset.append({"mood": "sad", "trackId": results['items'][i]['track']['id']})    
    
def get_stressful_playlist_tracks():
    results = spotify.user_playlist_tracks("afrah", "6SjlyZeHD8yV1vUPqzO3AW")
    for i in range (0,65):
        dataset.append({"mood": "stressful", "trackId": results['items'][i]['track']['id']})    
    
def get_calm_playlist_tracks():
    results = spotify.user_playlist_tracks("afrah", "0Rk9OM7ZnWLKfutDHJzk2O")
    for i in range (0,65):
        dataset.append({"mood": "calm", "trackId": results['items'][i]['track']['id']})    

def get_track_name(trackId):
    return spotify.track(trackId)['name']

def get_acoustic_feature(trackId, featureName):
    return spotify.audio_features(trackId)[0][featureName]

def get_track_artist(trackId):
    return spotify.track(trackId)['album']['artists'][0]['name']

# labelled dataset
dataset = []

get_happy_playlist_tracks()
get_sad_playlist_tracks()
get_stressful_playlist_tracks()
get_calm_playlist_tracks()

# return data as [mood, track_name, artist, track_id, acousticness, danceability, energy, instrumentalness,
# key, liveness, loudness, mode, speechiness, tempo, time_signature, valence]
data = [[]]
for i in dataset:
    mood = i.get("mood")
    trackId = i.get("trackId")
    data.append([mood, get_track_name(trackId), get_track_artist(trackId), trackId,
        get_acoustic_feature(trackId, 'acousticness'),
        get_acoustic_feature(trackId, 'danceability'),
        get_acoustic_feature(trackId, 'energy'),
        get_acoustic_feature(trackId, 'instrumentalness'),
        get_acoustic_feature(trackId, 'key'),
        get_acoustic_feature(trackId, 'liveness'),
        get_acoustic_feature(trackId, 'loudness'),
        get_acoustic_feature(trackId, 'mode'),
        get_acoustic_feature(trackId, 'speechiness'),
        get_acoustic_feature(trackId, 'tempo'),
        get_acoustic_feature(trackId, 'time_signature'),
        get_acoustic_feature(trackId, 'valence')           
       ])

# create pandas DataFrame
df = pd.DataFrame(data, columns = ['mood', 'track_name', 'artist','track_id', 'acousticness', 'danceability', 'energy',
                                   'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness',
                                   'tempo', 'time_signature', 'valence'])

# export to csv file
df.to_csv(r'C:\Users\Afrah\Desktop\music-mood-classifier\dataset.csv', index=False)
print(df)