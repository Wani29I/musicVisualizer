import librosa
import numpy as np
import pytube
import urllib
import re
import os
# import youtube_dl

def load_test_data(yt_link):
    X = []
    SR = []
    yt = pytube.YouTube(yt_link)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_file = audio_stream.download()
    x, sr = librosa.load(audio_file, sr=None)
    X.append(x[:])
    SR.append(sr)
    
    # Create a YouTube object from the link
    youtube = pytube.YouTube(yt_link)
    # Extract video ID, title, and length
    video_id = youtube.video_id
    video_title = youtube.title
    
    return np.asarray(X), np.asarray(SR), video_title, video_id

def old_load_test_data(song_file):
    
    X = []
    SR = []
    x, sr = librosa.load(song_file, sr=None)
    X.append(x[:])
    SR.append(sr)
    return np.asarray(X), np.asarray(SR)

def check_load_test_data(yt_link):
    X = []
    SR = []
    try:
        yt = pytube.YouTube(yt_link)
    except:
        # Assume yt_link is not a valid YouTube URL
        # Try to search YouTube instead and get the first video
        query = yt_link
        html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + query)
        video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
        if( not video_ids ):
            raise "error"
        
        yt_link = "https://www.youtube.com/watch?v=" + video_ids[0]
        yt = pytube.YouTube(yt_link)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_file = audio_stream.download()
    x, sr = librosa.load(audio_file, sr=None)
    X.append(x[:])
    SR.append(sr)
    os.remove(audio_file)
    
    return np.asarray(X), np.asarray(SR), yt_link

# x, y, video_title, video_id = load_test_data('https://www.youtube.com/watch?v=J0yO4n9lUbE')
# x, y = old_load_test_data('Favorite-Person-_feat.-Marsharis_-_128-kbps_.au')
x,y, ytlink = check_load_test_data('rickrool')

print(x)
print(y)
print(ytlink)
# print(video_title)
# print(video_id)