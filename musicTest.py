import librosa
import numpy as np
import pytube
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

x, y, video_title, video_id = load_test_data('https://www.youtube.com/watch?v=J0yO4n9lUbE')
# x, y = old_load_test_data('Favorite-Person-_feat.-Marsharis_-_128-kbps_.au')

print(x)
print(y)
print(video_title)
print(video_id)