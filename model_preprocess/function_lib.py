from __future__ import print_function
import numpy as np
import librosa
import model_preprocess.music_gen_lib as mgl
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pytube
import os
import re
import urllib


MGCNN = mgl.Music_Genre_CNN(mgl.baseline_model_96)
MGCNN.load_model("classify_model.h5")

# # au file
# def load_test_data(song_file):
    
#     X = []
#     SR = []
#     x, sr = librosa.load(song_file, sr=None)
#     X.append(x[:])
#     SR.append(sr)
#     return np.asarray(X), np.asarray(SR)

# # YouTube link
def load_test_data(yt_link):
    X = []
    SR = []
    try:
        yt = pytube.YouTube(yt_link)
    except:
        # Assume yt_link is not a valid YouTube URL
        # Try to search YouTube instead and get the first video
        query = urllib.parse.quote(yt_link)
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
    
    youtube = pytube.YouTube(yt_link)
    # Extract video ID, title, and length
    video_title = youtube.title
    
    return np.asarray(X), np.asarray(SR), yt_link, video_title

def predictTestColor(testX, testSR, colorDict):

    newTestX = mgl.batch_mel_spectrogram(testX, testSR)
    newTestX = np.log(newTestX+1)
    newTestX = newTestX[:, :, :, np.newaxis]
    
    predict_test1 = MGCNN.new_song_spectrogram_prediction_mid(newTestX[0])
    
    predict_test1 = [int(predict) for predict in predict_test1]
    cmapColorList1 = [colorDict[predict] for predict in predict_test1]
        
    return cmapColorList1, predict_test1

def create_color_bar(colors):
    color_hex = []
    for color in colors:
        color_hex.append('#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255)))
    return color_hex
    
if __name__ == '__main__':
    predictTestColor()