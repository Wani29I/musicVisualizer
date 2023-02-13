from __future__ import print_function
import numpy as np
import librosa
import model_preprocess.music_gen_lib as mgl
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

MGCNN = mgl.Music_Genre_CNN(mgl.baseline_model_96)
MGCNN.load_model("classify_model.h5")

def load_test_data(song_file):
    
    X = []
    SR = []
    x, sr = librosa.load(song_file, sr=None)
    X.append(x[:])
    SR.append(sr)
    return np.asarray(X), np.asarray(SR)

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