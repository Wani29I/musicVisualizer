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
MGCNN.load_model("mgcnn_rs_15_copy.h5")

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

def predictTestColor(testX, testSR, colorDict, song_duration):

    newTestX = mgl.batch_mel_spectrogram(testX, testSR)
    newTestX = np.log(newTestX+1)
    newTestX = newTestX[:, :, :, np.newaxis]
    # print('song data after mel: ',newTestX.shape[1])
    
    window_size = int(newTestX[0].shape[1]/song_duration)
    # print(window_size)
    
    predict_test1 = MGCNN.new_song_spectrogram_prediction_mid(newTestX[0], window_size)
    
    predict_test1 = [int(predict) for predict in predict_test1]
    cmapColorList1 = [colorDict[predict] for predict in predict_test1]
    
        
    return cmapColorList1, predict_test1, window_size, newTestX

def train_model_shift_frame(self, round, input_spectrograms, labels, cv=False,
                    validation_spectrograms=None, validation_labels=None,
                    small_batch_size=75):
        """
        train the CNN model
        :param input_spectrograms: number of training examplex * num of mel bands * number of fft windows * 1
            type: 4D numpy array
        :param labels: vectorized class labels
            type:
        :param cv: whether do cross validation
        :param validation_spectrograms: data used for cross validation
            type: as input_spectrogram
        :param validation_labels: used for cross validation
        :param small_batch_size: size of each training batch
        :param max_iteration:
            maximum number of iterations allowed for one training
        :return:
            trained model
        """
        validation_accuracy_list = []

        # split training data into even batches
        num_training_data = len(input_spectrograms)
        batch_idx = np.random.permutation(num_training_data)
        num_batches = int(num_training_data / small_batch_size)
        
        for jjj in range(num_batches - 1):
            
            minIndex = 0
            maxIndex = 512
        
            while( maxIndex < 2560 ): 
                training_idx = batch_idx[jjj * small_batch_size: (jjj + 1) * small_batch_size]
                training_data = input_spectrograms[training_idx, :, minIndex:maxIndex, :]
                training_label = labels[training_idx]
                self.model.train_on_batch(training_data, training_label)
                training_accuracy = self.model.evaluate(training_data, training_label)
                minIndex += 64
                maxIndex += 64

        if cv:
            validation_accuracy = self.model.evaluate(validation_spectrograms[:, :, minIndex:maxIndex, :], validation_labels)
            validation_accuracy_list.append(validation_accuracy[1])
        else:
            validation_accuracy = [-1.0, -1.0]
            
        if cv:
            return np.asarray(validation_accuracy_list)


songInp = 'https://www.youtube.com/watch?v=BxuY9FET9Y4'
x, sr, ytLink, ytTitle = load_test_data(songInp)
color_list, genre_list, window_size, newTestX = predictTestColor(x, sr, colorDict, song_duration)

# training the model
training_flag = True
max_iterations = 5
while training_flag and max_iterations >= 0:
    validation_accuracies = MGCNN.train_model_shift_frame(max_iterations, training_X, training_T, cv=True,
                                                validation_spectrograms=validation_X,
                                                validation_labels=validation_T)

    diff = np.mean(validation_accuracies[-10:]) - np.mean(validation_accuracies[:10])
    MGCNN.backup_model()  # backup in case error occurred
    if np.abs(diff) < 0.001:
        training_flag = False
    max_iterations -= 1

MGCNN.backup_model("mgcnn_rs_15_copy.h5")