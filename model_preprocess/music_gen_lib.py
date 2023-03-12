# store the function/object used in the project

# import modules
from __future__ import print_function
import numpy as np
import librosa
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from keras import regularizers
import time
from tensorflow.keras.layers import Layer
import pprint

# parameters
sr = 22050 # if sampling rate is different, resample it to this

# parameters for calculating spectrogram in mel scale
fmax = 10000 # maximum frequency considered
fft_window_points = 512
fft_window_dur = fft_window_points * 1.0 / sr # 23ms windows
hop_size = int(fft_window_points/ 2) # 50% overlap between consecutive frames
n_mels = 64

# segment duration
num_fft_windows = 512 # num fft windows per music segment
segment_in_points = num_fft_windows * 255 # number of data points that insure the spectrogram has size: 64 * 256
segment_dur = segment_in_points * 1.0 / sr

num_genres=10
input_shape=(64, 512, 1)

def baseline_model_32(num_genres=num_genres, input_shape=input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(decay=1e-5),
                  metrics=['accuracy'])
    return(model)

def baseline_model_64(num_genres=num_genres, input_shape=input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(decay=1e-5),
                  metrics=['accuracy'])
    return(model)

def baseline_model_96(num_genres=num_genres, input_shape=input_shape):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(3, 3),
                     activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(decay=1e-5),
                  metrics=['accuracy'])
    return(model)

def load_ice_data():
    import os
    data_folder = "/Users/Ice/OneDrive/Desktop/GradProj/relatedWork/music_genre_classification/genres"
    # genre_folders = [x[0] for x in os.walk(data_folder)]
    genre_folders = os.listdir(data_folder)
    X = []
    T = []
    SR = []
    min_length = 0
    genre_path = data_folder
    audio_files = os.listdir(genre_path)
    for sub_folder in genre_folders:
        genre_path = data_folder + "/" + sub_folder
        print(genre_path)
        audio_files = os.listdir(genre_path)
        for audio_name in audio_files:
            audio_path = genre_path + "/" + audio_name
            x, sr = librosa.core.load(audio_path)
            if x.shape[0] < 30 * sr:
                x = np.append(x, np.zeros(30*sr - x.shape[0])) # insure all files are exactly the same length
                if min_length < x.shape[0]:
                    min_length = x.shape[0] # report the duration of the minimum audio clip
                    print("This audio last %f seconds, zeros are padded at the end." % (x.shape[0]*1.0/sr))
            X.append(x[:30*sr])
            SR.append(sr)
            T.append(sub_folder)
    return np.asarray(X), np.asarray(SR), np.asarray(T, dtype=str)
    
class Music_Genre_CNN(object):

    def __init__(self, ann_model):
        self.model = ann_model()

    def load_model(self, model_path, custom_objects=None):
        self.model = load_model(model_path, custom_objects=custom_objects)

    def new_song_spectrogram_prediction_mid(self, song_mel_spectrogram):
        predict_array = []
        
        minIndex = 192
        maxIndex = 704
        
        while( maxIndex < (song_mel_spectrogram.shape[1] - 320)):        
            segment = []
            segment.append(song_mel_spectrogram[:, minIndex : maxIndex])
            segment_array = np.asarray(segment)[:, :, :, np.newaxis]
            
            # print(segment_array.shape,len(segment_array))
            predictions = self.model.predict(segment_array, batch_size=len(segment_array))
            summarized_prediction = np.argmax(predictions.sum(axis=0))
            predict_array.append(summarized_prediction)
            
            minIndex += 64
            maxIndex += 64
            
        return(predict_array)