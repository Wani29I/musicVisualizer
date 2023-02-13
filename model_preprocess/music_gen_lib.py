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
from keras.utils import np_utils
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


def split_data(T, split_idxes):
    """
    give the indexes of training, validation, and testing data
    :param T: label of all data
    :param split_idxes: splitting points of the data
    :return:
    """
    genres = np.unique(T)
    training_idxes = []
    validation_idxes = []
    testing_idxes = []
    for idx, music_genre in enumerate(genres):
        tmp_logidx = music_genre == T
        tmp_idx = np.flatnonzero(tmp_logidx)
        tmp_shuffled_idx = np.random.permutation(tmp_idx)
        tmp_num_examles = len(tmp_shuffled_idx)
        tmp_split_idxes = np.asarray(split_idxes * tmp_num_examles, dtype=np.int)
        training_idxes.append(tmp_shuffled_idx[tmp_split_idxes[0] : tmp_split_idxes[1]])
        validation_idxes.append(tmp_shuffled_idx[tmp_split_idxes[1] : tmp_split_idxes[2]])
        testing_idxes.append(tmp_shuffled_idx[tmp_split_idxes[2] : tmp_split_idxes[3]])
    return(np.concatenate(training_idxes), np.concatenate(validation_idxes), np.concatenate(testing_idxes))

def load_test_data():
    import os
    data_folder = "/Users/Ice/OneDrive/Desktop/GradProj/relatedWork/music_genre_classification/test_music"
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
            X.append(x[:180*sr])
            SR.append(sr)
            T.append(sub_folder)
    return np.asarray(X), np.asarray(SR), np.asarray(T, dtype=str)

def load_accuracy_test_data():
    """
    load original audio files
    :return:
    """
    import os
    data_folder = "/Users/Ice/OneDrive/Desktop/GradProj/relatedWork/music_genre_classification/musicForAccuracyTest"
    # genre_folders = [x[0] for x in os.walk(data_folder)]
    genre_folders = os.listdir(data_folder)
    X = []
    T = []
    SR = []
    min_length = 0
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

def load_original_data():
    """
    load original audio files
    :return:
    """
    import os
    data_folder = "/Users/Ice/OneDrive/Desktop/GradProj/relatedWork/music_genre_classification/filteredGenre"
    # genre_folders = [x[0] for x in os.walk(data_folder)]
    genre_folders = os.listdir(data_folder)
    X = []
    T = []
    SR = []
    min_length = 0
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

# calculate mel-spectrogram
def mel_spectrogram(ys, sr, n_mels=n_mels, hop_size=hop_size,win_length=fft_window_dur, fmax=fmax, pre_emphasis=False):
    """
    calculate the spectrogram in mel scale, refer to documentation of libriso and MFCC tutorial
    :param ys:
    :param sr:
    :param n_mels:
    :param hop_size:
    :param fmax:
    :param pre_emphasis:
    :return:
    """
    if pre_emphasis:
        ys = np.append(ys[0], ys[1:]-pre_emphasis*ys[:-1])
            
    result =  librosa.feature.melspectrogram(ys, sr,
                                            n_fft=fft_window_points,
                                            hop_length=hop_size, n_mels=n_mels,
                                            fmax=fmax)

    return result

# batch convert waveform into spectrogram in mel-scale
def batch_mel_spectrogram(X, SR):
    """
    convert all waveforms in R into time * 64 spectrogram in mel scale
    :param X:
    :param SR:
    :return:
    """
    melspec_list = []
    for idx in range(X.shape[0]):
        tmp_melspec = mel_spectrogram(X[idx], SR[idx])
        melspec_list.append(tmp_melspec)
    return np.asarray(melspec_list)


# def segment_spectrogram(input_spectrogram, num_fft_windows=num_fft_windows):
#     # given a spectrogram of a music that's longer than 3 seconds, segment it into relatively independent pieces
#     length_in_fft = input_spectrogram.shape[1]
#     num_segments = int(length_in_fft / num_fft_windows)
#     pass


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