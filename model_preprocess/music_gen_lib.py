# store the function/object used in the project

# import modules
from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
        
    def train_model_shift_frame(self, round, input_spectrograms, labels, cv=False,
                    validation_spectrograms=None, validation_labels=None,
                    small_batch_size=75, max_iteration=350, print_interval=1):
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
        for iii in range(max_iteration):

            st_time = time.time()

            # split training data into even batches
            num_training_data = len(input_spectrograms)
            batch_idx = np.random.permutation(num_training_data)
            num_batches = int(num_training_data / small_batch_size)
            
            # print(num_batches)
            # print(len(batch_idx))
            print( round, "round(s) left",  "[", "=" *(iii//5), " " * (70 - (iii//5)) , "]", iii, "/", max_iteration )
            # print( "round:", 10-round, "====================================", iii, "/", max_iteration, "====================================")
            
            
            for jjj in range(num_batches - 1):
                
                minIndex = 0
                maxIndex = 512
            
                while( maxIndex < 2560 ): 
                    # print('input_spectrograms.shape',input_spectrograms[0][1])
                    # sample_idx = np.random.randint(input_spectrograms.shape[2] - num_fft_windows)
                    training_idx = batch_idx[jjj * small_batch_size: (jjj + 1) * small_batch_size]
                    training_data = input_spectrograms[training_idx, :, minIndex:maxIndex, :]
                    training_label = labels[training_idx]
                    # pprint.pprint(training_data.shape)
                    # pprint.pprint('*'*100)
                    # pprint.pprint(training_data[0][0])
                    # pprint.pprint('*'*100)
                    # pprint.pprint(training_data[0][0].shape)
                    # pprint.pprint(training_data)
                    # pprint.pprint(training_data[1])
                    # pprint.pprint('*'*100)
                    # pprint.pprint(training_data[2])
                    # pprint.pprint('*'*100)
                    # pprint.pprint(training_data[3])
                    # pprint.pprint('*'*100)
                    # pprint.pprint(training_label)
                    self.model.train_on_batch(training_data, training_label)
                    training_accuracy = self.model.evaluate(training_data, training_label)
                    minIndex += 64
                    maxIndex += 64
                    # print("Training accuracy is: %f" % (training_accuracy))

                end_time = time.time()
                elapsed_time = end_time - st_time
                
            if cv:
                validation_accuracy = self.model.evaluate(validation_spectrograms[:, :, minIndex:maxIndex, :], validation_labels)
                validation_accuracy_list.append(validation_accuracy[1])
            else:
                validation_accuracy = [-1.0, -1.0]

            if iii % print_interval == 0:
                print("\nTime elapsed: %f; Training accuracy: %f, Validation accuracy: %f\n" %
                    (elapsed_time, training_accuracy[1], validation_accuracy[1]))
        if cv:
            return np.asarray(validation_accuracy_list)

    def train_model(self, input_spectrograms, labels, cv,
                    validation_spectrograms, validation_labels,
                    small_batch_size=150, max_iteration=500, print_interval=1):
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
        for iii in range(max_iteration):

            st_time = time.time()

            # split training data into even batches
            num_training_data = len(input_spectrograms)
            batch_idx = np.random.permutation(num_training_data)
            num_batches = int(num_training_data / small_batch_size)
            
            for jjj in range(num_batches - 1):
                print(input_spectrograms.shape[2], num_fft_windows, input_spectrograms.shape[2] - num_fft_windows)
                sample_idx = np.random.randint(input_spectrograms.shape[2] - num_fft_windows)
                training_idx = batch_idx[jjj * small_batch_size: (jjj + 1) * small_batch_size]
                training_data = input_spectrograms[training_idx, :, sample_idx:sample_idx+num_fft_windows, :]
                training_label = labels[training_idx]
                self.model.train_on_batch(training_data, training_label)
                training_accuracy = self.model.evaluate(training_data, training_label)
                # print("Training accuracy is: %f" % (training_accuracy))

            end_time = time.time()
            elapsed_time = end_time - st_time
            if cv:
                validation_accuracy = self.model.evaluate(validation_spectrograms[:, :, sample_idx:sample_idx+num_fft_windows, :], validation_labels)
                validation_accuracy_list.append(validation_accuracy[1])
            else:
                validation_accuracy = [-1.0, -1.0]

            if iii % print_interval == 0:
                print("\nTime elapsed: %f; Training accuracy: %f, Validation accuracy: %f\n" %
                      (elapsed_time, training_accuracy[1], validation_accuracy[1]))
        if cv:
            return np.asarray(validation_accuracy_list)


    def song_spectrogram_prediction(self, song_mel_spectrogram, overlap):
        """
        give the predicted_probability for each class and each segment
        :param song_mel_spectrogram:
            4D numpy array: num of time windows * mel bands * 1 (depth)
        :param overlap:
            overlap between segments, overlap = 0 means no overlap between segments
        :return:
            predictions: numpy array (number of segments * num classes)
        """
        # 1st segment spectrogram into sizes of 64 * 256
        largest_idx = song_mel_spectrogram.shape[1] - num_fft_windows - 1
        step_size = int((1 - overlap) * num_fft_windows)
        num_segments = int(largest_idx / step_size)
        segment_edges = np.arange(num_segments) * step_size
        segment_list = []
        for idx in segment_edges:
            segment = song_mel_spectrogram[:, idx : idx + num_fft_windows]
            segment_list.append(segment)
        segment_array = np.asarray(segment_list)[:, :, :, np.newaxis]
        predictions = self.model.predict(segment_array, batch_size=len(segment_array))
        summarized_prediction = np.argmax(predictions.sum(axis=0))
        return(summarized_prediction, predictions)
    
    def new_song_spectrogram_prediction(self, song_mel_spectrogram):
        predict_array = []
        
        minIndex = 0
        maxIndex = num_fft_windows
        
        while( maxIndex < (song_mel_spectrogram.shape[1] - num_fft_windows) ):        
            segment = []
            segment.append(song_mel_spectrogram[:, minIndex : maxIndex])
            segment_array = np.asarray(segment)[:, :, :, np.newaxis]
            
            predictions = self.model.predict(segment_array, batch_size=len(segment_array))
            summarized_prediction = np.argmax(predictions.sum(axis=0))
            predict_array.append(summarized_prediction)
            
            minIndex += num_fft_windows
            maxIndex += num_fft_windows
            
        return(predict_array)
    
    def new_song_spectrogram_prediction_2(self, song_mel_spectrogram):
        predict_array = []
        
        minIndex = 0
        maxIndex = 512
        
        while( maxIndex < (song_mel_spectrogram.shape[1] - 512) ):        
            segment = []
            segment.append(song_mel_spectrogram[:, minIndex : maxIndex])
            segment_array = np.asarray(segment)[:, :, :, np.newaxis]
            
            predictions = self.model.predict(segment_array, batch_size=len(segment_array))
            summarized_prediction = np.argmax(predictions.sum(axis=0))
            predict_array.append(summarized_prediction)
            
            minIndex += 64
            maxIndex += 64
            
        return(predict_array)
    
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
