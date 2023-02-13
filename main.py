from flask import Flask, request
from model_preprocess.function_lib import load_test_data, predictTestColor, create_color_bar
from flask import render_template
from flask import Flask
from pymongo import MongoClient

# import flask
# import keras
# import librosa
# import numpy
# import pymongo
# import sklearn
# import tensorflow

# print("flask",flask.__version__)
# print("keras",keras.__version__)
# print("librosa",librosa.__version__)
# print("numpy",numpy.__version__)
# print("pymongo",pymongo.__version__)
# print("sklearn",sklearn.__version__)
# print("tensorflow",tensorflow.__version__)

client = MongoClient("mongodb+srv://Wani29:Ice.31458@musicvector.xr7hdip.mongodb.net/?retryWrites=true&w=majority")

app = Flask(__name__, template_folder='templates')
musicvectorDb = client.MusicVector.musicvector

mock = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1, 1, 1, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 5, 5, 2, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 5, 1, 1, 5, 5, 5, 5, 2, 5, 5, 5, 1, 5, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 2, 2, 5, 5, 2, 2, 2, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 5, 5, 5, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 5, 5, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 5, 5, 5, 5, 2, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

def store_music_vector(song_genre_array, filename):
    
    # check if already stored
    prevSong = musicvectorDb.find_one({"song_genre_array": song_genre_array})
    if prevSong : return
    
    genreCount = [0,0,0,0,0,0,0,0,0,0]
    
    unit = 1/len(song_genre_array)
    
    for genre in song_genre_array:
        genreCount[genre] += unit
    print(genreCount)
    # print(filename)
    
    musicvectorDb.insert_one({
        "file_name": filename,
        "song_genre_array": song_genre_array,
        "genre_percentage": genreCount
        })
    
    return

@app.route('/create')
def create():
    
    # musicvectorDb.insert_one({0:0})
    return {}

@app.route('/')
def index():
    return render_template('mainTemplate.html')

@app.route('/', methods=['POST'])
def main_progress():
    
    if 'file' not in request.files:
        return 'No file part in the request'
    file = request.files['file']
    if file.filename == '':
        return 'No file selected'

    # Preprocess the file to extract features
    x, sr = load_test_data(file)
    
    rawColor = dict(request.form)
    
    colorDict = {
        0 : rawColor['color0'],
        1 : rawColor['color1'],
        2 : rawColor['color2'],
        3 : rawColor['color3'],
        4 : rawColor['color4'],
        5 : rawColor['color5'],
        6 : rawColor['color6'],
        7 : rawColor['color7'],
        8 : rawColor['color8'],
        9 : rawColor['color9'],
    }
    # print(colorDict)
    
    color_list, genre_list = predictTestColor(x, sr, colorDict)
        
    store_music_vector(list(genre_list), file.filename)
    
    # print(len(color_list))
    # print(color_list)
    
    
    colors = color_list
    
    defaultRGB = list(colorDict.values())

    # Example: return the color bar as a string
    return render_template('mainTemplate.html', colors=colors, filename=file.filename, newDefaultRGB=defaultRGB)
    

@app.route('/test', methods=['GET'])
def classify_file2():
    
    defaultRGB = [
        "rgba(36, 108, 183, 1)",
        "rgba(239, 234, 214, 1)",
        "rgba(132, 202, 235, 1)",
        "rgba(255, 0, 102, 1)",
        "rgba(66, 66, 66, 1)",
        "rgba(255, 159, 120, 1)",
        "rgba(0, 57, 178, 1)",
        "rgba(255, 106, 180, 1)",
        "rgba(102, 204, 0, 1)",
        "rgba(133, 194, 149, 1)",
      ];
    
    colors = ['#000000']

    # Example: return the color bar as a string
    
    return render_template('mainTemplate.html', colors=colors, filename='filename', newDefaultRGB=defaultRGB)

if __name__ == '__main__':
    app.run(debug=True)