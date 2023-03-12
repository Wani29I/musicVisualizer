from flask import Flask, request
from model_preprocess.function_lib import load_test_data, predictTestColor, create_color_bar
from flask import render_template
from flask import Flask
from pymongo import MongoClient
import pprint
from datetime import datetime, timedelta
from bson import ObjectId

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
reviewSongDb = client.MusicVector.reviewSong

mock = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1, 1, 1, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 5, 5, 2, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 5, 1, 1, 5, 5, 5, 5, 2, 5, 5, 5, 1, 5, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 2, 2, 5, 5, 2, 2, 2, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 5, 5, 5, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 5, 5, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 5, 5, 5, 5, 2, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

mock2 = [0,0.7954545454545483,0.03743315508021388,0,0,0.15240641711229977,0,0,0,0.014705882352941176]


def check_similarity(genreCount):
    similar_song_dict = {}
    
    x = musicvectorDb.aggregate([
    {
    '$project': {
      'song_genre_array': '$song_genre_array',
      'file_name': '$file_name',
      'youtube_link': '$youtube_link',
      'divergent_array': {
        '$map': {
          'input': {
            '$zip': {
              'inputs': [ "$genre_percentage", genreCount ]
            }
          },
          'as': "zipped",
          'in': {
            '$abs': {'$subtract': [ { '$arrayElemAt': [ "$$zipped", 0 ] }, { '$arrayElemAt': [ "$$zipped", 1 ] } ]}
          }
        },
      }
    }},
    {'$project':{
        'file_name': '$file_name',
        'youtube_link': '$youtube_link',
        'song_genre_array': '$song_genre_array',
        'divergent': { '$sum': '$divergent_array'},
        'similarity_percentage':{
           '$round': [{'$divide': [{'$subtract': [2, { '$sum': '$divergent_array'} ] }, 2 ]}, 4]
        },
      }
    },
    {
        '$sort':{'similarity_percentage': -1}
    },
    {
        '$limit': 11
    }
    ])
    
    y = list(x)
    
    # for i in y:
    #     print(i['similarity_percentage'])
    #     print(i['divergent'])
    
    return y[1:]
    
def store_music_vector(song_genre_array, link, filename):
    
    genreCount = [0,0,0,0,0,0,0,0,0,0]
    
    unit = 1/len(song_genre_array)
    
    for genre in song_genre_array:
        genreCount[genre] += unit
        
    # print(genreCount)
    # print(filename)
    
    songID = musicvectorDb.insert_one({
        "youtube_link": link,
        "file_name": filename,
        "song_genre_array": song_genre_array,
        "genre_percentage": genreCount
        })
        
    return genreCount, str(songID.inserted_id)

def checkURL(ytLink, colorDict):
    prevSong = musicvectorDb.find_one({"youtube_link": ytLink})
    if( prevSong ): 
        return prevSong['_id'], prevSong['song_genre_array'], prevSong['genre_percentage'], [colorDict[predict] for predict in prevSong['song_genre_array']]
    
    return None

@app.route('/create')
def create():
    x = check_similarity(mock2)
    # pprint.pprint(x)
    # print(len(x))
    # musicvectorDb.insert_one({0:0})
    return {}

@app.route('/')
def index():
    return render_template('mainTemplate.html')
  
@app.route('/review')
def reviewResult():
    
    args = dict(request.args)
    review = args['isLike']=='true' 
    songID = ObjectId(args['songID'])
    
    reviewSongDb.insert_one({
        "review": review,
        "timestamp": datetime.now(),
        "songID" : songID
    })
 
    return {'deatail':'review success'}

@app.route('/getReview')
def getReviewResult():
    
    days = request.args.get('days')
    queryDislike = {'review': False}
    queryLike = {'review': True}
 
    if(days):
        days = datetime.utcnow()-timedelta(days=days)
        queryLike['timestamp'] = {'$gte':days}
        queryDislike['timestamp'] = {'$gte':days}
    
    like = list(reviewSongDb.find(queryLike))
    dislike = list(reviewSongDb.find(queryDislike))
 
    return {'like':len(like), 'dislike':len(dislike)}

@app.route('/', methods=['POST'])
def main_progress():
    
    formData = dict(request.form)
    # Preprocess the file to extract features
    
    songInp = formData['songInp']
    
    x, sr, ytLink, ytTitle = load_test_data(songInp)
    
    colorDict = {
        0 : formData['color0'],
        1 : formData['color1'],
        2 : formData['color2'],
        3 : formData['color3'],
        4 : formData['color4'],
        5 : formData['color5'],
        6 : formData['color6'],
        7 : formData['color7'],
        8 : formData['color8'],
        9 : formData['color9'],
    }
    # print(colorDict)
    
    songData = checkURL(ytLink, colorDict)
    
    if(songData):
        songID, genre_list, genreCount, color_list = songData
    else:
        color_list, genre_list = predictTestColor(x, sr, colorDict)
        genreCount, songID = store_music_vector(list(genre_list), ytLink, ytTitle)
    
    similar_song_array = check_similarity(genreCount)
    
    colors = color_list
    
    defaultRGB = list(colorDict.values())

    # Example: return the color bar as a string
    return render_template('mainTemplate.html', genre_list=genre_list, filename=ytTitle, newDefaultRGB=defaultRGB, similar_song_array= similar_song_array, genre_percent=genreCount, songID=songID, ytLink=ytLink )
    

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

def add_song(songInp):
    
    x, sr, ytLink, ytTitle = load_test_data(songInp)
    
    songData = checkURL(ytLink, {
        0 : 0,
        1 : 0,
        2 : 0,
        3 : 0,
        4 : 0,
        5 : 0,
        6 : 0,
        7 : 0,
        8 : 0,
        9 : 0,
    })
    
    if(songData):
        songID, genre_list, genreCount, color_list = songData
    else:
        color_list, genre_list = predictTestColor(x, sr, {
        0 : 0,
        1 : 0,
        2 : 0,
        3 : 0,
        4 : 0,
        5 : 0,
        6 : 0,
        7 : 0,
        8 : 0,
        9 : 0,
    })
        genreCount, songID = store_music_vector(list(genre_list), ytLink, ytTitle)

if __name__ == '__main__':
        
    app.run(debug=True)