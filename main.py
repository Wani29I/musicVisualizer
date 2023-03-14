from flask import Flask, request
from model_preprocess.function_lib import load_test_data, predictTestColor
from flask import render_template
from flask import Flask
from pymongo import MongoClient
from datetime import datetime, timedelta
from bson import ObjectId

client = MongoClient("mongodb+srv://Wani29:Ice.31458@musicvector.xr7hdip.mongodb.net/?retryWrites=true&w=majority")

app = Flask(__name__, template_folder='templates')
musicvectorDb = client.MusicVector.newMusicvector
reviewSongDb = client.MusicVector.reviewSong

def check_similarity(genreCount):
    
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
        "genre_percentage": genreCount,
        "like": 0,
        "dislike": 0
        })
        
    return genreCount, str(songID.inserted_id)

def checkURL(ytLink, colorDict):
    prevSong = musicvectorDb.find_one({"youtube_link": ytLink})
    if( prevSong ): 
        return prevSong['_id'], prevSong['song_genre_array'], prevSong['genre_percentage'], [colorDict[predict] for predict in prevSong['song_genre_array']]
    
    return None

def getSongById(song_id, colorDict):
    prevSong = musicvectorDb.find_one({"_id": ObjectId(song_id)})
    if( prevSong ): 
        return prevSong['_id'], prevSong['song_genre_array'], prevSong['genre_percentage'], [colorDict[predict] for predict in prevSong['song_genre_array']], prevSong['file_name'],prevSong['youtube_link']
    
    return None

@app.route('/')
def index():
    return render_template('mainTemplate.html')
  
@app.route('/review')
def reviewResult():
    
    args = dict(request.args)
    id = ObjectId(args['songID'])
    review = args['isLike']=='true' 
    songID = ObjectId(args['songID'])
    
    update_data = {}
    
    if review:
        update_data['like'] = 1
    else:
        update_data['dislike'] = 1
    
    musicvectorDb.update_one({ '_id': id}, {
        '$inc': update_data
    })
    
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

@app.route('/song/<song_id>', methods=['GET'])
def main_progress2(song_id: str):

    formData = dict(request.args)
    
    
    colorDict = {
        0 : formData.get('color0',"rgba(36, 108, 183, 1)"),
        1 : formData.get('color1',"rgba(239, 234, 214, 1)"),
        2 : formData.get('color2',"rgba(132, 202, 235, 1)"),
        3 : formData.get('color3',"rgba(255, 0, 102, 1)"),
        4 : formData.get('color4',"rgba(66, 66, 66, 1)"),
        5 : formData.get('color5',"rgba(255, 159, 120, 1)"),
        6 : formData.get('color6',"rgba(0, 57, 178, 1)"),
        7 : formData.get('color7',"rgba(255, 106, 180, 1)"),
        8 : formData.get('color8',"rgba(102, 204, 0, 1)"),
        9 : formData.get('color9',"rgba(133, 194, 149, 1)"),
    }
    # print(colorDict)
    songData = getSongById(song_id,colorDict)

    
    if(songData):
        songID, genre_list, genreCount, color_list,ytTitle ,ytLink= songData
    else:
       return '<h1>404 not found</h1>'
        
    print('genre_list_num: ',len(genre_list))
    
    similar_song_array = check_similarity(genreCount)
    
    defaultRGB = list(colorDict.values())

    # Example: return the color bar as a string
    return render_template('songTemplate.html', genre_list=genre_list, filename=ytTitle, newDefaultRGB=defaultRGB, similar_song_array= similar_song_array, genre_percent=genreCount, songID=songID, ytLink=ytLink )
    
@app.route('/', methods=['POST'])
def main_progress():
    
    formData = dict(request.form)
    # Preprocess the file to extract features
    
    songInp = formData['songInp']
    
    x, sr, ytLink, ytTitle = load_test_data(songInp)
    
    # print('x: ', len(x[0]))
    # print('sr: ', sr)
    
    song_duration = int(len(x[0])/sr)
    
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
        color_list, genre_list = predictTestColor(x, sr, colorDict, song_duration)
        genreCount, songID = store_music_vector(list(genre_list), ytLink, ytTitle)
        
    print('genre_list_num: ',len(genre_list))
    
    similar_song_array = check_similarity(genreCount)
    
    defaultRGB = list(colorDict.values())

    # Example: return the color bar as a string
    return render_template('mainTemplate.html', genre_list=genre_list, filename=ytTitle, newDefaultRGB=defaultRGB, similar_song_array= similar_song_array, genre_percent=genreCount, songID=songID, ytLink=ytLink )

if __name__ == '__main__':
        
    app.run(debug=True)