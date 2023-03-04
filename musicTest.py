import pydub
import librosa

# get YouTube video audio stream
video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # replace with your YouTube video URL
audio_stream = pydub.AudioSegment.from_url(video_url)

# convert audio stream to numpy array
audio_data = librosa.load(pydub.AudioSegment.export(audio_stream, format="wav"), sr=22050)[0]

# extract music features
tempo, beat_frames = librosa.beat.beat_track(audio_data, sr=22050)
chroma_stft = librosa.feature.chroma_stft(audio_data, sr=22050)