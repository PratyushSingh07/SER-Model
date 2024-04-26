
### General Imports
import numpy as np
import pandas as pd
import time
import os
from collections import Counter

### Flask imports
import requests
from flask import Flask, render_template, session, request, redirect, flash, Response, jsonify

### Audio imports ###
from library.speech_emotion_recognition import *

from werkzeug.utils import secure_filename


# Flask config
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = ''
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

################################################################################
################################## INDEX #######################################
################################################################################

# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Audio Index
@app.route('/audio_index', methods=['POST'])
def audio_index():

    # Flash message
    flash("After pressing the button above, you will have 15sec to answer the question.")
    
    return render_template('audio.html', display_button=False)

# Audio Recording => can be fixed, pyaudio error rn
@app.route('/audio_recording', methods=("POST", "GET"))
def audio_recording():

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition()

    # Voice Recording
    rec_duration = 16 
    rec_sub_dir = os.path.join('tmp','voice_recording.wav')
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

    # Send Flash message
    flash("The recording is over! You now have the opportunity to do an analysis of your emotions. If you wish, you can also choose to record yourself again.")

    return render_template('audio.html', display_button=True)


@app.route('/audio_dash', methods=["POST", "GET"])
def audio_dash():
    major_emotion = None
    emotion_dist = []
    
    if request.method == 'POST':
        file = request.files.get('file')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Proceed with emotion recognition
            model_sub_dir = os.path.join('Models', 'audio.hdf5')
            SER = speechEmotionRecognition(model_sub_dir)
            
            # Predict emotion in voice
            step = 1  # in seconds
            sample_rate = 16000  # in Hz
            emotions, timestamp = SER.predict_emotion_from_file(file_path, chunk_step=step * sample_rate)
            
            # Get most common emotion during the recording
            if emotions:
                major_emotion = max(set(emotions), key=emotions.count)
                emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
            
                # Export emotion distribution for visualization
                df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
                dist_file_path = os.path.join('static/js/db', 'audio_emotions_dist.txt')
                df.to_csv(dist_file_path, sep=',')
                
                # Sleep (might not be necessary in production)
                time.sleep(0.5)
        else:
            if file:
                return render_template('audio_dash.html', error="Unsupported file type")
            else:
                return render_template('audio_dash.html', error="No file selected or file is not valid")

    # Handle both GET requests and unsuccessful POST requests.
    return render_template('audio_dash.html', emo=major_emotion, prob=emotion_dist)



if __name__ == '__main__':
    app.run(debug=True)
