
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


# Audio Dash
@app.route('/audio_dash', methods=["GET", "POST"])
def audio_dash():
    prob = []  # Initialize prob as an empty list

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            model_sub_dir = os.path.join('Models', 'audio.hdf5')
            SER = speechEmotionRecognition(model_sub_dir)
            emotions, timestamp = SER.predict_emotion_from_file(file_path, chunk_step=1*16000)
            # Calculate probabilities or any other necessary processing
             # Export predicted emotions to .txt format
            SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')
            # Sample calculation for demonstration
            # Get most common emotion during the interview
            major_emotion = max(set(emotions), key=emotions.count)
            # Calculate emotion distribution
            emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
            prob = emotion_dist  # Update prob with the calculated emotion distribution
            df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
            df.to_csv(os.path.join('static/js/db','audio_emotions_dist.txt'), sep=',')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If the user does not select a file, the browser may submit an empty part without a filename
        if file.filename == '':
            flash('No selected file')
            model_sub_dir = os.path.join('Models', 'audio.hdf5')
            SER = speechEmotionRecognition(model_sub_dir)
            emotions, timestamp = SER.predict_emotion_from_file(file_path, chunk_step=1*16000)
            # Calculate probabilities or any other necessary processing
             # Export predicted emotions to .txt format
            SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')
            # Sample calculation for demonstration
            # Get most common emotion during the interview
            major_emotion = max(set(emotions), key=emotions.count)
            # Calculate emotion distribution
            emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
            prob = emotion_dist  # Update prob with the calculated emotion distribution
            df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
            df.to_csv(os.path.join('static/js/db','audio_emotions_dist.txt'), sep=',')
            return redirect(request.url)
        
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            flash('File successfully uploaded')

            # Perform calculations on the uploaded file
            # For example, you can pass the file_path to your SER object for analysis
            model_sub_dir = os.path.join('Models', 'audio.hdf5')
            SER = speechEmotionRecognition(model_sub_dir)
            emotions, timestamp = SER.predict_emotion_from_file(file_path, chunk_step=1*16000)
            # Calculate probabilities or any other necessary processing
             # Export predicted emotions to .txt format
            SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')
            # Sample calculation for demonstration
            # Get most common emotion during the interview
            major_emotion = max(set(emotions), key=emotions.count)
            # Calculate emotion distribution
            emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
            prob = emotion_dist  # Update prob with the calculated emotion distribution
            df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
            df.to_csv(os.path.join('static/js/db','audio_emotions_dist.txt'), sep=',')
            print(prob)
            # Redirect to another page or render a template with the calculated values
            # return render_template('audio_dash.html', prob=prob, emo=major_emotion)
            return jsonify(pro = prob, emotion = major_emotion)

    return render_template('audio_dash.html', prob=[])


if __name__ == '__main__':
    app.run(debug=True)
