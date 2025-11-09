from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import sqlite3
import base64
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATABASE'] = 'database.db'

# Load your trained model
model = joblib.load('models/emotion_svm.pkl')

# Emotion classes (modify to match your model)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- DATABASE SETUP ---
def init_db():
    with sqlite3.connect(app.config['DATABASE']) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            image_path TEXT,
            emotion TEXT,
            confidence REAL
        )''')
init_db()

def save_to_db(name, image_path, emotion, confidence):
    with sqlite3.connect(app.config['DATABASE']) as conn:
        conn.execute("INSERT INTO users (name, image_path, emotion, confidence) VALUES (?, ?, ?, ?)",
                     (name, image_path, emotion, confidence))
        conn.commit()

# --- IMAGE PREDICTION FUNCTION ---
def predict_emotion(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No Face Detected", 0.0

    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48)) / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    preds = model.predict(face)[0]
    emotion = EMOTIONS[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return emotion, confidence

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']

    if 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', prediction="No file uploaded")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    elif 'captured_image' in request.form:
        image_data = request.form['captured_image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        filename = f"{name}_capture.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(img_bytes)

    else:
        return render_template('index.html', prediction="No image source found")

    emotion, confidence = predict_emotion(filepath)
    save_to_db(name, filename, emotion, confidence)

    return render_template('index.html', prediction=emotion, confidence=confidence, image_file=filename)

@app.route('/database')
def database():
    with sqlite3.connect(app.config['DATABASE']) as conn:
        data = conn.execute("SELECT * FROM users").fetchall()
    return render_template('database.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
