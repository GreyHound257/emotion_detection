from flask import Flask, render_template, request
import os
import sqlite3
import base64
from werkzeug.utils import secure_filename
from predict import predict  # <-- import the function from predict.py
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATABASE'] = 'emotion_data.db'

# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- DATABASE SETUP ---
def init_db():
    with sqlite3.connect(app.config['DATABASE']) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp TEXT,
            image_path TEXT,
            predicted_emotion TEXT,
            confidence REAL
        )''')
        conn.commit()
init_db()

def save_to_db(name, image_path, emotion, confidence):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(app.config['DATABASE']) as conn:
        conn.execute(
            "INSERT INTO users (name, timestamp, image_path, predicted_emotion, confidence) VALUES (?, ?, ?, ?, ?)",
            (name, timestamp, image_path, emotion, confidence)
        )
        conn.commit()

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    name = request.form['name']

    # Handle uploaded file
    if 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    # Handle captured image from webcam
    elif 'captured_image' in request.form:
        image_data = request.form['captured_image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        filename = f"{name}_capture.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(img_bytes)

    else:
        return render_template('index.html', prediction="No image source found")

    # --- Use predict.py for prediction ---
    try:
        emotion, confidence = predict(filepath)
        save_to_db(name, filename, emotion, confidence)
    except Exception as e:
        logging.exception("Prediction failed")
        emotion, confidence = "Prediction Error", 0.0

    return render_template('index.html', prediction=emotion, confidence=confidence, image_file=filename)

@app.route('/database')
def database():
    with sqlite3.connect(app.config['DATABASE']) as conn:
        data = conn.execute("SELECT * FROM users").fetchall()
    return render_template('database.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
