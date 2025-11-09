import sqlite3
from datetime import datetime
import os
import base64
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__, template_folder="templates", static_folder="static")

MODEL_PATH = "model.h5"
CLASSES_PATH = "classes.npy"
DB_PATH = "emotion_data.db"

# Load model & classes
model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True).tolist()

if not os.path.exists(DB_PATH):
    from init_database import init_db
    init_db()

# Database helper
def save_to_db(name, image_path, emotion, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO users (name, timestamp, image_path, predicted_emotion, confidence) VALUES (?, ?, ?, ?, ?)",
        (name, datetime.utcnow().isoformat(), image_path, emotion, confidence)
    )
    conn.commit()
    conn.close()

# Save base64 -> image
def save_image(b64data):
    os.makedirs("uploads", exist_ok=True)
    header, encoded = b64data.split(",", 1)
    data = base64.b64decode(encoded)
    filename = f"uploads/{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg"
    with open(filename, "wb") as f:
        f.write(data)
    return filename

def preprocess_image(img_path, size=(160,160)):
    img = Image.open(img_path).convert("RGB").resize(size)
    arr = np.asarray(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    if not payload or "image" not in payload:
        return jsonify({"error": "No image provided"}), 400

    name = payload.get("name", "Anonymous")
    image_b64 = payload["image"]
    image_path = save_image(image_b64)

    x = preprocess_image(image_path)
    preds = model.predict(x)[0]
    emotion = classes[np.argmax(preds)]
    confidence = float(np.max(preds))

    save_to_db(name, image_path, emotion, confidence)

    return jsonify({
        "emotion": emotion,
        "confidence": confidence,
        "all": dict(zip(classes, map(float, preds)))
    })
