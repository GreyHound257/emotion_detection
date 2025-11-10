import logging
from PIL import Image, ImageOps
import numpy as np
import joblib
import os

# Constants
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MODEL_PATH = "models/emotion_svm.pkl"

# Load the SVM model once
try:
    data = joblib.load(MODEL_PATH)
    model = data['model'] if isinstance(data, dict) and 'model' in data else data
except Exception as e:
    logging.error(f"Could not load model: {e}")
    model = None

def preprocess_image(image_path, size=(64, 64)):
    """
    Open image, convert to RGB if needed, resize, normalize, flatten.
    Returns a 2D array suitable for SVM input.
    """
    try:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = ImageOps.fit(img, size, Image.LANCZOS)  # Resize
        arr = np.asarray(img).astype("float32") / 255.0  # Normalize
        arr = arr.flatten()  # Flatten to 1D
        return arr.reshape(1, -1)
    except Exception as e:
        logging.exception(f"Image preprocessing failed for {image_path}")
        raise RuntimeError(f"Preprocessing failed: {e}")

def predict(image_path):
    """
    Predict emotion from an image using the preloaded SVM model.
    Returns (predicted_class, confidence)
    """
    if model is None:
        raise RuntimeError("Model not loaded!")

    features = preprocess_image(image_path)

    try:
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(features)[0]
            idx = int(np.argmax(preds))
            confidence = float(preds[idx] * 100)
        else:
            idx = model.predict(features)[0]
            confidence = None  # SVM without probabilities
            # If idx is numeric, map to CLASS_NAMES
            if isinstance(idx, int) and 0 <= idx < len(CLASS_NAMES):
                idx = idx
            else:
                idx = 0  # fallback to first class

        predicted_class = CLASS_NAMES[idx]

        # Debug logging
        if confidence:
            logging.info(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        else:
            logging.info(f"Prediction: {predicted_class} (confidence unavailable)")

        return predicted_class, confidence
    except Exception as e:
        logging.exception("Prediction failed")
        return "Prediction Error", 0.0

# CLI support
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
    else:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
        else:
            pred, conf = predict(image_path)
            if conf:
                print(f"Predicted: {pred} ({conf:.2f}%)")
            else:
                print(f"Predicted: {pred}")
