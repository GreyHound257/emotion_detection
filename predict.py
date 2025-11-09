# predict.py
import sys
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

MODEL_PATH = "saved_model/emotion_model.h5"
CLASS_NAMES = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def predict(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.fit(img, (224,224), Image.ANTIALIAS)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr)[0]
    idx = preds.argmax()
    print(f"Prediction: {CLASS_NAMES[idx]} ({preds[idx]*100:.2f}%)")
    for i,p in enumerate(preds):
        print(f" - {CLASS_NAMES[i]}: {p*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
    else:
        predict(sys.argv[1])
