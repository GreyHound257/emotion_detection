import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from tqdm import tqdm

# -------------------------------
# CONFIG
# -------------------------------
DATASET_DIR = "dataset"  # folder containing subfolders per emotion
MODEL_PATH = "models/emotion_svm.pkl"
os.makedirs("models", exist_ok=True)

# -------------------------------
# FEATURE EXTRACTION (HOG)
# -------------------------------
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    return hog.compute(gray).flatten()

# -------------------------------
# LOAD DATASET
# -------------------------------
def load_dataset(dataset_path):
    X, y = [], []
    for emotion in os.listdir(dataset_path):
        emotion_path = os.path.join(dataset_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        for img_file in tqdm(os.listdir(emotion_path), desc=f"Loading {emotion}"):
            try:
                img_path = os.path.join(emotion_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (64, 64))
                features = extract_hog_features(img)
                X.append(features)
                y.append(emotion)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    return np.array(X), np.array(y)

# -------------------------------
# TRAIN MODEL
# -------------------------------
print("📦 Loading dataset...")
X, y = load_dataset(DATASET_DIR)
print("✅ Dataset loaded:", X.shape, "samples")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("🚀 Training SVM model...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n✅ Model trained successfully!")
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save model + label encoder
dump(model, MODEL_PATH)
dump(le, "models/label_encoder.pkl")
print(f"\n💾 Model saved to {MODEL_PATH}")
