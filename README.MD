# Emotion Detection (HOG + SVM)

---
A simple web-based Emotion Detection system that uses a trained Machine Learning model to recognize emotions from input data or images.

## Setup
1. Create virtualenv:
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows

2. Install:
   pip install -r requirements.txt

3. Put training images in data/<label>/*.jpg (e.g. data/happy/1.jpg)

4. Train model:
   python model.py

5. Run app:
   streamlit run app.py



---

## 🚀 Features
- Train an emotion detection model using MLPClassifier (Scikit-learn)
- View evaluation metrics (accuracy, classification report)
- Save and reuse the trained model
- Interactive web interface using Streamlit