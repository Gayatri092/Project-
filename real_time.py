import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import hog
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model
model = load_model("C:/Users/Admin/Desktop/hiproject/Project/pothole_detector.h5")

# Load the scaler used during training
scaler = joblib.load("C:/Users/Admin/Desktop/hiproject/Project/scaler.pkl")  # Ensure you saved the scaler during training

# Function to extract HOG features from a video frame
def extract_hog_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray = cv2.resize(gray, (64, 64))  # Resize to match training images
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return features

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract HOG features
    features = extract_hog_features(frame)

    # Normalize using the same scaler used during training
    features = scaler.transform([features])  

    # Predict using the model
    pred = model.predict(features)[0][0]  # Output probability

    # Determine label
    label = "POTHOLE" if pred > 0.5 else "NORMAL"
    color = (0, 0, 255) if pred > 0.5 else (0, 255, 0)

    # Display result on frame
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Real-time Pothole Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
