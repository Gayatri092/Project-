import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import hog
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model
model = load_model("Project/pothole_detector.h5")

# Load the scaler used during training
scaler = joblib.load("C:/Users/Admin/Desktop/hiproject/Project/scaler.pkl")  # Ensure you saved the scaler during training

# Function to extract HOG features from a video frame
def extract_hog_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray = cv2.resize(gray, (64, 64))  # Resize to match training images
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return features

# Function to detect pothole region (contour detection)
def detect_pothole_contours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pothole_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 5000:  # Filter contours by area (adjust based on your case)
            pothole_contours.append(contour)
    
    return pothole_contours

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

    if label == "POTHOLE":
        pothole_contours = detect_pothole_contours(frame)

        for contour in pothole_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw bounding box
        
        cv2.putText(frame, "POTHOLE DETECTED!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save the captured frame
        cv2.imwrite("pothole_image.jpg", frame)
        print("✅ Frame Captured and Saved!")

    # Display result on frame
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Real-time Pothole Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
