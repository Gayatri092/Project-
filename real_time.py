
import os
import pandas as pd
import numpy as np
import cv2
from skimage.feature import hog

def classify_frame(frame):
    """ Process and classify a single frame """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    
    pred = svm.predict([features])[0]
    
    label = "POTHOLE" if pred == 1 else "NORMAL"
    color = (0, 0, 255) if pred == 1 else (0, 255, 0)

    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

cap = cv2.VideoCapture(0)  # Open webcam
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = classify_frame(frame)
    cv2.imshow("Real-time Pothole Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
