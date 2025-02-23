import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import Sequential      # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense   # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore

# Set the image folder path
image_folder = "C:/Users/Admin/Desktop/hiproject/Project/normal"

# Define image size (resize all images to the same size)
IMG_SIZE = (64, 64)  # You can change this based on your dataset

# Labels (Adjust based on your dataset)
label_map = {"label": 0, "path":1}  # Example labels

# Load images and labels
X, y = [], []

for label in os.listdir(image_folder):  # Folder names as labels
    label_path = os.path.join(image_folder, label)
    
    if os.path.isdir(label_path):  # Ensure it's a directory
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            image = cv2.imread(img_path)  # Read image
            image = cv2.resize(image, IMG_SIZE)  # Resize
            X.append(image)
            y.append(label_map[label])  # Assign label

# Convert lists to numpy arrays
X = np.array(X) / 255.0  # Normalize images (0-1)
y = np.array(y)
y = to_categorical(y, num_classes=len(label_map))  # Convert labels to one-hot encoding

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_map), activation='softmax')  # Output layer with class count
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
