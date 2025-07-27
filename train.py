import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Paths to data
IMAGE_DIR = "data/train_images"
LABELS_CSV = "data/labels.csv"

# Load labels
labels_data = pd.read_csv(LABELS_CSV)
labels_data['label'] = labels_data['label'].astype(int)

# Load images and labels
images, labels = [], []
for _, row in labels_data.iterrows():
    img_path = os.path.join(IMAGE_DIR, row['image_filename'])
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize images for consistency
            images.append(img)
            labels.append(row['label'])

images = np.array(images) / 255.0  # Normalize images
labels = to_categorical(labels)    # One-hot encode labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/weed_detection_model.h5")
print("Model saved to models/weed_detection_model.h5")
