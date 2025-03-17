import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define image size and batch size
image_size = (48, 48)
batch_size = 64

# Paths to dataset
train_dir = "dataset/train"
test_dir = "dataset/test"

# Data Augmentation and Preprocessing
datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.1,
                             zoom_range=0.1,
                             horizontal_flip=True)

train_generator = datagen.flow_from_directory(train_dir, target_size=image_size,
                                              batch_size=batch_size, color_mode="grayscale",
                                              class_mode="categorical")

test_generator = datagen.flow_from_directory(test_dir, target_size=image_size,
                                             batch_size=batch_size, color_mode="grayscale",
                                             class_mode="categorical")

# Define CNN Model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 25
history = model.fit(train_generator, validation_data=test_generator, epochs=epochs)

# Save trained model
model.save("emotion_detector.h5")
print("Training complete. Model saved as emotion_detector.h5")
