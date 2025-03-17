from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to test dataset
test_dir = "dataset/test"

# Load model
model = load_model("emotion_detector.h5")

# Data preprocessing
datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory(test_dir, target_size=(48, 48),
                                             batch_size=64, color_mode="grayscale",
                                             class_mode="categorical")

# Evaluate model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
