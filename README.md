# AI-Based Emotion Detection System

## 📌 Introduction
The **AI-Based Emotion Detection System** is a deep learning model designed to recognize human emotions from facial expressions. Using Convolutional Neural Networks (CNNs) trained on the FER-2013 dataset, this project aims to classify emotions into seven categories: **Happiness, Neutral, Sadness, Anger, Surprise, Disgust, and Fear**. The model can process images and live webcam feeds to predict emotions in real time.

## 🚀 Features
- ✅ Detects emotions from images and webcam video streams.
- ✅ Uses a **CNN-based deep learning model** for high accuracy.
- ✅ Supports **real-time emotion detection** using OpenCV.
- ✅ Pre-trained on the **FER-2013 dataset** with over 35,000 images.
- ✅ Python-based implementation with TensorFlow and OpenCV.

## 🛠️ Technologies Used
- **Programming Language:** Python 3.12
- **Frameworks & Libraries:**
  - TensorFlow / Keras
  - OpenCV
  - NumPy & Pandas
  - Matplotlib & Seaborn (for visualization)
- **Development Environments:**
  - PyCharm
  - VS Code
  - Anaconda (optional)

## 📂 Project Structure
```
📁 emotion-detection-system
│── 📂 dataset/               # Dataset for training and testing
│── 📂 models/                # Trained model files
│── 📂 src/                   # Source code files
│   ├── train_model.py              # Training the CNN model
│   ├── test_model.py               # Testing the model
│   ├── realtime_emotion_detector.py   # Real-time emotion detection
│── 📜 README.md              # Project documentation
```

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone (https://github.com/divyanshhkaushik/Emotion-Detection-System.git)

### 3️⃣ Run the Model
- **For real-time detection using webcam:**
```bash
python src/realtime_emotion_detector.py
```
- **For training a new model:**
```bash
python src/train_model.py
```
- **For testing with images:**
```bash
python src/test_model.py 
```

## 🎯 Dataset Used
The model is trained on the **FER-2013 (Facial Expression Recognition) dataset**, which contains **48x48 grayscale images** categorized into seven emotions.
- Dataset Source: [Kaggle - FER-2013](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)

## 🤝 Contributing
Contributions are welcome! If you'd like to contribute:
1. **Fork** this repository.
2. **Create a feature branch** (`git checkout -b feature-name`).
3. **Commit your changes** (`git commit -m 'Added new feature'`).
4. **Push to the branch** (`git push origin feature-name`).
5. **Create a Pull Request**.

---
🚀 **Happy Coding!** 😊
