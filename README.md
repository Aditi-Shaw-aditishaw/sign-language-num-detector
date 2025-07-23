# Sign Language Detection using MediaPipe & FFNN

This project detects static sign language gestures using webcam, MediaPipe hand landmarks, and a Feedforward Neural Network (FFNN).

## Features
- Real-time hand tracking
- Custom dataset collection
- Gesture classification using FFNN
- Live webcam-based detection

## Tech Stack
- Python, OpenCV
- MediaPipe
- TensorFlow / Keras
- scikit-learn

## How to Run
1. Collect gestures: `python collect_sign_data.py`
2. Combine CSVs: `python combine_csv.py`
3. Train model: `python train_ffnn.py`
4. Predict live: `python predict_realtime.py`

## Author
Aditi Shaw
