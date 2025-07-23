import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained model and label classes
model = load_model("gesture_model.h5")
label_classes = np.load("label_classes.npy")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

print("📷 Real-time gesture prediction started... Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip for selfie-view and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        wrist = hand.landmark[0]

        # Extract relative x, y coords
        features = []
        for lm in hand.landmark:
            features.append(lm.x - wrist.x)
            features.append(lm.y - wrist.y)

        if len(features) == 42:
            # Predict using the model
            prediction = model.predict(np.array([features]))[0]
            predicted_index = np.argmax(prediction)
            predicted_label = label_classes[predicted_index]
            confidence = prediction[predicted_index]

            # Display the result
            cv2.putText(frame, f"{predicted_label} ({confidence*100:.1f}%)",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Draw landmarks
        mp_draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

    # Show video
    cv2.imshow("Real-time Gesture Recognition", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
