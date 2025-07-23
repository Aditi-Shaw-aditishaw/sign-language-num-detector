import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Capture from webcam
cap = cv2.VideoCapture(0)

# Ask for label
label = input("Enter the gesture label (e.g., 1, 2): ")
filename = f"{label}_gesture_data.csv"
output_dir = "gesture_dataset"
os.makedirs(output_dir, exist_ok=True)

# Data list and sample count
data = []
target_samples = 200
count = 0

print(f"Starting data collection for label '{label}'...")

while count < target_samples:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        wrist = hand.landmark[0]

        # Extract 42 features (x, y for each landmark relative to wrist)
        row = []
        for lm in hand.landmark:
            row.append(round(lm.x - wrist.x, 5))
            row.append(round(lm.y - wrist.y, 5))

        row.append(label)
        data.append(row)
        count += 1

        # Draw hand landmarks on screen
        mp_draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        print(f"Captured {count}/{target_samples}")

    # Show frame
    cv2.putText(frame, f"Label: {label} ({count}/{target_samples})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Data Collection", frame)

    # Stop if ESC is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Write to CSV
header = [f'x{i}' if i % 2 == 0 else f'y{i//2}' for i in range(42)] + ['label']
file_path = os.path.join(output_dir, filename)

with open(file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

print(f"✅ Dataset saved to {file_path}")
cap.release()
cv2.destroyAllWindows()
