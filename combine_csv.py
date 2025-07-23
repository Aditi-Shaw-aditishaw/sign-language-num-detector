import pandas as pd
import os

data_dir = "gesture_dataset"  # or wherever your individual files are
combined = pd.DataFrame()

for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_dir, file))
        combined = pd.concat([combined, df], ignore_index=True)

combined = combined.sample(frac=1).reset_index(drop=True)  # shuffle
combined.to_csv("final_gesture_dataset.csv", index=False)

print("✅ Combined CSV saved as final_gesture_dataset.csv")
