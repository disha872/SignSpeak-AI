import os
import pandas as pd

print("==========================================")
print("     SignSpeak AI Simple - Status Check   ")
print("==========================================")

csv_path = "data/dataset.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("\nDataset CSV Found!")
    print(f"Total Recorded Frames: {len(df)}")
    print("\nRecorded Signs & Frame Counts:")
    print(df["label"].value_counts())
else:
    print(f"\nDataset CSV ('{csv_path}') NOT found yet.")
    print("Run 'python 1_collect_data.py' to record your signs!")

print("\n==========================================")
print("Models Folder Check:")
models_dir = "models"
if os.path.exists(models_dir):
    files = os.listdir(models_dir)
    print("Files in models/:", files)
    if "model.keras" in files and "label_encoder.pkl" in files:
        print("\nTrained Model & Encoder Found! Ready to run 3_app.py!")
    else:
        print("\nModel not trained yet. Run 'python 2_train_model.py'!")
else:
    print("Models directory not found.")
