import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "data", "dataset.csv")

if not os.path.exists(csv_path):
    csv_path = "data/dataset.csv"

if not os.path.exists(csv_path):
    print(f"Error: Dataset '{csv_path}' not found!")
    exit(1)

df = pd.read_csv(csv_path)
print("Dataset Loaded Successfully! Shape:", df.shape)
print("\nClass Counts:")
print(df["label"].value_counts())

labels = df["label"].values
features = df.drop(columns=["label"]).values.astype(np.float32)

# ---------------------------------------------------------
# 2. FEATURE NORMALIZATION & AUGMENTATION
# ---------------------------------------------------------
# Reshape to (N, 21, 3)
N = len(features)
coords = features.reshape(N, 21, 3)

# Wrist Centering (Point 0 subtraction)
coords = coords - coords[:, 0:1, :]

# Scale Normalization (Distance invariant)
scales = np.max(np.linalg.norm(coords, axis=2, keepdims=True), axis=1, keepdims=True)
scales[scales == 0] = 1.0
norm_coords = coords / scales

# Augmentation: Left/Right Hand & Camera Mirroring Invariance (Flip X axis)
flipped_coords = norm_coords.copy()
flipped_coords[:, :, 0] *= -1.0

# Combine Original + Flipped
aug_coords = np.vstack([norm_coords, flipped_coords])
aug_labels = np.concatenate([labels, labels])

X_final = aug_coords.reshape(len(aug_coords), 63)

# Encode Labels
label_encoder = LabelEncoder()
y_final = label_encoder.fit_transform(aug_labels)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.20, random_state=42, stratify=y_final
)

print(f"\nTraining Samples: {len(X_train)} | Testing Samples: {len(X_test)}")

# ---------------------------------------------------------
# 3. TRAIN RANDOM FOREST CLASSIFIER
# ---------------------------------------------------------
print("\nTraining Robust Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

test_acc = model.score(X_test, y_test)
print("==========================================")
print(f"  TEST ACCURACY: {test_acc * 100:.2f}%")
print("==========================================")

y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ---------------------------------------------------------
# 4. SAVE MODEL AND ENCODER
# ---------------------------------------------------------
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)
os.makedirs("models", exist_ok=True)

model_file = os.path.join(models_dir, "model.pkl")
encoder_file = os.path.join(models_dir, "label_encoder.pkl")

with open(model_file, "wb") as f:
    pickle.dump(model, f)

with open(encoder_file, "wb") as f:
    pickle.dump(label_encoder, f)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print(f"\nModel saved successfully to {model_file}")
print(f"Encoder saved successfully to {encoder_file}")
