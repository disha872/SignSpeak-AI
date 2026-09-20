import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------
CSV_PATH = "data/dataset.csv"

if not os.path.exists(CSV_PATH):
    print(f"Error: Dataset '{CSV_PATH}' not found. Run 1_collect_data.py first!")
    exit()

df = pd.read_csv(CSV_PATH)
print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)
print("\nClasses Count:")
print(df["label"].value_counts())

# Separate Features (X) and Labels (y)
X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values

# ---------------------------------------------------------
# 2. ENCODE LABELS
# ---------------------------------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print("\nClasses:", list(label_encoder.classes_))

# ---------------------------------------------------------
# 3. TRAIN / TEST SPLIT (80% Train, 20% Test)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

print(f"\nTraining Samples: {len(X_train)} | Testing Samples: {len(X_test)}")

# ---------------------------------------------------------
# 4. BUILD SIMPLE NEURAL NETWORK
# ---------------------------------------------------------
model = Sequential([
    Input(shape=(63,)),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------------------------------------
# 5. TRAIN MODEL
# ---------------------------------------------------------
print("\nTraining Neural Network...")
history = model.fit(
    X_train, y_train,
    validation_split=0.20,
    epochs=40,
    batch_size=16,
    verbose=1
)

# ---------------------------------------------------------
# 6. EVALUATE MODEL
# ---------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\n==========================================")
print(f"  TEST ACCURACY: {test_acc * 100:.2f}%")
print("==========================================")

y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ---------------------------------------------------------
# 7. SAVE MODEL AND ENCODER
# ---------------------------------------------------------
os.makedirs("models", exist_ok=True)
model.save("models/model.keras")

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nModel saved to: models/model.keras")
print("Encoder saved to: models/label_encoder.pkl")
