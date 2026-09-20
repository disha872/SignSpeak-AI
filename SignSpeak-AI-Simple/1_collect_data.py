import cv2
import csv
import os
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------
# 1. SETUP PATHS AND MEDIAPIPE DETECTOR
# ---------------------------------------------------------
MODEL_PATH = "models/hand_landmarker.task"
CSV_PATH = "data/dataset.csv"

# Initialize MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# ---------------------------------------------------------
# 2. ASK USER FOR SIGN NAME
# ---------------------------------------------------------
sign_name = input("Enter Sign Name to Collect (e.g., HELLO, YES, NO): ").strip().upper()
print(f"\nCollecting Data for Sign: '{sign_name}'")
print("Press SPACE to start/stop saving frames. Press Q to Quit.\n")

# ---------------------------------------------------------
# 3. SETUP CSV FILE
# ---------------------------------------------------------
os.makedirs("data", exist_ok=True)
file_exists = os.path.exists(CSV_PATH)
csv_file = open(CSV_PATH, mode="a", newline="")
writer = csv.writer(csv_file)

# Write Header if file is new (label + 63 features)
if not file_exists:
    header = ["label"] + [f"f{i+1}" for i in range(63)]
    writer.writerow(header)

# ---------------------------------------------------------
# 4. WEBCAM LOOP
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)
is_saving = False
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading webcam.")
        break

    frame = cv2.flip(frame, 1) # Mirror view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    # Check if hand detected
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        
        # Extract 21 points
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
        
        # WRIST CENTERING: Subtract wrist (point 0) from all points
        wrist = coords[0].copy()
        coords -= wrist
        
        features = coords.flatten() # 63 numbers

        # Save to CSV if saving is active
        if is_saving:
            writer.writerow([sign_name] + list(features))
            csv_file.flush()
            saved_count += 1

        # Draw hand dots on frame
        h, w, _ = frame.shape
        for lm in hand:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

    # Display instructions on screen
    status = f"SAVING ({saved_count} frames)" if is_saving else "READY"
    color = (0, 255, 0) if is_saving else (0, 255, 255)
    
    cv2.putText(frame, f"Sign: {sign_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, "SPACE = Toggle Saving | Q = Quit", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("SignSpeak Simple - Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        is_saving = not is_saving
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
detector.close()

print(f"\nData Collection Complete! Total frames saved for '{sign_name}': {saved_count}")
