import os
import time
import pickle
import numpy as np
import cv2
import pyttsx3
import threading
import streamlit as st
import mediapipe as mp

from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf

# ---------------------------------------------------------
# 1. STREAMLIT PAGE CONFIG & CSS
# ---------------------------------------------------------
st.set_page_config(page_title="SignSpeak AI", page_icon="🤟", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .title-banner { background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%); padding: 20px; border-radius: 12px; color: white; text-align: center; }
    .sentence-box { background: #1e1b4b; border: 2px solid #6366f1; border-radius: 12px; padding: 15px; font-size: 22px; color: #38bdf8; min-height: 70px; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-banner">
    <h1>🤟 SignSpeak AI</h1>
    <p>Real-Time Indian Sign Language Recognition & Text-to-Speech App</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. OFFLINE VOICE SYNTHESIZER (pyttsx3 Background Thread)
# ---------------------------------------------------------
def speak_text(text):
    if not text: return
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()

# ---------------------------------------------------------
# 3. LOAD MODEL & MEDIAPIPE DETECTOR
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    possible_dirs = [
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "SignSpeak-AI-Simple", "models"),
        "models",
        "SignSpeak-AI-Simple/models"
    ]

    model_path, encoder_path, hand_model_path = None, None, None

    for d in possible_dirs:
        m = os.path.join(d, "model.keras")
        e = os.path.join(d, "label_encoder.pkl")
        h = os.path.join(d, "hand_landmarker.task")
        if os.path.exists(m) and os.path.exists(e) and os.path.exists(h):
            model_path, encoder_path, hand_model_path = m, e, h
            break

    if not model_path:
        model_path = "models/model.keras"
        encoder_path = "models/label_encoder.pkl"
        hand_model_path = "models/hand_landmarker.task"

    model, encoder, detector = None, None, None
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        try:
            model = tf.keras.models.load_model(model_path)
            with open(encoder_path, "rb") as f:
                encoder = pickle.load(f)
        except Exception as ex:
            st.error(f"Error loading Keras model: {ex}")

    if os.path.exists(hand_model_path):
        try:
            base_options = python.BaseOptions(model_asset_path=hand_model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5
            )
            detector = vision.HandLandmarker.create_from_options(options)
        except Exception as ex:
            st.error(f"MediaPipe Initialization Error: {ex}")
    else:
        st.error(f"Hand landmarker model file not found at {hand_model_path}")

    return model, encoder, detector

model, encoder, detector = load_assets()

# -----------------------------
# SESSION STATE SETUP
# -----------------------------
if 'words_list' not in st.session_state:
    st.session_state.words_list = []

# ---------------------------------------------------------
# 4. MAIN APP INTERFACE
# ---------------------------------------------------------
if model is None or detector is None:
    st.error("⚠️ Model or MediaPipe Landmarker not initialized properly. Please verify model files exist and system libraries are installed!")
else:
    run_camera = st.checkbox("▶ Start Live Camera", value=False)

    col_video, col_info = st.columns([2, 1])

    with col_info:
        st.subheader("Live Recognition")
        pred_placeholder = st.empty()
        conf_placeholder = st.empty()

        st.subheader("Constructed Sentence")
        sentence_placeholder = st.empty()

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            if st.button("🔊 Speak"):
                sentence_text = " ".join(st.session_state.words_list)
                speak_text(sentence_text)
        with col_b2:
            if st.button("⌫ Undo"):
                if st.session_state.words_list:
                    st.session_state.words_list.pop()
        with col_b3:
            if st.button("🗑 Clear"):
                st.session_state.words_list.clear()

    with col_video:
        video_placeholder = st.empty()

    if run_camera:
        cap = cv2.VideoCapture(0)
        history_buffer = deque(maxlen=8)
        last_word = None
        last_word_time = 0.0

        while run_camera:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = detector.detect(mp_image)

            predicted_sign = "Neutral"
            confidence = 0.0

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                # Extract 21 points & Wrist Center
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
                coords -= coords[0] # Wrist subtraction
                features = coords.flatten().reshape(1, -1) # (1, 63)

                # Model Prediction
                probs = model.predict(features, verbose=0)[0]
                pred_idx = np.argmax(probs)
                confidence = float(probs[pred_idx])

                if confidence >= 0.70:
                    predicted_sign = str(encoder.inverse_transform([pred_idx])[0])

                # Draw dots on hand
                h, w, _ = frame.shape
                for lm in hand:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

            # Smooth predictions with rolling majority vote
            history_buffer.append(predicted_sign)
            counts = Counter(history_buffer)
            most_common_sign, count = counts.most_common(1)[0]

            if most_common_sign != "Neutral" and count >= 5:
                # Deduplication & Cooldown (1.8s)
                curr_time = time.time()
                if most_common_sign != last_word or (curr_time - last_word_time) > 1.8:
                    st.session_state.words_list.append(most_common_sign)
                    speak_text(most_common_sign)
                    last_word = most_common_sign
                    last_word_time = curr_time

            # Update UI
            if predicted_sign != "Neutral":
                pred_placeholder.markdown(f"### Sign: **{predicted_sign}**")
                conf_placeholder.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
                cv2.putText(frame, f"Sign: {predicted_sign} ({confidence*100:.0f}%)", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                pred_placeholder.markdown("### Sign: *[ Waiting for hand ]*")
                conf_placeholder.progress(0.0, text="Confidence: 0%")

            video_placeholder.image(frame, channels="BGR", use_container_width=True)

            sentence = " ".join(st.session_state.words_list).capitalize()
            if sentence: sentence += "."
            sentence_placeholder.markdown(f'<div class="sentence-box">{sentence or "<i>[ Waiting for signs... ]</i>"}</div>', unsafe_allow_html=True)

            time.sleep(0.01)

        cap.release()
