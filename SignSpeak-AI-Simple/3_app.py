import os
import time
import pickle
import io
import base64
import numpy as np
import cv2
import pyttsx3
import threading
import streamlit as st
import streamlit.components.v1 as components
import mediapipe as mp
from PIL import Image
from gtts import gTTS

from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf

# Check if running on Streamlit Cloud server
IS_STREAMLIT_CLOUD = os.path.exists("/mount/src") or "STREAMLIT_SERVER_MODE" in os.environ

# ---------------------------------------------------------
# 1. STREAMLIT PAGE CONFIG & CSS
# ---------------------------------------------------------
st.set_page_config(page_title="SignSpeak AI", page_icon="🤟", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .title-banner { background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%); padding: 20px; border-radius: 12px; color: white; text-align: center; margin-bottom: 20px; }
    .sentence-box { background: #1e1b4b; border: 2px solid #6366f1; border-radius: 12px; padding: 15px; font-size: 22px; color: #38bdf8; min-height: 70px; margin-top: 15px; }
    .notice-box { background: #1e293b; border: 1px solid #38bdf8; border-radius: 8px; padding: 12px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-banner">
    <h1>🤟 SignSpeak AI</h1>
    <p>Real-Time Indian Sign Language Recognition & Text-to-Speech App</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. VOICE SYNTHESIZER (Browser Web Speech API + gTTS + pyttsx3)
# ---------------------------------------------------------
def speak_text(text):
    if not text: return
    # 1. Browser Native Speech Synthesis (JavaScript)
    clean_text = text.replace('"', '\\"').replace("'", "\\'")
    js = f"""
        <script>
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var utterance = new SpeechSynthesisUtterance("{clean_text}");
                utterance.rate = 0.95;
                utterance.pitch = 1.0;
                window.speechSynthesis.speak(utterance);
            }}
        </script>
    """
    components.html(js, height=0, width=0)

    # 2. Offline pyttsx3 fallback for local PC execution
    def _speak_local():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_speak_local, daemon=True).start()

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

# Helper function to process a single frame
def process_frame(frame, model, encoder, detector):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)
    predicted_sign = "Neutral"
    confidence = 0.0

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
        coords -= coords[0] # Wrist subtraction
        features = coords.flatten().reshape(1, -1) # (1, 63)

        probs = model.predict(features, verbose=0)[0]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])

        if confidence >= 0.70:
            predicted_sign = str(encoder.inverse_transform([pred_idx])[0])

        h, w, _ = frame.shape
        for lm in hand:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

    return frame, predicted_sign, confidence

# ---------------------------------------------------------
# 4. MAIN APP INTERFACE
# ---------------------------------------------------------
if model is None or detector is None:
    st.error("⚠️ Model or MediaPipe Landmarker not initialized properly. Please verify model files exist!")
else:
    cam_mode = st.radio(
        "📷 Select Input Method:",
        ["🌐 Browser Webcam (Select for Website Deployment)", "💻 Local OpenCV Camera (Use only on Local Laptop)", "📁 Upload Image"],
        index=0,
        horizontal=True
    )

    col_video, col_info = st.columns([2, 1])

    with col_info:
        st.subheader("Live Recognition")
        pred_placeholder = st.empty()
        conf_placeholder = st.empty()

        st.subheader("Constructed Sentence")
        sentence_placeholder = st.empty()

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            if st.button("🔊 Speak", use_container_width=True):
                sentence_text = " ".join(st.session_state.words_list)
                if sentence_text:
                    speak_text(sentence_text)
                    try:
                        tts = gTTS(text=sentence_text, lang='en')
                        fp = io.BytesIO()
                        tts.write_to_fp(fp)
                        fp.seek(0)
                        st.audio(fp, format="audio/mp3", autoplay=True)
                    except Exception:
                        pass

        with col_b2:
            if st.button("⌫ Undo", use_container_width=True):
                if st.session_state.words_list:
                    st.session_state.words_list.pop()
        with col_b3:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.words_list.clear()

        sentence = " ".join(st.session_state.words_list).capitalize()
        if sentence: sentence += "."
        sentence_placeholder.markdown(f'<div class="sentence-box">{sentence or "<i>[ Waiting for signs... ]</i>"}</div>', unsafe_allow_html=True)

    with col_video:
        if cam_mode == "🌐 Browser Webcam (Select for Website Deployment)":
            st.info("👇 Click 'Take Photo' below to capture your hand sign from your browser camera.")
            img_file_buffer = st.camera_input("Browser Webcam Feed")

            if img_file_buffer is not None:
                bytes_data = img_file_buffer.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, 1)

                annotated_frame, predicted_sign, confidence = process_frame(frame, model, encoder, detector)

                if predicted_sign != "Neutral":
                    pred_placeholder.markdown(f"### Sign: **{predicted_sign}**")
                    conf_placeholder.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
                    if st.button(f"➕ Add '{predicted_sign}' & Speak", type="primary", use_container_width=True):
                        st.session_state.words_list.append(predicted_sign)
                        speak_text(predicted_sign)
                        try:
                            tts = gTTS(text=predicted_sign, lang='en')
                            fp = io.BytesIO()
                            tts.write_to_fp(fp)
                            fp.seek(0)
                            st.audio(fp, format="audio/mp3", autoplay=True)
                        except Exception:
                            pass
                        st.rerun()
                else:
                    pred_placeholder.markdown("### Sign: *[ No hand / low confidence ]*")
                    conf_placeholder.progress(0.0, text="Confidence: 0%")

                st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Processed Hand Landmark Frame", use_container_width=True)

        elif cam_mode == "💻 Local OpenCV Camera (Use only on Local Laptop)":
            if IS_STREAMLIT_CLOUD:
                st.warning("⚠️ **Note:** 'Local OpenCV Camera' only works when running the app locally on your laptop using `streamlit run app.py`. On this Streamlit Cloud website, please select **'🌐 Browser Webcam'** above!")

            run_camera = st.checkbox("▶ Start Local Camera Stream", value=False)
            video_placeholder = st.empty()

            if run_camera:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Cannot access local camera hardware. If running on Streamlit Cloud website, please select '🌐 Browser Webcam' above!")
                else:
                    history_buffer = deque(maxlen=8)
                    last_word = None
                    last_word_time = 0.0

                    while run_camera:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Cannot access local camera. Please select '🌐 Browser Webcam' above for Streamlit Cloud!")
                            break

                        frame = cv2.flip(frame, 1)
                        annotated_frame, predicted_sign, confidence = process_frame(frame, model, encoder, detector)

                        history_buffer.append(predicted_sign)
                        counts = Counter(history_buffer)
                        most_common_sign, count = counts.most_common(1)[0]

                        if most_common_sign != "Neutral" and count >= 5:
                            curr_time = time.time()
                            if most_common_sign != last_word or (curr_time - last_word_time) > 1.8:
                                st.session_state.words_list.append(most_common_sign)
                                speak_text(most_common_sign)
                                last_word = most_common_sign
                                last_word_time = curr_time

                        if predicted_sign != "Neutral":
                            pred_placeholder.markdown(f"### Sign: **{predicted_sign}**")
                            conf_placeholder.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
                            cv2.putText(annotated_frame, f"Sign: {predicted_sign} ({confidence*100:.0f}%)", (20, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        else:
                            pred_placeholder.markdown("### Sign: *[ Waiting for hand ]*")
                            conf_placeholder.progress(0.0, text="Confidence: 0%")

                        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

                        sentence = " ".join(st.session_state.words_list).capitalize()
                        if sentence: sentence += "."
                        sentence_placeholder.markdown(f'<div class="sentence-box">{sentence or "<i>[ Waiting for signs... ]</i>"}</div>', unsafe_allow_html=True)
                        time.sleep(0.01)

                    cap.release()

        elif cam_mode == "📁 Upload Image":
            uploaded_file = st.file_uploader("Upload an image of a hand sign", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                annotated_frame, predicted_sign, confidence = process_frame(frame, model, encoder, detector)

                if predicted_sign != "Neutral":
                    pred_placeholder.markdown(f"### Sign: **{predicted_sign}**")
                    conf_placeholder.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
                    if st.button(f"➕ Add '{predicted_sign}' & Speak", type="primary", use_container_width=True):
                        st.session_state.words_list.append(predicted_sign)
                        speak_text(predicted_sign)
                        try:
                            tts = gTTS(text=predicted_sign, lang='en')
                            fp = io.BytesIO()
                            tts.write_to_fp(fp)
                            fp.seek(0)
                            st.audio(fp, format="audio/mp3", autoplay=True)
                        except Exception:
                            pass
                        st.rerun()
                else:
                    pred_placeholder.markdown("### Sign: *[ No sign detected / Low confidence ]*")
                    conf_placeholder.progress(0.0, text="Confidence: 0%")

                st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Uploaded & Processed Frame", use_container_width=True)
