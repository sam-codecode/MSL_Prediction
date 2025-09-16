# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
from utils import speak_text
from model_utils import predict_live  # your stable/predict logic
import mediapipe as mp

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

current_text = ""

# --- MediaPipe hands (share instance to avoid re-init cost) ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

@app.route('/')
def index():
    return render_template("index.html")

@socketio.on('frame')
def handle_frame(data):
    """
    Receives base64 frame from frontend.
    Returns: live preview char and full stable text (so frontend can highlight)
    """
    global current_text
    try:
        img_bytes = base64.b64decode(data.split(",")[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        live_char = ""
        stable_char = None

        if results.multi_hand_landmarks:
            row = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
            # Pad for 2 hands (if only 1 detected)
            if len(results.multi_hand_landmarks) == 1:
                row.extend([0.0, 0.0, 0.0] * 21)
            row = np.array(row)

            live_char, stable_char_candidate = predict_live(row)
            live_char = live_char or ""

            # If stability returned a new stable char, append it
            if stable_char_candidate:
                current_text += stable_char_candidate

        # emit both live preview and the full stable string
        emit('prediction', {'live': live_char, 'stable': current_text})
    except Exception as e:
        # In case of error, emit empty live and current stable text
        print(f"[frame handler error] {e}")
        emit('prediction', {'live': "", 'stable': current_text})

@socketio.on('button')
def handle_button(data):
    """
    Button commands: delete, clear, space, read
    """
    global current_text
    btn = data.get('button', '')

    if btn == "delete":
        current_text = ""
    elif btn == "clear":
        current_text = current_text[:-1]
    elif btn == "space":
        current_text += " "
    elif btn == "read":
        # Read the full current_text using speak_text (blocking)
        # it's OK to block here because it's explicitly user-triggered
        speak_text(current_text)

    # send the updated text back to frontend
    emit('update_text', current_text)

if __name__ == "__main__":
    # `debug=False` recommended in production
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
