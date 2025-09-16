import torch
import joblib
import numpy as np
from model import HandSignModel
from collections import deque
import time

# --- Load model & preprocessing ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_mapping = joblib.load("label_mapping.pkl")
scaler = joblib.load("scaler.pkl")

model = HandSignModel(input_size=126, hidden_size=256, num_classes=len(label_mapping)).to(device)
model.load_state_dict(torch.load("hand_sign_model.pth", map_location=device))
model.eval()

# --- Stability logic ---
history = deque(maxlen=15)  # last 15 frames
stable_start_time = None
STABLE_DURATION = 0.5  # seconds
last_char_sent = None

def predict_live(features):
    """
    Returns live preview character and stable character (None if not stable)
    """
    global history, stable_start_time, last_char_sent

    X_scaled = scaler.transform(features.reshape(1, -1))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_idx = np.argmax(probs)
        predicted_label = label_mapping[predicted_idx]

    # --- Live preview ---
    live_char = predicted_label

    # --- Update history for stability ---
    history.append(predicted_label)
    counts = {k: history.count(k) for k in set(history)}
    stable_char = None
    for k, v in counts.items():
        if v / len(history) >= 0.7:  # 70% stability
            # Check duration
            if k == last_char_sent:
                stable_start_time = None  # already sent
            else:
                if stable_start_time is None:
                    stable_start_time = time.time()
                elif time.time() - stable_start_time >= STABLE_DURATION:
                    stable_char = k
                    last_char_sent = k
                    stable_start_time = None
    return live_char, stable_char
