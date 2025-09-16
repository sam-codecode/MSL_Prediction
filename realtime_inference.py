import cv2
import mediapipe as mp
import torch
import joblib
import numpy as np
from model import HandSignModel

# --- Load Model & Scaler ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_mapping = joblib.load("label_mapping.pkl")
scaler = joblib.load("scaler.pkl")

model = HandSignModel(input_size=126, hidden_size=256, num_classes=len(label_mapping)).to(device)
model.load_state_dict(torch.load("hand_sign_model.pth", map_location=device))
model.eval()

# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- Webcam ---
cap = cv2.VideoCapture(0)
print("✅ Real-time hand sign detection started! Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        row = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

        # Pad to ensure 2 hands (42 landmarks → 126 features)
        if len(results.multi_hand_landmarks) == 1:
            row.extend([0.0, 0.0, 0.0] * 21)

        # Scale + predict
        X = np.array(row).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            label = label_mapping[predicted.item()]

        # Display
        cv2.putText(frame, f"Prediction: {label}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Real-time Hand Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()