import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils

# Get label and CSV file
label = input("Enter label (A-Z or 1-10): ").strip()
csv_file = f"{label}.csv"

# Prepare CSV header
header = ["label", "handedness"] + [
    f"{side}_{axis}{i}"
    for side in ("left", "right")
    for i in range(21)
    for axis in ("x", "y", "z")
]

# Create CSV file if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Start video capture
cap = cv2.VideoCapture(0)
print(f"Recording for label '{label}'. Press 'q' in the video window or Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Initialize row with zeros
        row = [label, "unknown"] + [0] * 126

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                side = handedness.classification[0].label.lower()
                # Update row for this hand
                offset = 0 if side == "left" else 63
                for i, lm in enumerate(hand_landmarks.landmark):
                    idx = 2 + offset + i * 3
                    row[idx:idx+3] = [lm.x, lm.y, lm.z]
                row[1] = side
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Save row only if at least one hand was detected
        if results.multi_hand_landmarks:
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

        cv2.imshow("Capture", frame)
        # Use a slightly higher waitKey for better responsiveness
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Stopping capture…")
            break

except KeyboardInterrupt:
    print("\nCapture interrupted by user (Ctrl+C).")

finally:
    cap.release()
    cv2.destroyAllWindows()
