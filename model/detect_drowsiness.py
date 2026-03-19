import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import winsound

# ----------------- PATHS -----------------
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(SCRIPT_DIR, "drowsiness_model.h5")
ALARM_PATH = os.path.join(PROJECT_ROOT, "alarm.wav")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Run train_model.py first to create drowsiness_model.h5."
    )

# ----------------- LOAD MODEL -----------------
model = load_model(MODEL_PATH)
print("✅ Loaded model from:", MODEL_PATH)

# Our model was trained on 24x24 grayscale eye images
IMG_H, IMG_W = 24, 24

# ----------------- HAAR CASCADES (OpenCV) -----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

if face_cascade.empty() or eye_cascade.empty():
    raise RuntimeError("Failed to load Haar cascades.")

# ----------------- SETTINGS -----------------
PRED_THRESHOLD = 0.5       # model output < 0.5 = closed, >= 0.5 = open
FRAMES_THRESHOLD = 20      # how many consecutive closed frames => drowsy

def play_alarm():
    if os.path.exists(ALARM_PATH):
        winsound.PlaySound(ALARM_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
    else:
        print("⚠ alarm.wav not found:", ALARM_PATH)

# ----------------- WEBCAM LOOP -----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

closed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    status_text = "NO FACE"
    status_color = (0, 255, 255)

    for (x, y, w, h) in faces:
        # draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        face_roi_gray = gray[y:y + h, x:x + w]

        # detect eyes inside the face region
        eyes = eye_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
        )

        preds = []

        for (ex, ey, ew, eh) in eyes[:2]:  # use at most two eyes
            eye_img = face_roi_gray[ey:ey + eh, ex:ex + ew]

            # preprocess like training (24x24, grayscale, 0–1)
            eye_resized = cv2.resize(eye_img, (IMG_W, IMG_H))
            eye_norm = eye_resized.astype("float32") / 255.0
            eye_norm = eye_norm.reshape(1, IMG_H, IMG_W, 1)

            prob_open = float(model.predict(eye_norm, verbose=0)[0][0])
            preds.append(prob_open)

            # draw eye box (just for visual)
            cv2.rectangle(
                frame,
                (x + ex, y + ey),
                (x + ex + ew, y + ey + eh),
                (0, 255, 255),
                1,
            )

        if preds:
            # average prediction of both eyes
            prob_open_mean = sum(preds) / len(preds)

            if prob_open_mean < PRED_THRESHOLD:
                status_text = f"DROWSY ({prob_open_mean:.2f})"
                status_color = (0, 0, 255)
                closed_frames += 1
            else:
                status_text = f"AWAKE ({prob_open_mean:.2f})"
                status_color = (0, 255, 0)
                closed_frames = 0
        else:
            status_text = "FACE FOUND, NO EYES"
            status_color = (0, 255, 255)

        break  # only handle first face

    cv2.putText(
        frame,
        status_text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2,
    )

    if closed_frames >= FRAMES_THRESHOLD:
        cv2.putText(
            frame,
            "DROWSY! WAKE UP!",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
        )
        play_alarm()

    cv2.imshow("Driver Drowsiness Detection", frame)

    # ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
