import base64
import json
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

from config import (
    MODEL_PATH,
    MODEL_LABELS_PATH,
    SEQUENCE_LENGTH,
    FEATURE_SIZE,
    BASE_FEATURE_SIZE,
    HAND_FEATURE_SIZE,
    CONFIDENCE_THRESHOLD,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)

app = FastAPI()

mp_hands = mp.solutions.hands

model = tf.keras.models.load_model(MODEL_PATH)

with open(MODEL_LABELS_PATH, "r") as f:
    labels = json.load(f)

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

sequence = []
previous_base_features = None

ACTION_SIGNS = {"yes"}
WRIST_Y_INDEX = 64

# Change this only if phone frames reach backend sideways
ROTATE_INPUT_FRAME = False
ROTATE_CODE = cv2.ROTATE_90_CLOCKWISE

# Change this only if phone front camera becomes mirrored wrongly
FLIP_INPUT_FRAME = False


class FrameRequest(BaseModel):
    image: str


def normalize_hand(landmarks):
    wrist = landmarks[0]
    points = []

    for lm in landmarks:
        x = lm.x - wrist.x
        y = lm.y - wrist.y
        z = lm.z - wrist.z
        points.append([x, y, z])

    points = np.array(points, dtype=np.float32)

    scale = np.max(np.linalg.norm(points, axis=1))
    if scale < 1e-6:
        scale = 1.0

    points = points / scale

    hand_shape = points.flatten()

    wrist_position = np.array(
        [wrist.x, wrist.y, wrist.z],
        dtype=np.float32
    )

    return np.concatenate([hand_shape, wrist_position])


def extract_base_keypoints_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(frame_rgb)

    all_hands = []
    hand_count = 0

    if result.multi_hand_landmarks:
        detected_hands = result.multi_hand_landmarks
        hand_count = len(detected_hands)

        detected_hands = sorted(
            detected_hands,
            key=lambda hand: hand.landmark[0].x
        )

        for hand_landmarks in detected_hands[:2]:
            hand_features = normalize_hand(hand_landmarks.landmark)
            all_hands.append(hand_features)

    while len(all_hands) < 2:
        all_hands.append(np.zeros(HAND_FEATURE_SIZE, dtype=np.float32))

    base_features = np.concatenate(all_hands)

    if base_features.shape[0] != BASE_FEATURE_SIZE:
        raise ValueError(
            f"Base feature mismatch: got {base_features.shape[0]}, expected {BASE_FEATURE_SIZE}"
        )

    return base_features, hand_count


def detect_yes_nod(sequence_array):
    wrist_y = sequence_array[:, WRIST_Y_INDEX]

    kernel = np.ones(3) / 3
    wrist_y_smooth = np.convolve(wrist_y, kernel, mode="same")

    velocity = np.diff(wrist_y_smooth)

    threshold = 0.004

    direction = np.zeros_like(velocity)
    direction[velocity > threshold] = 1
    direction[velocity < -threshold] = -1

    reversals = 0
    last_dir = 0

    for d in direction:
        if d == 0:
            continue

        if last_dir != 0 and d != last_dir:
            reversals += 1

        last_dir = d

    nod_score = float(np.max(wrist_y_smooth) - np.min(wrist_y_smooth))
    nod_detected = reversals >= 2

    return nod_detected, nod_score, reversals


def decode_base64_image(image_base64):
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    image_bytes = base64.b64decode(image_base64)
    np_arr = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode image")

    if ROTATE_INPUT_FRAME:
        frame = cv2.rotate(frame, ROTATE_CODE)

    if FLIP_INPUT_FRAME:
        frame = cv2.flip(frame, 1)

    return frame


@app.get("/")
def home():
    return {
        "status": "running",
        "message": "SignSarthi backend is running",
        "labels": labels,
        "feature_size": FEATURE_SIZE,
    }


@app.post("/reset")
def reset():
    global sequence, previous_base_features

    sequence = []
    previous_base_features = None

    return {
        "status": "reset",
        "frames_collected": 0,
    }


@app.post("/predict")
def predict(request: FrameRequest):
    global sequence, previous_base_features

    frame = decode_base64_image(request.image)

    base_features, hand_count = extract_base_keypoints_from_frame(frame)

    if previous_base_features is None:
        motion_features = np.zeros(BASE_FEATURE_SIZE, dtype=np.float32)
    else:
        motion_features = base_features - previous_base_features

    previous_base_features = base_features.copy()

    final_features = np.concatenate(
        [base_features, motion_features]
    ).astype(np.float32)

    if final_features.shape[0] != FEATURE_SIZE:
        raise ValueError(
            f"Final feature mismatch: got {final_features.shape[0]}, expected {FEATURE_SIZE}"
        )

    sequence.append(final_features)

    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    if len(sequence) < SEQUENCE_LENGTH:
        return {
            "status": "collecting",
            "label": "WAITING",
            "confidence": 0.0,
            "frames_collected": len(sequence),
            "total_frames": SEQUENCE_LENGTH,
            "hand_count": hand_count,
            "probabilities": {},
        }

    sequence_array = np.array(sequence, dtype=np.float32)
    input_data = np.expand_dims(sequence_array, axis=0)

    prediction = model.predict(input_data, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index])
    predicted_label = labels[predicted_index]

    probabilities = {
        labels[i]: float(prediction[i])
        for i in range(len(labels))
    }
    print("HAND:", hand_count)
    print("PREDICTED:", predicted_label, confidence)
    print("PROBS:", probabilities)

    accepted = confidence >= CONFIDENCE_THRESHOLD
    message = "ok"

    if predicted_label in ACTION_SIGNS:
        nod_detected, nod_score, reversals = detect_yes_nod(sequence_array)

        if not nod_detected:
            accepted = False
            message = f"{predicted_label.upper()} needs motion"
    else:
        nod_score = 0.0
        reversals = 0

    if not accepted:
        display_label = "UNKNOWN"
    else:
        display_label = predicted_label.upper()

    return {
        "status": "ready",
        "label": display_label,
        "raw_label": predicted_label,
        "confidence": confidence,
        "frames_collected": len(sequence),
        "total_frames": SEQUENCE_LENGTH,
        "hand_count": hand_count,
        "probabilities": probabilities,
        "message": message,
    }


if __name__ == "__main__":
    print("Loaded labels:", labels)
    print("Feature size:", FEATURE_SIZE)
    print("Starting SignSarthi backend...")
    uvicorn.run(app, host="0.0.0.0", port=8000)