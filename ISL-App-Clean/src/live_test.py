import json
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from config import (
    MODEL_PATH,
    MODEL_LABELS_PATH,
    SEQUENCE_LENGTH,
    FEATURE_SIZE,
    BASE_FEATURE_SIZE,
    HAND_FEATURE_SIZE,
    CONFIDENCE_THRESHOLD,
    YES_MOTION_THRESHOLD,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

ACTION_SIGNS = {"yes"}

# Wrist Y position index inside base features for the first (or dominant) hand.
# Layout: [hand0: 63 shape + 3 wrist_pos | hand1: 63 shape + 3 wrist_pos]
# Wrist position for hand0 starts at index 63: [x=63, y=64, z=65]
WRIST_Y_INDEX = 64  # wrist Y of the first detected hand in base features


def detect_yes_nod(sequence_array):
    """
    Checks whether the sequence contains at least ONE clear vertical nod
    (down-then-up OR up-then-down) in wrist Y position.

    In MediaPipe, Y increases downward on screen.
    A "nod down" = wrist Y goes up (increases).
    A "nod up"   = wrist Y goes down (decreases).

    Returns (bool: nod_detected, float: nod_score for display)
    """
    # Extract wrist Y from base features (first half of each frame)
    wrist_y = sequence_array[:, WRIST_Y_INDEX]  # shape: (30,)

    # Smooth slightly to reduce noise
    kernel = np.ones(3) / 3
    wrist_y_smooth = np.convolve(wrist_y, kernel, mode="same")

    # Frame-to-frame vertical velocity
    velocity = np.diff(wrist_y_smooth)  # shape: (29,)

    # Threshold: ignore tiny jitter — only count real directional movement
    NOD_VELOCITY_THRESHOLD = 0.004

    # Convert velocity to direction: +1 down, -1 up, 0 noise
    direction = np.zeros_like(velocity)
    direction[velocity > NOD_VELOCITY_THRESHOLD] = 1
    direction[velocity < -NOD_VELOCITY_THRESHOLD] = -1

    # Find direction changes (reversals): that's one nod
    reversals = 0
    last_dir = 0
    for d in direction:
        if d == 0:
            continue
        if last_dir != 0 and d != last_dir:
            reversals += 1
        last_dir = d

    # "YES" requires at least 1 reversal (one down-up or up-down = one nod)
    # Two tilts = 2 reversals minimum
    nod_detected = reversals >= 2

    # Score: peak-to-peak amplitude of wrist Y movement
    nod_score = float(np.max(wrist_y_smooth) - np.min(wrist_y_smooth))

    return nod_detected, nod_score, reversals


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


def extract_base_keypoints_from_frame(frame, hands_detector):
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

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    while len(all_hands) < 2:
        all_hands.append(np.zeros(HAND_FEATURE_SIZE, dtype=np.float32))

    base_features = np.concatenate(all_hands)

    if base_features.shape[0] != BASE_FEATURE_SIZE:
        raise ValueError(
            f"Base feature mismatch: got {base_features.shape[0]}, expected {BASE_FEATURE_SIZE}"
        )

    return base_features, frame, hand_count


def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(MODEL_LABELS_PATH, "r") as f:
        labels = json.load(f)

    print("Loaded model:", MODEL_PATH)
    print("Loaded labels:", labels)
    print("Feature size:", FEATURE_SIZE)
    print("Base feature size:", BASE_FEATURE_SIZE)
    print("YES motion threshold (legacy):", YES_MOTION_THRESHOLD)
    print("YES now uses nod detection (reversal-based).")

    sequence = []
    previous_base_features = None

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam started.")
    print("Press R to reset sequence.")
    print("Press Q to quit.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands_detector:

        while True:
            success, frame = cap.read()

            if not success:
                print("Failed to read webcam frame.")
                break

            # Do not flip or rotate live webcam frame.
            # Training videos were fixed in extract.py.
            # frame = cv2.flip(frame, 1)

            base_features, frame, hand_count = extract_base_keypoints_from_frame(
                frame,
                hands_detector
            )

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

            display_text = "Collecting frames..."
            confidence_text = ""
            probability_text = ""
            hand_text = f"Hands detected: {hand_count}"
            motion_text = ""

            if len(sequence) == SEQUENCE_LENGTH:
                sequence_array = np.array(sequence, dtype=np.float32)

                input_data = np.expand_dims(sequence_array, axis=0)

                prediction = model.predict(input_data, verbose=0)[0]

                predicted_index = int(np.argmax(prediction))
                confidence = float(prediction[predicted_index])
                predicted_label = labels[predicted_index]

                # --- YES-specific nod validation ---
                if predicted_label in ACTION_SIGNS:
                    nod_detected, nod_score, reversals = detect_yes_nod(sequence_array)
                    motion_text = f"Nod score: {nod_score:.4f} | Reversals: {reversals}"

                    if not nod_detected:
                        display_text = f"UNKNOWN - YES NEEDS TILTING MOTION"
                    elif confidence >= CONFIDENCE_THRESHOLD:
                        display_text = predicted_label.upper()
                    else:
                        display_text = f"UNKNOWN ({predicted_label.upper()}?)"

                # --- All other signs: original logic ---
                else:
                    motion_part = sequence_array[:, BASE_FEATURE_SIZE:]
                    motion_score = float(np.mean(np.abs(motion_part)))
                    motion_text = f"Motion score: {motion_score:.5f}"

                    if confidence >= CONFIDENCE_THRESHOLD:
                        display_text = predicted_label.upper()
                    else:
                        display_text = f"UNKNOWN ({predicted_label.upper()}?)"

                confidence_text = f"Top confidence: {confidence * 100:.2f}%"

                probability_text = " | ".join(
                    [
                        f"{labels[i]}: {prediction[i] * 100:.1f}%"
                        for i in range(len(labels))
                    ]
                )

            height, width, _ = frame.shape

            cv2.rectangle(
                frame,
                (0, 0),
                (width, 175),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                display_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                confidence_text,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                motion_text,
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                probability_text,
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                hand_text,
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.imshow("ISL Live Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                sequence.clear()
                previous_base_features = None
                print("Sequence reset")

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()