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

# Signs that require motion validation before being accepted
ACTION_SIGNS = {"yes", "please", "need", "thank_you", "help"}

# Feature layout per hand: 63 shape + 3 wrist = 66
# Hand 0 wrist: x=63, y=64, z=65
# Hand 1 wrist: x=129, y=130, z=131
WRIST_X_0 = 63
WRIST_Y_0 = 64
WRIST_Z_0 = 65
WRIST_X_1 = 129
WRIST_Y_1 = 130
WRIST_Z_1 = 131

# Motion thresholds — tune these if needed
PLEASE_MOTION_THRESHOLD   = 1.108   # minimum wrist movement for please
NEED_MOTION_THRESHOLD     = 0.015   # minimum motion for need
NEED_CLOSING_THRESHOLD    = 0.003   # minimum finger-spread change for closing
THANK_YOU_Z_THRESHOLD     = 0.015   # minimum forward (Z-decrease) movement for thank_you
HELP_MOTION_THRESHOLD     = 0.003   # minimum motion score for help (slight movement is enough)
HELP_HANDS_RATIO          = 0.40    # both hands must appear in ≥40% of frames for help


# ── YES validator ─────────────────────────────────────────────────────────────
def detect_yes_nod(sequence_array):
    """
    Detects vertical wrist oscillation (tilt down-up twice).
    Returns (detected, score, reversals)
    """
    wrist_y = sequence_array[:, WRIST_Y_0]
    kernel = np.ones(3) / 3
    wrist_y_smooth = np.convolve(wrist_y, kernel, mode="same")
    velocity = np.diff(wrist_y_smooth)

    NOD_VELOCITY_THRESHOLD = 0.004
    direction = np.zeros_like(velocity)
    direction[velocity >  NOD_VELOCITY_THRESHOLD] =  1
    direction[velocity < -NOD_VELOCITY_THRESHOLD] = -1

    reversals = 0
    last_dir = 0
    for d in direction:
        if d == 0:
            continue
        if last_dir != 0 and d != last_dir:
            reversals += 1
        last_dir = d

    nod_detected = reversals >= 2
    nod_score = float(np.max(wrist_y_smooth) - np.min(wrist_y_smooth))
    return nod_detected, nod_score, reversals


# ── PLEASE validator ──────────────────────────────────────────────────────────
def detect_please_motion(sequence_array):
    """
    Please = circular rubbing motion on chest.
    Circular motion means the wrist moves in BOTH X and Y directions
    over the sequence — total path length must exceed threshold.
    Returns (detected, motion_score)
    """
    wrist_x = sequence_array[:, WRIST_X_0]
    wrist_y = sequence_array[:, WRIST_Y_0]

    # Smooth to reduce noise
    kernel = np.ones(3) / 3
    wx = np.convolve(wrist_x, kernel, mode="same")
    wy = np.convolve(wrist_y, kernel, mode="same")

    # Total path length of wrist across the sequence
    dx = np.diff(wx)
    dy = np.diff(wy)
    step_distances = np.sqrt(dx**2 + dy**2)
    total_path = float(np.sum(step_distances))

    # Also check that movement exists in both axes (not just one direction)
    x_range = float(np.max(wx) - np.min(wx))
    y_range = float(np.max(wy) - np.min(wy))
    both_axes = x_range > 0.01 and y_range > 0.01

    detected = total_path > PLEASE_MOTION_THRESHOLD and both_axes
    return detected, total_path, x_range, y_range


# ── NEED validator ────────────────────────────────────────────────────────────
def detect_need_motion(sequence_array):
    """
    Need = both hands near chest, closing palm (fingers coming together).
    Two checks:
    1. Both hands present in most frames
    2. Overall motion exists (hands moving as they close)
    Returns (detected, motion_score, both_hands_ratio)
    """
    # Check how many frames have both hands (second hand nonzero)
    second_hand = sequence_array[:, HAND_FEATURE_SIZE:BASE_FEATURE_SIZE]
    frames_with_both = int(np.sum(np.any(second_hand != 0, axis=1)))
    both_hands_ratio = frames_with_both / SEQUENCE_LENGTH

    # Overall motion from motion features (second half of feature vector)
    motion_part = sequence_array[:, BASE_FEATURE_SIZE:]
    motion_score = float(np.mean(np.abs(motion_part)))

    # Need requires both hands visible in at least 40% of frames
    # AND some motion (closing gesture)
    detected = both_hands_ratio >= 0.4 and motion_score > NEED_MOTION_THRESHOLD
    return detected, motion_score, both_hands_ratio


# ── THANK_YOU validator ───────────────────────────────────────────────────────
def detect_thankyou_motion(sequence_array):
    """
    Thank you = flat hand moves forward/outward from chin toward the camera.
    Detected when the wrist Z-coordinate decreases (Z drops = hand comes closer)
    from the start of the sequence to the end.
    Returns (detected, z_drop, z_range)
    """
    wrist_z = sequence_array[:, WRIST_Z_0]
    kernel  = np.ones(3) / 3
    wz      = np.convolve(wrist_z, kernel, mode="same")

    z_start = float(np.mean(wz[:5]))   # average of first 5 frames
    z_end   = float(np.mean(wz[-5:]))  # average of last 5 frames
    z_drop  = z_start - z_end          # positive = moved forward (Z decreased)
    z_range = float(np.max(wz) - np.min(wz))

    detected = z_drop > THANK_YOU_Z_THRESHOLD
    return detected, z_drop, z_range


# ── HELP validator ────────────────────────────────────────────────────────────
def detect_help_motion(sequence_array):
    """
    Help = one palm resting on the other, both hands lift upward slightly.
    Two checks:
    1. Both hands visible in at least 40% of frames.
    2. Any slight overall motion present.
    Returns (detected, motion_score, both_hands_ratio)
    """
    second_hand      = sequence_array[:, HAND_FEATURE_SIZE:BASE_FEATURE_SIZE]
    frames_with_both = int(np.sum(np.any(second_hand != 0, axis=1)))
    both_hands_ratio = frames_with_both / SEQUENCE_LENGTH

    motion_part  = sequence_array[:, BASE_FEATURE_SIZE:]
    motion_score = float(np.mean(np.abs(motion_part)))

    detected = both_hands_ratio >= HELP_HANDS_RATIO and motion_score > HELP_MOTION_THRESHOLD
    return detected, motion_score, both_hands_ratio


def normalize_hand(landmarks):
    wrist = landmarks[0]
    points = []
    for lm in landmarks:
        points.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    points = np.array(points, dtype=np.float32)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale < 1e-6:
        scale = 1.0
    points = points / scale
    hand_shape = points.flatten()
    wrist_position = np.array([wrist.x, wrist.y, wrist.z], dtype=np.float32)
    return np.concatenate([hand_shape, wrist_position])


def extract_base_keypoints_from_frame(frame, hands_detector):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(frame_rgb)

    all_hands = []
    hand_count = 0

    if result.multi_hand_landmarks:
        detected_hands = result.multi_hand_landmarks
        hand_count = len(detected_hands)
        detected_hands = sorted(detected_hands, key=lambda h: h.landmark[0].x)
        for hand_landmarks in detected_hands[:2]:
            all_hands.append(normalize_hand(hand_landmarks.landmark))
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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

    print("Loaded model  :", MODEL_PATH)
    print("Loaded labels :", labels)
    print("Action signs  :", ACTION_SIGNS)

    sequence = []
    previous_base_features = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam started. R = reset | Q = quit")

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

            base_features, frame, hand_count = extract_base_keypoints_from_frame(
                frame, hands_detector
            )

            if previous_base_features is None:
                motion_features = np.zeros(BASE_FEATURE_SIZE, dtype=np.float32)
            else:
                motion_features = base_features - previous_base_features
            previous_base_features = base_features.copy()

            final_features = np.concatenate([base_features, motion_features]).astype(np.float32)

            if final_features.shape[0] != FEATURE_SIZE:
                raise ValueError(
                    f"Final feature mismatch: got {final_features.shape[0]}, expected {FEATURE_SIZE}"
                )

            sequence.append(final_features)
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            display_text    = "Collecting frames..."
            confidence_text = ""
            probability_text = ""
            motion_text     = ""
            hand_text       = f"Hands detected: {hand_count}"

            if len(sequence) == SEQUENCE_LENGTH:
                sequence_array = np.array(sequence, dtype=np.float32)
                input_data = np.expand_dims(sequence_array, axis=0)
                prediction = model.predict(input_data, verbose=0)[0]

                predicted_index = int(np.argmax(prediction))
                confidence      = float(prediction[predicted_index])
                predicted_label = labels[predicted_index]

                # ── YES ───────────────────────────────────────────────────────
                if predicted_label == "yes":
                    nod_detected, nod_score, reversals = detect_yes_nod(sequence_array)
                    motion_text = f"Nod score: {nod_score:.4f} | Reversals: {reversals}"
                    if not nod_detected:
                        display_text = "UNKNOWN - YES NEEDS TILTING MOTION"
                    elif confidence >= CONFIDENCE_THRESHOLD:
                        display_text = "YES"
                    else:
                        display_text = f"UNKNOWN (YES?)"

                # ── PLEASE ────────────────────────────────────────────────────
                elif predicted_label == "please":
                    detected, path, xr, yr = detect_please_motion(sequence_array)
                    motion_text = f"Path: {path:.4f} | X: {xr:.3f} | Y: {yr:.3f}"
                    if not detected:
                        display_text = "UNKNOWN - PLEASE NEEDS CIRCULAR MOTION"
                    elif confidence >= CONFIDENCE_THRESHOLD:
                        display_text = "PLEASE"
                    else:
                        display_text = f"UNKNOWN (PLEASE?)"

                # ── NEED ──────────────────────────────────────────────────────
                elif predicted_label == "need":
                    detected, mot_score, both_ratio = detect_need_motion(sequence_array)
                    motion_text = f"Motion: {mot_score:.5f} | Both hands: {both_ratio*100:.0f}%"
                    if not detected:
                        display_text = "UNKNOWN - NEED REQUIRES BOTH HANDS + MOTION"
                    elif confidence >= CONFIDENCE_THRESHOLD:
                        display_text = "NEED"
                    else:
                        display_text = f"UNKNOWN (NEED?)"

                # ── THANK_YOU ────────────────────────────────────────────
                elif predicted_label == "thank_you":
                    detected, z_drop, z_range = detect_thankyou_motion(sequence_array)
                    motion_text = f"Z-drop: {z_drop:.4f} | Z-range: {z_range:.4f}"
                    if not detected:
                        display_text = "UNKNOWN - THANK YOU NEEDS FORWARD MOTION"
                    elif confidence >= CONFIDENCE_THRESHOLD:
                        display_text = "THANK_YOU"
                    else:
                        display_text = f"UNKNOWN (THANK_YOU?)"

                # ── HELP ─────────────────────────────────────────────────────
                elif predicted_label == "help":
                    detected, mot_score, both_ratio = detect_help_motion(sequence_array)
                    motion_text = f"Motion: {mot_score:.5f} | Both hands: {both_ratio*100:.0f}%"
                    if not detected:
                        display_text = "UNKNOWN - HELP REQUIRES BOTH HANDS + SLIGHT MOTION"
                    elif confidence >= CONFIDENCE_THRESHOLD:
                        display_text = "HELP"
                    else:
                        display_text = f"UNKNOWN (HELP?)"

                # ── All other signs ───────────────────────────────────────────
                else:
                    mot = float(np.mean(np.abs(sequence_array[:, BASE_FEATURE_SIZE:])))
                    motion_text = f"Motion score: {mot:.5f}"
                    if confidence >= CONFIDENCE_THRESHOLD:
                        display_text = predicted_label.upper()
                    else:
                        display_text = f"UNKNOWN ({predicted_label.upper()}?)"

                confidence_text  = f"Top confidence: {confidence * 100:.2f}%"
                probability_text = " | ".join(
                    f"{labels[i]}: {prediction[i]*100:.1f}%"
                    for i in range(len(labels))
                )

            height, width, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (width, 175), (0, 0, 0), -1)

            cv2.putText(frame, display_text,     (20, 35),  cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (0, 255, 0),   2)
            cv2.putText(frame, confidence_text,  (20, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, motion_text,      (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255),  2)
            cv2.putText(frame, probability_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
            cv2.putText(frame, hand_text,        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

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