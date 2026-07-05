import json
import cv2
import numpy as np
import mediapipe as mp

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    X_PATH,
    Y_PATH,
    LABELS_PATH,
    SEQUENCE_LENGTH,
    FEATURE_SIZE,
    BASE_FEATURE_SIZE,
    HAND_FEATURE_SIZE,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)


mp_hands = mp.solutions.hands

ROTATE_VIDEOS = True
ROTATE_CODE = cv2.ROTATE_90_CLOCKWISE


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

    if result.multi_hand_landmarks:
        detected_hands = sorted(
            result.multi_hand_landmarks,
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

    return base_features


def add_motion_features(base_sequence):
    """
    base_sequence shape: (30, 132)

    motion = current frame - previous frame

    final sequence shape:
    base features 132 + motion features 132 = 264
    """

    motion_sequence = np.zeros_like(base_sequence, dtype=np.float32)

    motion_sequence[1:] = base_sequence[1:] - base_sequence[:-1]

    final_sequence = np.concatenate(
        [base_sequence, motion_sequence],
        axis=1
    )

    return final_sequence.astype(np.float32)


def process_video(video_path, hands_detector):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"No frames found in video: {video_path}")
        cap.release()
        return None

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    ).astype(int)

    base_frame_features = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

        success, frame = cap.read()

        if not success:
            base_frame_features.append(
                np.zeros(BASE_FEATURE_SIZE, dtype=np.float32)
            )
            continue

        if ROTATE_VIDEOS:
            frame = cv2.rotate(frame, ROTATE_CODE)

        base_features = extract_base_keypoints_from_frame(
            frame,
            hands_detector
        )

        base_frame_features.append(base_features)

    cap.release()

    base_sequence = np.array(base_frame_features, dtype=np.float32)

    if base_sequence.shape != (SEQUENCE_LENGTH, BASE_FEATURE_SIZE):
        print(f"Bad base sequence shape for {video_path}: {base_sequence.shape}")
        return None

    final_sequence = add_motion_features(base_sequence)

    if final_sequence.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
        print(f"Bad final sequence shape for {video_path}: {final_sequence.shape}")
        return None

    return final_sequence


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    class_folders = [
        folder for folder in RAW_DATA_DIR.iterdir()
        if folder.is_dir()
    ]

    labels = sorted([folder.name for folder in class_folders])

    if len(labels) < 2:
        print("You need at least 2 classes.")
        return

    print("Detected classes:")
    for i, label in enumerate(labels):
        print(f"{i}: {label}")

    X = []
    y = []

    class_counts = {}

    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands_detector:

        for label_index, label in enumerate(labels):
            class_dir = RAW_DATA_DIR / label

            video_files = [
                file for file in class_dir.iterdir()
                if file.is_file() and file.suffix.lower() in video_extensions
            ]

            video_files = sorted(video_files)

            print(f"\nProcessing class: {label}")
            print(f"Videos found: {len(video_files)}")

            processed_count = 0

            if len(video_files) == 0:
                print(f"No videos found in {class_dir}")
                class_counts[label] = 0
                continue

            for video_path in video_files:
                sequence = process_video(video_path, hands_detector)

                if sequence is None:
                    print(f"Skipped: {video_path.name}")
                    continue

                X.append(sequence)
                y.append(label_index)

                processed_count += 1
                print(f"Processed: {video_path.name}")

            class_counts[label] = processed_count

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    if len(X) == 0:
        print("No data extracted.")
        return

    np.save(X_PATH, X)
    np.save(Y_PATH, y)

    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=4)

    print("\nExtraction completed.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Labels: {labels}")

    print("\nClass counts:")
    for label, count in class_counts.items():
        print(f"{label}: {count}")

    print(f"\nSaved X to: {X_PATH}")
    print(f"Saved y to: {Y_PATH}")
    print(f"Saved labels to: {LABELS_PATH}")


if __name__ == "__main__":
    main()