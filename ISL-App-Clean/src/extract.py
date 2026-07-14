import json
import cv2
import numpy as np
import mediapipe as mp
import time
from pathlib import Path

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

# Cache directory — stores per-video .npy so re-runs skip already-processed videos
CACHE_DIR = PROCESSED_DATA_DIR / "cache"


def cache_path_for(video_path: Path) -> Path:
    size = video_path.stat().st_size
    label = video_path.parent.name
    key = f"{label}__{video_path.stem}__{size}"
    return CACHE_DIR / f"{key}.npy"


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
    if result.multi_hand_landmarks:
        detected_hands = sorted(
            result.multi_hand_landmarks,
            key=lambda hand: hand.landmark[0].x
        )
        for hand_landmarks in detected_hands[:2]:
            all_hands.append(normalize_hand(hand_landmarks.landmark))
    while len(all_hands) < 2:
        all_hands.append(np.zeros(HAND_FEATURE_SIZE, dtype=np.float32))
    return np.concatenate(all_hands)


def add_motion_features(base_sequence):
    motion_sequence = np.zeros_like(base_sequence, dtype=np.float32)
    motion_sequence[1:] = base_sequence[1:] - base_sequence[:-1]
    return np.concatenate([base_sequence, motion_sequence], axis=1).astype(np.float32)


def process_video(video_path, hands_detector):
    # Cache hit
    cp = cache_path_for(video_path)
    if cp.exists():
        try:
            sequence = np.load(str(cp))
            if sequence.shape == (SEQUENCE_LENGTH, FEATURE_SIZE):
                return sequence, True
        except Exception:
            pass

    # Extract from video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, False

    frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH).astype(int)
    base_frame_features = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()
        if not success:
            base_frame_features.append(np.zeros(BASE_FEATURE_SIZE, dtype=np.float32))
            continue
        if ROTATE_VIDEOS:
            frame = cv2.rotate(frame, ROTATE_CODE)
        base_features = extract_base_keypoints_from_frame(frame, hands_detector)
        base_frame_features.append(base_features)

    cap.release()

    base_sequence = np.array(base_frame_features, dtype=np.float32)
    if base_sequence.shape != (SEQUENCE_LENGTH, BASE_FEATURE_SIZE):
        return None, False

    final_sequence = add_motion_features(base_sequence)
    if final_sequence.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
        return None, False

    np.save(str(cp), final_sequence)
    return final_sequence, False


def main():
    start_time = time.time()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    class_folders = [f for f in RAW_DATA_DIR.iterdir() if f.is_dir()]
    labels = sorted([f.name for f in class_folders])

    if len(labels) < 2:
        print("Need at least 2 classes.")
        return

    print(f"Detected classes: {labels}")

    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    all_tasks = []
    for label_index, label in enumerate(labels):
        class_dir = RAW_DATA_DIR / label
        video_files = sorted([
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ])
        for vf in video_files:
            all_tasks.append((vf, label_index))

    total = len(all_tasks)
    already_cached = sum(1 for vf, _ in all_tasks if cache_path_for(vf).exists())
    to_extract = total - already_cached

    print(f"\nTotal videos   : {total}")
    print(f"Already cached : {already_cached} (will load instantly)")
    print(f"To extract     : {to_extract} (running MediaPipe on these)\n")

    X = []
    y = []
    processed = 0
    cache_hits = 0
    skipped = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands_detector:

        for i, (video_path, label_index) in enumerate(all_tasks, start=1):
            sequence, from_cache = process_video(video_path, hands_detector)

            if sequence is not None:
                X.append(sequence)
                y.append(label_index)
                processed += 1
                if from_cache:
                    cache_hits += 1
            else:
                skipped += 1
                print(f"  Skipped: {video_path.name}")

            if i % 20 == 0 or i == total:
                elapsed = time.time() - start_time
                fresh = processed - cache_hits
                rate = fresh / elapsed if elapsed > 0 and fresh > 0 else 0.1
                remaining_fresh = to_extract - fresh
                eta = remaining_fresh / rate if rate > 0 else 0
                print(
                    f"[{i:>4}/{total}] "
                    f"Done: {processed} | "
                    f"Cached: {cache_hits} | "
                    f"Fresh: {fresh} | "
                    f"Skipped: {skipped} | "
                    f"Elapsed: {elapsed:.0f}s | "
                    f"ETA: {eta:.0f}s"
                )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    if len(X) == 0:
        print("No data extracted.")
        return

    np.save(X_PATH, X)
    np.save(Y_PATH, y)

    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=4)

    total_time = time.time() - start_time
    print(f"\nDone in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"Labels  : {labels}")
    print(f"Processed: {processed} | Cached: {cache_hits} | Fresh: {processed - cache_hits} | Skipped: {skipped}")
    print(f"\nSaved X to      : {X_PATH}")
    print(f"Saved y to      : {Y_PATH}")
    print(f"Saved labels to : {LABELS_PATH}")
    print(f"Cache folder    : {CACHE_DIR}")


if __name__ == "__main__":
    main()