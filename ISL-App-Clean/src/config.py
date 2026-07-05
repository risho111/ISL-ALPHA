from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "model"

X_PATH = PROCESSED_DATA_DIR / "X.npy"
Y_PATH = PROCESSED_DATA_DIR / "y.npy"
LABELS_PATH = PROCESSED_DATA_DIR / "labels.json"

MODEL_PATH = MODEL_DIR / "isl_model.h5"
MODEL_LABELS_PATH = MODEL_DIR / "labels.json"

SEQUENCE_LENGTH = 30

NUM_HANDS = 2
NUM_LANDMARKS = 21
NUM_COORDS = 3

# One hand:
# 21 landmarks × 3 normalized coordinates = 63
# raw wrist position x, y, z = 3
# total per hand = 66
HAND_FEATURE_SIZE = (NUM_LANDMARKS * NUM_COORDS) + 3

# Two hands:
# 66 × 2 = 132
BASE_FEATURE_SIZE = NUM_HANDS * HAND_FEATURE_SIZE

# Final input:
# base features 132 + motion features 132 = 264
FEATURE_SIZE = BASE_FEATURE_SIZE * 2

MIN_DETECTION_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3

CONFIDENCE_THRESHOLD = 0.70

# Used in live_test.py to stop static fist from being accepted as YES
YES_MOTION_THRESHOLD = 0.006
