import cv2
from pathlib import Path
import sys

# Usage:
# python src/check_video.py yes
# python src/check_video.py idle
# python src/check_video.py water
# python src/check_video.py i

if len(sys.argv) > 1:
    CLASS_NAME = sys.argv[1]
else:
    CLASS_NAME = "yes"

ROTATE_VIDEOS = True
ROTATE_CODE = cv2.ROTATE_90_CLOCKWISE

WINDOW_NAME = f"Check {CLASS_NAME} Videos - N next, Q quit"

video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

class_dir = Path("data") / "raw" / CLASS_NAME

if not class_dir.exists():
    print(f"Folder not found: {class_dir}")
    exit()

video_files = [
    file for file in class_dir.iterdir()
    if file.is_file() and file.suffix.lower() in video_extensions
]

video_files = sorted(video_files)

if len(video_files) == 0:
    print(f"No videos found in {class_dir}")
    exit()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 700, 500)

for video_number, video_path in enumerate(video_files, start=1):
    print(f"Opening {video_number}/{len(video_files)}: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path.name}")
        continue

    while True:
        success, frame = cap.read()

        if not success:
            break

        if ROTATE_VIDEOS:
            frame = cv2.rotate(frame, ROTATE_CODE)

        cv2.putText(
            frame,
            f"{CLASS_NAME.upper()} | {video_number}/{len(video_files)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            video_path.name,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "N = next video | Q = quit",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("n"):
            break

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()

cv2.destroyAllWindows()
#heeeehaaaa

