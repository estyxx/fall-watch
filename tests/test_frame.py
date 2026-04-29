"""
Test YOLOv8 pose detection on a single static image.

Usage:
    uv run python tests/test_frame.py path/to/image.jpg

Shows:
- Detected keypoints overlaid on the image
- Bounding box: green = ok, red = ON FLOOR
- Prints detection details to terminal
"""

import argparse
import logging
from pathlib import Path

import cv2

from fall_watch.detector import _is_lying_down, load_model

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test YOLOv8 pose detection on a single static image.",
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the image file to analyse",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    args = _parse_args()
    image_path: Path = args.image

    if not image_path.exists():
        logger.error("❌ File not found: %s", image_path)
        raise SystemExit(1)

    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error("❌ Could not read image: %s", image_path)
        raise SystemExit(1)

    logger.info("📷 Image: %s (%dx%dpx)", image_path.name, frame.shape[1], frame.shape[0])
    logger.info("🔍 Running YOLOv8 pose detection...")

    model = load_model()
    results = model(frame, verbose=False)
    display = frame.copy()

    people_found = 0
    people_on_floor = 0

    for result in results:
        if result.keypoints is None:
            continue

        keypoints_data = result.keypoints.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

        for i, person_kps in enumerate(keypoints_data):
            people_found += 1
            on_floor = _is_lying_down(person_kps, frame.shape[0])
            if on_floor:
                people_on_floor += 1

            color = (0, 0, 255) if on_floor else (0, 200, 0)
            status = "⚠ ON FLOOR" if on_floor else "ok"
            conf = f"{confidences[i]:.0%}" if i < len(confidences) else "?"

            logger.info("  Person %d: %s (confidence: %s)", people_found, status, conf)

            # Bounding box
            if i < len(boxes):
                x1, y1, x2, y2 = boxes[i].astype(int)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    display,
                    f"{status} {conf}",
                    (x1, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )

            # Keypoints with labels
            kp_names = [
                "nose",
                "l_eye",
                "r_eye",
                "l_ear",
                "r_ear",
                "l_shoulder",
                "r_shoulder",
                "l_elbow",
                "r_elbow",
                "l_wrist",
                "r_wrist",
                "l_hip",
                "r_hip",
                "l_knee",
                "r_knee",
                "l_ankle",
                "r_ankle",
            ]
            for j, kp in enumerate(person_kps):
                x, y, conf_kp = kp
                if conf_kp > 0.3:
                    cv2.circle(display, (int(x), int(y)), 5, color, -1)
                    if j in (5, 6, 11, 12):  # highlight shoulders + hips
                        cv2.circle(display, (int(x), int(y)), 9, color, 2)
                        name = kp_names[j] if j < len(kp_names) else str(j)
                        cv2.putText(
                            display,
                            name,
                            (int(x) + 6, int(y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            1,
                        )

    if people_found == 0:
        logger.info("  No people detected in this frame")
        cv2.putText(
            display,
            "No person detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 165, 255),
            2,
        )
    else:
        logger.info("✅ %d person(s) detected, %d on floor", people_found, people_on_floor)

    # Scale down for display if image is large
    h, w = display.shape[:2]
    max_dim = 900
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        display = cv2.resize(display, (int(w * scale), int(h * scale)))

    logger.info("Press Q or Esc in the image window to quit.")
    cv2.imshow(f"fall-watch — {image_path.name} (Q to quit)", display)
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # Q, q, or Esc
            break
        if (
            cv2.getWindowProperty(
                f"fall-watch — {image_path.name} (Q to quit)", cv2.WND_PROP_VISIBLE
            )
            < 1
        ):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
