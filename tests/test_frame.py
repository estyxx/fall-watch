"""
Test YOLOv8 pose detection on a single static image.

Usage:
    uv run python tests/test_frame.py path/to/image.jpg

Shows:
- Detected keypoints overlaid on the image
- Bounding box: green = ok, red = ON FLOOR
- Floor polygon drawn in translucent green when FLOOR_ROI is set in .env
- Prints detection details to terminal
"""

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from fall_watch.config import parse_polygon
from fall_watch.detector import FrameAnalysis, PersonDetection, analyse_frame, load_model

logger = logging.getLogger(__name__)

_KP_NAMES = [
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
_HIGHLIGHT_KPS = {5, 6, 11, 12}  # shoulders + hips


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


def _render_floor_roi(display: np.ndarray, floor_roi: tuple[tuple[int, int], ...]) -> None:
    """Draw the floor polygon as a semi-transparent green fill with an outline."""
    contour = np.array(floor_roi, dtype=np.int32)
    overlay = display.copy()
    cv2.fillPoly(overlay, [contour], (0, 255, 100))
    cv2.addWeighted(overlay, 0.20, display, 0.80, 0, display)
    cv2.polylines(display, [contour], True, (0, 255, 100), 2)


def _render_person(display: np.ndarray, person: PersonDetection, index: int) -> None:
    """Draw bounding box, label, and keypoints for one detected person."""
    color = (0, 0, 255) if person.on_floor else (0, 200, 0)
    status = "⚠ ON FLOOR" if person.on_floor else "ok"
    x1, y1, x2, y2 = person.box

    cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        display,
        f"{status} {person.box_confidence:.0%}",
        (x1, y1 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )

    for j, kp in enumerate(person.keypoints):
        x, y, conf_kp = kp
        if conf_kp > 0.3:
            cv2.circle(display, (int(x), int(y)), 5, color, -1)
            if j in _HIGHLIGHT_KPS:
                cv2.circle(display, (int(x), int(y)), 9, color, 2)
                name = _KP_NAMES[j] if j < len(_KP_NAMES) else str(j)
                cv2.putText(
                    display,
                    name,
                    (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )


def main() -> None:
    load_dotenv()
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

    floor_roi = parse_polygon(os.getenv("FLOOR_ROI"))

    logger.info("📷 Image: %s (%dx%dpx)", image_path.name, frame.shape[1], frame.shape[0])
    if floor_roi is not None:
        logger.info("📐 Using FLOOR_ROI (%d points)", len(floor_roi))
    logger.info("🔍 Running YOLOv8 pose detection...")

    model = load_model()
    analysis: FrameAnalysis = analyse_frame(model, frame, floor_roi)
    display = frame.copy()

    if floor_roi is not None:
        _render_floor_roi(display, floor_roi)

    for i, person in enumerate(analysis.people, start=1):
        _render_person(display, person, index=i)
        logger.info(
            "  Person %d: %s (confidence: %.0f%%)",
            i,
            "⚠ ON FLOOR" if person.on_floor else "ok",
            person.box_confidence * 100,
        )

    if not analysis.people:
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
        people_on_floor = sum(p.on_floor for p in analysis.people)
        logger.info("✅ %d person(s) detected, %d on floor", len(analysis.people), people_on_floor)

    # Scale down for display if image is large
    h, w = display.shape[:2]
    max_dim = 900
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        display = cv2.resize(display, (int(w * scale), int(h * scale)))

    logger.info("Press Q or Esc in the image window to quit.")
    win = f"fall-watch — {image_path.name} (Q to quit)"
    cv2.imshow(win, display)
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
