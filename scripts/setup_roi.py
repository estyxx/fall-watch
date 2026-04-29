"""
Interactive ROI (Region of Interest) setup tool.

Click 4 points to define the floor or bed zone visible from the camera.

Usage:
    uv run python scripts/setup_roi.py image.jpg                  # captures FLOOR_ROI (default)
    uv run python scripts/setup_roi.py image.jpg --zone bed       # captures BED_ROI

Controls:
    - On startup: any existing ROI for the selected zone is loaded and shown
    - Left click: add a point (up to 4)
    - R: reset points
    - Enter: confirm and save to .env
    - Q: quit without saving
"""

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from fall_watch.config import parse_polygon

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively define a floor or bed ROI by clicking 4 points on an image."
    )
    parser.add_argument("image", type=Path, help="Path to a reference frame from the camera")
    parser.add_argument(
        "--zone",
        choices=["floor", "bed"],
        default="floor",
        help="Which ROI to capture (default: floor, preserves backwards compat).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    args = _parse_args()
    image_path: Path = args.image
    env_var = f"{args.zone.upper()}_ROI"

    if not image_path.exists():
        logger.error("❌ File not found: %s", image_path)
        raise SystemExit(1)

    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error("❌ Could not read image: %s", image_path)
        raise SystemExit(1)

    # Scale down for display if image is large
    h, w = frame.shape[:2]
    max_dim = 900
    scale = min(max_dim / max(h, w), 1.0)
    display_w, display_h = int(w * scale), int(h * scale)
    display_base = cv2.resize(frame, (display_w, display_h))

    points: list[tuple[int, int]] = []
    try:
        saved = parse_polygon(os.getenv(env_var))
        if saved is not None:
            points = [(int(x * scale), int(y * scale)) for x, y in saved]
            logger.info(
                "📐 Loaded existing %s (%d points) — R to reset, Q to keep unchanged",
                env_var,
                len(points),
            )
    except ValueError as e:
        logger.warning("⚠️  Ignoring malformed %s in .env: %s", env_var, e)

    def draw() -> np.ndarray:
        img = display_base.copy()

        # Instructions overlay
        instructions = [
            f"Click 4 points to define the {args.zone.upper()} zone",
            "R = reset | Enter = save | Q = quit",
            f"Points: {len(points)}/4",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(img, text, (10, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(
                img, text, (10, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1
            )

        # Draw polygon
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(img, points[i], points[i + 1], (0, 255, 100), 2)
            if len(points) == 4:
                cv2.line(img, points[3], points[0], (0, 255, 100), 2)
                overlay = img.copy()
                cv2.fillPoly(overlay, [np.array(points)], (0, 255, 100))
                cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
                cv2.putText(
                    img,
                    "✓ Press Enter to save",
                    (10, display_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 100),
                    2,
                )

        # Draw points
        for i, pt in enumerate(points):
            cv2.circle(img, pt, 7, (0, 255, 100), -1)
            cv2.circle(img, pt, 7, (0, 0, 0), 1)
            cv2.putText(
                img,
                str(i + 1),
                (pt[0] + 10, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 100),
                2,
            )

        return img

    def on_click(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.imshow(win, draw())

    win = f"fall-watch — ROI setup ({args.zone} zone)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    cv2.imshow(win, draw())

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"):
            points.clear()
            cv2.imshow(win, draw())

        elif key in (13, 10):  # Enter
            if len(points) != 4:
                logger.warning("⚠️  Need exactly 4 points, got %d", len(points))
                continue

            # Scale back to original image coordinates
            real_points = [(int(x / scale), int(y / scale)) for x, y in points]
            roi_str = ";".join(f"{x},{y}" for x, y in real_points)

            # Write to .env
            env_path = Path(".env")
            if env_path.exists():
                existing = env_path.read_text()
                if env_var in existing:
                    lines = [
                        f"{env_var}={roi_str}" if line.startswith(f"{env_var}=") else line
                        for line in existing.splitlines()
                    ]
                    env_path.write_text("\n".join(lines) + "\n")
                else:
                    with env_path.open("a") as f:
                        f.write(f"\n{env_var}={roi_str}\n")
            else:
                env_path.write_text(f"{env_var}={roi_str}\n")

            logger.info("✅ ROI saved to .env: %s=%s", env_var, roi_str)
            logger.info("   Points (original resolution): %s", real_points)
            break

        elif key == ord("q"):
            logger.info("👋 Quit without saving")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
