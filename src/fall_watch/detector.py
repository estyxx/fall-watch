import logging
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

logger = logging.getLogger(__name__)

# COCO keypoint indices
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_HIP = 11
_RIGHT_HIP = 12


@dataclass(frozen=True)
class FrameAnalysis:
    person_on_floor: bool


def load_model() -> YOLO:
    """Load YOLOv8 nano pose model — downloads ~6MB on first run."""
    model = YOLO("yolov8n-pose.pt")
    logger.info("✅ AI model loaded")
    return model


def _keypoint(kps: np.ndarray, idx: int, min_conf: float = 0.3) -> np.ndarray | None:
    """Return (x, y) for a keypoint if confidence is high enough, else None."""
    kp = kps[idx]
    return kp[:2] if kp[2] >= min_conf else None


def _is_lying_down(kps: np.ndarray, frame_height: int) -> bool:
    """
    Heuristic: person is on the floor when their body keypoints are more
    spread horizontally than vertically, OR shoulders and hips are at a
    similar vertical level (flat body).
    """
    left_shoulder, right_shoulder, left_hip, right_hip = (
        _keypoint(kps, _LEFT_SHOULDER),
        _keypoint(kps, _RIGHT_SHOULDER),
        _keypoint(kps, _LEFT_HIP),
        _keypoint(kps, _RIGHT_HIP),
    )
    visible = [p for p in (left_shoulder, right_shoulder, left_hip, right_hip) if p is not None]

    if len(visible) < 2:
        return False

    y_coords = [p[1] for p in visible]
    x_coords = [p[0] for p in visible]
    vertical_spread = max(y_coords) - min(y_coords)
    horizontal_spread = max(x_coords) - min(x_coords)

    is_horizontal = horizontal_spread > vertical_spread * 1.5

    shoulder_ys = [p[1] for p in (left_shoulder, right_shoulder) if p is not None]
    hip_ys = [p[1] for p in (left_hip, right_hip) if p is not None]

    is_flat = (
        bool(shoulder_ys and hip_ys)
        and abs(float(np.mean(shoulder_ys)) - float(np.mean(hip_ys))) < frame_height * 0.15
    )

    return is_horizontal or is_flat


def _hip_in_zone(kps: np.ndarray, polygon: tuple[tuple[int, int], ...]) -> bool:
    """True if at least one visible hip keypoint is inside the polygon.

    Fails safe: returns False when no hip keypoints are visible, so an
    ambiguous frame never triggers a false positive in bed.
    """
    contour = np.array(polygon, dtype=np.int32)
    for idx in (_LEFT_HIP, _RIGHT_HIP):
        if (point := _keypoint(kps, idx)) is not None:
            if cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0:
                return True
    return False


def _keypoints_array(result: Results) -> np.ndarray:
    """Return all person keypoints as a numpy array, handling the Tensor | ndarray union."""
    assert result.keypoints is not None
    data = result.keypoints.data
    if isinstance(data, np.ndarray):
        return data
    return data.cpu().numpy()


def analyse_frame(
    model: YOLO,
    frame: np.ndarray,
    floor_roi: tuple[tuple[int, int], ...] | None = None,
) -> FrameAnalysis:
    """Analyse one frame and return signals derived from pose estimation.

    If `floor_roi` is provided, only detections whose hips fall inside the
    polygon count as on-floor — this excludes the bed and any off-floor zones.
    """
    results: list[Results] = model(frame, verbose=False)

    person_on_floor = any(
        _is_lying_down(person_kps, frame.shape[0])
        and (floor_roi is None or _hip_in_zone(person_kps, floor_roi))
        for result in results
        if result.keypoints is not None
        for person_kps in _keypoints_array(result)
    )
    return FrameAnalysis(person_on_floor=person_on_floor)
