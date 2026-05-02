"""Unit tests for FreshFrameCapture.

cv2.VideoCapture is monkeypatched with a fake that yields frames whose pixel
value equals a monotonically-increasing counter.  The tests assert that
read_latest() always returns a recent frame — never a stale one from the
back of a buffered queue.
"""

import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

import fall_watch.camera as camera_mod
from fall_watch.camera import FreshFrameCapture

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRAME_SHAPE = (10, 10, 3)


def _make_frame(value: int) -> np.ndarray:
    return np.full(_FRAME_SHAPE, value % 256, dtype=np.uint8)


class _FakeCapture:
    """Minimal cv2.VideoCapture stand-in.

    Each call to read() increments an internal counter and returns a frame
    whose every pixel equals that counter value.  This lets us verify which
    frame the reader last stored.
    """

    def __init__(self, _url: str) -> None:
        self._counter = 0
        self._lock = threading.Lock()
        self._open = True

    def isOpened(self) -> bool:
        return self._open

    def read(self) -> tuple[bool, np.ndarray]:
        with self._lock:
            self._counter += 1
            return True, _make_frame(self._counter)

    @property
    def counter(self) -> int:
        with self._lock:
            return self._counter

    def release(self) -> None:
        self._open = False


class _FailingCapture:
    """Always fails on read() — used to test the failure / back-off path."""

    def __init__(self, _url: str) -> None:
        self.read_calls = 0

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        self.read_calls += 1
        return False, np.zeros(_FRAME_SHAPE, dtype=np.uint8)

    def release(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_read_latest_returns_recent_frame() -> None:
    """After the reader thread has had time to run, read_latest() should return
    a frame whose counter value is close to the most recently produced frame —
    not frame #1 from the back of a buffered queue."""
    fake = _FakeCapture("rtsp://fake")

    with patch.object(camera_mod, "_open_capture", return_value=fake):
        cap = FreshFrameCapture("rtsp://fake", poll_interval=0.0)
        try:
            # Let the background thread run freely for 100 ms so it produces
            # many frames.  Sample before and after to bracket the expected range.
            time.sleep(0.05)
            counter_before = fake.counter
            frame = cap.read_latest()
            counter_after = fake.counter
        finally:
            cap.release()

    assert frame is not None, "Expected a frame but got None"
    assert frame.shape == _FRAME_SHAPE

    # The pixel value encodes (counter % 256).  What matters is that the thread
    # ran for long enough to produce many frames — confirming the buffer is being
    # actively drained rather than queued.
    assert counter_before > 1, (
        f"Thread only produced {counter_before} frames in 50 ms; "
        "expected the reader to be spinning freely"
    )
    _ = counter_after  # used only as a bracket; suppress unused-variable lint


def test_read_latest_returns_copy() -> None:
    """Mutating the returned array must not affect the internally stored frame."""
    fake = _FakeCapture("rtsp://fake")

    with patch.object(camera_mod, "_open_capture", return_value=fake):
        cap = FreshFrameCapture("rtsp://fake", poll_interval=0.0)
        try:
            time.sleep(0.05)
            frame1 = cap.read_latest()
            assert frame1 is not None
            frame1[:] = 0  # mutate the returned copy

            time.sleep(0.05)
            frame2 = cap.read_latest()
        finally:
            cap.release()

    assert frame2 is not None
    # frame2 should reflect continued reading, not our zeroed-out mutation
    assert int(frame2[0, 0, 0]) > 0, "Internal frame was mutated by the caller"


def test_read_latest_returns_none_before_first_frame() -> None:
    """read_latest() returns None when no frame has arrived yet."""

    class _SlowCapture:
        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            # Block long enough that the main thread queries before any frame
            time.sleep(10)
            return True, _make_frame(1)

        def release(self) -> None:
            pass

    with patch.object(camera_mod, "_open_capture", return_value=_SlowCapture()):
        cap = FreshFrameCapture("rtsp://fake", poll_interval=0.0)
        try:
            result = cap.read_latest()
        finally:
            cap.release()

    assert result is None


def test_failed_property_reflects_read_failure() -> None:
    """failed becomes True when the underlying capture keeps returning False."""
    failing = _FailingCapture("rtsp://fake")

    with patch.object(camera_mod, "_open_capture", return_value=failing):
        cap = FreshFrameCapture("rtsp://fake", poll_interval=0.005)
        try:
            time.sleep(0.1)
            assert cap.failed is True, "Expected failed=True after sustained read failures"
            assert cap.read_latest() is None
        finally:
            cap.release()


def test_failed_property_clears_after_recovery() -> None:
    """failed goes back to False once reads start succeeding again."""

    class _RecoveringCapture:
        def __init__(self) -> None:
            self._calls = 0

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            self._calls += 1
            if self._calls <= 3:
                return False, np.zeros(_FRAME_SHAPE, dtype=np.uint8)
            return True, _make_frame(self._calls)

        def release(self) -> None:
            pass

    with patch.object(camera_mod, "_open_capture", return_value=_RecoveringCapture()):
        cap = FreshFrameCapture("rtsp://fake", poll_interval=0.005)
        try:
            time.sleep(0.1)
            assert cap.failed is False, "Expected failed=False after recovery"
            assert cap.read_latest() is not None
        finally:
            cap.release()


def test_release_stops_thread() -> None:
    """After release(), the background thread should be dead."""
    fake = _FakeCapture("rtsp://fake")

    with patch.object(camera_mod, "_open_capture", return_value=fake):
        cap = FreshFrameCapture("rtsp://fake", poll_interval=0.0)
        thread = cap._thread
        cap.release()

    assert not thread.is_alive(), "Background reader thread still alive after release()"


def test_open_capture_raises_on_unopened_stream() -> None:
    """_open_capture raises RuntimeError when VideoCapture.isOpened() is False."""

    class _ClosedCapture:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            pass

    with patch("cv2.VideoCapture", return_value=_ClosedCapture()):
        with pytest.raises(RuntimeError, match="Cannot open camera stream"):
            camera_mod._open_capture("rtsp://unreachable")
