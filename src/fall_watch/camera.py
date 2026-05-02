"""Camera I/O — background thread keeps the VideoCapture buffer drained.

OpenCV's FFmpeg backend queues ~150 frames between calls to cap.read() (at
30 fps x 5 s sleep = 150 frames buffered).  A naïve read-then-sleep loop falls
another 5 s behind real time on every cycle.  After a minute the monitor is
permanently watching the past.

FreshFrameCapture dedicates a daemon thread to reading frames as fast as the
camera produces them, keeping only the most recent one.  The main loop calls
read_latest() to get the freshest frame without ever blocking on the camera's
internal queue.
"""

import logging
import os
import threading
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Configure the FFmpeg backend for low-latency RTSP *before* any VideoCapture
# is opened.  The os.environ.setdefault call is a no-op when the variable is
# already present, so a user can still override from the environment or .env.
#
#   rtsp_transport;tcp  — reliable delivery over TCP instead of UDP
#   fflags;nobuffer     — disable the demuxer input buffer
#   flags;low_delay     — enable the decoder's low-delay mode
#   max_delay;0         — zero tolerance for buffered frames
_FFMPEG_LOW_LATENCY_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _FFMPEG_LOW_LATENCY_OPTIONS)


class FrameReader(Protocol):
    """Minimal interface for camera wrappers.

    main.py depends only on this abstraction — mirroring how notifier.Notifier
    decouples the watchers from TelegramNotifier.
    """

    @property
    def failed(self) -> bool:
        """True when the last read attempt failed (stream dead or disconnected)."""
        ...

    def read_latest(self) -> np.ndarray | None:
        """Return a copy of the most recent decoded frame, or None if none yet."""
        ...

    def release(self) -> None:
        """Stop the background reader and release the underlying VideoCapture."""
        ...


class FreshFrameCapture:
    """VideoCapture wrapper with a background reader thread.

    The background thread continuously calls cap.read() and overwrites a
    shared slot with each new frame.  read_latest() returns a copy of whatever
    is in that slot.  The main loop therefore always gets the freshest decoded
    frame regardless of how long it slept between calls.
    """

    def __init__(self, url: str, poll_interval: float = 0.01) -> None:
        """Open *url* and start the background reader.

        Args:
            url: RTSP (or any OpenCV-compatible) stream URL.
            poll_interval: Seconds to back off after a failed read before
                retrying.  In the happy path the reader spins without sleeping
                to drain the buffer as fast as possible.
        """
        self._url = url
        self._poll_interval = poll_interval
        self._cap: cv2.VideoCapture = _open_capture(url)
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._failed = False
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="camera-reader",
        )
        self._thread.start()
        logger.info("📷 Camera stream connected — background reader started")

    # ------------------------------------------------------------------
    # Background reader
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._latest_frame = frame
                    self._failed = False
            else:
                logger.warning(
                    "⚠️  Camera reader: read failed, backing off %.2fs", self._poll_interval
                )
                with self._lock:
                    self._failed = True
                # Sleep only on failure so we don't busy-spin while the stream
                # is down.  In the happy path we read without any delay to keep
                # draining the buffer.
                self._stop_event.wait(self._poll_interval)

    # ------------------------------------------------------------------
    # Public interface (satisfies FrameReader Protocol)
    # ------------------------------------------------------------------

    @property
    def failed(self) -> bool:
        """True when the background reader last reported a read failure."""
        with self._lock:
            return self._failed

    def read_latest(self) -> np.ndarray | None:
        """Return a copy of the most recent frame, or None if none received yet."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def release(self) -> None:
        """Signal the reader thread to stop and release the VideoCapture."""
        self._stop_event.set()
        self._thread.join(timeout=3.0)
        self._cap.release()
        logger.info("📷 Camera stream released")


# ------------------------------------------------------------------
# Module-level helper (not part of the public API)
# ------------------------------------------------------------------


def _open_capture(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera stream: {url}")
    return cap
