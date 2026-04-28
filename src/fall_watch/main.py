import os
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
from dotenv import load_dotenv
from nonno_watch.detector import analyse_frame, load_model
from nonno_watch.notifier import send_all_clear, send_fall_alert, send_startup

load_dotenv()

RTSP_URL = os.environ["RTSP_URL"]
FALL_THRESHOLD_MINUTES = float(os.getenv("FALL_THRESHOLD_MINUTES", "3"))
ALERT_COOLDOWN_MINUTES = float(os.getenv("ALERT_COOLDOWN_MINUTES", "15"))
FRAME_INTERVAL_SECONDS = 5
NOT_ON_FLOOR_STREAK_MAX = 3  # consecutive "ok" frames before declaring all clear


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _open_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera stream: {url}")
    return cap


def main() -> None:
    _log("🟢 OcchioSuNonno starting...")
    model = load_model()
    _log("✅ AI model loaded")

    send_startup()
    _log("✅ Startup message sent")

    cap = _open_stream(RTSP_URL)
    _log("📷 Camera stream connected")

    on_floor_since: datetime | None = None
    alert_sent_at: datetime | None = None
    last_floor_frame: np.ndarray | None = None
    was_on_floor = False
    not_on_floor_streak = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            _log("⚠️  Lost camera connection, retrying in 10s...")
            cap.release()
            time.sleep(10)
            cap = _open_stream(RTSP_URL)
            continue

        now = datetime.now()
        person_on_floor = analyse_frame(model, frame)

        if person_on_floor:
            not_on_floor_streak = 0

            if on_floor_since is None:
                on_floor_since = now
                _log("⚠️  Person on floor — timer started")

            last_floor_frame = frame.copy()
            minutes_on_floor = (now - on_floor_since).total_seconds() / 60
            cooldown_ok = alert_sent_at is None or (
                now - alert_sent_at > timedelta(minutes=ALERT_COOLDOWN_MINUTES)
            )

            if minutes_on_floor >= FALL_THRESHOLD_MINUTES and cooldown_ok:
                _log(f"🚨 Alerting! On floor for {minutes_on_floor:.1f}min")
                send_fall_alert(minutes_on_floor, frame)
                alert_sent_at = now

            was_on_floor = True

        else:
            if was_on_floor:
                not_on_floor_streak += 1
                _log(f"📊 Off floor streak: {not_on_floor_streak}/{NOT_ON_FLOOR_STREAK_MAX}")

                if not_on_floor_streak >= NOT_ON_FLOOR_STREAK_MAX:
                    _log("✅ Person got up")
                    if alert_sent_at is not None:
                        send_all_clear(last_floor_frame)
                    on_floor_since = None
                    alert_sent_at = None
                    last_floor_frame = None
                    was_on_floor = False
                    not_on_floor_streak = 0

        time.sleep(FRAME_INTERVAL_SECONDS)

    cap.release()


if __name__ == "__main__":
    main()
