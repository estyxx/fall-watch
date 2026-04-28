import os
import time
from datetime import datetime, timedelta

import cv2
from dotenv import load_dotenv

from fall_watch.detector import analyse_frame, load_model
from fall_watch.notifier import send_all_clear, send_fall_alert, send_startup

load_dotenv()

RTSP_URL = os.environ["RTSP_URL"]
FALL_THRESHOLD_MINUTES = float(os.getenv("FALL_THRESHOLD_MINUTES", "3"))
ALERT_COOLDOWN_MINUTES = float(os.getenv("ALERT_COOLDOWN_MINUTES", "15"))
FRAME_INTERVAL_SECONDS = 5


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
    _log("✅ Telegram test message sent")

    cap = _open_stream(RTSP_URL)
    _log("📷 Camera stream connected")

    on_floor_since: datetime | None = None
    alert_sent_at: datetime | None = None
    was_on_floor = False

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

        match (person_on_floor, was_on_floor):
            case (True, _):
                if on_floor_since is None:
                    on_floor_since = now
                    _log("⚠️  Person on floor — timer started")

                minutes_on_floor = (now - on_floor_since).total_seconds() / 60
                cooldown_ok = alert_sent_at is None or (
                    now - alert_sent_at > timedelta(minutes=ALERT_COOLDOWN_MINUTES)
                )

                if minutes_on_floor >= FALL_THRESHOLD_MINUTES and cooldown_ok:
                    _log(f"🚨 Alerting! On floor for {minutes_on_floor:.1f}min")
                    send_fall_alert(minutes_on_floor)
                    alert_sent_at = now

                was_on_floor = True

            case (False, True):
                _log("✅ Person got up")
                if alert_sent_at is not None:
                    send_all_clear()
                on_floor_since = None
                alert_sent_at = None
                was_on_floor = False

            case _:
                pass

        time.sleep(FRAME_INTERVAL_SECONDS)

    cap.release()


if __name__ == "__main__":
    main()
