import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import cv2
import numpy as np
from dotenv import load_dotenv

from fall_watch.detector import analyse_frame, load_model
from fall_watch.notifier import (
    poll_commands,
    send_all_clear,
    send_fall_alert,
    send_startup,
    send_status_reply,
)

load_dotenv()

RTSP_URL = os.environ["RTSP_URL"]
FALL_THRESHOLD_MINUTES = float(os.getenv("FALL_THRESHOLD_MINUTES", "3"))
ALERT_COOLDOWN_MINUTES = float(os.getenv("ALERT_COOLDOWN_MINUTES", "15"))
FRAME_INTERVAL_SECONDS = 5
NOT_ON_FLOOR_STREAK_MAX = 3  # consecutive "ok" frames before declaring all clear


@dataclass
class FallState:
    on_floor_since: datetime | None = None
    alert_sent_at: datetime | None = None
    last_floor_frame: np.ndarray | None = field(default=None, repr=False)
    latest_frame: np.ndarray | None = field(default=None, repr=False)
    was_on_floor: bool = False
    not_on_floor_streak: int = 0


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _open_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera stream: {url}")
    return cap


def _handle_commands(update_offset: int, state: FallState) -> int:
    """Poll Telegram for commands and reply to any /status requests."""
    commands, new_offset = poll_commands(update_offset)
    for chat_id, cmd in commands:
        match cmd:
            case "/status":
                _log(f"📲 /status requested by chat {chat_id}")
                send_status_reply(chat_id, state.latest_frame, state.on_floor_since)
            case _:
                _log(f"⚙️  Unknown command '{cmd}' from chat {chat_id} — ignored")
    return new_offset


def main() -> None:
    _log("🟢 OcchioSuNonno starting...")
    model = load_model()
    _log("✅ AI model loaded")

    send_startup()
    _log("✅ Startup message sent")

    cap = _open_stream(RTSP_URL)
    _log("📷 Camera stream connected")

    state = FallState()
    update_offset = 0

    try:
        while True:
            update_offset = _handle_commands(update_offset, state)

            ret, frame = cap.read()

            if not ret:
                _log("⚠️  Lost camera connection, retrying in 10s...")
                cap.release()
                time.sleep(10)
                try:
                    cap = _open_stream(RTSP_URL)
                except RuntimeError as e:
                    _log(f"❌ Reconnect failed: {e} — will retry")
                continue

            state.latest_frame = frame
            now = datetime.now()
            person_on_floor = analyse_frame(model, frame)

            if person_on_floor:
                state.not_on_floor_streak = 0

                if state.on_floor_since is None:
                    state.on_floor_since = now
                    _log("⚠️  Person on floor — timer started")

                state.last_floor_frame = frame.copy()
                minutes_on_floor = (now - state.on_floor_since).total_seconds() / 60
                cooldown_ok = state.alert_sent_at is None or (
                    now - state.alert_sent_at > timedelta(minutes=ALERT_COOLDOWN_MINUTES)
                )

                if minutes_on_floor >= FALL_THRESHOLD_MINUTES and cooldown_ok:
                    _log(f"🚨 Alerting! On floor for {minutes_on_floor:.1f}min")
                    send_fall_alert(minutes_on_floor, frame)
                    state.alert_sent_at = now

                state.was_on_floor = True

            else:
                if state.was_on_floor:
                    state.not_on_floor_streak += 1
                    _log(
                        f"📊 Off floor streak: {state.not_on_floor_streak}/{NOT_ON_FLOOR_STREAK_MAX}"
                    )

                    if state.not_on_floor_streak >= NOT_ON_FLOOR_STREAK_MAX:
                        _log("✅ Person got up")
                        if state.alert_sent_at is not None:
                            send_all_clear(state.last_floor_frame)
                        state = FallState()

            time.sleep(FRAME_INTERVAL_SECONDS)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
