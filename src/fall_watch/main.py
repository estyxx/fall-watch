import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from dotenv import load_dotenv

# Must run before local imports so Config.load() reads populated os.environ
load_dotenv()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from fall_watch.config import Config  # noqa: E402
from fall_watch.detector import analyse_frame, load_model  # noqa: E402
from fall_watch.notifier import (  # noqa: E402
    poll_commands,
    send_all_clear,
    send_fall_alert,
    send_startup,
    send_status_reply,
)


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


def _handle_commands(config: Config, update_offset: int, state: FallState) -> int:
    """Poll Telegram for commands and reply to any /status requests."""
    commands, new_offset = poll_commands(config, update_offset)
    for chat_id, cmd in commands:
        match cmd:
            case "/status":
                _log(f"📲 /status requested by chat {chat_id}")
                send_status_reply(config, chat_id, state.latest_frame, state.on_floor_since)
            case _:
                _log(f"⚙️  Unknown command '{cmd}' from chat {chat_id} — ignored")
    return new_offset


def main() -> None:
    _log("🟢 OcchioSuNonno starting...")
    config = Config.load()

    model = load_model()
    _log("✅ AI model loaded")

    send_startup(config)
    _log("✅ Startup message sent")

    cap = _open_stream(config.rtsp_url)
    _log("📷 Camera stream connected")

    state = FallState()
    update_offset = 0

    try:
        while True:
            update_offset = _handle_commands(config, update_offset, state)

            ret, frame = cap.read()

            if not ret:
                _log("⚠️  Lost camera connection, retrying in 10s...")
                cap.release()
                time.sleep(10)
                try:
                    cap = _open_stream(config.rtsp_url)
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
                    now - state.alert_sent_at > timedelta(minutes=config.alert_cooldown_minutes)
                )

                if minutes_on_floor >= config.fall_threshold_minutes and cooldown_ok:
                    _log(f"🚨 Alerting! On floor for {minutes_on_floor:.1f}min")
                    send_fall_alert(config, minutes_on_floor, frame)
                    state.alert_sent_at = now

                state.was_on_floor = True

            else:
                if state.was_on_floor:
                    state.not_on_floor_streak += 1
                    _log(
                        f"📊 Off floor streak: {state.not_on_floor_streak}/{config.not_on_floor_streak_max}"
                    )

                    if state.not_on_floor_streak >= config.not_on_floor_streak_max:
                        _log("✅ Person got up")
                        if state.alert_sent_at is not None:
                            send_all_clear(config, state.last_floor_frame)
                        state = FallState()

            time.sleep(config.frame_interval_seconds)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
