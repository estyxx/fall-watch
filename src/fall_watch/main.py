import logging
import logging.config
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from datetime import time as dt_time

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

logger = logging.getLogger(__name__)


@dataclass
class FallState:
    on_floor_since: datetime | None = None
    alert_sent_at: datetime | None = None
    last_floor_frame: np.ndarray | None = field(default=None, repr=False)
    latest_frame: np.ndarray | None = field(default=None, repr=False)
    was_on_floor: bool = False
    not_on_floor_streak: int = 0


def _setup_logging() -> None:
    log_file = os.getenv("LOG_FILE", "fall-watch.log")
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "console": {"format": "[%(asctime)s] %(message)s", "datefmt": "%H:%M:%S"},
                "file": {
                    "format": "[%(asctime)s] %(levelname)s %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {"class": "logging.StreamHandler", "formatter": "console"},
                "file": {
                    "class": "logging.handlers.TimedRotatingFileHandler",
                    "filename": log_file,
                    "when": "D",  # daily, time set by atTime
                    "atTime": dt_time(20, 0),  # rotate at 20:00
                    "backupCount": 7,  # keep one week of daily logs
                    "encoding": "utf-8",
                    "formatter": "file",
                },
            },
            "root": {"level": "INFO", "handlers": ["console", "file"]},
        }
    )
    logger.info("📝 Logging to file: %s", log_file)


def _open_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera stream: {url}")
    logger.info("📷 Camera stream connected")
    return cap


def _reconnect(config: Config, cap: cv2.VideoCapture) -> cv2.VideoCapture:
    logger.warning("⚠️  Lost camera connection, retrying in 10s...")
    cap.release()
    time.sleep(10)
    try:
        return _open_stream(config.rtsp_url)
    except RuntimeError as e:
        logger.error("❌ Reconnect failed: %s — will retry", e)
        return cap


def _handle_commands(config: Config, update_offset: int, state: FallState) -> int:
    """Poll Telegram for commands and reply to any /status requests."""
    commands, new_offset = poll_commands(config, update_offset)
    for chat_id, cmd in commands:
        match cmd:
            case "/status":
                logger.info("📲 /status requested by chat %s", chat_id)
                send_status_reply(config, chat_id, state.latest_frame, state.on_floor_since)
            case _:
                logger.warning("⚙️  Unknown command '%s' from chat %s — ignored", cmd, chat_id)
    return new_offset


def _on_floor(config: Config, state: FallState, frame: np.ndarray, now: datetime) -> None:
    state.not_on_floor_streak = 0

    if state.on_floor_since is None:
        state.on_floor_since = now
        logger.warning("⚠️  Person on floor — timer started")

    state.last_floor_frame = frame.copy()
    minutes_on_floor = (now - state.on_floor_since).total_seconds() / 60
    cooldown_ok = state.alert_sent_at is None or (
        now - state.alert_sent_at > timedelta(minutes=config.alert_cooldown_minutes)
    )

    if minutes_on_floor >= config.fall_threshold_minutes and cooldown_ok:
        logger.warning("🚨 Alerting! On floor for %.1fmin", minutes_on_floor)
        send_fall_alert(config, minutes_on_floor, frame)
        state.alert_sent_at = now

    state.was_on_floor = True


def _off_floor(config: Config, state: FallState) -> FallState:
    if not state.was_on_floor:
        return state

    state.not_on_floor_streak += 1
    logger.info(
        "📊 Off floor streak: %d/%d", state.not_on_floor_streak, config.not_on_floor_streak_max
    )

    if state.not_on_floor_streak >= config.not_on_floor_streak_max:
        logger.info("✅ Person got up")
        if state.alert_sent_at is not None:
            send_all_clear(config, state.last_floor_frame)
        return FallState()

    return state


def main() -> None:
    _setup_logging()
    logger.info("🟢 OcchioSuNonno starting...")

    config = Config.load()
    model = load_model()
    send_startup(config)
    cap = _open_stream(config.rtsp_url)

    state = FallState()
    update_offset = 0

    try:
        while True:
            update_offset = _handle_commands(config, update_offset, state)

            ret, frame = cap.read()
            if not ret:
                cap = _reconnect(config, cap)
                continue

            state.latest_frame = frame
            now = datetime.now()

            if analyse_frame(model, frame):
                _on_floor(config, state, frame, now)
            else:
                state = _off_floor(config, state)

            time.sleep(config.frame_interval_seconds)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
