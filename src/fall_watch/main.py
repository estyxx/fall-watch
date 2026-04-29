import logging
import logging.config
import os
import time
from datetime import datetime
from datetime import time as dt_time

from dotenv import load_dotenv

# Must run before local imports so Config.load() reads populated os.environ
load_dotenv()

import cv2  # noqa: E402

from fall_watch.config import Config  # noqa: E402
from fall_watch.detector import FrameAnalysis, analyse_frame, load_model  # noqa: E402
from fall_watch.fall_watcher import FallWatcher  # noqa: E402
from fall_watch.notifier import Notifier, TelegramNotifier  # noqa: E402

logger = logging.getLogger(__name__)


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


def _handle_commands(notifier: Notifier, watcher: FallWatcher, offset: int) -> int:
    """Poll Telegram for commands and reply to any /status requests."""
    commands, new_offset = notifier.poll_commands(offset)
    for chat_id, cmd in commands:
        match cmd:
            case "/status":
                logger.info("📲 /status requested by chat %s", chat_id)
                watcher.handle_status_request(chat_id)
            case _:
                logger.warning("⚙️  Unknown command '%s' from chat %s — ignored", cmd, chat_id)
    return new_offset


def main() -> None:
    _setup_logging()
    logger.info("🟢 OcchioSuNonno starting...")

    config = Config.load()
    model = load_model()
    notifier = TelegramNotifier(config)
    watcher = FallWatcher(config, notifier)
    notifier.send_startup()
    cap = _open_stream(config.rtsp_url)

    update_offset = 0

    try:
        while True:
            update_offset = _handle_commands(notifier, watcher, update_offset)

            ret, frame = cap.read()
            if not ret:
                cap = _reconnect(config, cap)
                continue

            now = datetime.now()
            analysis: FrameAnalysis = analyse_frame(
                model, frame, floor_roi=config.floor_roi, bed_polygon=config.bed_roi
            )
            watcher.observe(analysis.person_on_floor, frame, now)
            if analysis.person_climbing_out:
                logger.info("🪜 Climb-out posture detected (no alert yet — wiring lands next)")

            time.sleep(config.frame_interval_seconds)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
