import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    rtsp_url: str
    telegram_token: str
    telegram_chat_id: str
    fall_threshold_minutes: float = 3.0
    alert_cooldown_minutes: float = 15.0
    frame_interval_seconds: int = 5
    not_on_floor_streak_max: int = 3

    @classmethod
    def load(cls) -> "Config":
        return cls(
            rtsp_url=os.environ["RTSP_URL"],
            telegram_token=os.environ["TELEGRAM_TOKEN"],
            telegram_chat_id=os.environ["TELEGRAM_CHAT_ID"],
            fall_threshold_minutes=float(os.getenv("FALL_THRESHOLD_MINUTES", "3")),
            alert_cooldown_minutes=float(os.getenv("ALERT_COOLDOWN_MINUTES", "15")),
            frame_interval_seconds=int(os.getenv("FRAME_INTERVAL_SECONDS", "5")),
            not_on_floor_streak_max=int(os.getenv("NOT_ON_FLOOR_STREAK_MAX", "3")),
        )
