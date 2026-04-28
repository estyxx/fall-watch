import os
from datetime import datetime

import cv2
import numpy as np
import requests


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _token_and_chat() -> tuple[str, str]:
    return os.environ["TELEGRAM_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]


def _send_text(text: str) -> bool:
    token, chat_id = _token_and_chat()
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"[{_now()}] ❌ Telegram error: {e}")
        return False


def _send_photo(frame: np.ndarray | None, caption: str) -> bool:
    """Encode frame as JPEG and send it to Telegram with a caption."""
    if frame is None:
        return _send_text(caption)

    token, chat_id = _token_and_chat()
    try:
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return _send_text(caption)  # fallback to text if encoding fails

        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
            files={"photo": ("alert.jpg", buffer.tobytes(), "image/jpeg")},
            timeout=15,
        )
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"[{_now()}] ❌ Telegram photo error: {e}")
        return _send_text(caption)  # fallback to text on failure


def send_fall_alert(minutes_on_floor: float, frame: np.ndarray | None = None) -> bool:
    caption = (
        f"🚨 <b>ATTENZIONE — Nonno a terra!</b>\n\n"
        f"A terra da <b>{minutes_on_floor:.0f} minuti</b>. Controllare subito!\n\n"
        f"🕐 {_now()}"
    )
    return _send_photo(frame, caption)


def send_all_clear(frame: np.ndarray | None = None) -> bool:
    caption = f"✅ <b>Tutto ok</b> — il nonno si è rialzato.\n🕐 {_now()}"
    return _send_photo(frame, caption) if frame is not None else _send_text(caption)


def send_startup() -> bool:
    return _send_text("👋 <b>OcchioSuNonno attivo!</b>\nIl sistema di monitoraggio è operativo. 🟢")
