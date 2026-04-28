import os
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import requests


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _token_and_chat() -> tuple[str, str]:
    return os.environ["TELEGRAM_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]


def _send_text(text: str, to_chat_id: str | None = None) -> bool:
    token, default_chat_id = _token_and_chat()
    chat_id = to_chat_id or default_chat_id
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


def _send_photo(
    frame: np.ndarray | None,
    caption: str,
    to_chat_id: str | None = None,
) -> bool:
    """Encode frame as JPEG and send it to Telegram with a caption."""
    if frame is None:
        return _send_text(caption, to_chat_id)

    token, default_chat_id = _token_and_chat()
    chat_id = to_chat_id or default_chat_id
    try:
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return _send_text(caption, to_chat_id)

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
        return _send_text(caption, to_chat_id)


def poll_commands(offset: int) -> tuple[list[tuple[str, str]], int]:
    """
    Poll Telegram getUpdates (non-blocking, timeout=0).

    Returns a list of (chat_id, command) pairs for any bot commands found,
    and the next offset to pass on the following call to avoid reprocessing.
    """
    token = os.environ["TELEGRAM_TOKEN"]
    try:
        # POST is accepted by the Bot API and avoids params serialisation issues
        r = requests.post(
            f"https://api.telegram.org/bot{token}/getUpdates",
            json={"offset": offset, "timeout": 0, "allowed_updates": ["message"]},
            timeout=5,
        )
        r.raise_for_status()
        data: Any = r.json()  # untyped Bot API response
    except requests.RequestException as e:
        print(f"[{_now()}] ❌ Telegram poll error: {e}")
        return [], offset

    commands: list[tuple[str, str]] = []
    new_offset = offset

    for update in data.get("result", []):
        update_id: int = update["update_id"]
        new_offset = max(new_offset, update_id + 1)

        msg: Any = update.get("message", {})
        text: str = str(msg.get("text", ""))
        chat_id: str = str(msg.get("chat", {}).get("id", ""))

        if text.startswith("/") and chat_id:
            # Strip bot @mention: /status@MyBot → /status
            cmd = text.split("@")[0].split()[0]
            commands.append((chat_id, cmd))

    return commands, new_offset


def send_status_reply(
    to_chat_id: str,
    frame: np.ndarray | None,
    on_floor_since: datetime | None,
) -> bool:
    """Reply to a /status command with the latest frame and current state."""
    if on_floor_since is not None:
        minutes = (datetime.now() - on_floor_since).total_seconds() / 60
        status_line = (
            f"⚠️ <b>Nonno è a terra da {minutes:.0f} minut{'o' if minutes < 2 else 'i'}!</b>"
        )
    else:
        status_line = "✅ <b>Nonno sta bene.</b>"

    caption = f"{status_line}\n🕐 {_now()}"
    return _send_photo(frame, caption, to_chat_id)


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
    return _send_text(
        "👋 <b>OcchioSuNonno attivo!</b>\nIl sistema di monitoraggio è operativo. 🟢\n"
        "Invia /status per ricevere uno screenshot live."
    )
