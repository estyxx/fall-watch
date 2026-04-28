import os
from datetime import datetime

import requests


def _post(token: str, chat_id: str, text: str) -> bool:
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


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _send(text: str) -> bool:
    token = os.environ["TELEGRAM_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    return _post(token, chat_id, text)


def send_fall_alert(minutes_on_floor: float) -> bool:
    return _send(
        f"🚨 <b>ATTENZIONE — Nonno a terra!</b>\n\n"
        f"Il nonno sembra essere a terra da <b>{minutes_on_floor:.0f} minuti</b>.\n"
        f"Controllare subito!\n\n"
        f"🕐 {_now()}"
    )


def send_all_clear() -> bool:
    return _send(f"✅ <b>Tutto ok</b> — il nonno si è rialzato.\n🕐 {_now()}")


def send_startup() -> bool:
    return _send("👋 <b>OcchioSuNonno attivo!</b>\nIl sistema di monitoraggio è operativo. 🟢")
