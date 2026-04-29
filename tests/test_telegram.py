"""Quick smoke test — run this before touching the camera."""

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

from fall_watch.config import Config  # noqa: E402
from fall_watch.notifier import TelegramNotifier  # noqa: E402

config = Config.load()
notifier = TelegramNotifier(config)

# Dummy 480p frame — grey with a white rectangle, just to test photo sending
dummy_frame = np.full((480, 640, 3), 80, dtype=np.uint8)
dummy_frame[100:380, 200:440] = 200  # light rectangle simulating a person shape

print("Sending startup message...")
notifier.send_startup()

print("Sending fake fall alert with dummy frame...")
notifier.send_fall_alert(minutes_on_floor=4, frame=dummy_frame)

print("Sending fake fall alert without frame (text fallback)...")
notifier.send_fall_alert(minutes_on_floor=4, frame=None)

print("Sending all clear...")
notifier.send_all_clear()

print("Done! Check Sentinelle di Nonno on Telegram 🎯")
