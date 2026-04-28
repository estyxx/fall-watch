"""Quick smoke test — run this before touching the camera."""
from dotenv import load_dotenv

from fall_watch.notifier import send_all_clear, send_fall_alert, send_startup

load_dotenv()

print("Sending startup message...")
send_startup()

print("Sending fake fall alert (4 minutes)...")
send_fall_alert(minutes_on_floor=4)

print("Sending all clear...")
send_all_clear()

print("Done! Check Sentinelle di Nonno on Telegram 🎯")
