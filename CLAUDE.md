# CLAUDE.md

This file is read by Claude Code. Follow everything here precisely.

---

## Project overview

`fall-watch` monitors an RTSP camera stream and alerts a Telegram group when a
person has been lying on the floor for too long. It runs 24/7 on a Raspberry Pi 5.

Alerts go to a private family Telegram group. Camera: EZVIZ, accessed via RTSP.

**Stack:** Python 3.13 · YOLOv8-pose · OpenCV · Telegram Bot API · uv · ruff · mypy

---

## Commands

```bash
uv sync                                   # install all deps (including dev)
uv run fall-watch                         # run the monitor
uv run python tests/test_detector.py      # integration test: webcam + detection + Telegram
uv run python tests/test_telegram.py      # Telegram-only smoke test (no camera needed)
uv run python tests/test_frame.py image.jpg    # test detection on a static image
uv run python scripts/setup_roi.py image.jpg   # interactive floor ROI setup (click 4 points)

uv run ruff check src/ tests/            # lint
uv run ruff format src/                  # format
uv run mypy src/                         # type check
uv run pytest                            # run tests
```

Always run `ruff check` and `mypy` before finishing any task. Fix all warnings —
do not suppress them unless there is a very good reason, and if so, leave a comment.

---

## Architecture

```
src/fall_watch/
├── main.py       # entry point, main loop, state machine (match/case)
├── config.py     # Config dataclass, all env-var loading
├── detector.py   # YOLOv8 model + lying-down heuristic + ROI support
└── notifier.py   # Telegram sendMessage / sendPhoto / poll_commands

tests/
├── test_detector.py   # integration test: webcam + detection + Telegram
├── test_telegram.py   # Telegram-only smoke test (no camera needed)
└── test_frame.py      # test detection on a single static image

scripts/
└── setup_roi.py       # interactive floor ROI setup tool (click 4 points)
```

Keep these four concerns strictly separated. `main.py` orchestrates, `config.py`
owns all configuration, `detector.py` does vision, `notifier.py` does IO.
Do not mix them.

---

## Key design decisions

- **Frame interval:** analyse one frame every 5 seconds (`FRAME_INTERVAL_SECONDS`)
  to keep CPU load manageable on the Pi
- **Hysteresis:** require 3 consecutive "not on floor" frames (`NOT_ON_FLOOR_STREAK_MAX`)
  before sending an all-clear, to avoid flapping on ambiguous detections
- **Fall threshold:** alert only fires after `FALL_THRESHOLD_MINUTES` (default 3 min)
  continuously on the floor
- **Alert cooldown:** once alerted, wait `ALERT_COOLDOWN_MINUTES` (default 15 min)
  before alerting again, to avoid spam
- **Photo alerts:** both fall alert and all-clear include a JPEG snapshot, not just text
- **Floor ROI:** a configurable polygon zone (`ROI_POINTS` env var) restricts
  detection to the floor area, excluding the bed to avoid false positives
- **/status command:** family can send `/status` to the Telegram group at any time
  to get a live screenshot and current state

---

## Code style

Priority order: **readability → simplicity → brevity**. Never sacrifice
readability for cleverness.

### Python version

Target Python 3.13. Use modern features actively:

```python
# match/case instead of if/elif chains
match (person_on_floor, was_on_floor):
    case (True, _): ...
    case (False, True): ...

# X | None instead of Optional[X]
def get_point(idx: int) -> np.ndarray | None: ...

# Union types with |
def process(value: int | float) -> str: ...

# Walrus operator where it genuinely reduces repetition
if chunk := data.read(8192): ...

# f-strings always, no .format() or %
msg = f"On floor for {minutes:.1f} minutes"
```

### Types

- mypy strict is enabled — all functions must have full type annotations
- No `Any` unless wrapping a truly untyped third-party API, and always with a comment
- Prefer `TypeAlias` for complex repeated types

### Naming

- Functions and variables: `snake_case`
- Private module-level helpers: prefix with `_` (e.g. `_is_lying_down`)
- Constants: `UPPER_SNAKE_CASE`
- No abbreviations unless they are universally understood (`url`, `id`, `kp` for keypoint)

### Functions

- One responsibility per function — if you need "and" to describe what it does, split it
- Keep functions under ~30 lines; if longer, extract helpers
- Prefer returning early over deeply nested conditionals

```python
# good
def analyse_frame(model: YOLO, frame: np.ndarray) -> bool:
    results = model(frame, verbose=False)
    return any(
        _is_lying_down(person_kps, frame.shape[0])
        for result in results
        if result.keypoints is not None
        for person_kps in result.keypoints.data.cpu().numpy()
    )

# avoid
def analyse_frame(model, frame):
    results = model(frame, verbose=False)
    found = False
    for result in results:
        if result.keypoints is not None:
            for kps in result.keypoints.data.cpu().numpy():
                if _is_lying_down(kps, frame.shape[0]):
                    found = True
    return found
```

### Comprehensions and generators

Prefer comprehensions and generators over explicit loops when the intent is
clear. Avoid them when they become hard to read — a loop is fine.

### Imports

- Standard library first, third-party second, local last (ruff/isort enforces this)
- No wildcard imports
- No relative imports (use `from fall_watch.detector import ...`)

### Error handling

- Catch specific exceptions, never bare `except:`
- Log errors with context (e.g. `logger.error("❌ Failed: %s", e)`) before re-raising
  or recovering
- For camera disconnects and Telegram failures, the code should retry gracefully —
  a transient failure must not crash the whole monitor

### Configuration

- All config comes from environment variables via `.env` / `os.environ`
- `os.environ["KEY"]` for required values (raises clearly if missing)
- `os.getenv("KEY", "default")` for optional values with a default
- Never hardcode IPs, tokens, thresholds, or any deployment-specific value
- All env-var loading lives in `config.py` — nowhere else

### Logging

Use the standard `logging` module via `logging.getLogger(__name__)`. The root
logger is configured in `main.py` (`_setup_logging()`) with a console handler
and a daily-rotating file handler. Do not use `print()` directly outside of
scripts. Do not add extra logging frameworks unless the project grows substantially.

---

## What to avoid

- **No `Optional[X]`** — use `X | None`
- **No `.format()` or `%s`** — use f-strings
- **No `Any`** without a comment explaining why
- **No mutable default arguments**
- **No God functions** — if `main()` grows beyond orchestration, extract
- **No silent failures** — if something goes wrong, log it
- **No hardcoded secrets or IPs** — everything via `.env`
- **No new dependencies without a reason** — this runs on a Pi, keep it lean

---

## Adding a new notification channel

`notifier.py` currently supports Telegram only. If adding another channel
(e.g. email, PagerDuty), extract a `Notifier` protocol:

```python
from typing import Protocol

class Notifier(Protocol):
    def send_fall_alert(self, minutes: float) -> bool: ...
    def send_all_clear(self) -> bool: ...
    def send_startup(self) -> bool: ...
```

Then inject it into `main()` rather than importing the Telegram functions directly.

---

## Raspberry Pi notes

- Target hardware: Raspberry Pi 5 (4 GB)
- Python 3.13 must be installed via `uv` (it manages its own Python builds)
- Run as a systemd service — see `README.md` for the unit file
- The YOLOv8 nano model (`yolov8n-pose.pt`) downloads automatically on first run
  and is cached locally; it is gitignored (`.pt` files)
