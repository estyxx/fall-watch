# CLAUDE.md

This file is read by Claude Code. Follow everything here precisely.

---

## Project overview

`fall-watch` monitors an RTSP camera stream and alerts a Telegram group when a
person has been lying on the floor for too long. It runs 24/7 on a Raspberry Pi.

**Stack:** Python 3.13 · YOLOv8-pose · OpenCV · Telegram Bot API · uv · ruff · mypy

---

## Commands

```bash
uv sync                        # install all deps (including dev)
uv run fall-watch              # run the monitor
uv run python tests/test_telegram.py  # smoke-test Telegram without a camera

uv run ruff check src/         # lint
uv run ruff format src/        # format
uv run mypy src/               # type check
uv run pytest                  # run tests
```

Always run `ruff check` and `mypy` before finishing any task. Fix all warnings —
do not suppress them unless there is a very good reason, and if so, leave a comment.

---

## Architecture

```
src/fall_watch/
├── main.py       # entry point, main loop, state machine
├── detector.py   # YOLOv8 model loading + pose analysis logic
└── notifier.py   # Telegram message sending
tests/
└── test_telegram.py  # manual smoke test (not pytest)
```

Keep these three concerns strictly separated. `main.py` orchestrates,
`detector.py` does vision, `notifier.py` does IO. Do not mix them.

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
- Log errors with context (`_log(f"❌ Failed: {e}")`) before re-raising or recovering
- For camera disconnects and Telegram failures, the code should retry gracefully —
  a transient failure must not crash the whole monitor

### Configuration

- All config comes from environment variables via `.env` / `os.environ`
- `os.environ["KEY"]` for required values (raises clearly if missing)
- `os.getenv("KEY", "default")` for optional values with a default
- Never hardcode IPs, tokens, thresholds, or any deployment-specific value

### Logging

Use the `_log()` helper in `main.py` — it prefixes every line with a timestamp.
Do not use `print()` directly outside of scripts. Do not add a full logging
framework unless the project grows substantially.

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

- Python 3.13 must be installed via `uv` (it manages its own Python builds)
- Run as a systemd service — see `README.md` for the unit file
- The YOLOv8 nano model (`yolov8n-pose.pt`) downloads automatically on first run
  and is cached locally; it is gitignored (`.pt` files)
- Pi 4 (4GB) is sufficient; Pi 5 is faster if available
