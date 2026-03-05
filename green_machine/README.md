# Green Machine

Ball-peak shot release assistant for NBA 2K. No timing meters, no delay calibration.
It watches where the ball physically is and releases when it hits the peak of the arc.

## How it works

1. Captures your screen at 60fps
2. Isolates the basketball using HSV color filtering (orange range)
3. Tracks the ball's Y position frame by frame
4. When the ball stops going up — peak detected — fires the shot release
5. Reads Early/Late/Green feedback text via OCR and auto-adjusts the trigger point

## Why this beats timer-based approaches

Every shot type (quick jumper, slow post fade, hop step) reaches the same physical
release point — the peak of the arc. A timer fires at a fixed delay regardless of
shot animation. Ball tracking fires at the actual physics moment, every time.

## Setup

```bash
pip install -r requirements.txt
```

Tesseract OCR is required for feedback reading:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt install tesseract-ocr`
- Mac: `brew install tesseract`

## Usage

```bash
# With controller connected
python green_machine.py --port COM3

# Dry run (no controller, for tuning)
python green_machine.py

# Full options
python green_machine.py --port /dev/ttyUSB0 --monitor 1 --fps 60 --offset 0
```

## Tuning

Edit `Config` in `green_machine.py`:

| Parameter | What it does |
|-----------|-------------|
| `ball_roi` | Screen region to search for the ball |
| `hsv_lower/upper` | Color range for ball detection |
| `min_ball_area` | Filters out small noise blobs |
| `peak_confirm_frames` | Frames of downward motion before trigger |
| `peak_offset` | Manual Y offset for trigger point |
| `offset_step` | How much Early/Late shifts the offset |

## Feedback loop

After each shot the system reads the result text:
- **Green** → no change, keep going
- **Early** → raises the trigger point (waits longer)
- **Late** → lowers the trigger point (fires sooner)
