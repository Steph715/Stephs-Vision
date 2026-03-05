# Green Machine

Hand-peak shot release assistant for NBA 2K. No timing meters, no delay calibration,
no ball color dependency. It watches where the shooting hand is and releases when the
fingertips hit the peak of the arc.

## How it works

1. Captures from the Elgato capture card (or screen) at 60fps
2. Runs Mediapipe Hand Landmarks on each frame — tracks the hand skeleton
3. Monitors the Y position of the middle fingertip (landmark 12) frame by frame
4. When the hand stops rising — peak detected — fires the shot release to the Titan Two
5. Reads Early/Late/Green feedback via OCR and auto-adjusts the trigger point

## Why hand tracking beats ball color tracking

Ball color changes per arena (some balls are darker/lighter), per skin tone it blends with,
and per defender contact. The hand is always the same shape. Mediapipe doesn't care about
skin tone or lighting — it finds the hand skeleton regardless.

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
# Find your Titan Two port and Elgato device index first
python green_machine.py --scan-ports
python green_machine.py --scan-devices

# Full setup: Elgato capture + Titan Two auto-detected
python green_machine.py --capture-device 1

# Explicit port
python green_machine.py --port COM4 --capture-device 1

# Dry run (no controller, just ball tracking)
python green_machine.py --capture-device 1

# Screen capture fallback (no Elgato)
python green_machine.py --port COM4 --monitor 1
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
