# Green Machine - Project Context for Claude

## What this is
Ball-peak shot release assistant for NBA 2K. Tracks the basketball's Y position
frame-by-frame using OpenCV HSV color filtering. When the ball reaches its peak
arc height, it sends a release command to the controller over serial.

## Architecture
- **green_machine.py** — single self-contained Python file, no remote downloads
- **No compiled extensions** — pure Python + OpenCV + pyserial
- **No obfuscated code** — everything is readable and modifiable

## Key classes
- `Config` — all tunable parameters in one dataclass
- `BallTracker` — HSV masking → contour detection → Y-position history → peak detection
- `FeedbackReader` — OCR (pytesseract) reads Early/Late/Green text post-shot
- `Controller` — serial writes to hardware controller
- `GreenMachine` — main loop wiring everything together

## Tuning workflow
1. Adjust `ball_roi` to cover only the shooting area on screen
2. Tune `hsv_lower`/`hsv_upper` for the basketball color in your arena
3. Run and watch Early/Late feedback auto-adjust `peak_offset`
4. `peak_confirm_frames=1` is fastest; increase to 2-3 for stability

## Dependencies
See requirements.txt. Install with: `pip install -r requirements.txt`
Tesseract binary also required: https://github.com/tesseract-ocr/tesseract

## Running
```bash
python green_machine.py --port COM3 --monitor 1 --fps 60
```
Omit `--port` for dry-run mode (no controller needed for testing).
