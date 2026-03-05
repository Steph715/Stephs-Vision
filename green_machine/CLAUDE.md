# Green Machine - Project Context for Claude

## What this is
Hand-peak shot release assistant for NBA 2K. Tracks the shooting hand's Y position
frame-by-frame using Mediapipe Hand Landmarks. When the hand reaches its peak arc
height (wrist snap point), it sends a release command to the Titan Two over serial.

No color dependency — works across every arena, ball skin, jersey, and lighting
because it tracks the hand skeleton, not pixels.

## Architecture
- **green_machine.py** — single self-contained Python file, no remote downloads
- **No obfuscated code** — everything is readable and modifiable

## Key classes
- `Config` — all tunable parameters in one dataclass
- `HandTracker` — Mediapipe Hands → landmark Y-position history → peak detection
- `FeedbackReader` — OCR (pytesseract) reads Early/Late/Green text post-shot
- `Controller` — serial writes to Titan Two with VID/PID auto-detection
- `GreenMachine` — main loop wiring everything together

## Tracked landmark
Default is landmark 12 (middle fingertip — highest point at full extension).
- `--landmark 12` = middle fingertip (default, fires latest)
- `--landmark 0`  = wrist (fires earlier, more conservative)
- `--landmark 20` = pinky tip

## Titan Two GPC companion script

The Python side sends a single byte (`0x58`) over serial. A GPC script must
be loaded on the Titan Two that reads that byte and maps it to the shot button.

Minimal GPC script to load in Gtuner IV:
```c
int cmd;

init {
    // Enable serial I/O on Titan Two programming port
}

main {
    cmd = iser();          // read one byte from serial (-1 if nothing)
    if(cmd == 0x58) {      // 0x58 = release trigger from Python
        combo_run(DoRelease);
    }
}

combo DoRelease {
    set_val(BUTTON_X, 100);   // X = shoot button — change to your mapping
    wait(50);
    set_val(BUTTON_X, 0);
    wait(50);
}
```
Load this via Gtuner IV → Script → Build & Run, then run `green_machine.py`.

## Titan Two port detection
The Titan Two shows up as USB VID 0x04D8 (Microchip / ConsoleTuner).
It exposes two COM ports — auto-detection tries the lower-numbered one first
(GPC iser() I/O port), then falls back to the higher one (programming port).
Run `python green_machine.py --scan-ports` to see all ports and confirm.
If auto-detect picks the wrong one, pass `--port COM4` explicitly.

## Elgato HD60 S+ setup
The Elgato shows up as a VideoCapture device. Run:
```bash
python green_machine.py --scan-devices
```
to find its index, then use `--capture-device <index>`.

## Tuning workflow
1. Run `--scan-ports` → confirm Titan Two is found
2. Run `--scan-devices` → find Elgato device index
3. Adjust `ball_roi` to cover only the shooting area on screen
4. Tune `hsv_lower`/`hsv_upper` for the basketball color in your arena
5. Run and watch Early/Late feedback auto-adjust `peak_offset`
6. `peak_confirm_frames=1` is fastest; increase to 2-3 for stability

## Dependencies
See requirements.txt. Install with: `pip install -r requirements.txt`
Tesseract binary also required: https://github.com/tesseract-ocr/tesseract

## Running
```bash
# Full setup: Elgato capture + Titan Two auto-detected
python green_machine.py --capture-device 1

# Explicit port
python green_machine.py --port COM4 --capture-device 1

# Dry run (no hardware — for ball tracking tuning only)
python green_machine.py --capture-device 1
```
