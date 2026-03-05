"""
Green Machine - Ball Peak Tracker for NBA 2K
=============================================
Tracks the basketball's Y position frame-by-frame using HSV color filtering.
When the ball reaches its peak (stops going up), fires the shot release.
No timing guesswork - triggers on the actual physical release point.

Hardware: Titan Two connected via USB (ConsoleTuner VID 0x04D8)
Capture:  Elgato HD60 S+ (or any capture card) via cv2.VideoCapture,
          or screen capture via mss as fallback.

Titan Two GPC companion script required on device (see CLAUDE.md).
"""

import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple
import mss
import pytesseract


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # ROI for ball tracking (x, y, width, height) - tune to your setup
    ball_roi: Tuple[int, int, int, int] = (400, 100, 500, 600)

    # ROI for shot feedback text (Early / Late / Green)
    feedback_roi: Tuple[int, int, int, int] = (550, 650, 300, 80)

    # HSV range for basketball (orange/brown) - tweak per arena/lighting
    hsv_lower: Tuple[int, int, int] = (5, 100, 100)
    hsv_upper: Tuple[int, int, int] = (25, 255, 255)

    # Minimum ball contour area to consider a valid detection (px²)
    min_ball_area: int = 200

    # How many frames the ball must be moving down before triggering
    # 1 = trigger immediately at first downward frame (fastest)
    # 2-3 = slightly more stable, small extra delay
    peak_confirm_frames: int = 1

    # Serial port for Titan Two (None = auto-detect by VID 0x04D8)
    serial_port: Optional[str] = None
    # Titan Two programming port baud rate
    baud_rate: int = 9600

    # Single byte command sent to Titan Two GPC script to fire release.
    # Must match the value read by iser() in the companion GPC script.
    release_button: bytes = b'\x58'  # 0x58 = arbitrary trigger byte

    # Duration to hold the release button (seconds)
    release_duration: float = 0.05

    # Peak trigger Y-offset adjustment (pixels, + = trigger later/lower)
    # Shifts in response to Early/Late feedback
    peak_offset: int = 0

    # How much to shift offset per feedback event
    offset_step: int = 3

    # Elgato / capture card device index for cv2.VideoCapture.
    # None = use mss screen capture instead (slower, display only).
    # 0, 1, 2 ... = VideoCapture device index (run --scan-devices to find it).
    capture_device: Optional[int] = None

    # Capture monitor index (used only when capture_device is None)
    monitor_index: int = 1

    # Target frame rate for capture loop
    capture_fps: int = 60


# ---------------------------------------------------------------------------
# Ball Tracker
# ---------------------------------------------------------------------------

class BallTracker:
    """Tracks basketball Y position and detects the peak of the arc."""

    def __init__(self, config: Config):
        self.cfg = config
        self.lower = np.array(config.hsv_lower, dtype=np.uint8)
        self.upper = np.array(config.hsv_upper, dtype=np.uint8)
        self.y_history: deque = deque(maxlen=10)
        self.down_count = 0

    def find_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Returns (cx, cy) of ball centroid in frame coordinates, or None."""
        x, y, w, h = self.cfg.ball_roi
        roi = frame[y:y+h, x:x+w]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Take the largest contour
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.cfg.min_ball_area:
            return None

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00']) + x
        cy = int(M['m01'] / M['m00']) + y
        return cx, cy

    def update(self, cy: int) -> bool:
        """
        Feed the current ball Y position.
        Returns True when peak is detected (trigger moment).
        """
        self.y_history.append(cy)

        if len(self.y_history) < 2:
            return False

        prev_y = self.y_history[-2]
        curr_y = self.y_history[-1]

        # In screen coords Y increases downward, so ball going UP = decreasing Y
        # Peak = ball was going up and is now going down (curr_y > prev_y)
        if curr_y > prev_y + self.cfg.peak_offset:
            self.down_count += 1
        else:
            self.down_count = 0

        return self.down_count >= self.cfg.peak_confirm_frames

    def reset(self):
        self.y_history.clear()
        self.down_count = 0


# ---------------------------------------------------------------------------
# Feedback Reader
# ---------------------------------------------------------------------------

class FeedbackReader:
    """Reads Early/Late/Green text from the screen after each shot."""

    PATTERN = re.compile(r'\b(early|late|green|slightly early|slightly late)\b', re.IGNORECASE)

    def __init__(self, config: Config):
        self.cfg = config

    def read(self, frame: np.ndarray) -> Optional[str]:
        x, y, w, h = self.cfg.feedback_roi
        roi = frame[y:y+h, x:x+w]

        # Upscale for better OCR accuracy
        roi_up = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi_up, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        text = pytesseract.image_to_string(thresh, config='--psm 7')
        match = self.PATTERN.search(text)
        return match.group(0).lower() if match else None


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class Controller:
    """Sends button commands to the Titan Two over serial."""

    # ConsoleTuner (Titan Two) USB vendor ID
    TITAN_TWO_VID = 0x04D8

    def __init__(self, config: Config):
        self.cfg = config
        self.ser: Optional[serial.Serial] = None

    def connect(self) -> bool:
        port = self.cfg.serial_port or self._auto_detect_titan_two()
        if not port:
            print("[Controller] Titan Two not found. Run --scan-ports to list devices.")
            print("[Controller] Running in DRY RUN mode.")
            return False
        try:
            self.ser = serial.Serial(
                port,
                self.cfg.baud_rate,
                timeout=1,
                write_timeout=1,
            )
            # Brief settle after open — Titan Two resets on DTR
            time.sleep(0.1)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            print(f"[Controller] Titan Two connected on {port} @ {self.cfg.baud_rate} baud")
            return True
        except serial.SerialException as e:
            print(f"[Controller] Failed to open {port}: {e}")
            print("[Controller] Running in DRY RUN mode.")
            return False

    def _auto_detect_titan_two(self) -> Optional[str]:
        """Find the Titan Two by USB VID (0x04D8). Returns the programming port."""
        ports = serial.tools.list_ports.comports()
        candidates = [p for p in ports if p.vid == self.TITAN_TWO_VID]

        if not candidates:
            return None

        # Titan Two exposes two COM ports; the higher-numbered one is typically
        # the programming/data port used for GPC serial I/O.
        candidates.sort(key=lambda p: p.device)
        chosen = candidates[-1]
        print(f"[Controller] Auto-detected Titan Two: {chosen.device} ({chosen.description})")
        return chosen.device

    @staticmethod
    def scan_ports():
        """Print all available serial ports with VID/PID. Use to find Titan Two port."""
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("No serial ports found.")
            return
        print(f"{'Device':<15} {'VID':<8} {'PID':<8} Description")
        print("-" * 60)
        for p in ports:
            vid = hex(p.vid) if p.vid else "None"
            pid = hex(p.pid) if p.pid else "None"
            titan = " <-- Titan Two" if p.vid == 0x04D8 else ""
            print(f"{p.device:<15} {vid:<8} {pid:<8} {p.description}{titan}")

    def release_shot(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(self.cfg.release_button)
                self.ser.flush()
                time.sleep(self.cfg.release_duration)
            except serial.SerialException as e:
                print(f"[Controller] Write failed: {e}")
        else:
            print("[DRY RUN] Shot released")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

class GreenMachine:
    def __init__(self, config: Config = None):
        self.cfg = config or Config()
        self.tracker = BallTracker(self.cfg)
        self.feedback = FeedbackReader(self.cfg)
        self.controller = Controller(self.cfg)
        self.shot_count = 0
        self.green_count = 0
        self.triggered = False
        self.trigger_cooldown = 0

    def run(self):
        self.controller.connect()

        frame_interval = 1.0 / self.cfg.capture_fps

        print("[Green Machine] Running. Press Ctrl+C to stop.")
        print(f"[Green Machine] Peak offset: {self.cfg.peak_offset}px")

        if self.cfg.capture_device is not None:
            self._run_capture_card(frame_interval)
        else:
            self._run_screen_capture(frame_interval)

    def _run_capture_card(self, frame_interval: float):
        """Capture from Elgato or any VideoCapture device."""
        cap = cv2.VideoCapture(self.cfg.capture_device, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # CAP_DSHOW is Windows-only; fall back to default backend
            cap = cv2.VideoCapture(self.cfg.capture_device)
        if not cap.isOpened():
            print(f"[Capture] Could not open device {self.cfg.capture_device}. "
                  f"Run --scan-devices to list available capture devices.")
            self.controller.close()
            return

        # Request native resolution from Elgato (1080p)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.capture_fps)
        print(f"[Capture] Elgato device {self.cfg.capture_device} opened  "
              f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
              f"@ {int(cap.get(cv2.CAP_PROP_FPS))}fps")

        try:
            while True:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    print("[Capture] Frame read failed — check Elgato connection.")
                    time.sleep(0.5)
                    continue
                self._process_frame(frame, t0, frame_interval)
        except KeyboardInterrupt:
            print(f"\n[Green Machine] Stopped. {self.green_count}/{self.shot_count} greens.")
        finally:
            cap.release()
            self.controller.close()

    def _run_screen_capture(self, frame_interval: float):
        """Capture from the display using mss (fallback when no capture card)."""
        with mss.mss() as sct:
            monitor = sct.monitors[self.cfg.monitor_index]
            try:
                while True:
                    t0 = time.perf_counter()
                    raw = sct.grab(monitor)
                    frame = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
                    self._process_frame(frame, t0, frame_interval)
            except KeyboardInterrupt:
                print(f"\n[Green Machine] Stopped. {self.green_count}/{self.shot_count} greens.")
            finally:
                self.controller.close()

    def _process_frame(self, frame: np.ndarray, t0: float, frame_interval: float):
        """Core per-frame logic shared by both capture paths."""
        if self.trigger_cooldown > 0:
            self.trigger_cooldown -= 1
            result = self.feedback.read(frame)
            if result:
                self._apply_feedback(result)
            self._sleep_remainder(t0, frame_interval)
            return

        pos = self.tracker.find_ball(frame)
        if pos is None:
            self.tracker.reset()
            self._sleep_remainder(t0, frame_interval)
            return

        cx, cy = pos
        if self.tracker.update(cy):
            self.controller.release_shot()
            self.shot_count += 1
            self.tracker.reset()
            self.trigger_cooldown = 90
            print(f"[Shot #{self.shot_count}] Released at Y={cy}  offset={self.cfg.peak_offset}")

        self._sleep_remainder(t0, frame_interval)

    def _apply_feedback(self, result: str):
        """Shift the peak trigger offset based on shot feedback."""
        self.shot_count  # already incremented at trigger

        if 'green' in result:
            self.green_count += 1
            print(f"  -> GREEN  ({self.green_count} total) offset={self.cfg.peak_offset}")
        elif 'early' in result:
            # Released too early = ball hadn't peaked yet = trigger later (raise offset)
            self.cfg.peak_offset += self.cfg.offset_step
            print(f"  -> EARLY  offset adjusted to {self.cfg.peak_offset}")
        elif 'late' in result:
            # Released too late = ball past peak = trigger sooner (lower offset)
            self.cfg.peak_offset -= self.cfg.offset_step
            print(f"  -> LATE   offset adjusted to {self.cfg.peak_offset}")

    @staticmethod
    def _sleep_remainder(t0: float, interval: float):
        elapsed = time.perf_counter() - t0
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def scan_capture_devices(max_index: int = 10):
    """Print available VideoCapture devices (cameras, capture cards, etc.)."""
    print("Scanning VideoCapture devices...")
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(f"  Device {i}: {w}x{h} @ {fps}fps")
            found.append(i)
            cap.release()
    if not found:
        print("  No VideoCapture devices found.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Green Machine - Ball Peak Shot Releaser')
    parser.add_argument('--port', help='Titan Two serial port (e.g. COM3). Auto-detected if omitted.')
    parser.add_argument('--capture-device', type=int, default=None,
                        help='Elgato/capture card device index (from --scan-devices). '
                             'Omit to use screen capture.')
    parser.add_argument('--monitor', type=int, default=1, help='Monitor index (screen capture only)')
    parser.add_argument('--fps', type=int, default=60, help='Capture frame rate')
    parser.add_argument('--offset', type=int, default=0, help='Initial peak Y offset')
    parser.add_argument('--scan-ports', action='store_true',
                        help='List all serial ports with VID/PID and exit')
    parser.add_argument('--scan-devices', action='store_true',
                        help='List all VideoCapture devices and exit')
    args = parser.parse_args()

    if args.scan_ports:
        Controller.scan_ports()
        raise SystemExit(0)

    if args.scan_devices:
        scan_capture_devices()
        raise SystemExit(0)

    cfg = Config(
        serial_port=args.port,
        capture_device=args.capture_device,
        monitor_index=args.monitor,
        capture_fps=args.fps,
        peak_offset=args.offset,
    )

    GreenMachine(cfg).run()
