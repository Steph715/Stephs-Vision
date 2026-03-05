"""
Green Machine - Ball Peak Tracker for NBA 2K
=============================================
Tracks the basketball's Y position frame-by-frame using HSV color filtering.
When the ball reaches its peak (stops going up), fires the shot release.
No timing guesswork - triggers on the actual physical release point.

Hardware: Controller connected via serial (e.g. Titan Two / Arduino)
Capture:  Any screen capture or HDMI capture card feeding frames
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

    # Serial port for controller output (None = auto-detect)
    serial_port: Optional[str] = None
    baud_rate: int = 115200

    # Button code sent to controller for shot release (customize per device)
    release_button: bytes = b'\x01'

    # Duration to hold the release button (seconds)
    release_duration: float = 0.05

    # Peak trigger Y-offset adjustment (pixels, + = trigger later/lower)
    # Shifts in response to Early/Late feedback
    peak_offset: int = 0

    # How much to shift offset per feedback event
    offset_step: int = 3

    # Capture monitor index (1 = primary)
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
    """Sends button commands over serial to the hardware controller."""

    def __init__(self, config: Config):
        self.cfg = config
        self.ser: Optional[serial.Serial] = None

    def connect(self) -> bool:
        port = self.cfg.serial_port or self._auto_detect()
        if not port:
            print("[Controller] No serial port found. Running in DRY RUN mode.")
            return False
        try:
            self.ser = serial.Serial(port, self.cfg.baud_rate, timeout=1)
            print(f"[Controller] Connected on {port}")
            return True
        except serial.SerialException as e:
            print(f"[Controller] Failed to open {port}: {e}")
            return False

    def _auto_detect(self) -> Optional[str]:
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if 'ttyUSB' in p.device or 'ttyACM' in p.device or 'COM' in p.device:
                return p.device
        return None

    def release_shot(self):
        if self.ser and self.ser.is_open:
            self.ser.write(self.cfg.release_button)
            time.sleep(self.cfg.release_duration)
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

        mon = {"mon": self.cfg.monitor_index}
        frame_interval = 1.0 / self.cfg.capture_fps

        print("[Green Machine] Running. Press Ctrl+C to stop.")
        print(f"[Green Machine] Peak offset: {self.cfg.peak_offset}px")

        with mss.mss() as sct:
            monitors = sct.monitors
            monitor = monitors[self.cfg.monitor_index]

            try:
                while True:
                    t0 = time.perf_counter()

                    # Grab frame
                    raw = sct.grab(monitor)
                    frame = np.array(raw)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Cooldown after trigger (avoid double-fire)
                    if self.trigger_cooldown > 0:
                        self.trigger_cooldown -= 1

                        # Read feedback during cooldown window
                        result = self.feedback.read(frame)
                        if result:
                            self._apply_feedback(result)

                        self._sleep_remainder(t0, frame_interval)
                        continue

                    # Find ball
                    pos = self.tracker.find_ball(frame)

                    if pos is None:
                        self.tracker.reset()
                        self._sleep_remainder(t0, frame_interval)
                        continue

                    cx, cy = pos

                    # Check for peak
                    if self.tracker.update(cy):
                        self.controller.release_shot()
                        self.shot_count += 1
                        self.triggered = True
                        self.tracker.reset()
                        # Wait ~90 frames (~1.5s) before reading feedback
                        self.trigger_cooldown = 90
                        print(f"[Shot #{self.shot_count}] Released at Y={cy}  offset={self.cfg.peak_offset}")

                    self._sleep_remainder(t0, frame_interval)

            except KeyboardInterrupt:
                print(f"\n[Green Machine] Stopped. {self.green_count}/{self.shot_count} greens.")
            finally:
                self.controller.close()

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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Green Machine - Ball Peak Shot Releaser')
    parser.add_argument('--port', help='Serial port (e.g. COM3 or /dev/ttyUSB0)')
    parser.add_argument('--monitor', type=int, default=1, help='Monitor index to capture')
    parser.add_argument('--fps', type=int, default=60, help='Capture frame rate')
    parser.add_argument('--offset', type=int, default=0, help='Initial peak Y offset')
    args = parser.parse_args()

    cfg = Config(
        serial_port=args.port,
        monitor_index=args.monitor,
        capture_fps=args.fps,
        peak_offset=args.offset,
    )

    GreenMachine(cfg).run()
