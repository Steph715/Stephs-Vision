"""
Green Machine - Hand Peak Tracker for NBA 2K
=============================================
Tracks the shooting hand's Y position using Mediapipe Hand Landmarks.
When the hand reaches its peak arc — the moment before the wrist snap —
fires the shot release via the Titan Two controller.

No color dependency. Works across every arena, ball skin, jersey, and
court lighting because it tracks the hand skeleton, not pixels.

Hardware: Titan Two connected via USB (ConsoleTuner VID 0x04D8)
Capture:  Elgato HD60 S+ via cv2.VideoCapture, or mss screen capture.

Titan Two GPC companion script required on device (see CLAUDE.md).
"""

import cv2
import numpy as np
import mediapipe as mp
import serial
import serial.tools.list_ports
import time
import re
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import mss
import pytesseract


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # ROI for hand tracking (x, y, width, height) — crop to shooting arm area
    # Narrowing this improves speed and avoids tracking the defender's hand.
    # (0, 0, 1920, 1080) = full frame, safe default
    hand_roi: Tuple[int, int, int, int] = (0, 0, 1920, 1080)

    # Which hand landmark to track for peak detection.
    # 12 = middle fingertip (highest point at full extension)
    # 0  = wrist (a bit lower, fires slightly earlier)
    # 20 = pinky tip
    track_landmark: int = 12

    # ROI for shot feedback text (Early / Late / Green)
    feedback_roi: Tuple[int, int, int, int] = (550, 650, 300, 80)

    # How many consecutive frames the hand must be moving DOWN before triggering.
    # 1 = fastest (fire on first downward frame)
    # 2 = slightly more stable
    peak_confirm_frames: int = 1

    # Mediapipe detection/tracking confidence thresholds.
    # 2K hands are rendered, not real — lower values help detection.
    detection_confidence: float = 0.4
    tracking_confidence: float = 0.4

    # Show a live debug preview window (slower, use for tuning only)
    debug: bool = False

    # Serial port for Titan Two (None = auto-detect by VID 0x04D8)
    serial_port: Optional[str] = None
    # Titan Two baud rate — 9600 is the GPC iser() default
    baud_rate: int = 9600

    # Single byte sent to Titan Two GPC script to fire release.
    # Must match the value read by iser() in the companion GPC script.
    release_button: bytes = b'\x58'  # 0x58 = trigger byte

    # How long to hold the release signal (seconds)
    release_duration: float = 0.05

    # Peak trigger Y-offset (pixels). Auto-adjusted by Early/Late feedback.
    # Positive = trigger later (hand must drop further before firing)
    peak_offset: int = 0

    # How much Early/Late shifts the offset per shot
    offset_step: int = 3

    # Elgato/capture card device index for cv2.VideoCapture.
    # None = use mss screen capture (display only, no capture card).
    # Run --scan-devices to find the right index.
    capture_device: Optional[int] = None

    # Monitor index used only when capture_device is None (mss fallback)
    monitor_index: int = 1

    # Target frame rate
    capture_fps: int = 60


# ---------------------------------------------------------------------------
# Hand Tracker
# ---------------------------------------------------------------------------

class HandTracker:
    """
    Tracks the shooting hand Y position using Mediapipe Hand Landmarks.
    Detects peak: moment the tracked landmark stops rising and starts falling.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )
        self.y_history: deque = deque(maxlen=8)
        self.down_count = 0
        self._last_result = None   # stored for debug drawing

    def find_hand(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Returns pixel (x, y) of the tracked landmark in frame coordinates,
        or None if no hand is detected.
        """
        rx, ry, rw, rh = self.cfg.hand_roi

        # Mediapipe expects RGB
        rgb = cv2.cvtColor(frame[ry:ry+rh, rx:rx+rw], cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        self._last_result = result

        if not result.multi_hand_landmarks:
            return None

        lm = result.multi_hand_landmarks[0].landmark[self.cfg.track_landmark]
        px = int(lm.x * rw) + rx
        py = int(lm.y * rh) + ry
        return px, py

    def draw_debug(self, frame: np.ndarray, tracked_pos: Optional[Tuple[int, int]]) -> np.ndarray:
        """Return a downscaled debug frame with landmarks and status overlaid."""
        dbg = frame.copy()
        rx, ry, rw, rh = self.cfg.hand_roi

        # ROI rectangle
        cv2.rectangle(dbg, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)

        # Draw hand skeleton if detected
        if self._last_result and self._last_result.multi_hand_landmarks:
            for hand_lm in self._last_result.multi_hand_landmarks:
                # Re-map normalized coords into full-frame pixel space
                h, w = frame.shape[:2]
                for lm in hand_lm.landmark:
                    cx = int(lm.x * rw) + rx
                    cy = int(lm.y * rh) + ry
                    cv2.circle(dbg, (cx, cy), 3, (0, 255, 0), -1)
            # Highlight tracked landmark
            if tracked_pos:
                cv2.circle(dbg, tracked_pos, 8, (0, 0, 255), -1)
                cv2.putText(dbg, f"Y={tracked_pos[1]}", (tracked_pos[0]+10, tracked_pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        status = "HAND DETECTED" if tracked_pos else "no hand"
        color = (0, 255, 0) if tracked_pos else (0, 0, 255)
        cv2.putText(dbg, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Downscale to 960x540 so it doesn't fill the whole screen
        return cv2.resize(dbg, (960, 540))

    def update(self, py: int) -> bool:
        """
        Feed current hand Y position.
        Returns True when peak is detected (trigger now).
        """
        self.y_history.append(py)

        if len(self.y_history) < 2:
            return False

        prev_y = self.y_history[-2]
        curr_y = self.y_history[-1]

        # Screen Y increases downward — hand going UP = decreasing Y.
        # Peak = hand was rising and is now falling (curr_y > prev_y + offset).
        if curr_y > prev_y + self.cfg.peak_offset:
            self.down_count += 1
        else:
            self.down_count = 0

        return self.down_count >= self.cfg.peak_confirm_frames

    def reset(self):
        self.y_history.clear()
        self.down_count = 0

    def close(self):
        self.hands.close()


# ---------------------------------------------------------------------------
# Feedback Reader
# ---------------------------------------------------------------------------

class FeedbackReader:
    """Reads Early / Late / Green text from the screen after each shot."""

    PATTERN = re.compile(r'\b(early|late|green|slightly early|slightly late)\b', re.IGNORECASE)

    def __init__(self, config: Config):
        self.cfg = config

    def read(self, frame: np.ndarray) -> Optional[str]:
        x, y, w, h = self.cfg.feedback_roi
        roi = frame[y:y+h, x:x+w]

        roi_up = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi_up, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        text = pytesseract.image_to_string(thresh, config='--psm 7')
        match = self.PATTERN.search(text)
        return match.group(0).lower() if match else None


# ---------------------------------------------------------------------------
# Controller (Titan Two)
# ---------------------------------------------------------------------------

class Controller:
    """Sends the release command to the Titan Two over serial."""

    # ConsoleTuner USB vendor ID
    TITAN_TWO_VID = 0x04D8

    def __init__(self, config: Config):
        self.cfg = config
        self.ser: Optional[serial.Serial] = None

    def connect(self) -> bool:
        port = self.cfg.serial_port or self._auto_detect_titan_two()
        if not port:
            print("[Controller] Titan Two not found. Run --scan-ports to see all ports.")
            print("[Controller] Running in DRY RUN mode.")
            return False
        return self._open(port)

    def _open(self, port: str) -> bool:
        try:
            self.ser = serial.Serial(
                port,
                self.cfg.baud_rate,
                timeout=1,
                write_timeout=1,
                # Disable hardware flow control — T2 doesn't use it and
                # some drivers will hold the port locked without this.
                rtscts=False,
                dsrdtr=False,
            )
            # Pull RTS/DTR low so the T2 doesn't reset on connect
            self.ser.rts = False
            self.ser.dtr = False
            time.sleep(0.15)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            print(f"[Controller] Titan Two connected on {port} @ {self.cfg.baud_rate} baud")
            return True
        except serial.SerialException as e:
            print(f"[Controller] Failed to open {port}: {e}")
            self.ser = None
            return False

    def connect(self) -> bool:
        if self.cfg.serial_port:
            return self._open(self.cfg.serial_port)

        ports = serial.tools.list_ports.comports()
        candidates = sorted(
            [p for p in ports if p.vid == self.TITAN_TWO_VID],
            key=lambda p: p.device,
        )

        if not candidates:
            print("[Controller] Titan Two not found. Run --scan-ports to see all ports.")
            print("[Controller] Running in DRY RUN mode.")
            return False

        # Try each T2 port until one opens — lower-numbered is GPC I/O port
        for candidate in candidates:
            print(f"[Controller] Trying {candidate.device} ({candidate.description}) ...")
            if self._open(candidate.device):
                return True

        print("[Controller] All Titan Two ports failed. Running in DRY RUN mode.")
        return False

    @staticmethod
    def scan_ports():
        """List all serial ports with VID/PID. Titan Two marked with arrow."""
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("No serial ports found.")
            return
        print(f"{'Device':<15} {'VID':<8} {'PID':<8} Description")
        print("-" * 65)
        for p in ports:
            vid = hex(p.vid) if p.vid else "None"
            pid = hex(p.pid) if p.pid else "None"
            tag = " <-- Titan Two" if p.vid == 0x04D8 else ""
            print(f"{p.device:<15} {vid:<8} {pid:<8} {p.description}{tag}")

    def release_shot(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(self.cfg.release_button)
                self.ser.flush()
                time.sleep(self.cfg.release_duration)
            except serial.SerialException as e:
                print(f"[Controller] Write error: {e}")
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
        self.tracker = HandTracker(self.cfg)
        self.feedback = FeedbackReader(self.cfg)
        self.controller = Controller(self.cfg)
        self.shot_count = 0
        self.green_count = 0
        self.trigger_cooldown = 0

    def run(self):
        self.controller.connect()

        frame_interval = 1.0 / self.cfg.capture_fps
        print("[Green Machine] Running — tracking hand peak. Ctrl+C to stop.")
        print(f"[Green Machine] Landmark: {self.cfg.track_landmark}  "
              f"Peak offset: {self.cfg.peak_offset}px")

        try:
            if self.cfg.capture_device is not None:
                self._run_capture_card(frame_interval)
            else:
                self._run_screen_capture(frame_interval)
        finally:
            self.tracker.close()

    def _run_capture_card(self, frame_interval: float):
        cap = cv2.VideoCapture(self.cfg.capture_device, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.cfg.capture_device)
        if not cap.isOpened():
            print(f"[Capture] Cannot open device {self.cfg.capture_device}. "
                  "Run --scan-devices to list available devices.")
            self.controller.close()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.capture_fps)
        print(f"[Capture] Device {self.cfg.capture_device}  "
              f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  "
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
            print(f"\n[Green Machine] Done. {self.green_count}/{self.shot_count} greens.")
        finally:
            cap.release()
            self.controller.close()

    def _run_screen_capture(self, frame_interval: float):
        with mss.mss() as sct:
            monitor = sct.monitors[self.cfg.monitor_index]
            try:
                while True:
                    t0 = time.perf_counter()
                    frame = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
                    self._process_frame(frame, t0, frame_interval)
            except KeyboardInterrupt:
                print(f"\n[Green Machine] Done. {self.green_count}/{self.shot_count} greens.")
            finally:
                self.controller.close()

    def _process_frame(self, frame: np.ndarray, t0: float, frame_interval: float):
        if self.trigger_cooldown > 0:
            self.trigger_cooldown -= 1
            result = self.feedback.read(frame)
            if result:
                self._apply_feedback(result)
            self._sleep_remainder(t0, frame_interval)
            return

        pos = self.tracker.find_hand(frame)

        if self.cfg.debug:
            dbg = self.tracker.draw_debug(frame, pos)
            cv2.imshow("Green Machine - Debug", dbg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        if pos is None:
            self.tracker.reset()
            self._sleep_remainder(t0, frame_interval)
            return

        _, py = pos
        if self.tracker.update(py):
            self.controller.release_shot()
            self.shot_count += 1
            self.tracker.reset()
            self.trigger_cooldown = 90
            print(f"[Shot #{self.shot_count}] Hand peak at Y={py}  offset={self.cfg.peak_offset}")

        self._sleep_remainder(t0, frame_interval)

    def _apply_feedback(self, result: str):
        if 'green' in result:
            self.green_count += 1
            print(f"  -> GREEN  ({self.green_count} total)")
        elif 'early' in result:
            self.cfg.peak_offset += self.cfg.offset_step
            print(f"  -> EARLY  offset -> {self.cfg.peak_offset}")
        elif 'late' in result:
            self.cfg.peak_offset -= self.cfg.offset_step
            print(f"  -> LATE   offset -> {self.cfg.peak_offset}")

    @staticmethod
    def _sleep_remainder(t0: float, interval: float):
        remaining = interval - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def scan_capture_devices(max_index: int = 10):
    """List VideoCapture devices (capture cards, webcams, etc.)."""
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

    parser = argparse.ArgumentParser(description='Green Machine — Hand Peak Shot Releaser')
    parser.add_argument('--port', help='Titan Two serial port (e.g. COM3). Auto-detected if omitted.')
    parser.add_argument('--capture-device', type=int, default=None,
                        help='Elgato device index (from --scan-devices). Omit for screen capture.')
    parser.add_argument('--monitor', type=int, default=1, help='Monitor index (screen capture only)')
    parser.add_argument('--fps', type=int, default=60, help='Capture frame rate')
    parser.add_argument('--offset', type=int, default=0, help='Initial peak Y offset')
    parser.add_argument('--landmark', type=int, default=12,
                        help='Hand landmark to track (12=middle tip, 0=wrist, 20=pinky tip)')
    parser.add_argument('--confidence', type=float, default=0.4,
                        help='Mediapipe detection confidence (0.1-1.0, lower = more permissive)')
    parser.add_argument('--debug', action='store_true',
                        help='Show live preview window with hand skeleton overlay')
    parser.add_argument('--scan-ports', action='store_true',
                        help='List serial ports with VID/PID and exit')
    parser.add_argument('--scan-devices', action='store_true',
                        help='List VideoCapture devices and exit')
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
        track_landmark=args.landmark,
        detection_confidence=args.confidence,
        tracking_confidence=args.confidence,
        debug=args.debug,
    )

    GreenMachine(cfg).run()
