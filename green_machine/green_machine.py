"""
Green Machine — 2K26 Ball Tracker
===================================
Tracks the shooting hand's Y position using Mediapipe Hand Landmarks.
Fires the shot release via Titan Two when the hand peaks.

Hardware: Titan Two (ConsoleTuner VID 0x04D8)
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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import mss
import pytesseract


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # ROI for hand tracking (x, y, width, height)
    # Narrowing to just the shooting side reduces false detections.
    # (0, 0, 1920, 1080) = full frame
    hand_roi: Tuple[int, int, int, int] = (0, 0, 1920, 1080)

    # Landmark to track.  12 = middle fingertip (highest at full extension)
    # 0 = wrist (fires a bit earlier),  20 = pinky tip
    track_landmark: int = 12

    # ROI for shot feedback text (Early / Late / Green)
    feedback_roi: Tuple[int, int, int, int] = (550, 650, 300, 80)

    # Frames hand must be moving DOWN before trigger fires (1 = fastest)
    peak_confirm_frames: int = 1

    # Mediapipe confidence — rendered 2K hands need lower values than real hands
    detection_confidence: float = 0.4
    tracking_confidence: float = 0.4

    # Serial port for Titan Two (None = auto-detect by VID 0x04D8)
    serial_port: Optional[str] = None
    baud_rate: int = 9600

    # Byte sent to GPC script to trigger the shot release
    release_button: bytes = b'\x58'
    release_duration: float = 0.05

    # Peak Y-offset (auto-adjusted by Early/Late feedback)
    peak_offset: int = 0
    offset_step: int = 3

    # Elgato / capture card device index (None = mss screen capture)
    capture_device: Optional[int] = None
    monitor_index: int = 1
    capture_fps: int = 60

    # Optional player name list shown in the overlay (top-left)
    overlay_names: List[str] = field(default_factory=list)

    # Show the live overlay window
    show_window: bool = True


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

IDLE       = "IDLE"
TRACKING   = "TRACKING"
TRIGGERED  = "TRIGGERED"

STATE_COLORS = {
    IDLE:      (0, 255,  0),    # green
    TRACKING:  (0, 165, 255),   # orange
    TRIGGERED: (0, 255, 255),   # cyan/yellow
}


# ---------------------------------------------------------------------------
# Hand Tracker
# ---------------------------------------------------------------------------

class HandTracker:
    """Mediapipe hand landmark tracker + peak detector."""

    def __init__(self, config: Config):
        self.cfg = config
        self._mp_hands = mp.solutions.hands
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )
        self.y_history: deque = deque(maxlen=8)
        self.down_count = 0
        self.last_result = None

    def find_hand(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Returns pixel (x, y) of the tracked landmark, or None."""
        rx, ry, rw, rh = self.cfg.hand_roi
        rgb = cv2.cvtColor(frame[ry:ry+rh, rx:rx+rw], cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.last_result = self.hands.process(rgb)

        if not self.last_result.multi_hand_landmarks:
            return None

        lm = self.last_result.multi_hand_landmarks[0].landmark[self.cfg.track_landmark]
        return int(lm.x * rw) + rx, int(lm.y * rh) + ry

    def update(self, py: int) -> bool:
        """Feed Y position. Returns True at peak (trigger now)."""
        self.y_history.append(py)
        if len(self.y_history) < 2:
            return False
        if self.y_history[-1] > self.y_history[-2] + self.cfg.peak_offset:
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
    PATTERN = re.compile(r'\b(early|late|green|slightly early|slightly late)\b', re.IGNORECASE)

    def __init__(self, config: Config):
        self.cfg = config

    def read(self, frame: np.ndarray) -> Optional[str]:
        x, y, w, h = self.cfg.feedback_roi
        roi = cv2.resize(frame[y:y+h, x:x+w], None, fx=3, fy=3,
                         interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY
        )
        text = pytesseract.image_to_string(thresh, config='--psm 7')
        m = self.PATTERN.search(text)
        return m.group(0).lower() if m else None


# ---------------------------------------------------------------------------
# Controller (Titan Two)
# ---------------------------------------------------------------------------

class Controller:
    TITAN_TWO_VID = 0x04D8

    def __init__(self, config: Config):
        self.cfg = config
        self.ser: Optional[serial.Serial] = None

    def connect(self) -> bool:
        if self.cfg.serial_port:
            return self._open(self.cfg.serial_port)

        candidates = sorted(
            [p for p in serial.tools.list_ports.comports()
             if p.vid == self.TITAN_TWO_VID],
            key=lambda p: p.device,
        )
        if not candidates:
            print("[Controller] Titan Two not found. Run --scan-ports to see all ports.")
            print("[Controller] DRY RUN mode.")
            return False

        for c in candidates:
            print(f"[Controller] Trying {c.device} ({c.description}) ...")
            if self._open(c.device):
                return True

        print("[Controller] All T2 ports failed. DRY RUN mode.")
        return False

    def _open(self, port: str) -> bool:
        try:
            self.ser = serial.Serial(
                port, self.cfg.baud_rate,
                timeout=1, write_timeout=1,
                rtscts=False, dsrdtr=False,
            )
            self.ser.rts = False
            self.ser.dtr = False
            time.sleep(0.15)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            print(f"[Controller] Connected on {port} @ {self.cfg.baud_rate} baud")
            return True
        except serial.SerialException as e:
            print(f"[Controller] {port} failed: {e}")
            self.ser = None
            return False

    @staticmethod
    def scan_ports():
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
# Overlay renderer  (Helios-style window)
# ---------------------------------------------------------------------------

# Display dimensions
_DW, _DH   = 960, 540   # game frame
_INFO_H    = 90          # black info strip height
_INSET_W   = 160
_INSET_H   = 90

def _txt(img, text, pos, scale, color, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv2.LINE_AA)

def render_window(frame: np.ndarray,
                  tracker: HandTracker,
                  pos: Optional[Tuple[int, int]],
                  state: str,
                  shot_count: int,
                  green_count: int,
                  peak_offset: int,
                  overlay_names: List[str]) -> np.ndarray:
    """Build the full display frame (game feed + overlays + info strip)."""
    cfg = tracker.cfg
    scale_x = _DW / frame.shape[1]
    scale_y = _DH / frame.shape[0]
    sc = STATE_COLORS.get(state, (255, 255, 255))

    # --- Resize game frame ---
    disp = cv2.resize(frame, (_DW, _DH))

    # --- ROI bounding box ---
    rx, ry, rw, rh = cfg.hand_roi
    cv2.rectangle(
        disp,
        (int(rx * scale_x), int(ry * scale_y)),
        (int((rx + rw) * scale_x), int((ry + rh) * scale_y)),
        (0, 255, 0), 2,
    )

    # --- Hand landmark dots ---
    if tracker.last_result and tracker.last_result.multi_hand_landmarks:
        for hand_lm in tracker.last_result.multi_hand_landmarks:
            for lm in hand_lm.landmark:
                cx = int(lm.x * rw * scale_x + rx * scale_x)
                cy = int(lm.y * rh * scale_y + ry * scale_y)
                cv2.circle(disp, (cx, cy), 3, (0, 200, 0), -1)

    # --- Tracking circle + crosshair at tracked landmark ---
    if pos:
        cx = int(pos[0] * scale_x)
        cy = int(pos[1] * scale_y)
        cv2.circle(disp, (cx, cy), 22, (0, 220, 255), 2)                # yellow ring
        cv2.line(disp, (cx - 38, cy), (cx + 38, cy), (0, 255, 0), 1)   # H crosshair
        cv2.line(disp, (cx, cy - 38), (cx, cy + 38), (0, 255, 0), 1)   # V crosshair
        _txt(disp, str(pos[1]), (cx + 26, cy - 8), 0.45, (0, 220, 255), 1)

    # --- State badge top-left ---
    _txt(disp, f'[{state}]', (10, 28), 0.8, sc, 2)

    # --- Player name list ---
    for i, name in enumerate(overlay_names):
        _txt(disp, f'* {name}', (10, 54 + i * 22), 0.5, (255, 255, 255), 1)

    # --- Shot stats top-right ---
    pct = int(green_count / shot_count * 100) if shot_count else 0
    stats = f'{green_count}/{shot_count} ({pct}%)'
    (tw, _), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    _txt(disp, stats, (_DW - tw - 10, 28), 0.55, (255, 255, 255), 1)

    # --- Mini ROI inset (bottom-left corner of game frame) ---
    roi_crop = frame[ry:ry + rh, rx:rx + rw]
    if roi_crop.size > 0:
        inset = cv2.resize(roi_crop, (_INSET_W, _INSET_H))
        y1, y2 = _DH - _INSET_H - 5, _DH - 5
        x1, x2 = 5, 5 + _INSET_W
        disp[y1:y2, x1:x2] = inset
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 200, 200), 1)

    # --- Black info strip ---
    strip = np.zeros((_INFO_H, _DW, 3), dtype=np.uint8)
    cv2.line(strip, (0, 0), (_DW, 0), (55, 55, 55), 1)   # separator
    y_val = pos[1] if pos else 0
    _txt(strip, f'State: {state}',         ( 20, 55), 0.75, sc,              2)
    _txt(strip, f'Ball Y: {y_val}',        (340, 55), 0.75, (255, 255, 255), 2)
    _txt(strip, f'Offset: {peak_offset}px',(680, 55), 0.75, (255, 255, 255), 2)

    return np.vstack([disp, strip])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class GreenMachine:
    def __init__(self, config: Config = None):
        self.cfg = config or Config()
        self.tracker   = HandTracker(self.cfg)
        self.feedback  = FeedbackReader(self.cfg)
        self.controller = Controller(self.cfg)
        self.shot_count  = 0
        self.green_count = 0
        self.trigger_cooldown = 0
        self.state = IDLE
        self._state_timer = 0  # frames left showing TRIGGERED badge

    def run(self):
        self.controller.connect()
        if self.cfg.show_window:
            cv2.namedWindow("Green Machine — 2K26 Ball Tracker", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Green Machine — 2K26 Ball Tracker", _DW, _DH + _INFO_H)

        frame_interval = 1.0 / self.cfg.capture_fps
        print("[Green Machine] Running. Press Q in window or Ctrl+C to stop.")

        try:
            if self.cfg.capture_device is not None:
                self._run_capture_card(frame_interval)
            else:
                self._run_screen_capture(frame_interval)
        finally:
            self.tracker.close()
            if self.cfg.show_window:
                cv2.destroyAllWindows()

    # ---- capture paths ------------------------------------------------

    def _run_capture_card(self, frame_interval: float):
        cap = cv2.VideoCapture(self.cfg.capture_device, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.cfg.capture_device)
        if not cap.isOpened():
            print(f"[Capture] Cannot open device {self.cfg.capture_device}. "
                  "Run --scan-devices to list devices.")
            self.controller.close()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
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
                    print("[Capture] Frame read failed.")
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

    # ---- per-frame logic ----------------------------------------------

    def _process_frame(self, frame: np.ndarray, t0: float, frame_interval: float):
        # State timer — hold TRIGGERED badge for ~20 frames then revert
        if self._state_timer > 0:
            self._state_timer -= 1
            if self._state_timer == 0:
                self.state = IDLE

        # Shot feedback cooldown
        if self.trigger_cooldown > 0:
            self.trigger_cooldown -= 1
            result = self.feedback.read(frame)
            if result:
                self._apply_feedback(result)
            self._show(frame, None)
            self._sleep_remainder(t0, frame_interval)
            return

        pos = self.tracker.find_hand(frame)

        if pos is None:
            if self._state_timer == 0:
                self.state = IDLE
            self.tracker.reset()
        else:
            if self.state == IDLE:
                self.state = TRACKING
            _, py = pos
            if self.tracker.update(py):
                self.controller.release_shot()
                self.shot_count += 1
                self.state = TRIGGERED
                self._state_timer = 20
                self.tracker.reset()
                self.trigger_cooldown = 90
                print(f"[Shot #{self.shot_count}] Peak Y={py}  offset={self.cfg.peak_offset}")

        self._show(frame, pos)
        self._sleep_remainder(t0, frame_interval)

    def _show(self, frame: np.ndarray, pos: Optional[Tuple[int, int]]):
        if not self.cfg.show_window:
            return
        img = render_window(
            frame, self.tracker, pos,
            self.state, self.shot_count, self.green_count,
            self.cfg.peak_offset, self.cfg.overlay_names,
        )
        cv2.imshow("Green Machine — 2K26 Ball Tracker", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

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

    parser = argparse.ArgumentParser(description='Green Machine — 2K26 Ball Tracker')
    parser.add_argument('--port',           help='Titan Two COM port. Auto-detected if omitted.')
    parser.add_argument('--capture-device', type=int, default=None,
                        help='Elgato device index (from --scan-devices).')
    parser.add_argument('--monitor',        type=int, default=1,
                        help='Monitor index (screen capture only)')
    parser.add_argument('--fps',            type=int, default=60)
    parser.add_argument('--offset',         type=int, default=0,
                        help='Initial peak Y offset')
    parser.add_argument('--landmark',       type=int, default=12,
                        help='12=middle tip  0=wrist  20=pinky tip')
    parser.add_argument('--confidence',     type=float, default=0.4,
                        help='Mediapipe detection confidence (0.1-1.0)')
    parser.add_argument('--names',          nargs='*', default=[],
                        help='Player names shown in overlay  e.g. --names Steph Leeky')
    parser.add_argument('--no-window',      action='store_true',
                        help='Disable the overlay window (headless mode)')
    parser.add_argument('--scan-ports',     action='store_true')
    parser.add_argument('--scan-devices',   action='store_true')
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
        overlay_names=args.names,
        show_window=not args.no_window,
    )

    GreenMachine(cfg).run()
