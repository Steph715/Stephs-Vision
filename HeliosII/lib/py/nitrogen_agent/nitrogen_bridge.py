import sys
import os
import io
import contextlib
import time
import signal
import argparse
import ctypes
import ctypes.wintypes
import threading
import queue

print(f"Python: {sys.executable} ({sys.version.split()[0]})")
sys.stdout.flush()

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_reader import FrameReader
from controller_writer import ControllerWriter
from action_mapper import map_actions

kernel32 = ctypes.windll.kernel32
WAIT_OBJECT_0 = 0x00000000

_shutdown = False
_devnull = io.StringIO()


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True


def check_shutdown_event(event_handle):
    if event_handle and kernel32.WaitForSingleObject(event_handle, ctypes.wintypes.DWORD(0)) == WAIT_OBJECT_0:
        return True
    return _shutdown


def create_inference_direct(checkpoint, timesteps_override):
    import torch
    from nitrogen.inference_session import load_model, InferenceSession

    print("Loading model...")
    sys.stdout.flush()

    os.environ["TQDM_DISABLE"] = "1"
    import logging
    logging.disable(logging.WARNING)

    with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
        model, tokenizer, img_proc, ckpt_config, game_mapping, action_downsample_ratio = load_model(checkpoint)

    logging.disable(logging.NOTSET)

    if timesteps_override and timesteps_override > 0:
        model.num_inference_timesteps = timesteps_override

    with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
        session = InferenceSession(
            model=model,
            ckpt_path=checkpoint,
            tokenizer=tokenizer,
            img_proc=img_proc,
            ckpt_config=ckpt_config,
            game_mapping=game_mapping,
            selected_game=None,
            old_layout=False,
            cfg_scale=1.0,
            action_downsample_ratio=action_downsample_ratio,
        )
        session.reset()

    try:
        session.model = torch.compile(session.model, mode="reduce-overhead")
        print("Model compiled with torch.compile (reduce-overhead)")
    except Exception as e:
        print(f"torch.compile unavailable, using eager mode: {e}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {params:,} parameters")
    print(f"Diffusion steps: {model.num_inference_timesteps}")
    sys.stdout.flush()

    return session, action_downsample_ratio


def create_inference_zmq(host, port):
    from nitrogen.inference_client import ModelClient

    client = ModelClient(host=host, port=port)
    client.reset()
    info = client.info()
    action_downsample_ratio = info.get("action_downsample_ratio", 1)
    return client, action_downsample_ratio


def preprocess_frame(frame_bgr):
    resized = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def inference_worker(reader, policy, action_queue, shutdown_event_handle, warmup_frames=3):
    inference_count = 0
    warmup_done = False

    while not check_shutdown_event(shutdown_event_handle):
        result = reader.wait_for_frame(timeout_ms=100)
        if result is None:
            continue

        frame_bgr, width, height, channels, fmt = result
        obs = preprocess_frame(frame_bgr)

        t0 = time.perf_counter()
        with contextlib.redirect_stdout(_devnull):
            prediction = policy.predict(obs)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        actions = map_actions(prediction)
        inference_count += 1

        if not warmup_done and inference_count >= warmup_frames:
            warmup_done = True
            print("Warmup complete")
            sys.stdout.flush()

        try:
            action_queue.put((actions, inference_ms, warmup_done), timeout=0.5)
        except queue.Full:
            pass


def main():
    parser = argparse.ArgumentParser(description="NitroGen Bridge for Helios")
    parser.add_argument("--ring_buffer_name", required=True)
    parser.add_argument("--controller_output_name", required=True)
    parser.add_argument("--controller_report_name", default="")
    parser.add_argument("--shutdown_event", default="")
    parser.add_argument("--mode", choices=["direct", "zmq"], default="direct")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--server_host", default="localhost")
    parser.add_argument("--server_port", type=int, default=5555)
    parser.add_argument("--target_fps", type=int, default=60)
    parser.add_argument("--timesteps", type=int, default=0)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    shutdown_event_handle = None
    if args.shutdown_event:
        wide = ctypes.create_unicode_buffer(args.shutdown_event)
        shutdown_event_handle = kernel32.OpenEventW(0x00100000, False, wide)

    target_fps = max(1, args.target_fps)
    action_period = 1.0 / target_fps
    print(f"NitroGen bridge starting in {args.mode} mode (target {target_fps} FPS)")

    reader = FrameReader()
    writer = ControllerWriter()

    try:
        reader.connect(args.ring_buffer_name)
        print(f"Connected to ring buffer: {args.ring_buffer_name}")
    except Exception as e:
        print(f"ERROR: Failed to connect to ring buffer: {e}", file=sys.stderr)
        return 1

    try:
        writer.connect(args.controller_output_name, args.controller_report_name)
        print(f"Connected to controller output: {args.controller_output_name}")
        if args.controller_report_name:
            print(f"Connected to controller report: {args.controller_report_name}")
    except Exception as e:
        print(f"ERROR: Failed to connect to controller output: {e}", file=sys.stderr)
        reader.disconnect()
        return 1

    try:
        if args.mode == "direct":
            if not args.checkpoint:
                print("ERROR: Checkpoint path required for direct mode", file=sys.stderr)
                return 1
            print(f"Loading model from: {args.checkpoint}")
            ts = args.timesteps if args.timesteps > 0 else None
            policy, action_downsample_ratio = create_inference_direct(args.checkpoint, ts)
        else:
            print(f"Connecting to ZMQ server at {args.server_host}:{args.server_port}")
            policy, action_downsample_ratio = create_inference_zmq(args.server_host, args.server_port)
        print("Inference ready")
    except Exception as e:
        print(f"ERROR: Failed to initialize inference: {e}", file=sys.stderr)
        reader.disconnect()
        writer.disconnect()
        return 1

    action_queue = queue.Queue(maxsize=2)

    worker = threading.Thread(
        target=inference_worker,
        args=(reader, policy, action_queue, shutdown_event_handle),
        daemon=True,
    )
    worker.start()

    action_count = 0
    report_start = time.perf_counter()
    last_inference_ms = 0.0

    print("Running pipelined loop (async inference, paced output)...")
    sys.stdout.flush()

    try:
        while not check_shutdown_event(shutdown_event_handle):
            try:
                actions, inference_ms, warmup_done = action_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if inference_ms > 0:
                last_inference_ms = inference_ms

            for buttons, axes in actions:
                if check_shutdown_event(shutdown_event_handle):
                    break
                action_start = time.perf_counter()
                for _ in range(action_downsample_ratio):
                    writer.write(buttons, axes)
                    action_count += 1

                    elapsed = time.perf_counter() - action_start
                    remaining = action_period - elapsed
                    if remaining > 0.001:
                        time.sleep(remaining - 0.0005)
                    while time.perf_counter() - action_start < action_period:
                        pass
                    action_start = time.perf_counter()

            now = time.perf_counter()
            report_elapsed = now - report_start
            if warmup_done and report_elapsed >= 2.0:
                effective_fps = action_count / report_elapsed
                nonzero_btns = sum(1 for b in buttons if abs(b) > 0.01)
                nonzero_axes = sum(1 for a in axes if abs(a) > 0.01)
                print(f"FPS: {effective_fps:.1f} | inference {last_inference_ms:.0f}ms | btns:{nonzero_btns} axes:{nonzero_axes}")
                sys.stdout.flush()
                action_count = 0
                report_start = now

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        global _shutdown
        _shutdown = True
        reader.disconnect()
        writer.disconnect()
        if shutdown_event_handle:
            kernel32.CloseHandle(shutdown_event_handle)
        print(f"NitroGen bridge stopped ({action_count} actions)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
