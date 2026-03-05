#!/usr/bin/env python3
import sys
import os
import shutil
import ctypes
import ctypes.wintypes as wintypes
import threading
import time
import signal

_hidden_hwnd = None
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle SIGTERM signal for graceful shutdown"""
    global _shutdown_requested
    print(f"Received signal {signum} - requesting shutdown")
    _shutdown_requested = True


class POINT(ctypes.Structure):
    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG)
    ]

class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", POINT)
    ]

class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", ctypes.WINFUNCTYPE(wintypes.LONG, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)),
        ("cbClsExtra", wintypes.INT),
        ("cbWndExtra", wintypes.INT),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HANDLE),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HANDLE),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR)
    ]


def load_gcv_wrapper():
    try:
        import GCVWrapper
        print(f"Loaded GCVWrapper for Python {sys.version_info.major}.{sys.version_info.minor}")
        return GCVWrapper
    except ImportError as e:
        raise RuntimeError(f"Failed to import GCVWrapper: {e}")

def set_performance_optimizations():
    try:
        kernel32 = ctypes.windll.kernel32
        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        THREAD_PRIORITY_ABOVE_NORMAL = 1
        
        PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
        ProcessPowerThrottling = 4

        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [
                ("Version", wintypes.ULONG),
                ("ControlMask", wintypes.ULONG),
                ("StateMask", wintypes.ULONG),
            ]

        handle = kernel32.GetCurrentProcess()
        kernel32.SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS)

        # Disable power throttling
        kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, wintypes.ULONG, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        
        throttling = PROCESS_POWER_THROTTLING_STATE()
        throttling.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION
        throttling.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        throttling.StateMask = 0
        kernel32.SetProcessInformation(handle, ProcessPowerThrottling, ctypes.byref(throttling), ctypes.sizeof(throttling))

        thread_handle = kernel32.GetCurrentThread()
        kernel32.SetThreadPriority(thread_handle, THREAD_PRIORITY_ABOVE_NORMAL)
    except Exception:
        pass

def main():
    global _shutdown_requested
    try:
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal_handler)
        
        if sys.platform == "win32":
            set_performance_optimizations()
        GCVWrapper = load_gcv_wrapper()
        if len(sys.argv) >= 12 and sys.argv[1] == "--script_path":
            script_path = sys.argv[2]
            process_id = sys.argv[4]
            video_ring_buffer_name = sys.argv[6]
            gcv_data_name = sys.argv[8]
            controller_input_name = sys.argv[10]
            controller_report_name = sys.argv[12]
            # Check for optional shutdown event parameter
            shutdown_event_name = None
            if len(sys.argv) >= 15 and sys.argv[13] == "--shutdown_event_name_param":
                shutdown_event_name = sys.argv[14]
        else:
            script_path = sys.argv[1]
            process_id = sys.argv[2]
            video_ring_buffer_name = sys.argv[3]
            gcv_data_name = sys.argv[4]
            controller_input_name = sys.argv[5]
            controller_report_name = sys.argv[6]
            shutdown_event_name = None
            if len(sys.argv) >= 8:
                shutdown_event_name = sys.argv[7]
        
        # Pass shutdown flag reference to GCVWrapper
        GCVWrapper.main(script_path, process_id, video_ring_buffer_name,
                       gcv_data_name, controller_input_name, controller_report_name,
                       shutdown_event_name, lambda: _shutdown_requested)
    except ImportError as e:
        print("Error Code: SYS_001 - Failed to import GCVWrapper module")
        print("Make sure the GCVWrapper.cp*-win_amd64.pyd file for your Python version is available")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Shutdown requested via keyboard interrupt")
        sys.exit(0)
    except Exception as e:
        # Check if this looks like a user script error or system error
        error_msg = str(e)
        if "Script Error:" in error_msg or "Error Code:" in error_msg:
            # Already formatted error from GCVWrapper
            print(error_msg)
        else:
            # Unhandled system error
            print(f"Error Code: SYS_999 - Unexpected launcher error: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()