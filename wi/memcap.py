"""Hard RAM cap for every pipeline stage and the demo.

A daemon thread checks this process's resident memory every second; if it exceeds WI_MEM_CAP_GB
(default 6) the process stops with a clear message instead of pushing the Mac into swap. All stages
are resumable (features/phones save chunks) or take minutes (experiments), so stopping is safe.
GPU (MPS) memory is capped separately with PYTORCH_MPS_HIGH/LOW_WATERMARK_RATIO (see Makefile).
"""
import os
import sys
import threading
import time

import psutil

_started = False


def install(cap_gb=None, interval=1.0):
    global _started
    if _started:
        return
    _started = True
    cap = float(cap_gb or os.environ.get("WI_MEM_CAP_GB", 6)) * 1e9
    proc = psutil.Process()

    def watch():
        while True:
            rss = proc.memory_info().rss
            if rss > cap:
                sys.stderr.write(f"\n[memcap] RSS {rss / 1e9:.1f} GB exceeded the "
                                 f"{cap / 1e9:.1f} GB cap (WI_MEM_CAP_GB); stopping.\n")
                sys.stderr.flush()
                os._exit(137)
            time.sleep(interval)

    threading.Thread(target=watch, daemon=True, name="memcap").start()
