"""Per-process runtime environment capture.

The environment must be recorded BY THE WORKER, at the moment it runs. Reading
`OMP_NUM_THREADS` from a reporting script afterwards captures that script's
shell, not the training process's -- so the paper would state a thread setting
no worker ever used.

Every script that writes a result JSON calls `snapshot()` and stores the result
under an `env` key. `record_compute.py` then reads the real values back and
flags any run whose settings differ from the rest.
"""
from __future__ import annotations

import os
import platform
import sys
import time


def snapshot() -> dict:
    """Capture what THIS process is actually using."""
    env = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pid": os.getpid(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        # thread-control variables as seen BY THIS PROCESS
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS", "unset"),
        "argv": " ".join(sys.argv),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["torch_num_threads"] = torch.get_num_threads()
        env["torch_num_interop_threads"] = torch.get_num_interop_threads()
        env["device"] = "cpu"
        env["cuda_available"] = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None)
        env["mps_available"] = bool(mps and mps.is_available())
        env["mps_used"] = False
    except Exception:
        env["torch"] = "not imported"
    try:
        import numpy
        env["numpy"] = numpy.__version__
    except Exception:
        pass
    # Peak resident memory. Measured LAST: this function imports torch, which
    # costs ~180 MB, so capturing earlier reported a "peak" BELOW the process's
    # own current RSS -- which a cross-check against `ps` caught.
    # On macOS ru_maxrss is bytes; on Linux it is kilobytes. Getting that wrong
    # is a 1024x error, so the platform is checked rather than assumed.
    try:
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        env["peak_rss_mb"] = round(
            raw / (1024 ** 2) if sys.platform == "darwin" else raw / 1024, 1)
        ch = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        if ch:
            env["peak_rss_children_mb"] = round(
                ch / (1024 ** 2) if sys.platform == "darwin" else ch / 1024, 1)
        env["peak_rss_note"] = ("ru_maxrss for this process at save time; "
                                "excludes shared library pages that `ps` RSS "
                                "counts, so it reads slightly lower than ps.")
    except Exception:
        pass
    return env
