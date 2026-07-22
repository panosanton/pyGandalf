"""
Benchmark instrumentation shared by the interactive tests and the headless
benchmark script.

Two things live here:

  FrameMeter  -- rolling wall-clock frame timer that reports median fps per
                 phase ("idle" before a cut, "cutting" while the blade is
                 sweeping, "settling" afterwards).  Median rather than mean so
                 a single JIT stall or a Windows scheduler hiccup does not
                 dominate the number quoted in the thesis.

  mem_report  -- process RSS via psutil plus per-process GPU memory via
                 nvidia-smi.  nvidia-smi is queried with
                 --query-compute-apps so the number is THIS process's VRAM,
                 not the whole desktop's.

Both are measurement-only: nothing here is imported by the physics path.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------
# Everything worth keeping from an interactive session is appended here, so a
# run can be summarised into one JSON file instead of being scraped back out
# of the terminal. Written on exit when --bench-json <path> is passed.
_EVENTS: list = []


def _argv_value(flag: str, default=None):
    """Read `--flag value` straight from argv (argparse runs too late for us)."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


_JSON_PATH = _argv_value('--bench-json')


def record_event(name: str, **fields):
    """Add a named entry to the run log (e.g. cut setup time, mesh sizes)."""
    _EVENTS.append({'event': name, **fields})


# ---------------------------------------------------------------------------
# Frame timing
# ---------------------------------------------------------------------------
class FrameMeter:
    """
    Accumulates per-frame wall times and prints a rolling summary.

    Feed it the frame delta every frame via `tick(dt, phase)`.  It prints one
    line per `report_every` seconds with the median / p95 frame time and the
    implied fps for the current phase, and keeps a full per-phase history so a
    final summary can be printed on exit.

    The caller is responsible for passing an honest dt.  With vsync enabled the
    frame time is clamped to the monitor refresh interval and the fps number is
    meaningless as a benchmark -- run with --no-vsync when measuring.
    """

    def __init__(self, report_every: float = 1.0, window: int = 240,
                 warmup_frames: int = 30):
        self.report_every  = report_every
        self.window        = window
        self.warmup_frames = warmup_frames

        self._recent   = []            # (dt, phase) for the rolling window
        self._history  = {}            # phase -> list of dt
        self._n_frames = 0
        self._t_last   = time.perf_counter()
        self._enabled  = False

    def enable(self):
        self._enabled = True
        self._t_last  = time.perf_counter()

    def tick(self, dt: float, phase: str = 'idle'):
        if not self._enabled:
            return
        self._n_frames += 1
        if self._n_frames <= self.warmup_frames or dt <= 0.0:
            return

        self._recent.append((dt, phase))
        if len(self._recent) > self.window:
            self._recent.pop(0)
        self._history.setdefault(phase, []).append(dt)

        now = time.perf_counter()
        if now - self._t_last >= self.report_every:
            self._t_last = now
            self._print_rolling(phase)

    def _print_rolling(self, phase: str):
        vals = np.array([d for d, p in self._recent if p == phase], dtype=np.float64)
        if vals.size < 5:
            return
        med = float(np.median(vals))
        p95 = float(np.percentile(vals, 95))
        print(f"[Bench/{phase}] {med*1000:6.2f} ms median  "
              f"({1.0/med:6.1f} fps)   p95 {p95*1000:6.2f} ms   "
              f"n={vals.size}",
              flush=True)

    def summary(self):
        """
        Print a per-phase summary over the whole run, and write the run log to
        --bench-json if that flag was passed.

        Registered with atexit, so it also runs when the window is closed.
        """
        if not self._history:
            return
        print("=" * 70, flush=True)
        print("[Bench] frame-time summary (wall clock, whole run)", flush=True)
        print("=" * 70, flush=True)

        phases = {}
        for phase, vals in self._history.items():
            a = np.array(vals, dtype=np.float64)
            med = float(np.median(a))
            phases[phase] = {
                'n_frames':  int(a.size),
                'median_ms': med * 1000.0,
                'mean_ms':   float(a.mean()) * 1000.0,
                'p95_ms':    float(np.percentile(a, 95)) * 1000.0,
                'min_ms':    float(a.min()) * 1000.0,
                'fps_median': 1.0 / med if med > 0 else None,
            }
            print(f"  {phase:10s} n={a.size:5d}  "
                  f"median {med*1000:6.2f} ms ({1.0/med:6.1f} fps)  "
                  f"mean {a.mean()*1000:6.2f} ms  "
                  f"p95 {np.percentile(a, 95)*1000:6.2f} ms  "
                  f"min {a.min()*1000:6.2f} ms",
                  flush=True)

        if _JSON_PATH:
            try:
                with open(_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump({'phases': phases, 'events': _EVENTS}, f, indent=2)
                print(f"[Bench] wrote {_JSON_PATH}", flush=True)
            except OSError as e:
                print(f"[Bench] could not write {_JSON_PATH}: {e}", flush=True)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
def _nvsmi(query: str) -> list | None:
    """Run an nvidia-smi CSV query, returning a list of row strings."""
    try:
        out = subprocess.run(
            ['nvidia-smi', f'--query-{query}', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return [l.strip() for l in out.stdout.splitlines() if l.strip()]


def gpu_proc_mem_mb() -> float | None:
    """
    VRAM charged to THIS process, or None if the driver will not report it.

    NOTE: on a consumer Windows GPU running in WDDM mode (the normal case for
    a GeForce laptop), nvidia-smi reports used_gpu_memory as [N/A] for every
    process on the system -- the display driver owns the allocations, not the
    compute context. This therefore returns None on such machines and the
    whole-device delta below is the only usable signal. It is kept because it
    does work under TCC / on Linux, where it is the number you actually want.
    """
    rows = _nvsmi('compute-apps=pid,used_memory')
    if rows is None:
        return None
    pid = os.getpid()
    for line in rows:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 2:
            continue
        try:
            if int(parts[0]) == pid:
                return float(parts[1])
        except ValueError:      # "[N/A]" under WDDM
            continue
    return None


def gpu_device_used_mb() -> float | None:
    """Total VRAM in use across the whole device, all processes."""
    rows = _nvsmi('gpu=memory.used')
    if not rows:
        return None
    try:
        return float(rows[0].split(',')[0])
    except ValueError:
        return None


# Device-wide VRAM at import time, before Taichi has allocated anything.
# Later readings are reported as a delta against this so the desktop's own
# consumption (~1 GB of browser / shell / compositor) is subtracted out.
_GPU_BASELINE_MB = gpu_device_used_mb()


_TI_DTYPE_BYTES = {
    'f16': 2, 'f32': 4, 'f64': 8,
    'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8,
    'u8': 1, 'u16': 2, 'u32': 4, 'u64': 8,
}


def taichi_field_bytes(obj) -> int | None:
    """
    Sum the device footprint of every Taichi field held by `obj`.

    Walks the object's attributes rather than hardcoding a field list, so it
    stays correct as fields are added or renamed. This is the honest "size of
    the simulation state on the GPU" number: unlike a device-wide delta it is
    not polluted by Taichi's runtime pool, the JIT, or the desktop.

    Returns None if Taichi is not importable or the object holds no fields.
    """
    try:
        import taichi as ti
    except ImportError:
        return None

    total = 0
    found = False
    for value in vars(obj).values():
        try:
            if not isinstance(value, ti.Field):
                continue
            n_elem = 1
            for d in value.shape:
                n_elem *= int(d)
            # Vector / matrix fields carry n x m components per element.
            n_elem *= int(getattr(value, 'n', 1)) * int(getattr(value, 'm', 1))
            key = str(getattr(value, 'dtype', '')).split('.')[-1].strip()
            total += n_elem * _TI_DTYPE_BYTES.get(key, 4)
            found = True
        except Exception:                                     # noqa: BLE001
            continue
    return total if found else None


def mem_report(tag: str = '', sim=None) -> dict:
    """
    Print and return a memory snapshot.

    Keys: rss_mb (host), fields_mb (Taichi field footprint, needs `sim`),
    gpu_proc_mb (per-process VRAM, usually None on Windows), gpu_device_mb and
    gpu_delta_mb (device-wide usage, and its rise since process start).
    """
    rss_mb = None
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception as e:                                    # noqa: BLE001
        print(f"[Bench/mem] psutil unavailable: {e}", flush=True)

    fields_mb = None
    if sim is not None:
        b = taichi_field_bytes(sim)
        if b is not None:
            fields_mb = b / (1024 * 1024)

    gpu_proc_mb   = gpu_proc_mem_mb()
    gpu_device_mb = gpu_device_used_mb()
    gpu_delta_mb  = (gpu_device_mb - _GPU_BASELINE_MB
                     if gpu_device_mb is not None and _GPU_BASELINE_MB is not None
                     else None)

    def _f(v):
        return f"{v:8.1f} MB" if v is not None else "     n/a"

    label = f" {tag}" if tag else ""
    gpu_s = _f(gpu_proc_mb) if gpu_proc_mb is not None else f"{_f(gpu_delta_mb)} (device delta)"
    print(f"[Bench/mem]{label}  host RSS {_f(rss_mb)}   "
          f"ti fields {_f(fields_mb)}   GPU {gpu_s}", flush=True)

    snapshot = {'rss_mb': rss_mb, 'fields_mb': fields_mb,
                'gpu_proc_mb': gpu_proc_mb, 'gpu_device_mb': gpu_device_mb,
                'gpu_delta_mb': gpu_delta_mb}
    record_event('memory', tag=tag, **snapshot)
    return snapshot
