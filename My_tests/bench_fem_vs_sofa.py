"""
Benchmark FEMMethod (pyGandalf, Taichi GPU) vs SofaMethod (SOFA v23, CPU)
on the same bunny tet mesh. Runs both backends sequentially in the same
process for identical mesh input, reports:

  1. Per-step time (uncut, static FEM)
  2. Cut setup time (setup_cut duration)
  3. Per-step time (during a cut, blade actively removing tets)
  4. Peak RSS memory delta per backend

Usage:
  ../.venv/Scripts/python.exe My_tests/bench_fem_vs_sofa.py [--mesh _bunny_jacobson]

Warm-up runs before every measurement window so JIT / solver caches are
primed. Each measurement is repeated N times and we report min/median/max
across runs.

Note: Taichi runs on GPU (arch=cuda) via pyGandalf's default init. SOFA runs
on CPU. Peak memory numbers are process RSS -- they include Taichi's
allocated GPU buffers only if the CUDA driver counts them, which varies.
"""
import argparse
import gc
import os
import statistics
import sys
import time

import numpy as np
import psutil

from pyGandalf.utilities.definitions import MODELS_PATH
from pyGandalf.utilities.mesh_lib import MeshLib
from pyGandalf.thesis_utilities.simulation_method import FEMMethod
from pyGandalf.thesis_utilities.sofa_method import SofaMethod


MESH_CONFIG = {
    '_bunny_jacobson':      15000,
    '_bunny_jacobson_full':  None,
    'sphere':                None,
}
MESH_FILE = {'_bunny_jacobson_full': '_bunny_jacobson'}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def build_cut_plane(verts, seed=5):
    """Same random-plane recipe as test_fem_cut.py so cuts are reproducible."""
    rng = np.random.default_rng(seed)
    normal = rng.standard_normal(3).astype(np.float32)
    normal /= np.linalg.norm(normal)
    depth = float(rng.uniform(-0.3, 0.3))
    origin = (normal * depth + verts.mean(axis=0)).astype(np.float32)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(normal, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    blade_dir = ref - float(np.dot(ref, normal)) * normal
    blade_dir /= np.linalg.norm(blade_dir)
    return normal, origin, blade_dir


def measure_step_times(method, n_warmup, n_measure):
    """Warm up, then time n_measure calls to method.step(dt). Return list of ms."""
    dt = float(getattr(method, "time_step", 0.01))
    for _ in range(n_warmup):
        method.step(dt)
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        method.step(dt)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def summarize(name, times_ms):
    med = statistics.median(times_ms)
    lo  = min(times_ms)
    hi  = max(times_ms)
    return f"{name:22s} min={lo:7.2f} ms  median={med:7.2f} ms  max={hi:7.2f} ms  (n={len(times_ms)})"


# ---------------------------------------------------------------------------
# One backend, four windows: uncut, setup_cut, during-cut, teardown.
# ---------------------------------------------------------------------------

def bench_backend(name, method_ctor, tet_mesh, cut_plane, params,
                  n_warmup_uncut, n_measure_uncut,
                  n_warmup_cut,   n_measure_cut,
                  n_blade_steps):
    print(f"\n{'='*72}\n  {name}\n{'='*72}")

    rss_before = rss_mb()
    peak_rss = rss_before

    method = method_ctor()
    method.initialize(tet_mesh, params)
    peak_rss = max(peak_rss, rss_mb())

    # --- Uncut per-step ---
    uncut_times = measure_step_times(method, n_warmup_uncut, n_measure_uncut)
    peak_rss = max(peak_rss, rss_mb())
    print(summarize(f"{name} uncut", uncut_times))

    # --- Cut setup time ---
    normal, origin, blade_dir = cut_plane
    t0 = time.perf_counter()
    ok = method.setup_cut(normal, origin, blade_dir)
    setup_ms = (time.perf_counter() - t0) * 1000.0
    peak_rss = max(peak_rss, rss_mb())
    print(f"{name:22s} setup_cut = {setup_ms:.2f} ms  (ok={ok})")

    # --- During-cut per-step: advance blade a bit each step ---
    # For a plane sweep, blade_travel_max is roughly the projection of the
    # mesh diagonal onto blade_dir. Divide by n_blade_steps for uniform
    # per-step advance.
    verts = np.asarray(tet_mesh.vertices, dtype=np.float64)
    proj = (verts - origin) @ blade_dir
    travel_range = float(proj.max() - proj.min())
    blade_step = travel_range / max(1, n_blade_steps)

    dt = float(params.get('time_step', 0.01))
    # Warmup cut steps
    for i in range(n_warmup_cut):
        travel = float(proj.min()) + (i + 1) * blade_step
        method.advance_blade(travel)
        method.step(dt)
    # Measure cut steps
    cut_times = []
    for i in range(n_measure_cut):
        travel = float(proj.min()) + (n_warmup_cut + i + 1) * blade_step
        t0 = time.perf_counter()
        method.advance_blade(travel)
        method.step(dt)
        cut_times.append((time.perf_counter() - t0) * 1000.0)
    peak_rss = max(peak_rss, rss_mb())
    print(summarize(f"{name} during cut", cut_times))

    rss_delta = peak_rss - rss_before
    print(f"{name:22s} peak RSS delta = {rss_delta:8.1f} MB  "
          f"(base {rss_before:.1f} -> peak {peak_rss:.1f})")

    # Teardown: drop the method so Taichi/SOFA can release memory before the
    # next backend is initialized.
    method = None
    gc.collect()

    return {
        'name':      name,
        'uncut_ms':  uncut_times,
        'setup_ms':  setup_ms,
        'cut_ms':    cut_times,
        'rss_delta': rss_delta,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', default='_bunny_jacobson',
                        choices=list(MESH_CONFIG.keys()))
    parser.add_argument('--tet-scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--n-warmup-uncut', type=int, default=10)
    parser.add_argument('--n-measure-uncut', type=int, default=50)
    parser.add_argument('--n-warmup-cut',   type=int, default=5)
    parser.add_argument('--n-measure-cut',  type=int, default=30)
    parser.add_argument('--n-blade-steps',  type=int, default=100)
    parser.add_argument('--skip-fem', action='store_true')
    parser.add_argument('--skip-sofa', action='store_true')
    args = parser.parse_args()

    mesh_file    = MESH_FILE.get(args.mesh, args.mesh)
    target_faces = MESH_CONFIG[args.mesh]

    print(f"Building tet mesh from {mesh_file}.obj (target_faces={target_faces}, "
          f"tet_scale={args.tet_scale})")
    tet_mesh = MeshLib().build_tetrahedral(
        f'{args.mesh}_bench',
        MODELS_PATH / f'{mesh_file}.obj',
        target_faces=target_faces,
        tet_scale=args.tet_scale,
    )
    n_verts = len(tet_mesh.vertices)
    n_tets  = len(tet_mesh.tetrahedra)
    print(f"  verts={n_verts}, tets={n_tets}")

    cut_plane = build_cut_plane(np.asarray(tet_mesh.vertices), seed=args.seed)

    # Shared physics params. Match test_fem_cut.py where reasonable, adapt
    # SOFA-specific keys to equivalent semantics.
    fem_params = dict(
        time_step     = 0.01,
        gravity       = [0.0, 0.0, 0.0],
        young_modulus = 5.0e4,
        poisson_ratio = 0.4,
        density       = 1000.0,
        cg_iters      = 20,
        sliver_vol_threshold = 1e-5,
    )
    sofa_params = dict(
        time_step     = 0.01,
        gravity       = [0.0, 0.0, 0.0],
        young_modulus = 5.0e4,
        poisson_ratio = 0.4,
        mass_density  = 1000.0 * 1e-3,   # density in kg/m3 -> DiagonalMass massDensity
    )

    results = []
    if not args.skip_fem:
        results.append(bench_backend(
            "FEMMethod (Taichi GPU)", FEMMethod, tet_mesh, cut_plane, fem_params,
            args.n_warmup_uncut, args.n_measure_uncut,
            args.n_warmup_cut,   args.n_measure_cut,
            args.n_blade_steps,
        ))
    if not args.skip_sofa:
        results.append(bench_backend(
            "SofaMethod (SOFA CPU)", SofaMethod, tet_mesh, cut_plane, sofa_params,
            args.n_warmup_uncut, args.n_measure_uncut,
            args.n_warmup_cut,   args.n_measure_cut,
            args.n_blade_steps,
        ))

    # --- Final summary table ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for r in results:
        med_uncut = statistics.median(r['uncut_ms']) if r['uncut_ms'] else float('nan')
        med_cut   = statistics.median(r['cut_ms'])   if r['cut_ms']   else float('nan')
        print(f"{r['name']:22s}: uncut {med_uncut:7.2f} ms/step  "
              f"setup {r['setup_ms']:7.2f} ms  "
              f"cut {med_cut:7.2f} ms/step  "
              f"peak RSS +{r['rss_delta']:.1f} MB")


if __name__ == "__main__":
    main()
