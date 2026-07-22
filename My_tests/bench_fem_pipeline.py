"""
Headless FEM benchmark -- the numbers that go in the thesis Results slides.

No window, no rendering, no ECS: this measures the physics only, which is what
makes it comparable to the SOFA C++ numbers (runSofa -g batch, which also
renders nothing). The interactive figure with rendering included comes from
`test_fem_cut.py --bench --no-vsync` instead.

Reported per phase:
  uncut     -- steady-state FEM step cost on the intact mesh
               (compare against SOFA's TetrahedronFEMForceField batch runs)
  cut setup -- one-shot wall time of setup_cut()
  cutting   -- step cost while the blade sweeps (topology grows every frame in
               progressive mode; compare against the SOFA carving scene)
  settling  -- step cost after the blade finishes, mesh still in two halves

Two timing methodologies are printed for every phase:
  per-step  -- ti.sync() after each step, median over the run. Robust to
               outliers, but the sync barrier adds a small fixed overhead.
  block     -- one sync at the end, total wall time / n steps. This is exactly
               how the SOFA batch numbers were produced, so THIS is the number
               to put next to them in the comparison table.

Usage:
    python My_tests/bench_fem_pipeline.py --mesh _bunny_jacobson
    python My_tests/bench_fem_pipeline.py --mesh _bunny_jacobson --steps 200 --json out.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import taichi as ti

from pyGandalf.utilities.mesh_lib import MeshLib
from pyGandalf.utilities.definitions import MODELS_PATH
from pyGandalf.thesis_utilities.simulation_method import FEMMethod
from pyGandalf.thesis_utilities.bench_utils import mem_report


# Must stay identical to test_fem_cut.py, and --tet_scale must match whatever
# export_tet_vtk.py used for the VTK handed to SOFA, or the comparison is void.
MESH_CONFIG = {
    'sphere':                   None,
    'Armadillo_verysimplified': None,
    'dragon_clean':             1500,
    'liver-smooth':             3500,
    'Armadillo_simplified':     None,
    '_bunny_jacobson':          15000,
}

SIM_PARAMS = dict(
    time_step             = 0.01,
    substeps              = 1,
    stiffness             = 200.0,
    damping               = 2.0,
    total_mass            = 100.0,
    gravity               = [0.0, 0.0, 0.0],
    opening_speed         = 5.0,
    v_max                 = 20.0,
    spring_damping        = 0.0,
    blade_speed           = 0.5,
    opening_ramp_frames   = 20,
)


def _random_cut_plane(rng: np.random.Generator, depth_range: float = 0.7):
    normal = rng.standard_normal(3).astype(np.float32)
    normal /= np.linalg.norm(normal)
    offset = rng.uniform(-depth_range, depth_range)
    origin = (normal * offset).astype(np.float32)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(normal, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    blade_dir = ref - float(np.dot(ref, normal)) * normal
    blade_dir /= np.linalg.norm(blade_dir)
    return normal, origin, blade_dir


def _stats(samples_s: list, label: str) -> dict:
    """Median / mean / p95 / min of a list of per-step wall times in seconds."""
    a = np.array(samples_s, dtype=np.float64)
    if a.size == 0:
        return {}
    med = float(np.median(a))
    out = {
        'n_steps':   int(a.size),
        'median_ms': med * 1000.0,
        'mean_ms':   float(a.mean()) * 1000.0,
        'p95_ms':    float(np.percentile(a, 95)) * 1000.0,
        'min_ms':    float(a.min()) * 1000.0,
        'fps_median': 1.0 / med if med > 0 else float('inf'),
    }
    print(f"  {label:14s} per-step: median {out['median_ms']:7.2f} ms "
          f"({out['fps_median']:7.2f} fps)  mean {out['mean_ms']:7.2f}  "
          f"p95 {out['p95_ms']:7.2f}  min {out['min_ms']:7.2f}  "
          f"n={out['n_steps']}", flush=True)
    return out


def _timed_block(method, dt: float, n_steps: int, label: str) -> dict:
    """
    Run n_steps with a single sync at the end -- the SOFA batch methodology.

    Also collects per-step times so both methodologies come from the same run.
    """
    per_step = []
    ti.sync()
    t_block = time.perf_counter()
    for _ in range(n_steps):
        t0 = time.perf_counter()
        method.step(dt)
        ti.sync()
        per_step.append(time.perf_counter() - t0)
    ti.sync()
    block_s = time.perf_counter() - t_block

    out = _stats(per_step, label)
    out['block_total_s']  = block_s
    out['block_ms_step']  = block_s / max(n_steps, 1) * 1000.0
    out['block_fps']      = n_steps / block_s if block_s > 0 else float('inf')
    print(f"  {label:14s} block   : {out['block_ms_step']:7.2f} ms/step "
          f"({out['block_fps']:7.2f} fps)  total {block_s:.2f} s"
          "   <-- SOFA-comparable", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description='Headless FEM benchmark')
    p.add_argument('--mesh', default='_bunny_jacobson', choices=list(MESH_CONFIG.keys()))
    p.add_argument('--tet_scale', type=float, default=1.0,
                   help='Must match the value used by export_tet_vtk.py for the SOFA VTK')
    p.add_argument('--seed', type=int, default=5, help='Cut plane RNG seed (matches test_fem_cut)')
    p.add_argument('--cg_iters', type=int, default=25,
                   help='CG iterations (default: 25, matching the CGLinearSolver cap in the '
                        'SOFA scenes). Note ours runs a FIXED count -- the solver is zero-sync, '
                        'so it never reads a residual back to test convergence -- whereas SOFA '
                        'treats 25 as a cap and stops early, so this figure is conservative.')
    p.add_argument('--warmup', type=int, default=30, help='Untimed steps to absorb JIT')
    p.add_argument('--steps', type=int, default=100, help='Timed steps for the uncut phase')
    p.add_argument('--settle', type=int, default=100, help='Timed steps after the blade finishes')
    p.add_argument('--max_cut_frames', type=int, default=4000, help='Safety cap on the blade sweep')
    p.add_argument('--no_cut', action='store_true', help='Uncut phase only')
    p.add_argument('--gravity', type=float, nargs=3, default=None,
                   metavar=('GX', 'GY', 'GZ'),
                   help='Override gravity, to match a loaded SOFA scene. Our per-step cost '
                        'is insensitive to this (the PCG runs a fixed iteration count), but '
                        'SOFA\'s is not, so both sides must be run under the same load for '
                        'the comparison to mean anything.')
    p.add_argument('--json', default=None, help='Write results to this JSON file')
    args = p.parse_args()

    if args.gravity is not None:
        SIM_PARAMS['gravity'] = list(args.gravity)
    results_gravity = list(SIM_PARAMS['gravity'])

    results = {'mesh': args.mesh, 'tet_scale': args.tet_scale, 'seed': args.seed,
               'cg_iters': args.cg_iters, 'gravity': results_gravity}

    rng = np.random.default_rng(args.seed)
    normal, origin, blade_dir = _random_cut_plane(rng)

    print("=" * 78)
    print(f"Headless FEM benchmark -- {args.mesh} (tet_scale={args.tet_scale})")
    print("=" * 78)

    t_mesh = time.perf_counter()
    tet_mesh = MeshLib().build_tetrahedral(
        f'{args.mesh}_tet',
        MODELS_PATH / f'{args.mesh}.obj',
        target_faces=MESH_CONFIG[args.mesh],
        tet_scale=args.tet_scale,
    )
    t_mesh = time.perf_counter() - t_mesh

    n_v, n_t = len(tet_mesh.vertices), len(tet_mesh.tetrahedra)
    print(f"  Tetrahedralization: {t_mesh:.2f} s")
    print(f"  Vertices:   {n_v:,}")
    print(f"  Tetrahedra: {n_t:,}")
    results.update({'n_verts': n_v, 'n_tets': n_t, 'tetgen_s': t_mesh})

    method = FEMMethod(cg_iters=args.cg_iters, sliver_vol_threshold=1e-5,
                       progressive_cut=True)
    method.initialize(tet_mesh, SIM_PARAMS)
    dt = method.time_step

    results['mem_after_init'] = mem_report('after init', method._simulator)

    # ---- Phase 1: uncut steady state -------------------------------------
    print(f"\n[Phase 1] uncut  (warmup {args.warmup}, timed {args.steps})")
    for _ in range(args.warmup):
        method.step(dt)
    ti.sync()
    results['uncut'] = _timed_block(method, dt, args.steps, 'uncut')
    results['mem_after_uncut'] = mem_report('after uncut phase', method._simulator)

    if args.no_cut:
        _finish(results, args)
        return

    # ---- Phase 2: cut setup ----------------------------------------------
    print(f"\n[Phase 2] cut setup")
    ti.sync()
    t0 = time.perf_counter()
    ok = method.setup_cut(normal, origin, blade_dir)
    ti.sync()
    t_setup = time.perf_counter() - t0
    if not ok:
        print("  Cut plane missed the mesh -- rerun with a different --seed.")
        _finish(results, args)
        return
    print(f"  setup_cut wall time: {t_setup:.3f} s", flush=True)
    results['cut_setup_s'] = t_setup
    results['mem_after_setup'] = mem_report('after cut setup', method._simulator)

    # ---- Phase 3: blade sweep --------------------------------------------
    # Mirrors _setup_progressive_cut's start-position logic: in progressive
    # mode there are no seam pairs yet, so fall back to the earliest scheduled
    # split, otherwise the blade burns thousands of frames traveling to the
    # mesh from travel=0.
    blade_travel = method.initial_blade_travel
    if blade_travel == 0.0 and getattr(method, '_progressive_cut', False):
        sched, cross = method._split_schedule, method._crossing_mask
        if sched is not None and cross is not None and cross.any():
            blade_travel = float(sched[cross].min()) - 1e-3
    blade_speed = method.blade_speed

    print(f"\n[Phase 3] cutting  (blade_speed={blade_speed}, start travel={blade_travel:.4f})")
    per_step = []
    ti.sync()
    t_block = time.perf_counter()
    frames, done = 0, False
    while not done and frames < args.max_cut_frames:
        t0 = time.perf_counter()
        blade_travel += blade_speed * dt
        done = method.advance_blade(blade_travel)
        method.step(dt)
        ti.sync()
        per_step.append(time.perf_counter() - t0)
        frames += 1
    block_s = time.perf_counter() - t_block

    if frames >= args.max_cut_frames:
        print(f"  WARNING: hit --max_cut_frames ({args.max_cut_frames}); "
              f"the cut did not report completion.", flush=True)

    cut = _stats(per_step, 'cutting')
    cut['block_total_s'] = block_s
    cut['block_ms_step'] = block_s / max(frames, 1) * 1000.0
    cut['block_fps']     = frames / block_s if block_s > 0 else float('inf')
    cut['frames']        = frames
    cut['completed']     = bool(done)
    print(f"  {'cutting':14s} block   : {cut['block_ms_step']:7.2f} ms/step "
          f"({cut['block_fps']:7.2f} fps)  total {block_s:.2f} s over {frames} frames"
          "   <-- SOFA-comparable", flush=True)
    results['cutting'] = cut
    results['mem_after_cut'] = mem_report('after blade sweep', method._simulator)

    print(f"  Vertices after cut: {method.vertex_count:,} "
          f"(was {results['n_verts']:,})")
    results['n_verts_after_cut'] = int(method.vertex_count)

    # ---- Phase 4: settling ------------------------------------------------
    print(f"\n[Phase 4] settling  (timed {args.settle})")
    results['settling'] = _timed_block(method, dt, args.settle, 'settling')
    results['mem_final'] = mem_report('final', method._simulator)

    _finish(results, args)


def _finish(results: dict, args):
    print("\n" + "=" * 78)
    print("SUMMARY (block methodology -- directly comparable to runSofa -g batch)")
    print("=" * 78)
    print(f"  mesh                : {results['mesh']}  "
          f"{results.get('n_verts', 0):,} verts / {results.get('n_tets', 0):,} tets")
    for phase in ('uncut', 'cutting', 'settling'):
        if phase in results and 'block_fps' in results[phase]:
            r = results[phase]
            print(f"  {phase:20s}: {r['block_ms_step']:8.2f} ms/step  "
                  f"{r['block_fps']:8.2f} fps")
    if 'cut_setup_s' in results:
        print(f"  {'cut setup':20s}: {results['cut_setup_s']:8.3f} s")
    for key, label in (('mem_after_init', 'mem uncut'),
                       ('mem_final', 'mem after cut')):
        m = results.get(key) or {}
        if m.get('rss_mb') is None:
            continue
        fields = f"{m['fields_mb']:.1f} MB" if m.get('fields_mb') is not None else "n/a"
        # Per-process VRAM where the driver reports it, device-wide rise since
        # process start otherwise (Windows/WDDM). Labelled so the two are never
        # confused in the write-up.
        if m.get('gpu_proc_mb') is not None:
            gpu = f"{m['gpu_proc_mb']:.0f} MB (process)"
        elif m.get('gpu_delta_mb') is not None:
            gpu = f"{m['gpu_delta_mb']:.0f} MB (device delta)"
        else:
            gpu = "n/a"
        print(f"  {label:20s}: host {m['rss_mb']:8.1f} MB   "
              f"ti fields {fields:>12s}   GPU {gpu}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == '__main__':
    main()
