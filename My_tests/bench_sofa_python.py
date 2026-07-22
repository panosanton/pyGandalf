"""
SofaPython3 benchmark -- the Python-SOFA column of the thesis comparison.

Deliberately SOFA-only. The FEM numbers come from `bench_fem_pipeline.py` on
this branch; the `sofa-comparison` branch predates the current FEM work, so
running its FEM side would compare against stale code.

REQUIRES `pyGandalf/thesis_utilities/sofa_method.py`, which lives on the
`tetrahedral-meshes` / `sofa-comparison` branches and is NOT part of this one.
To run this script here, copy it across first:
    git show tetrahedral-meshes:pyGandalf/thesis_utilities/sofa_method.py \
        > pyGandalf/thesis_utilities/sofa_method.py
Note that pyGandalf is installed editable against this checkout, so running
from a second worktree imports THIS tree's code regardless of directory --
copying the file is the working approach, not switching directories.

Methodology is copied from bench_fem_pipeline.py on purpose -- same mesh, same
cut plane recipe and seed, same warmup, same per-step and block statistics --
so the two JSON files can be read side by side.

Two things about the numbers that MUST be carried into the write-up:

  1. `removeTetrahedra` is not bound in SofaPython3 v23, so SofaMethod removes
     tetrahedra by rewriting the topology container's Data field. Whether SOFA
     propagates that to the force field and the surface mapping is asserted in
     a comment there, not verified. Treat the during-cut figure as indicative.
  2. Each advance_blade call that removes anything rebuilds a signature lookup
     over EVERY current tetrahedron in Python, because SOFA renumbers tets on
     removal. Measured at 170k tets under gravity: a removing step costs about
     1,208 ms against 829 ms for a plain step, so the Python bookkeeping is
     roughly 380 ms, about a third of the cost -- significant, but the solve
     still dominates. Do not claim bookkeeping is the whole story.

Usage:
    python My_tests/bench_sofa_python.py --mesh _bunny_jacobson --json sofa_bunny.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyGandalf.utilities.mesh_lib import MeshLib
from pyGandalf.utilities.definitions import MODELS_PATH
from pyGandalf.thesis_utilities.sofa_method import SofaMethod
from pyGandalf.thesis_utilities.bench_utils import mem_report

MESH_CONFIG = {
    'sphere':                   None,
    'dragon_clean':             1500,
    'liver-smooth':             3500,
    '_bunny_jacobson':          15000,
}

# Matched to the FEM run: same E, nu, dt, and the same CG cap (25) is already
# hardcoded in SofaMethod's scene template. `mass_density` feeds DiagonalMass,
# so it is the kg/m^3 density scaled the way the original bench scaled it.
SIM_PARAMS = dict(
    time_step     = 0.01,
    gravity       = [0.0, 0.0, 0.0],
    young_modulus = 5.0e4,
    poisson_ratio = 0.4,
    mass_density  = 1.0,
)


def _random_cut_plane(rng: np.random.Generator, depth_range: float = 0.7):
    """Identical to bench_fem_pipeline.py so both runs cut the same plane."""
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
    a = np.array(samples_s, dtype=np.float64)
    if a.size == 0:
        return {}
    med = float(np.median(a))
    out = {
        'n_steps':    int(a.size),
        'median_ms':  med * 1000.0,
        'mean_ms':    float(a.mean()) * 1000.0,
        'p95_ms':     float(np.percentile(a, 95)) * 1000.0,
        'min_ms':     float(a.min()) * 1000.0,
        'fps_median': 1.0 / med if med > 0 else float('inf'),
    }
    print(f"  {label:14s} per-step: median {out['median_ms']:8.2f} ms "
          f"({out['fps_median']:7.2f} fps)  mean {out['mean_ms']:8.2f}  "
          f"p95 {out['p95_ms']:8.2f}  min {out['min_ms']:8.2f}  "
          f"n={out['n_steps']}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description='SofaPython3 benchmark')
    p.add_argument('--mesh', default='_bunny_jacobson', choices=list(MESH_CONFIG.keys()))
    p.add_argument('--tet_scale', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=5)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--steps', type=int, default=100, help='Timed steps, uncut phase')
    p.add_argument('--cut_steps', type=int, default=60,
                   help='Timed steps in the during-cut window. Kept small because each '
                        'removal rebuilds an O(n_tets) lookup in Python.')
    p.add_argument('--no_cut', action='store_true')
    p.add_argument('--gravity', type=float, nargs=3, default=None,
                   metavar=('GX', 'GY', 'GZ'),
                   help='Override gravity. IMPORTANT: with zero gravity the mesh sits at '
                        'rest, SOFA\'s CGLinearSolver meets its 1e-9 tolerance almost '
                        'immediately and exits after ~1 iteration instead of 25, so the '
                        'per-step cost is not representative. Pass "0 -1 0" to load the '
                        'body and make the solver do real work.')
    p.add_argument('--json', default=None)
    args = p.parse_args()

    if args.gravity is not None:
        SIM_PARAMS['gravity'] = list(args.gravity)

    results = {'backend': 'SofaPython3', 'mesh': args.mesh, 'seed': args.seed,
               'params': {k: v for k, v in SIM_PARAMS.items()}}

    rng = np.random.default_rng(args.seed)
    normal, origin, blade_dir = _random_cut_plane(rng)

    print("=" * 78)
    print(f"SofaPython3 benchmark -- {args.mesh}  gravity={SIM_PARAMS['gravity']}")
    print("=" * 78)

    tet_mesh = MeshLib().build_tetrahedral(
        f'{args.mesh}_tet',
        MODELS_PATH / f'{args.mesh}.obj',
        target_faces=MESH_CONFIG[args.mesh],
        tet_scale=args.tet_scale,
    )
    n_v, n_t = len(tet_mesh.vertices), len(tet_mesh.tetrahedra)
    print(f"  Vertices:   {n_v:,}")
    print(f"  Tetrahedra: {n_t:,}")
    results.update({'n_verts': n_v, 'n_tets': n_t})

    mem_before = mem_report('before SOFA init')

    t0 = time.perf_counter()
    method = SofaMethod()
    method.initialize(tet_mesh, SIM_PARAMS)
    results['init_s'] = time.perf_counter() - t0
    print(f"  SOFA scene build + init: {results['init_s']:.2f} s")

    dt = float(SIM_PARAMS['time_step'])
    results['mem_after_init'] = mem_report('after SOFA init')

    # ---- Phase 1: uncut ---------------------------------------------------
    print(f"\n[Phase 1] uncut  (warmup {args.warmup}, timed {args.steps})")
    for _ in range(args.warmup):
        method.step(dt)
    per_step = []
    t_block = time.perf_counter()
    for _ in range(args.steps):
        t = time.perf_counter()
        method.step(dt)
        per_step.append(time.perf_counter() - t)
    block_s = time.perf_counter() - t_block
    uncut = _stats(per_step, 'uncut')
    uncut['block_total_s'] = block_s
    uncut['block_ms_step'] = block_s / max(args.steps, 1) * 1000.0
    uncut['block_fps']     = args.steps / block_s if block_s > 0 else float('inf')
    print(f"  {'uncut':14s} block   : {uncut['block_ms_step']:8.2f} ms/step "
          f"({uncut['block_fps']:7.2f} fps)   <-- comparable to our FEM uncut", flush=True)
    results['uncut'] = uncut
    results['mem_after_uncut'] = mem_report('after uncut phase')

    if args.no_cut:
        _finish(results, args, mem_before)
        return

    # ---- Phase 2: cut setup ----------------------------------------------
    print("\n[Phase 2] cut setup")
    t0 = time.perf_counter()
    ok = method.setup_cut(normal, origin, blade_dir)
    t_setup = time.perf_counter() - t0
    print(f"  setup_cut wall time: {t_setup:.3f} s  (ok={ok})")
    results['cut_setup_s'] = t_setup
    results['n_crossing_tets'] = int(len(method._crossing_travel))

    # ---- Phase 3: during-cut window --------------------------------------
    # Same blade speed as the FEM run (0.5 m/s at dt=0.01), started just before
    # the first crossing tet so the window contains actual removals.
    blade_speed  = 0.5
    blade_travel = float(np.min(method._crossing_travel)) - 1e-3
    tets_before  = len(method._topo.tetrahedra.array())

    print(f"\n[Phase 3] cutting  (timed {args.cut_steps}, blade_speed={blade_speed})")
    per_step, done = [], False
    for _ in range(args.cut_steps):
        t = time.perf_counter()
        blade_travel += blade_speed * dt
        done = method.advance_blade(blade_travel)
        method.step(dt)
        per_step.append(time.perf_counter() - t)
        if done:
            break
    tets_after = len(method._topo.tetrahedra.array())

    cut = _stats(per_step, 'cutting')
    cut['tets_removed']  = int(tets_before - tets_after)
    cut['cut_completed'] = bool(done)
    cut['blade_travel_reached'] = float(blade_travel)
    print(f"  tets removed in window: {cut['tets_removed']:,} "
          f"({tets_before:,} -> {tets_after:,}); cut complete = {done}")
    print("  NOTE: each removing step rebuilds an O(n_tets) Python lookup, so this "
          "figure is Python bookkeeping + SOFA, not SOFA alone.", flush=True)
    results['cutting'] = cut
    results['mem_after_cut'] = mem_report('after cut window')

    _finish(results, args, mem_before)


def _finish(results: dict, args, mem_before: dict):
    print("\n" + "=" * 78)
    print("SUMMARY -- SofaPython3")
    print("=" * 78)
    print(f"  mesh              : {results.get('n_verts', 0):,} verts / "
          f"{results.get('n_tets', 0):,} tets")
    if 'uncut' in results:
        r = results['uncut']
        print(f"  uncut             : {r['block_ms_step']:8.2f} ms/step  "
              f"{r['block_fps']:8.2f} fps")
    if 'cutting' in results and results['cutting']:
        r = results['cutting']
        print(f"  during cut window : {r['median_ms']:8.2f} ms/step  "
              f"{r['fps_median']:8.2f} fps   ({r['tets_removed']:,} tets removed)")
    if 'cut_setup_s' in results:
        print(f"  cut setup         : {results['cut_setup_s']:8.3f} s  "
              f"({results.get('n_crossing_tets', 0):,} crossing tets)")
    base = (mem_before or {}).get('rss_mb')
    last = (results.get('mem_after_cut') or results.get('mem_after_uncut') or {}).get('rss_mb')
    if base is not None and last is not None:
        print(f"  host RSS          : {base:8.1f} -> {last:8.1f} MB  "
              f"(delta {last - base:+.1f} MB)")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == '__main__':
    main()
