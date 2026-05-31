"""
Dataset generation for the GNN surrogate model.

Runs N headless spring-mass simulations with randomised cut parameters
and saves each trajectory as a compressed .npz file.

Usage:
    python My_tests/generate_dataset.py --n_runs 3000 --out dataset/ --seed 42

Each run produces one file: dataset/run_XXXXXX.npz
A metadata.json is written once at the end.

The .npz contains everything needed to train a graph-based learned simulator:
    positions        (T, N, 3)   float32   vertex positions at each recorded frame
    velocities       (T, N, 3)   float32   vertex velocities
    blade_progress   (T,)        float32   blade travel distance at each frame
    edges            (E, 2)      int32     spring connectivity (structural + cutting)
    rest_lengths     (E,)        float32   per-spring rest length
    stiffnesses_max  (E,)        float32   initial stiffness (0 after spring breaks)
    spring_break_at  (E,)        float32   blade_progress when spring broke (inf = never)
    masses           (N,)        float32   per-vertex mass
    fixed            (N,)        int32     1 = fixed vertex, 0 = free
    cut_normal       (3,)        float32
    cut_origin       (3,)        float32
    n_orig           scalar      int       original vertex count (before cut)
    n_phys           scalar      int       vertex count after cut
    n_structural     scalar      int       number of structural springs
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


class _Tee:
    """Writes to multiple streams simultaneously (terminal + log file)."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def _setup_logging(output_dir: Path) -> object:
    log_path = output_dir / 'generation.log'
    log_file = open(log_path, 'w', buffering=1, encoding='utf-8')   # line-buffered
    tee = _Tee(sys.__stdout__, log_file)
    sys.stdout = tee
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"Logging to {log_path}")
    return log_file

from pyGandalf.utilities.mesh_lib import MeshLib
from pyGandalf.utilities.definitions import MODELS_PATH
from pyGandalf.thesis_utilities.simulation_method import SpringMassMethod
from pyGandalf.thesis_utilities.headless_sim import HeadlessSim


# ---------------------------------------------------------------------------
# Simulation parameters — mirror test_random_cut.py exactly
# ---------------------------------------------------------------------------
SIM_PARAMS = dict(
    stiffness           = 200.0,
    damping             = 3.5,
    total_mass          = 100.0,
    time_step           = 0.005,
    substeps            = 4,
    gravity             = [0.0, 0.0, 0.0],
    opening_speed       = 4.0,
    blade_speed         = 0.5,
    v_max               = 4.0,
    spring_damping      = 2.0,
    opening_ramp_frames = 20,
)

SETTLE_STEPS  = 200   # physics frames to record after blade finishes
RECORD_EVERY  = 1     # record every N frames (1 = every frame)
DEPTH_RANGE   = 0.7   # max cut plane offset from mesh centre


# ---------------------------------------------------------------------------
# Random cut plane (same logic as test_random_cut.py)
# ---------------------------------------------------------------------------
def _random_cut_plane(rng: np.random.Generator, depth_range: float = DEPTH_RANGE):
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


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def generate_dataset(n_runs: int, output_dir: Path, seed: int | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = _setup_logging(output_dir)

    rng = np.random.default_rng(seed)

    print("Loading tetrahedral mesh...")
    tet_mesh = MeshLib().build_tetrahedral('sphere_tet', MODELS_PATH / 'sphere.obj')
    print(f"  Vertices:   {len(tet_mesh.vertices):,}")
    print(f"  Tetrahedra: {len(tet_mesh.tetrahedra):,}")
    print(f"Generating {n_runs} simulation runs → {output_dir}/")
    print("=" * 60)

    saved      = 0
    skipped    = 0
    t_start    = time.perf_counter()

    for run_idx in range(n_runs):
        normal, origin, blade_dir = _random_cut_plane(rng)

        method = SpringMassMethod()
        method.initialize(tet_mesh, SIM_PARAMS)
        sim = HeadlessSim(method, normal, origin, blade_dir)

        data = sim.run_and_record(settle_steps=SETTLE_STEPS, record_every=RECORD_EVERY)

        if data is None:
            skipped += 1
            print(f"[{run_idx+1:5d}/{n_runs}] SKIPPED (cut missed mesh)")
            continue

        out_path = output_dir / f"run_{saved:06d}.npz"
        np.savez_compressed(out_path, **data)
        saved += 1

        elapsed = time.perf_counter() - t_start
        eta     = elapsed / saved * (n_runs - run_idx - 1)
        T       = data['positions'].shape[0]
        N       = data['positions'].shape[1]
        E       = data['edges'].shape[0]
        print(f"[{run_idx+1:5d}/{n_runs}] saved run_{saved-1:06d}.npz  "
              f"T={T} N={N} E={E}  "
              f"elapsed {elapsed/60:.1f}min  ETA {eta/60:.1f}min")

    # --- Metadata ---
    metadata = {
        'n_runs_requested': n_runs,
        'n_runs_saved':     saved,
        'n_runs_skipped':   skipped,
        'mesh':             'sphere.obj',
        'settle_steps':     SETTLE_STEPS,
        'record_every':     RECORD_EVERY,
        'depth_range':      DEPTH_RANGE,
        'seed':             seed,
        'sim_params':       SIM_PARAMS,
        'data_fields': {
            'positions':       '(T, N, 3) float32 — vertex positions at each recorded frame',
            'velocities':      '(T, N, 3) float32 — vertex velocities',
            'blade_progress':  '(T,)      float32 — blade travel distance at each frame',
            'edges':           '(E, 2)    int32   — spring connectivity',
            'rest_lengths':    '(E,)      float32 — per-spring rest length',
            'stiffnesses_max': '(E,)      float32 — initial stiffness (zero after breaking)',
            'spring_break_at': '(E,)      float32 — blade_progress when broken (inf=never)',
            'masses':          '(N,)      float32 — per-vertex mass',
            'fixed':           '(N,)      int32   — 1=fixed, 0=free',
            'cut_normal':      '(3,)      float32',
            'cut_origin':      '(3,)      float32',
            'n_orig':          'int — original vertex count before cut',
            'n_phys':          'int — vertex count after cut',
            'n_structural':    'int — structural spring count (cutting springs follow)',
        },
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    total_time = time.perf_counter() - t_start
    print("=" * 60)
    print(f"Done. {saved} runs saved, {skipped} skipped.")
    print(f"Total time: {total_time/60:.1f} min  ({total_time/max(saved,1):.1f} s/run)")
    print(f"Output: {output_dir.resolve()}")
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate GNN training dataset')
    parser.add_argument('--n_runs', type=int,   default=3000,
                        help='Number of simulation runs (default: 3000)')
    parser.add_argument('--out',    type=str,   default='dataset',
                        help='Output directory (default: dataset/)')
    parser.add_argument('--seed',   type=int,   default=None,
                        help='RNG seed for reproducibility (default: random)')
    args = parser.parse_args()

    generate_dataset(
        n_runs     = args.n_runs,
        output_dir = Path(args.out),
        seed       = args.seed,
    )
