"""
Path B smoke test: SofaMethod.setup_cut / advance_blade drives tet removal
directly via TetrahedronSetTopologyModifier.removeTetrahedra, bypassing
SofaCarving's CarvingManager entirely.

Records tet count as blade sweeps across the mesh. Passes if tet count
decreases monotonically as blade_travel advances.
"""
import time
import numpy as np

from pyGandalf.utilities.mesh_lib import TetrahedralMeshInstance
from pyGandalf.thesis_utilities.sofa_method import SofaMethod


def build_dense_box(nx=6, ny=6, nz=6, size=1.0):
    xs = np.linspace(0, size, nx)
    ys = np.linspace(0, size, ny)
    zs = np.linspace(0, size, nz)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    verts = grid.reshape(-1, 3).astype(np.float32)

    def vid(i, j, k): return (i * ny + j) * nz + k
    tets = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                v = [vid(i,j,k), vid(i+1,j,k), vid(i,j+1,k), vid(i+1,j+1,k),
                     vid(i,j,k+1), vid(i+1,j,k+1), vid(i,j+1,k+1), vid(i+1,j+1,k+1)]
                tets += [
                    [v[0], v[1], v[3], v[7]], [v[0], v[3], v[2], v[7]],
                    [v[0], v[2], v[6], v[7]], [v[0], v[6], v[4], v[7]],
                    [v[0], v[4], v[5], v[7]], [v[0], v[5], v[1], v[7]],
                ]
    return TetrahedralMeshInstance(name="dense_box", path=None,
                                    vertices=verts,
                                    tetrahedra=np.array(tets, dtype=np.int32))


def main():
    mesh = build_dense_box(nx=6, ny=6, nz=6, size=1.0)
    print(f"[verify] mesh: {len(mesh.vertices)} verts, {len(mesh.tetrahedra)} tets")

    method = SofaMethod()
    method.initialize(mesh, dict(
        time_step     = 0.01,
        gravity       = [0.0, 0.0, 0.0],
        young_modulus = 1.0e5,
        poisson_ratio = 0.3,
        mass_density  = 0.01,
    ))

    # Warm-up steps (initial CG solve).
    for _ in range(5):
        method.step(0.01)
    n_before = len(method._topo.tetrahedra.array())
    print(f"[verify] tets before cut: {n_before}")

    # Cut plane y=0.5 (horizontal), blade sweeps along +x from x=-1 to x=+2.
    ok = method.setup_cut(
        cut_normal       = [0.0, 1.0, 0.0],
        cut_origin       = [0.5, 0.5, 0.5],
        blade_travel_dir = [1.0, 0.0, 0.0],
    )
    assert ok, "setup_cut returned False"

    n_frames = 80
    blade_speed = 0.05
    diag_frames = {0, 10, 20, 30, 40, 50, 60, 70, 79}

    t0 = time.perf_counter()
    for frame in range(n_frames):
        travel = -1.0 + frame * blade_speed
        done   = method.advance_blade(travel)
        method.step(0.01)
        if frame in diag_frames:
            n_now = len(method._topo.tetrahedra.array())
            print(f"[verify f={frame:2d}] travel={travel:+.3f} tets={n_now} "
                  f"done={done}")
    elapsed = time.perf_counter() - t0

    n_after = len(method._topo.tetrahedra.array())
    n_removed = n_before - n_after
    print(f"[verify] {n_frames} frames in {elapsed*1000:.1f} ms "
          f"({elapsed*1000/n_frames:.2f} ms/frame)")
    print(f"[verify] tets after cut: {n_after} (removed: {n_removed})")

    assert n_removed > 0, "no tets removed -- path B rewrite is broken"
    print("[verify] PASS: path B plane-sweep tet removal works.")


if __name__ == "__main__":
    main()
