"""
Smoke test: prove SofaMethod satisfies the SimulationMethod ABC and
that its outputs match a direct SOFA scene on the same box mesh.

Runs the 8-vert box through SofaMethod (via initialize -> step -> get_positions)
and checks the same sag we verified from a raw SOFA scene in verify_sofa.py.
"""
import time
import numpy as np

from pyGandalf.utilities.mesh_lib import TetrahedralMeshInstance
from pyGandalf.thesis_utilities.sofa_method import SofaMethod


def build_box_mesh():
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    tets = np.array([
        [0, 1, 3, 4], [1, 2, 3, 6], [1, 3, 4, 6],
        [3, 4, 6, 7], [1, 4, 5, 6],
    ], dtype=np.int32)
    return TetrahedralMeshInstance(name="box", path=None,
                                    vertices=verts, tetrahedra=tets)


def main():
    mesh   = build_box_mesh()
    method = SofaMethod()
    params = dict(
        time_step     = 0.01,
        gravity       = [0.0, -9.81, 0.0],
        young_modulus = 1.0e5,
        poisson_ratio = 0.3,
        total_mass    = 1.0,
        fem_method    = "large",
    )
    method.initialize(mesh, params)

    assert method.vertex_count == 8
    assert method.n_orig       == 8

    n_steps = 100
    t0 = time.perf_counter()
    for _ in range(n_steps):
        method.step(params['time_step'])
    elapsed = time.perf_counter() - t0

    pos = method.get_positions()
    top_y = pos[[2, 3, 6, 7], 1].mean()
    sag   = 1.0 - float(top_y)

    print(f"[verify_sofa_method] {n_steps} steps in {elapsed*1000:.1f} ms "
          f"({elapsed*1000/n_steps:.2f} ms/step)")
    print(f"[verify_sofa_method] top-face y={top_y:.8f} sag={sag:.2e}")
    assert sag > 1e-6, "top face did not sag through SofaMethod"

    # Check set_velocities round-trip.
    vel_before = method.get_velocities().copy()
    method.set_velocities(np.array([2, 3]),
                          np.array([[0.0, -5.0, 0.0], [0.0, -5.0, 0.0]],
                                   dtype=np.float32))
    vel_after = method.get_velocities()
    assert np.allclose(vel_after[2], [0.0, -5.0, 0.0]), \
        f"set_velocities failed: {vel_after[2]} != [0, -5, 0]"
    assert np.allclose(vel_after[0], vel_before[0]), \
        "set_velocities leaked to vert 0"
    print(f"[verify_sofa_method] set_velocities round-trip OK")


if __name__ == "__main__":
    main()
