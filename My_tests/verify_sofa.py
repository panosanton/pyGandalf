"""
Smoke test: prove SOFA is not just importable but actually simulates.

Builds a minimal SOFA scene: 8-vertex box, 5 tetrahedra, corotational
FEM, implicit Euler. Runs a fixed number of steps and prints timing +
final positions. If this runs without a traceback, SOFA is ready to
back a SimulationMethod implementation.
"""
import time
import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime


def build_scene():
    """A 1x1x1 box split into 5 tets, bottom face fixed."""
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    tets = np.array([
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 3, 4, 6],
        [3, 4, 6, 7],
        [1, 4, 5, 6],
    ], dtype=np.int32)
    # Bottom face indices (y == 0): 0, 1, 4, 5
    fixed = [0, 1, 4, 5]

    root = Sofa.Core.Node("root")
    root.gravity = [0.0, -9.81, 0.0]
    root.dt = 0.01

    # Solver + integrator
    SofaRuntime.importPlugin("Sofa.Component.LinearSolver.Direct")
    SofaRuntime.importPlugin("Sofa.Component.ODESolver.Backward")
    SofaRuntime.importPlugin("Sofa.Component.SolidMechanics.FEM.Elastic")
    SofaRuntime.importPlugin("Sofa.Component.StateContainer")
    SofaRuntime.importPlugin("Sofa.Component.Topology.Container.Dynamic")
    SofaRuntime.importPlugin("Sofa.Component.Constraint.Projective")
    SofaRuntime.importPlugin("Sofa.Component.Mass")

    root.addObject("EulerImplicitSolver", name="integrator")
    root.addObject("SparseLDLSolver", name="solver")

    root.addObject("MechanicalObject", name="dofs",
                   position=verts.tolist(),
                   template="Vec3d")
    root.addObject("TetrahedronSetTopologyContainer",
                   name="topo",
                   tetrahedra=tets.tolist(),
                   position=verts.tolist())
    root.addObject("TetrahedronFEMForceField",
                   name="fem",
                   youngModulus=1e5,
                   poissonRatio=0.3,
                   method="large")   # corotational
    root.addObject("UniformMass", totalMass=1.0)
    root.addObject("FixedConstraint", indices=fixed)

    return root


def main():
    root = build_scene()
    Sofa.Simulation.init(root)

    dofs = root.getObject("dofs")
    n_verts = len(dofs.position.array())
    print(f"[verify_sofa] scene built: {n_verts} verts, "
          f"{len(root.getObject('topo').tetrahedra.array())} tets")

    n_steps = 100
    dt = root.dt.value

    t0 = time.perf_counter()
    for _ in range(n_steps):
        Sofa.Simulation.animate(root, dt)
    elapsed = time.perf_counter() - t0

    pos = np.asarray(dofs.position.array())
    top_y = pos[[2, 3, 6, 7], 1].mean()
    sag   = 1.0 - top_y
    print(f"[verify_sofa] {n_steps} steps in {elapsed*1000:.1f} ms "
          f"({elapsed*1000/n_steps:.2f} ms/step)")
    print(f"[verify_sofa] top-face y after {n_steps} steps: {top_y:.8f} "
          f"(sag = {sag:.2e} m, expected ~1e-4 for E=1e5, mass=1kg)")
    assert sag > 1e-6, "top face did not sag -- gravity or FEM not applied"


if __name__ == "__main__":
    main()
