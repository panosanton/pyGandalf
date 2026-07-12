"""
SofaMethod: SOFA v23.12.00 backend behind the SimulationMethod ABC.

Wraps a SOFA scene (corotational tet FEM, implicit Euler, LDL solver) so
pyGandalf's ECS and headless runner can drive it interchangeably with
FEMMethod / SpringMassMethod. Exists to let the thesis benchmark
pyGandalf's Taichi GPU FEM against SOFA's CPU FEM on identical meshes.

Import is intentionally lazy -- SofaPython3 is heavy (~2s import cost and
pulls in QT/GL DLLs). It only loads when SofaMethod is instantiated so
users running FEMMethod or SpringMassMethod pay nothing for its presence.
"""
from __future__ import annotations

import numpy as np

from .simulation_method import SimulationMethod


# SOFA v23 plugin names required for the corotational-FEM + implicit-Euler
# scene we build. Kept as a module constant so cutting-related plugins can
# be appended later without disturbing the base list.
_SOFA_BASE_PLUGINS = (
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.Mass",
    "Sofa.Component.AnimationLoop",
)


class SofaMethod(SimulationMethod):
    """
    SOFA-backed FEM. Corotational tet FEM with implicit Euler and a sparse
    LDL solver on the CPU.

    params (all optional):
        time_step      : float = 0.01     integration timestep (s)
        gravity        : list  = [0,0,0]
        young_modulus  : float = 1.0e5    Pa
        poisson_ratio  : float = 0.3
        total_mass     : float = 1.0      distributed uniformly over the mesh
        fem_method     : str   = "large"  "small" (linear), "large" (corotational),
                                          "polar", "svd" -- see TetrahedronFEMForceField

    Cutting: setup_cut / advance_blade currently return placeholder values;
    the removal-based cut path will be added in a follow-up commit once the
    static-FEM comparison is validated.
    """

    def __init__(self, **overrides):
        self._params         = {}
        self._init_overrides = overrides
        # Handles to the SOFA scene objects set by initialize().
        self._root       = None
        self._dofs       = None  # MechanicalObject
        self._topo       = None  # TetrahedronSetTopologyContainer
        self._fem        = None  # TetrahedronFEMForceField
        # Cached identity captured at initialize time.
        self._n_orig_val = None
        self._current_tets = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, tet_mesh, params: dict) -> None:
        # Lazy import so pyGandalf runs that use other backends do not pay
        # the ~2s SofaPython3 import cost.
        import Sofa
        import Sofa.Core
        import Sofa.Simulation
        import SofaRuntime

        self._params = dict(params)
        self._params.update(self._init_overrides)

        for plugin in _SOFA_BASE_PLUGINS:
            SofaRuntime.importPlugin(plugin)

        verts = np.asarray(tet_mesh.vertices, dtype=np.float64)
        tets  = np.asarray(tet_mesh.tetrahedra, dtype=np.int32)
        self._current_tets = tets.copy()
        self._n_orig_val   = len(verts)

        # Fixed-vert heuristic mirrors FEMMethod: bottom 5% of the y-range.
        y = verts[:, 1]
        threshold  = y.min() + (y.max() - y.min()) * 0.05
        fixed_idx  = np.where(y < threshold)[0].tolist()

        root = Sofa.Core.Node("root")
        root.gravity = list(self._params.get('gravity', [0.0, 0.0, 0.0]))
        root.dt      = float(self._params.get('time_step', 0.01))

        # DefaultAnimationLoop must be explicit; SOFA warns and inserts one
        # automatically otherwise. Explicit is clearer.
        root.addObject("DefaultAnimationLoop")

        root.addObject("EulerImplicitSolver", name="integrator",
                       rayleighStiffness=0.1, rayleighMass=0.1)
        # CompressedRowSparseMatrixMat3x3d silences the SparseLDLSolver template
        # suggestion and is the correct block layout for Vec3d DOFs.
        root.addObject("SparseLDLSolver", name="solver",
                       template="CompressedRowSparseMatrixMat3x3d")

        root.addObject("MechanicalObject", name="dofs",
                       position=verts.tolist(), template="Vec3d")
        root.addObject("TetrahedronSetTopologyContainer", name="topo",
                       tetrahedra=tets.tolist(), position=verts.tolist())
        # Topology modifier is required for later removeTetrahedra() calls;
        # harmless when unused.
        root.addObject("TetrahedronSetTopologyModifier", name="topo_mod")

        root.addObject("TetrahedronFEMForceField", name="fem",
                       youngModulus=float(self._params.get('young_modulus', 1.0e5)),
                       poissonRatio=float(self._params.get('poisson_ratio', 0.3)),
                       method=str(self._params.get('fem_method', 'large')))

        root.addObject("UniformMass",
                       totalMass=float(self._params.get('total_mass', 1.0)))

        if fixed_idx:
            root.addObject("FixedConstraint", indices=fixed_idx)

        Sofa.Simulation.init(root)

        self._root = root
        self._dofs = root.getObject("dofs")
        self._topo = root.getObject("topo")
        self._fem  = root.getObject("fem")

        print(f"[SofaMethod] scene init: {len(verts)} verts, {len(tets)} tets, "
              f"fixed={len(fixed_idx)}, E={self._params.get('young_modulus', 1e5):.1e} Pa")

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        import Sofa.Simulation
        Sofa.Simulation.animate(self._root, float(dt))

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_positions(self) -> np.ndarray:
        return np.asarray(self._dofs.position.array(), dtype=np.float32)

    def get_velocities(self) -> np.ndarray:
        return np.asarray(self._dofs.velocity.array(), dtype=np.float32)

    def set_velocities(self, indices: np.ndarray, velocities: np.ndarray) -> None:
        # SOFA Data write requires a full-array assignment (in-place index
        # writes on the DataView do not persist back to the underlying field
        # in v23). Read, modify, write.
        with self._dofs.velocity.writeableArray() as vel:
            vel[np.asarray(indices, dtype=np.int64)] = np.asarray(velocities,
                                                                  dtype=np.float64)

    # ------------------------------------------------------------------
    # Cutting (placeholder for the first pass)
    # ------------------------------------------------------------------

    def setup_cut(self, cut_normal, cut_origin, blade_travel_dir) -> bool:
        # Cutting will use removeTetrahedra on tets whose centroid has
        # crossed the blade plane. Not implemented yet; report failure so
        # callers that require cut support know not to proceed.
        print("[SofaMethod] setup_cut not yet implemented -- static-FEM only.")
        return False

    def advance_blade(self, blade_travel: float) -> bool:
        return True  # nothing to progress; report "done"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vertex_count(self) -> int:
        return int(len(self._dofs.position.array()))

    @property
    def n_orig(self) -> int | None:
        return self._n_orig_val
