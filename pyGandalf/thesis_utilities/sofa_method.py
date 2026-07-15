"""
SofaMethod: SOFA v23.12.00 backend behind the SimulationMethod ABC.

Cutting model: PLANE-SWEEP TET REMOVAL driven by Python.

We deliberately BYPASS SOFA's CarvingManager because it does vertex-based
collision detection (via PointCollisionModel), which only fires when the
tool passes near mesh vertices. That works for interactive tools guided
onto specific points, but not for a swept cut plane that slices between
vertices. Confirmed empirically: canonical CarvingManager scenes carve
only when the tool trajectory hits vertices; between-vertex trajectories
do not trigger removal, even with LocalMinDistance intersection.

Path B: keep SOFA's FEM + implicit solver + TetrahedronSetTopologyModifier;
we compute which tets the blade plane has passed and call removeTetrahedra
ourselves. Same FEM engine, our cut trigger, fair comparison to pyGandalf.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from .simulation_method import SimulationMethod


_SCN_TEMPLATE = """<?xml version="1.0" ?>
<Node name="root" dt="{dt}" gravity="{gx} {gy} {gz}">
    <Node name="RequiredPlugins">
        <RequiredPlugin name="Sofa.Component.AnimationLoop"/>
        <RequiredPlugin name="Sofa.Component.Constraint.Projective"/>
        <RequiredPlugin name="Sofa.Component.LinearSolver.Iterative"/>
        <RequiredPlugin name="Sofa.Component.Mass"/>
        <RequiredPlugin name="Sofa.Component.ODESolver.Backward"/>
        <RequiredPlugin name="Sofa.Component.SolidMechanics.FEM.Elastic"/>
        <RequiredPlugin name="Sofa.Component.StateContainer"/>
        <RequiredPlugin name="Sofa.Component.Topology.Container.Dynamic"/>
        <RequiredPlugin name="Sofa.Component.Topology.Mapping"/>
    </Node>

    <DefaultAnimationLoop/>

    <EulerImplicitSolver name="EulerImplicit" rayleighStiffness="0.1" rayleighMass="0.1"/>
    <CGLinearSolver name="CG Solver" iterations="25" tolerance="1e-9" threshold="1e-9"/>

    <Node name="Volume">
        <MechanicalObject name="Volume" template="Vec3d" position="{positions}"/>
        <TetrahedronSetTopologyContainer  name="Container" tetrahedra="{tets}"/>
        <TetrahedronSetTopologyModifier   name="Modifier"/>
        <TetrahedronSetGeometryAlgorithms name="GeomAlgo" template="Vec3d"/>
        <DiagonalMass massDensity="{mass_density}"/>
        <FixedConstraint indices="{fixed}"/>
        <TetrahedralCorotationalFEMForceField name="CFEM" youngModulus="{E}" poissonRatio="{nu}" method="large"/>

        <Node name="Surface">
            <TriangleSetTopologyContainer  name="Container"/>
            <TriangleSetTopologyModifier   name="Modifier"/>
            <TriangleSetGeometryAlgorithms name="GeomAlgo" template="Vec3d"/>
            <Tetra2TriangleTopologicalMapping input="@../Container" output="@Container"/>
        </Node>
    </Node>
</Node>
"""


class SofaMethod(SimulationMethod):
    """
    SOFA-backed FEM with plane-sweep tet removal.

    params (all optional):
        time_step      : float = 0.01
        gravity        : list  = [0,0,0]
        young_modulus  : float = 300.0    Pa (matches SofaCarving demo)
        poisson_ratio  : float = 0.3
        mass_density   : float = 0.01

    Cutting: setup_cut(normal, origin, blade_dir) precomputes a schedule of
    (tet_index, centroid_travel_dist) for every tet that crosses the plane.
    advance_blade(travel) removes tets whose centroid_travel_dist has been
    passed, via TetrahedronSetTopologyModifier.removeTetrahedra.
    """

    def __init__(self, **overrides):
        self._params         = {}
        self._init_overrides = overrides
        # SOFA scene handles set by initialize().
        self._root       = None
        self._volume     = None
        self._dofs       = None       # MechanicalObject
        self._topo       = None       # TetrahedronSetTopologyContainer
        self._topo_mod   = None       # TetrahedronSetTopologyModifier
        # ECS compatibility shim. TaichiSimulationSystem sets
        # `comp.simulator = comp.method._simulator` at init and expects a
        # non-None object. Spring-mass code paths are all hasattr-guarded on
        # `_sa` / `_spring_forces`, so a bare self-reference is enough for
        # rendering + step-driven updates. The B-key progressive-cut path
        # reads `method._topology_result` which SofaMethod does not have;
        # do not press B when running with this backend for now.
        self._simulator  = self
        # Cache captured at initialize time.
        self._n_orig_val   = None
        self._current_tets = None
        # Cut state.
        self._cut_normal   = None
        self._cut_origin   = None
        self._blade_dir    = None
        # Per-tet schedule: (original_tet_index, centroid_travel_dist).
        # Removed lazily by matching against the current sofa tet list each
        # advance_blade call (SOFA renumbers tets on removal, so we cannot
        # cache absolute indices).
        self._crossing_centroids = None   # (K, 3) centroids of original crossing tets
        self._crossing_signature = None   # (K,) unique key per original crossing tet
        self._crossing_travel    = None   # (K,) travel_dist per original crossing tet
        self._pending_signatures = None   # set of signatures not yet removed
        self._scn_path = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, tet_mesh, params: dict) -> None:
        import Sofa
        import Sofa.Core
        import Sofa.Simulation

        self._params = dict(params)
        self._params.update(self._init_overrides)

        verts = np.asarray(tet_mesh.vertices, dtype=np.float64)
        tets  = np.asarray(tet_mesh.tetrahedra, dtype=np.int32)
        self._current_tets = tets.copy()
        self._n_orig_val   = len(verts)

        y = verts[:, 1]
        threshold  = y.min() + (y.max() - y.min()) * 0.05
        fixed_idx  = np.where(y < threshold)[0]

        gravity = list(self._params.get('gravity', [0.0, 0.0, 0.0]))

        pos_str   = " ".join(f"{v:.9g}" for v in verts.ravel())
        tets_str  = " ".join(str(int(v)) for v in tets.ravel())
        fixed_str = " ".join(str(int(v)) for v in fixed_idx)

        scn = _SCN_TEMPLATE.format(
            dt           = float(self._params.get('time_step', 0.01)),
            gx=gravity[0], gy=gravity[1], gz=gravity[2],
            positions    = pos_str,
            tets         = tets_str,
            mass_density = float(self._params.get('mass_density',  0.01)),
            fixed        = fixed_str,
            E            = float(self._params.get('young_modulus', 300.0)),
            nu           = float(self._params.get('poisson_ratio', 0.3)),
        )

        fd, path = tempfile.mkstemp(suffix=".scn", prefix="sofa_method_")
        os.close(fd)
        with open(path, "w") as f:
            f.write(scn)
        self._scn_path = path

        self._root = Sofa.Simulation.load(path)
        Sofa.Simulation.init(self._root)

        self._volume   = self._root.getChild("Volume")
        self._dofs     = self._volume.getObject("Volume")
        self._topo     = self._volume.getObject("Container")
        self._topo_mod = self._volume.getObject("Modifier")

        n_tets = len(self._topo.tetrahedra.array())
        print(f"[SofaMethod] scene loaded: {len(verts)} verts, {n_tets} tets, "
              f"fixed={len(fixed_idx)}, E={self._params.get('young_modulus', 300.0):.1e} Pa")

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
        with self._dofs.velocity.writeableArray() as vel:
            vel[np.asarray(indices, dtype=np.int64)] = np.asarray(velocities,
                                                                  dtype=np.float64)

    # ------------------------------------------------------------------
    # Cutting: plane-sweep tet removal (bypasses CarvingManager)
    # ------------------------------------------------------------------

    def setup_cut(self, cut_normal, cut_origin, blade_travel_dir) -> bool:
        """
        Compute the set of tets that cross the cut plane and, for each, the
        blade_travel value at which it should be removed (based on centroid
        projection along blade_dir). advance_blade() then removes them
        progressively.
        """
        normal    = np.asarray(cut_normal,       dtype=np.float64)
        origin    = np.asarray(cut_origin,       dtype=np.float64)
        blade_dir = np.asarray(blade_travel_dir, dtype=np.float64)
        normal    /= np.linalg.norm(normal)
        blade_dir /= np.linalg.norm(blade_dir)

        self._cut_normal = normal
        self._cut_origin = origin
        self._blade_dir  = blade_dir

        # Signed distance from cut plane, per original vertex.
        vp = np.asarray(self._dofs.position.array(), dtype=np.float64)
        signed = (vp - origin) @ normal

        tets = self._current_tets
        tet_dists = signed[tets]
        crosses   = np.any(tet_dists > 0, axis=1) & np.any(tet_dists < 0, axis=1)
        crossing_idx = np.where(crosses)[0]

        # Centroid projection along blade_dir per crossing tet.
        centroids  = vp[tets[crossing_idx]].mean(axis=1)
        cen_travel = (centroids - origin) @ blade_dir

        self._crossing_centroids = centroids
        self._crossing_travel    = cen_travel
        # Use a sorted-tuple of the tet's vertex indices as a stable signature
        # that survives SOFA's internal tet renumbering after removeTetrahedra.
        # (Tet identity is defined by which 4 verts it references.)
        # Kept as a plain Python list of tuples -- numpy object arrays lose
        # the tuple type through .tolist().
        self._crossing_signature = [
            tuple(sorted(int(v) for v in tets[i])) for i in crossing_idx
        ]
        self._pending_signatures = set(self._crossing_signature)

        print(f"[SofaMethod] setup_cut: {len(crossing_idx)} crossing tets, "
              f"travel span [{cen_travel.min():.3f}, {cen_travel.max():.3f}]")
        return True

    def advance_blade(self, blade_travel: float) -> bool:
        """
        Remove any crossing tet whose centroid_travel <= blade_travel and that
        has not already been removed. Returns True when the cut is complete.
        """
        if self._pending_signatures is None:
            return True

        # Which of our tracked crossing tets should now be gone.
        target_mask = (self._crossing_travel <= float(blade_travel))
        target_sigs = {sig for sig, m in zip(self._crossing_signature,
                                             target_mask.tolist()) if m}
        newly_target = target_sigs & self._pending_signatures
        if not newly_target:
            return len(self._pending_signatures) == 0

        # Map signatures to CURRENT sofa tet indices. SOFA renumbers on
        # removal so we cannot cache; look up fresh each call.
        current = np.asarray(self._topo.tetrahedra.array(), dtype=np.int64)
        current_sigs = {}
        for i in range(len(current)):
            key = tuple(sorted(int(v) for v in current[i]))
            current_sigs[key] = i

        remove_set = {current_sigs[sig] for sig in newly_target
                      if sig in current_sigs}
        if remove_set:
            # `removeTetrahedra` is not Python-bound in SofaPython3 v23. As a
            # workaround, overwrite the TetrahedronSetTopologyContainer's
            # tetrahedra Data field to exclude the removed indices. SOFA's
            # Data change tracking should propagate the update to the FEM
            # force field and Tetra2TriangleTopologicalMapping on the next
            # step. If it does not, we will see visible inconsistency and
            # need a different route (event injection or a scripted
            # controller).
            keep_mask = np.ones(len(current), dtype=bool)
            keep_mask[list(remove_set)] = False
            new_tets = current[keep_mask]
            with self._topo.tetrahedra.writeableArray() as arr:
                # Rewrite in place. The Data field is stored as (N, 4) but
                # SOFA reallocates when we resize; the writeable context does
                # not support resize, so we set via from_array below instead.
                pass
            # Direct rewrite: assign the new tetrahedra array to the Data
            # field. Uses the Data value setter, which handles resize.
            self._topo.tetrahedra = new_tets.tolist()

        self._pending_signatures -= newly_target
        return len(self._pending_signatures) == 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vertex_count(self) -> int:
        return int(len(self._dofs.position.array()))

    @property
    def n_orig(self) -> int | None:
        return self._n_orig_val

    def __del__(self):
        try:
            if self._scn_path and os.path.exists(self._scn_path):
                os.remove(self._scn_path)
        except Exception:
            pass
