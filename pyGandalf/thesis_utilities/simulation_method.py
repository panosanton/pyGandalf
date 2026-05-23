"""
Simulation method interface for soft-body tetrahedral mesh simulation.

Any physics backend (spring-mass, FEM, neural surrogate) implements this
interface so the ECS system and headless runner are backend-agnostic.

Implementations
---------------
SpringMassMethod   — Taichi GPU spring-mass (current backend)
FEMMethod          — Corotational FEM (planned)
NeuralMethod       — GNN surrogate (planned, trained on spring-mass trajectories)

Design rules
------------
- The interface defines WHAT, not HOW. "setup_cut" means "split topology along
  this plane" — the backend decides whether to use cutting springs (spring-mass),
  remove element face constraints (FEM), or condition a graph (neural).
- `params` at initialize() is a plain dict.  Each implementation reads the keys
  it needs and ignores the rest, so callers can pass a superset without errors.
- `step(dt)` is one full simulation frame.  The implementation handles substeps
  internally.  For neural methods this may be a no-op if advance_blade() drives
  state prediction directly.
- `advance_blade(blade_travel)` is called every frame while the blade is active.
  It returns True once all cutting is complete so the caller can stop calling it.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from .taichi_simulation_system import _SpringMassSimulator, _cut_topology_physics


class SimulationMethod(ABC):
    """
    Abstract base class for soft-body simulation backends.

    Lifecycle
    ---------
    1. Instantiate the concrete class.
    2. Call initialize(tet_mesh, params) once.
    3. Each frame:
       a. If blade is active: call advance_blade(blade_travel).
       b. Call step(dt).
       c. Call get_positions() / get_velocities() to read state.
    4. To cut: call setup_cut(normal, origin, blade_dir) once on the first
       B-press, then advance_blade() every frame thereafter.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self, tet_mesh, params: dict) -> None:
        """
        Set up the simulation from a tetrahedral mesh.

        Parameters
        ----------
        tet_mesh : TetrahedralMeshInstance
            Vertices (N, 3) and tetrahedra (M, 4).
        params : dict
            Method-specific configuration.  Each implementation reads only
            the keys it needs.

            Spring-mass keys:
                stiffness, damping, total_mass, gravity, v_max,
                spring_damping, time_step, substeps, opening_speed,
                blade_speed, opening_ramp_frames

            FEM keys (planned):
                young_modulus, poisson_ratio, density, damping, time_step

            Neural keys (planned):
                model_path, device
        """

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, dt: float) -> None:
        """
        Advance the simulation by one frame of duration dt seconds.

        For force-based methods (spring-mass, FEM) this integrates physics.
        For neural methods this may be a no-op if advance_blade() drives
        state prediction directly.
        """

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @abstractmethod
    def get_positions(self) -> np.ndarray:
        """
        Current vertex positions.

        Returns
        -------
        np.ndarray, shape (N, 3), dtype float32
        """

    @abstractmethod
    def get_velocities(self) -> np.ndarray:
        """
        Current vertex velocities.

        Returns
        -------
        np.ndarray, shape (N, 3), dtype float32
        """

    @abstractmethod
    def set_velocities(self, indices: np.ndarray, velocities: np.ndarray) -> None:
        """
        Overwrite velocities for specific vertices.

        Used for poke impulses and opening velocity application.

        Parameters
        ----------
        indices    : (K,) int array — vertex indices to update
        velocities : (K, 3) float32 array — new velocity per vertex
                     (added to existing velocity, not replaced, if
                     the implementation prefers additive semantics —
                     document your choice in the concrete class)
        """

    # ------------------------------------------------------------------
    # Cutting
    # ------------------------------------------------------------------

    @abstractmethod
    def setup_cut(self,
                  cut_normal:       np.ndarray,
                  cut_origin:       np.ndarray,
                  blade_travel_dir: np.ndarray) -> bool:
        """
        Split mesh topology along the cut plane and prepare progressive cutting.

        Called once on the first B-press.  After this returns True, call
        advance_blade() every frame to progressively separate the two halves.

        Parameters
        ----------
        cut_normal       : (3,) float32 unit vector — plane orientation
        cut_origin       : (3,) float32 — a point on the plane
        blade_travel_dir : (3,) float32 unit vector — direction blade moves
                           within the cut plane

        Returns
        -------
        bool — True if the plane intersects the mesh and setup succeeded;
               False if the plane misses the mesh entirely (caller should
               ignore the B-press in that case).

        Post-conditions
        ---------------
        - get_positions() returns positions for the expanded vertex set
          (original + intersection + seam-dup vertices).
        - vertex_count and n_orig are valid.
        - advance_blade() is ready to be called.
        """

    @abstractmethod
    def advance_blade(self, blade_travel: float) -> bool:
        """
        Update internal state for the blade cursor at the given travel distance.

        Called every frame while the blade is active.  The implementation
        is responsible for:
        - Breaking/removing cutting constraints behind the cursor.
        - Applying opening forces or impulses to newly separated vertices.

        Parameters
        ----------
        blade_travel : float — cumulative distance the blade has traveled
                       along blade_travel_dir since setup_cut().

        Returns
        -------
        bool — True when all cutting constraints are broken (blade done).
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def vertex_count(self) -> int:
        """
        Current total vertex count.

        Equals the original mesh vertex count before any cut.
        Grows after setup_cut() due to intersection and seam-dup vertices.
        """

    @property
    def n_orig(self) -> int | None:
        """
        Original vertex count before the first cut.

        Returns None if setup_cut() has not been called yet.
        Vertices with index < n_orig are original mesh vertices.
        Vertices with index >= n_orig were inserted by the cut.

        Default implementation returns None; override after setup_cut().
        """
        return None


# ---------------------------------------------------------------------------
# SpringMassMethod
# ---------------------------------------------------------------------------

class SpringMassMethod(SimulationMethod):
    """
    Taichi GPU spring-mass backend.

    params keys (all optional, defaults shown)
    ------------------------------------------
    stiffness      : float = 100.0   spring stiffness (N/m)
    damping        : float = 1.0     velocity damping coefficient
    spring_damping : float = 0.0     per-spring viscous damping
    total_mass     : float = 1.0     total object mass (kg), distributed uniformly
    gravity        : list  = [0,0,0] gravity vector (m/s²)
    v_max          : float = 1.5     velocity cap (m/s)
    time_step      : float = 0.001   integration timestep (s)
    substeps       : int   = 10      sub-steps per frame
    opening_speed  : float = 1.0     velocity (m/s) given to seam vertices on break
    """

    def __init__(self):
        self._simulator          = None
        self._params             = {}
        self._n_orig_val         = None
        self._n_split            = None
        self._current_tets       = None
        self._cut_normal         = None
        self._cut_origin         = None
        self._blade_dir          = None
        self._seam_pairs         = []
        self._opening_ramp_queue = []

    def initialize(self, tet_mesh, params: dict) -> None:
        self._params = dict(params)

        y          = tet_mesh.vertices[:, 1]
        threshold  = y.min() + (y.max() - y.min()) * 0.05
        fixed_mask = (y < threshold).astype(np.int32)
        self._current_tets = tet_mesh.tetrahedra.copy()

        gravity = np.array(self._params.get('gravity', [0.0, 0.0, 0.0]), dtype=np.float32)

        self._simulator = _SpringMassSimulator(
            vertices   = tet_mesh.vertices,
            tetrahedra = tet_mesh.tetrahedra,
            fixed_mask = fixed_mask,
            stiffness  = float(self._params.get('stiffness', 100.0)),
            gravity    = gravity,
            total_mass = float(self._params.get('total_mass', 1.0)),
        )

    def step(self, dt: float) -> None:
        self._process_opening_ramp()
        substeps       = int(self._params.get('substeps', 10))
        sub_dt         = dt / substeps
        damping        = float(self._params.get('damping', 1.0))
        spring_damping = float(self._params.get('spring_damping', 0.0))
        v_max          = float(self._params.get('v_max', 1.5))
        for _ in range(substeps):
            self._simulator.step(sub_dt, damping, spring_damping, v_max)

    def _process_opening_ramp(self) -> None:
        if not self._opening_ramp_queue:
            return
        normal = self._cut_normal
        vels   = self._simulator.velocities.to_numpy()
        active = []
        for entry in self._opening_ramp_queue:
            vels[entry['above']] += entry['step_speed'] * normal
            vels[entry['below']] -= entry['step_speed'] * normal
            entry['left'] -= 1
            if entry['left'] > 0:
                active.append(entry)
        self._simulator.velocities.from_numpy(vels.astype(np.float32))
        self._opening_ramp_queue = active

    def get_positions(self) -> np.ndarray:
        return self._simulator.positions.to_numpy()

    def get_velocities(self) -> np.ndarray:
        return self._simulator.velocities.to_numpy()

    def set_velocities(self, indices: np.ndarray, velocities: np.ndarray) -> None:
        vels = self._simulator.velocities.to_numpy()
        vels[np.asarray(indices)] = velocities
        self._simulator.velocities.from_numpy(vels.astype(np.float32))

    def setup_cut(self, cut_normal, cut_origin, blade_travel_dir) -> bool:
        """
        Split topology along the cut plane and add cutting springs between seam pairs.
        Returns True on success, False if the plane doesn't divide the mesh.
        """
        normal    = np.array(cut_normal,       dtype=np.float32)
        origin    = np.array(cut_origin,       dtype=np.float32)
        blade_dir = np.array(blade_travel_dir, dtype=np.float32)
        normal    /= np.linalg.norm(normal)
        blade_dir /= np.linalg.norm(blade_dir)

        stiffness = float(self._params.get('stiffness', 100.0))
        gravity   = np.array(self._params.get('gravity', [0.0, 0.0, 0.0]), dtype=np.float32)

        result = _cut_topology_physics(
            self._current_tets,
            self._simulator.positions.to_numpy(),
            self._simulator.velocities.to_numpy(),
            self._simulator._masses.to_numpy(),
            self._simulator._fixed.to_numpy(),
            stiffness, gravity, origin, normal,
        )
        if result is None:
            return False

        (new_sim, final_pos, _final_vel, _final_mass, _final_fixed,
         all_tets, n_orig, n_split, shared_list, remap, _inter_data, _orig_surf_set) = result

        self._simulator    = new_sim
        self._current_tets = all_tets
        self._n_orig_val   = n_orig
        self._n_split      = n_split
        self._cut_normal   = normal
        self._cut_origin   = origin
        self._blade_dir    = blade_dir

        # Add a zero-rest-length cutting spring for every seam pair.
        n_structural = len(new_sim._sa)
        sa_cut = np.array(shared_list, dtype=np.int32)
        sb_cut = np.array([remap[v] for v in shared_list], dtype=np.int32)
        sr_cut = np.zeros(len(shared_list), dtype=np.float32)
        sk_cut = np.full(len(shared_list), stiffness * 0.5, dtype=np.float32)
        new_sim.extend_springs(sa_cut, sb_cut, sr_cut, sk_cut)

        # Build ordered manifest of seam pairs for advance_blade().
        seam_pairs = []
        for i, v_above in enumerate(shared_list):
            v_below     = int(remap[v_above])
            spring_idx  = n_structural + i
            pos_seam    = final_pos[v_above]
            travel_dist = float(np.dot(pos_seam - origin, blade_dir))
            seam_pairs.append({
                'v_above':     v_above,
                'v_below':     v_below,
                'spring_idx':  spring_idx,
                'travel_dist': travel_dist,
                'broken':      False,
            })
        seam_pairs.sort(key=lambda p: p['travel_dist'])
        self._seam_pairs = seam_pairs

        return True

    def advance_blade(self, blade_travel: float) -> bool:
        """
        Break all cutting springs behind blade_travel, apply opening impulse.

        If opening_ramp_frames > 1 (default 1), the impulse is spread across
        that many future step() calls instead of applied all at once.

        Returns True when all springs are broken (cut complete).
        """
        if not self._seam_pairs:
            return True

        opening_speed = float(self._params.get('opening_speed', 1.0))
        ramp_frames   = int(self._params.get('opening_ramp_frames', 1))
        normal        = self._cut_normal

        newly_broken = [p for p in self._seam_pairs
                        if not p['broken'] and p['travel_dist'] <= blade_travel]

        if newly_broken:
            for p in newly_broken:
                self._simulator._sk[p['spring_idx']] = 0.0
                p['broken'] = True

            if ramp_frames <= 1:
                vels = self._simulator.velocities.to_numpy()
                for p in newly_broken:
                    vels[p['v_above']] += opening_speed * normal
                    vels[p['v_below']] -= opening_speed * normal
                self._simulator.velocities.from_numpy(vels.astype(np.float32))
            else:
                self._opening_ramp_queue.append({
                    'above':      np.array([p['v_above'] for p in newly_broken], dtype=np.int32),
                    'below':      np.array([p['v_below'] for p in newly_broken], dtype=np.int32),
                    'step_speed': opening_speed / ramp_frames,
                    'left':       ramp_frames,
                })

        return all(p['broken'] for p in self._seam_pairs)

    @property
    def vertex_count(self) -> int:
        return self._simulator.positions.shape[0]

    @property
    def n_orig(self) -> int | None:
        return self._n_orig_val


# ---------------------------------------------------------------------------
# FEMMethod — planned, not yet implemented
# ---------------------------------------------------------------------------

class FEMMethod(SimulationMethod):
    """
    Corotational FEM backend (planned).

    Key differences from SpringMassMethod:
    - Forces computed from per-tet deformation gradient and polar decomposition,
      not from per-edge spring stretch.
    - Material params: young_modulus (Pa) + poisson_ratio instead of stiffness.
    - Implicit integration: solves K*dx = f per frame (conjugate gradient).
      Unconditionally stable — no v_max clamp needed.
    - Cutting: halves separate by removing shared face coupling in the stiffness
      matrix rather than zeroing spring stiffness entries.
    """

    def initialize(self, tet_mesh, params: dict) -> None:
        raise NotImplementedError("FEMMethod not yet implemented.")

    def step(self, dt: float) -> None:
        raise NotImplementedError

    def get_positions(self) -> np.ndarray:
        raise NotImplementedError

    def get_velocities(self) -> np.ndarray:
        raise NotImplementedError

    def set_velocities(self, indices, velocities) -> None:
        raise NotImplementedError

    def setup_cut(self, cut_normal, cut_origin, blade_travel_dir) -> bool:
        raise NotImplementedError

    def advance_blade(self, blade_travel: float) -> bool:
        raise NotImplementedError

    @property
    def vertex_count(self) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# NeuralMethod — planned, not yet implemented
# ---------------------------------------------------------------------------

class NeuralMethod(SimulationMethod):
    """
    GNN surrogate backend (planned).

    Key differences from force-based methods:
    - step() is a no-op: the network predicts state from blade_travel directly,
      not by integrating forces forward in time.
    - advance_blade() does the real work: feeds (current_graph_state,
      blade_travel) into the GNN and writes the predicted positions/velocities.
    - setup_cut() still runs the same topology change as SpringMassMethod
      (vertex duplication, seam detection) so the graph structure is correct.
    - initialize() loads the trained model weights from params['model_path'].
    """

    def initialize(self, tet_mesh, params: dict) -> None:
        raise NotImplementedError("NeuralMethod not yet implemented — train the GNN first.")

    def step(self, dt: float) -> None:
        pass   # intentional no-op: GNN predicts state in advance_blade()

    def get_positions(self) -> np.ndarray:
        raise NotImplementedError

    def get_velocities(self) -> np.ndarray:
        raise NotImplementedError

    def set_velocities(self, indices, velocities) -> None:
        raise NotImplementedError

    def setup_cut(self, cut_normal, cut_origin, blade_travel_dir) -> bool:
        raise NotImplementedError

    def advance_blade(self, blade_travel: float) -> bool:
        raise NotImplementedError

    @property
    def vertex_count(self) -> int:
        raise NotImplementedError
