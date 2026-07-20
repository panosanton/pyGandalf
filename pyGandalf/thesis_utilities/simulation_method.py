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
from collections import defaultdict

import numpy as np

from .taichi_cut_utils import _SpringMassSimulator, _cut_topology_physics
from . import taichi_cut_utils as _tcu  # for _QUIET_CUT_LOGS toggle in progressive mode


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

    def is_cut_complete(self, blade_travel: float) -> bool:
        """
        Return True when the blade has finished cutting.

        Default implementation matches the legacy one-shot pipeline: the cut
        is complete once every seam pair created at setup_cut() time has been
        broken. Concrete backends that build seam pairs incrementally (e.g.
        FEM progressive cut) MUST override this so an initially-empty
        seam_pairs list doesn't cause a false-positive "done" at B-press
        (Python's all([]) is vacuously True).

        Callers: FEMMethod.advance_blade uses this as its return value, and
        the ECS _advance_progressive_blade uses it to decide when to stop
        the blade cursor / print "Cut complete".
        """
        return all(p['broken'] for p in getattr(self, '_seam_pairs', []))

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

    @property
    def time_step(self) -> float:
        """Integration timestep in seconds, read from params."""
        return float(self._params.get('time_step', 0.005))

    @property
    def blade_speed(self) -> float:
        """Blade travel speed in m/s, read from params."""
        return float(self._params.get('blade_speed', 0.5))

    @property
    def initial_blade_travel(self) -> float:
        """
        Recommended starting blade_travel value for HeadlessSim.

        Returns the travel distance of the first seam pair minus a small
        epsilon so the blade starts just before the first cut point.
        Returns 0.0 if setup_cut() has not been called yet.
        """
        pairs = getattr(self, '_seam_pairs', None)
        if pairs:
            return pairs[0]['travel_dist'] - 1e-3
        return 0.0

    def get_graph_data(self) -> dict:
        """
        Return the static graph structure needed for GNN input recording.

        Called once after setup_cut() by HeadlessSim.run_and_record().
        The returned dict is merged into the trajectory .npz alongside
        per-frame positions and velocities.

        Concrete methods that support recording must override this.
        Methods that do not support recording leave this unimplemented;
        calling it raises NotImplementedError at runtime.

        Returns
        -------
        dict with at minimum the keys needed by the GNN trainer.
        The exact keys are method-defined -- document them in the override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support graph recording. "
            "Override get_graph_data() to enable HeadlessSim recording."
        )


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
        active = []
        for entry in self._opening_ramp_queue:
            dv = (entry['step_speed'] * normal).astype(np.float32)
            above = np.asarray(entry['above'], dtype=np.int32)
            below = np.asarray(entry['below'], dtype=np.int32)
            self._simulator._add_vel_scalar(above,  float(dv[0]),  float(dv[1]),  float(dv[2]))
            self._simulator._add_vel_scalar(below, -float(dv[0]), -float(dv[1]), -float(dv[2]))
            entry['left'] -= 1
            if entry['left'] > 0:
                active.append(entry)
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
            blade_dir=blade_dir,
        )
        if result is None:
            return False

        (new_sim, final_pos, _final_vel, _final_mass, _final_fixed,
         all_tets, n_orig, n_split, shared_list, remap, _inter_data, _orig_surf_set,
         _phantom_keys, _side_label, _all_tets_parent) = result

        self._simulator       = new_sim
        self._current_tets    = all_tets
        self._n_orig_val      = n_orig
        self._n_split         = n_split
        self._cut_normal      = normal
        self._cut_origin      = origin
        self._blade_dir       = blade_dir
        self._topology_result = result  # cached for ECS surface computation

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
        self._seam_pairs         = seam_pairs
        self._opening_ramp_queue = []

        return True

    def get_graph_data(self) -> dict:
        """
        Snapshot the spring graph after setup_cut() for GNN recording.

        Returns
        -------
        dict with keys:
            edges            (E, 2)  int32   -- spring pairs [structural | cutting]
            rest_lengths     (E,)    float32 -- rest length per spring
            stiffnesses_max  (E,)    float32 -- initial stiffness per spring
            spring_break_at  (E,)    float32 -- blade_travel when broken (inf = never)
            masses           (N,)    float32 -- per-vertex mass
            fixed            (N,)    int32   -- 1 = fixed, 0 = free
            n_structural     int     -- number of structural springs
        """
        sim  = self._simulator
        sa   = np.array(sim._sa, dtype=np.int32)
        sb   = np.array(sim._sb, dtype=np.int32)
        sr   = np.array(sim._sr, dtype=np.float32)
        sk   = np.array(sim._sk, dtype=np.float32)

        n_springs    = len(sa)
        n_structural = n_springs - len(self._seam_pairs)

        spring_break_at = np.full(n_springs, np.inf, dtype=np.float32)
        for p in self._seam_pairs:
            spring_break_at[p['spring_idx']] = p['travel_dist']

        return {
            'edges':           np.stack([sa, sb], axis=1),
            'rest_lengths':    sr,
            'stiffnesses_max': sk,
            'spring_break_at': spring_break_at,
            'masses':          sim._masses.to_numpy().astype(np.float32),
            'fixed':           sim._fixed.to_numpy().astype(np.int32),
            'n_structural':    np.int32(n_structural),
        }

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
                dv = (opening_speed * normal).astype(np.float32)
                above_np = np.array([p['v_above'] for p in newly_broken], dtype=np.int32)
                below_np = np.array([p['v_below'] for p in newly_broken], dtype=np.int32)
                self._simulator._add_vel_scalar(above_np,  float(dv[0]),  float(dv[1]),  float(dv[2]))
                self._simulator._add_vel_scalar(below_np, -float(dv[0]), -float(dv[1]), -float(dv[2]))
            else:
                self._opening_ramp_queue.append({
                    'above':      np.array([p['v_above'] for p in newly_broken], dtype=np.int32),
                    'below':      np.array([p['v_below'] for p in newly_broken], dtype=np.int32),
                    'step_speed': opening_speed / ramp_frames,
                    'left':       ramp_frames,
                })

        return self.is_cut_complete(blade_travel)

    @property
    def vertex_count(self) -> int:
        return self._simulator.positions.shape[0]

    @property
    def n_orig(self) -> int | None:
        return self._n_orig_val


# ---------------------------------------------------------------------------
# FEMMethod
# ---------------------------------------------------------------------------

class FEMMethod(SimulationMethod):
    """
    Taichi GPU corotational FEM backend.

    Unconditionally stable — uses an implicit Euler integrator with a
    matrix-free conjugate gradient solver.  One step per frame; no substeps.

    params keys (all optional, defaults shown)
    ------------------------------------------
    young_modulus  : float = 5e4    stiffness in Pa
    poisson_ratio  : float = 0.4    material compressibility (0 = incompressible limit)
    density        : float = 1000.0 kg/m³ (distributed by tet volume)
    damping        : float = 1.0    velocity damping per second
    gravity        : list  = [0,0,0]
    v_max          : float = 10.0   velocity cap (m/s)
    time_step      : float = 0.01   integration timestep (larger than spring-mass is fine)
    cg_iters       : int   = 50     max CG iterations per step
    cg_eps         : float = 1e-6   CG convergence tolerance
    stiffness             : float = 200.0  spring stiffness for cutting springs (N/m)
    opening_speed         : float = 1.0    velocity (m/s) given to seam vertices on break
    opening_ramp_frames   : int   = 1
    sliver_vol_threshold  : float = 0.0    volume below which cut-adjacent tets are removed
                                           from the FEM (0 = keep all tets). Set to 1e-5
                                           to suppress force explosions from near-degenerate
                                           tets created when the plane nearly grazes a vertex.
    """

    def __init__(self, **overrides):
        self._simulator          = None
        self._params             = {}
        self._init_overrides     = overrides  # e.g. FEMMethod(cg_iters=50)
        self._n_orig_val         = None
        self._n_split            = None
        self._current_tets       = None
        self._cut_normal         = None
        self._cut_origin         = None
        self._blade_dir          = None
        self._seam_pairs          = []
        self._opening_ramp_queue  = []
        self._orphan_constraints  = {}   # zero-mass vert -> master vert (post-cut)
        self._debug_orphan_verts  = set()  # indices of all orphaned verts after cut (for purple overlay)
        # Phase 1b progressive-cut state (populated by setup_cut; unused when
        # progressive_cut=False, in which case _cut_topology_physics runs once
        # with split_mask=None at B-press as before).
        self._progressive_cut     = bool(overrides.get('progressive_cut', False))
        self._pristine_tets       = None   # (n_tets, 4) pre-cut tet array; input to per-frame split
        self._pristine_n_orig     = None   # pristine vertex count (never changes after initialize)
        self._split_schedule      = None   # (n_tets,) float32; blade_travel at which each crossing tet splits
        self._crossing_mask       = None   # (n_tets,) bool; True where the plane crosses (non-crossing tets are ignored by the schedule)
        self._split_mask          = None   # (n_tets,) bool; monotonically grown; True once tet has been split
        self._topology_version    = 0      # bumps each time _split_mask grows; ECS watches this to trigger render-side rebuild
        # Debug state
        self._debug_frames_since_cut = None   # None = pre-cut; int = frames since cut
        self._debug_fixed_orig_pos   = None   # fixed-vert positions at cut time
        self._debug_fixed_mask       = None   # fixed mask snapshot at cut time
        self._debug_pre_cut_force    = None   # max force on fixed verts last frame before cut
        self._debug_n_orig           = None   # n_orig at cut time (new-cut verts: >= n_orig)
        self._debug_cut_pos          = None   # positions of all verts at cut time

    def initialize(self, tet_mesh, params: dict) -> None:
        from pyGandalf.thesis_utilities.fem_simulator import _FEMSimulator
        self._params = dict(params)
        self._params.update(self._init_overrides)  # caller kwargs take priority

        y          = tet_mesh.vertices[:, 1]
        threshold  = y.min() + (y.max() - y.min()) * 0.05
        fixed_mask = (y < threshold).astype(np.int32)
        self._current_tets = tet_mesh.tetrahedra.copy()

        gravity = np.array(self._params.get('gravity', [0.0, 0.0, 0.0]), dtype=np.float32)

        self._simulator = _FEMSimulator(
            vertices      = tet_mesh.vertices,
            tetrahedra    = tet_mesh.tetrahedra,
            fixed_mask    = fixed_mask,
            young_modulus = float(self._params.get('young_modulus', 5e4)),
            poisson_ratio = float(self._params.get('poisson_ratio', 0.4)),
            density       = float(self._params.get('density', 1000.0)),
            gravity       = gravity,
            damping       = float(self._params.get('damping', 1.0)),
        )
        self._pristine_n_orig   = len(tet_mesh.vertices)
        self._pristine_positions = tet_mesh.vertices.astype(np.float32).copy()

    def step(self, dt: float) -> None:
        self._process_opening_ramp()
        self._simulator.step(
            dt,
            damping  = float(self._params.get('damping',  1.0)),
            v_max    = float(self._params.get('v_max',    10.0)),
            cg_iters = int(self._params.get('cg_iters',  20)),
        )
        # Progressive path: kinematically slave sliver-orphaned INTER/DUPs to
        # their master ORIG each step, on-GPU (no numpy round-trip).
        if getattr(self, '_orphan_ids', None) is not None and len(self._orphan_ids) > 0:
            self._simulator.apply_orphan_kinematics(
                self._orphan_ids,
                self._orphan_masters,
                self._orphan_offsets,
            )
        # Legacy one-shot path uses _apply_orphan_constraints (numpy). It's a
        # no-op when _orphan_constraints is empty (progressive doesn't populate it).
        self._apply_orphan_constraints()
        self._debug_step()
        self._debug_scan_high_velocity()
        self._debug_scan_seam_separation()

    def _debug_scan_high_velocity(self) -> None:
        """
        Flag verts whose speed exceeds a multiple of the configured
        opening_speed. Progressive mode only (this is where the current
        force problems live).

        Threshold is controlled by params['debug_vel_threshold_mult']
        (default 10x). Prints the top 5 offenders each frame the threshold
        is exceeded, with vertex classification (PRISTINE / INTER / DUP)
        and connected seam info when relevant.
        """
        if not self._progressive_cut:
            return
        opening_speed = float(self._params.get('opening_speed', 1.0))
        # v_max in the sim caps velocities at ~20 m/s by default, so 10x an
        # opening_speed of 5 would never fire. Default to 2.5x so we catch
        # anything meaningfully above the impulse magnitude.
        mult          = float(self._params.get('debug_vel_threshold_mult', 2.5))
        threshold     = mult * max(opening_speed, 1e-6)

        n_active = int(self._simulator.n_verts)
        vel = self._simulator.velocities.to_numpy()[:n_active]
        vmag = np.linalg.norm(vel, axis=1)

        bad_mask = vmag > threshold
        n_bad = int(bad_mask.sum())
        if n_bad == 0:
            return

        # Throttle: skip if it's been repeatedly firing with roughly the same
        # max speed. Print when the max grows by 20% or every 60 frames.
        vmax = float(vmag.max())
        last_max = getattr(self, '_debug_last_vmax', 0.0)
        last_frame_count = getattr(self, '_debug_last_frame_count', 0)
        cur_frame_count = last_frame_count + 1
        self._debug_last_frame_count = cur_frame_count
        if vmax < last_max * 1.2 and (cur_frame_count % 60 != 0):
            return
        self._debug_last_vmax = vmax

        # Classify each vert by index range and current side label if available.
        n_p     = self._pristine_n_orig or 0
        n_split = self._n_split         or 0
        top_idx = np.argsort(vmag)[::-1][:5]
        print(f"[VelDebug] frame={cur_frame_count}  {n_bad} verts exceed "
              f"{threshold:.2f} m/s ({mult:.1f}x opening_speed={opening_speed:.2f})  "
              f"max={vmax:.2f}  n_active={n_active}",
              flush=True)
        for i_arr in top_idx:
            i = int(i_arr)
            if i < n_p:
                cls = "PRISTINE"
            elif i < n_split:
                cls = "INTER   "
            else:
                cls = "DUP     "
            fixed_str = ""
            try:
                fx = int(self._simulator._fixed.to_numpy()[i])
                if fx != 0:
                    fixed_str = "  FIXED"
            except Exception:
                pass
            print(f"[VelDebug]   vert {i:>6d} [{cls}]  |v|={vmag[i]:8.2f}  "
                  f"v=({vel[i,0]:+8.2f},{vel[i,1]:+8.2f},{vel[i,2]:+8.2f})"
                  f"{fixed_str}",
                  flush=True)

    def _debug_scan_seam_separation(self) -> None:
        """
        For broken seam pairs, report:
          - INTER-DUP distance (how far the cut has opened locally).
          - pristine ORIG endpoints' current distance vs their PRISTINE
            rest distance (how much the surrounding mesh is stretching).

        A healthy progressive cut has the first quantity growing slowly and
        the second staying close to 1.0 (rest ratio). If both blow up, the
        FEM's restoring force isn't matching the impulses.

        Throttled to print every N frames or on threshold-crossing events.
        """
        if not self._progressive_cut or not self._seam_pairs:
            return
        broken_pairs = [p for p in self._seam_pairs if p['broken']]
        if not broken_pairs:
            return

        n_active = int(self._simulator.n_verts)
        pos      = self._simulator.positions.to_numpy()[:n_active]

        v_a = np.array([int(p['v_above']) for p in broken_pairs], dtype=np.int32)
        v_b = np.array([int(p['v_below']) for p in broken_pairs], dtype=np.int32)
        sep = np.linalg.norm(pos[v_a] - pos[v_b], axis=1)

        # Recover the two pristine ORIG endpoints for each seam pair from
        # inter_data cached in _topology_result. Skip if unavailable.
        stretch_ratio_max = None
        stretch_edge      = None
        stretched_over_2  = 0
        if self._topology_result is not None:
            inter_data = self._topology_result[10]  # list of (nid, vi, vj, t)
            inter_to_endpoints = {int(nid): (int(vi), int(vj))
                                  for nid, vi, vj, _t in inter_data}
            endpoints = []
            for p in broken_pairs:
                ep = inter_to_endpoints.get(int(p['v_above']))
                if ep is None:
                    endpoints.append(None)
                else:
                    endpoints.append(ep)
            ratios = []
            vi_list = []
            vj_list = []
            for ep in endpoints:
                if ep is None:
                    continue
                vi, vj = ep
                cur_d = float(np.linalg.norm(pos[vi] - pos[vj]))
                rest_d = float(np.linalg.norm(self._pristine_positions[vi]
                                              - self._pristine_positions[vj]))
                if rest_d > 1e-6:
                    ratios.append(cur_d / rest_d)
                    vi_list.append(vi)
                    vj_list.append(vj)
            if ratios:
                r_arr = np.asarray(ratios, dtype=np.float32)
                idx_max = int(np.argmax(r_arr))
                stretch_ratio_max = float(r_arr[idx_max])
                stretch_edge      = (vi_list[idx_max], vj_list[idx_max])
                stretched_over_2  = int((r_arr > 2.0).sum())

        # Bounding box scale so the seam-separation threshold is mesh-relative.
        n_p = int(self._pristine_n_orig or 0)
        if n_p > 0:
            mesh_scale = float(np.linalg.norm(
                self._pristine_positions.max(axis=0)
                - self._pristine_positions.min(axis=0)))
        else:
            mesh_scale = 1.0
        sep_threshold = 0.15 * mesh_scale   # 15% of mesh scale = clearly excessive

        sep_max = float(sep.max())
        n_over  = int((sep > sep_threshold).sum())

        # Throttle: print every 60 frames, or when max separation grows by 30%.
        last_seep_max = getattr(self, '_debug_last_seep_max', 0.0)
        last_frame    = getattr(self, '_debug_last_seep_frame', 0)
        cur_frame     = last_frame + 1
        self._debug_last_seep_frame = cur_frame
        should_print  = (sep_max > last_seep_max * 1.3
                         or (cur_frame % 60 == 0)
                         or (stretch_ratio_max is not None and stretch_ratio_max > 2.0
                             and stretched_over_2 > getattr(self, '_debug_last_stretched_2', -1)))
        if not should_print:
            return
        self._debug_last_seep_max     = sep_max
        self._debug_last_stretched_2  = stretched_over_2

        # Top-3 by INTER-DUP separation.
        top_sep = np.argsort(sep)[::-1][:3]
        print(f"[SeamDebug] frame={cur_frame}  broken_seams={len(broken_pairs)}  "
              f"max_INTER_DUP_sep={sep_max:.4f}  "
              f"(threshold {sep_threshold:.4f} = 15% of mesh_scale={mesh_scale:.4f})  "
              f"n_over={n_over}",
              flush=True)
        for k in top_sep:
            i = int(k)
            p = broken_pairs[i]
            print(f"[SeamDebug]   pair (v_above={int(p['v_above'])}, "
                  f"v_below={int(p['v_below'])})  sep={sep[i]:.4f}",
                  flush=True)
        if stretch_ratio_max is not None:
            print(f"[SeamDebug] max pristine-edge stretch ratio={stretch_ratio_max:.2f}x  "
                  f"({stretched_over_2} edges > 2x)  worst edge=({stretch_edge})",
                  flush=True)

    def _apply_orphan_constraints(self) -> None:
        if not self._orphan_constraints:
            return
        pos = self._simulator.positions.to_numpy()
        vel = self._simulator.velocities.to_numpy()

        for orphan, (master, offset) in self._orphan_constraints.items():
            pos[orphan] = pos[master] + offset
            vel[orphan] = vel[master]
        self._simulator.positions.from_numpy(pos.astype(np.float32))
        self._simulator.velocities.from_numpy(vel.astype(np.float32))

    def _debug_step(self):
        fixed_mask = self._simulator._fixed.to_numpy()
        n_fixed    = int(fixed_mask.sum())

        if self._debug_frames_since_cut is None:
            # Pre-cut baseline: track last force on fixed verts
            if n_fixed > 0:
                forces = self._simulator._forces.to_numpy()
                self._debug_pre_cut_force = float(
                    np.linalg.norm(forces[fixed_mask == 1], axis=1).max()
                )
            return

        if self._debug_frames_since_cut >= 30:
            return

        # Progressive mode: n_orig and _debug_cut_pos grow whenever the split
        # mask grows, so the "compare current post-cut positions to setup-time
        # positions" assumption breaks. All prints in this block are commented
        # out, so just skip the computation entirely under progressive_cut.
        if self._progressive_cut:
            self._debug_frames_since_cut += 1
            return

        f   = self._debug_frames_since_cut
        n_v = self._simulator.n_verts
        pos = self._simulator.positions.to_numpy()[:n_v]
        vel = self._simulator.velocities.to_numpy()[:n_v]
        fixed_mask = fixed_mask[:n_v]

        # --- fixed vert check (constraint) ---
        if n_fixed > 0:
            fixed_disp = np.linalg.norm(pos[fixed_mask == 1] - self._debug_fixed_orig_pos, axis=1)
            max_fixed_d = float(fixed_disp.max())
        else:
            max_fixed_d = 0.0

        # --- new-cut vert check (n_orig .. N-1) ---
        n_orig = self._debug_n_orig
        N      = pos.shape[0]
        if n_orig is not None and n_orig < N:
            new_idx      = np.arange(n_orig, N)
            new_pos      = pos[new_idx]
            new_pos_ref  = self._debug_cut_pos[new_idx]
            disp         = np.linalg.norm(new_pos - new_pos_ref, axis=1)
            max_disp     = float(disp.max())
            min_disp     = float(disp.min())
            mean_disp    = float(disp.mean())
            n_stuck      = int((disp < 1e-6).sum())
            n_expanding  = int((disp > 0.5).sum())

            new_vel_mag  = np.linalg.norm(vel[new_idx], axis=1)
            max_vel      = float(new_vel_mag.max())
            mean_vel     = float(new_vel_mag.mean())

            # NaN / inf check
            n_nan = int(np.isnan(new_pos).any(axis=1).sum())
            n_inf = int(np.isinf(new_pos).any(axis=1).sum())

            # Top-3 most displaced
            top3 = np.argsort(disp)[-3:][::-1]
            top3_global = new_idx[top3]
            top3_info   = [(int(top3_global[i]), float(disp[top3[i]]),
                            new_pos[top3[i]].tolist()) for i in range(len(top3))]

            # print(
            #     f"[DBG f+{f:02d}] new-cut({N - n_orig}): "
            #     f"disp max={max_disp:.4f} min={min_disp:.6f} mean={mean_disp:.4f} "
            #     f"stuck={n_stuck} expand>{0.5}={n_expanding} "
            #     f"vel max={max_vel:.4f} mean={mean_vel:.6f} "
            #     f"nan={n_nan} inf={n_inf} | "
            #     f"fixed max_disp={max_fixed_d:.6f}",
            #     flush=True
            # )
            # if f <= 2 or (f % 5 == 0):
            #     for vi, di, pi in top3_info:
            #         print(f"  top vert {vi}: disp={di:.4f}  pos={[round(x,4) for x in pi]}",
            #               flush=True)
        # else:
            # print(f"[DBG f+{f:02d}] (no new-cut verts)  fixed max_disp={max_fixed_d:.6f}",
            #       flush=True)

        self._debug_frames_since_cut += 1

    def _process_opening_ramp(self) -> None:
        if not self._opening_ramp_queue:
            return
        normal = self._cut_normal

        # In progressive mode entries store pristine edge keys; build a
        # fresh edge -> (v_above, v_below) map from the CURRENT seam_pairs
        # so impulses always land on the right verts even after rebuilds.
        edge_to_pair = None
        if self._progressive_cut:
            edge_to_pair = {p['edge']: (int(p['v_above']), int(p['v_below']))
                            for p in self._seam_pairs
                            if p.get('edge') is not None}

        active = []
        for entry in self._opening_ramp_queue:
            dv = (entry['step_speed'] * normal).astype(np.float32)
            if 'edges' in entry:
                above_list = []
                below_list = []
                for e in entry['edges']:
                    pair = edge_to_pair.get(e) if edge_to_pair is not None else None
                    if pair is None:
                        continue
                    above_list.append(pair[0])
                    below_list.append(pair[1])
                if above_list:
                    above = np.asarray(above_list, dtype=np.int32)
                    below = np.asarray(below_list, dtype=np.int32)
                    self._simulator._add_vel_scalar(above,  float(dv[0]),  float(dv[1]),  float(dv[2]))
                    self._simulator._add_vel_scalar(below, -float(dv[0]), -float(dv[1]), -float(dv[2]))
            else:
                above = np.asarray(entry['above'], dtype=np.int32)
                below = np.asarray(entry['below'], dtype=np.int32)
                self._simulator._add_vel_scalar(above,  float(dv[0]),  float(dv[1]),  float(dv[2]))
                self._simulator._add_vel_scalar(below, -float(dv[0]), -float(dv[1]), -float(dv[2]))
            entry['left'] -= 1
            if entry['left'] > 0:
                active.append(entry)
        self._opening_ramp_queue = active

    def get_positions(self) -> np.ndarray:
        n = self._simulator.n_verts
        return self._simulator.positions.to_numpy()[:n]

    def get_velocities(self) -> np.ndarray:
        n = self._simulator.n_verts
        return self._simulator.velocities.to_numpy()[:n]

    def set_velocities(self, indices: np.ndarray, velocities: np.ndarray) -> None:
        vels = self._simulator.velocities.to_numpy()
        vels[np.asarray(indices)] = velocities
        self._simulator.velocities.from_numpy(vels.astype(np.float32))

    def setup_cut(self, cut_normal, cut_origin, blade_travel_dir) -> bool:
        normal    = np.array(cut_normal,       dtype=np.float32)
        origin    = np.array(cut_origin,       dtype=np.float32)
        blade_dir = np.array(blade_travel_dir, dtype=np.float32)
        normal    /= np.linalg.norm(normal)
        blade_dir /= np.linalg.norm(blade_dir)

        gravity = np.array(self._params.get('gravity', [0.0, 0.0, 0.0]), dtype=np.float32)

        # Active-size views of the current simulator state. Fields are
        # capacity-allocated, so slicing strips zeroed padding before handing
        # arrays to topology code that assumes shape == vertex_count.
        n_active = self._simulator.n_verts
        cur_pos    = self._simulator.positions.to_numpy()[:n_active]
        cur_vel    = self._simulator.velocities.to_numpy()[:n_active]
        cur_masses = self._simulator._masses.to_numpy()[:n_active]
        cur_fixed  = self._simulator._fixed.to_numpy()[:n_active]

        # Check 4a: fixed count before cut
        n_fixed_before = int(cur_fixed.sum())
        _pre_force_str = f"{self._debug_pre_cut_force:.4f}" if self._debug_pre_cut_force is not None else "N/A"
        print(f"[DEBUG 4] Fixed verts before cut: {n_fixed_before}  (pre-cut max force on fixed: {_pre_force_str})", flush=True)

        # --- Phase 1b: capture pristine tets + schedule for progressive driver ---
        # Save the pre-cut tet array so advance_blade can re-invoke
        # _cut_topology_physics with a growing split_mask each frame. The schedule
        # is also computed inside _cut_topology_physics for logging; recomputed
        # here so it is available on self after setup_cut returns.
        self._pristine_tets = self._current_tets.copy()
        _signed_dist        = (cur_pos - origin) @ normal
        _tet_dists          = _signed_dist[self._current_tets]
        _crossing_mask      = ~(np.all(_tet_dists >= 0, axis=1)
                                | np.all(_tet_dists <= 0, axis=1))
        _sched              = np.full(len(self._current_tets), np.inf, dtype=np.float32)
        _cross_idx          = np.where(_crossing_mask)[0]
        if len(_cross_idx) > 0:
            _proj = (cur_pos[self._current_tets[_cross_idx]] - origin) @ blade_dir
            _sched[_cross_idx] = _proj.min(axis=1)
        self._crossing_mask = _crossing_mask
        self._split_schedule = _sched
        # Initial mask: nothing split yet in progressive mode; all crossings
        # split in the legacy path. Progressive mode also caches a full
        # precompute LAZILY on the first _maybe_grow_split call (see
        # _ensure_full_precompute) so setup_cut itself stays cheap.
        _initial_mask = np.zeros(len(self._current_tets), dtype=bool) \
            if self._progressive_cut else None

        result = _cut_topology_physics(
            self._current_tets,
            cur_pos, cur_vel, cur_masses, cur_fixed,
            0.0,          # stiffness unused — FEM doesn't need this for topology
            gravity, origin, normal,
            build_simulator=False,  # FEM discards new_sim; skip the build cost
            blade_dir=blade_dir,
            split_mask=_initial_mask,
        )
        if result is None:
            return False

        # Store the mask used for THIS call. Progressive path starts all-False
        # and grows monotonically in advance_blade. Legacy path uses all-True
        # (nothing to progress).
        self._split_mask = (_initial_mask.copy() if _initial_mask is not None
                            else np.ones(len(self._current_tets), dtype=bool))

        (_, final_pos, final_vel, _, final_fixed,
         all_tets, n_orig, n_split, shared_list, remap,
         _inter_data, _orig_surf_set, _phantom_keys, side_label,
         all_tets_parent) = result
        self._full_all_tets_parent = all_tets_parent  # precompute-and-subset cache
        # Progressive lazy cache -- populated on first _maybe_grow_split call.
        self._full_precompute_cache = None
        self._precomp_render_cache  = None
        # Stable-superset delta path state (populated on first grow).
        self._superset_initialized  = False
        self._parent_unsplit_slot   = None
        self._parent_split_slots    = None
        self._parent_split_slots_off = None
        self._active_verts_used     = None
        # Sliver / orphan state (populated by _build_sliver_and_orphan_data).
        self._sliver_slots   = None
        self._orphan_ids     = None
        self._orphan_masters = None
        self._orphan_offsets = None
        self._orphan_set     = set()
        # Cleared here so a re-setup rebuilds it against the new topology.
        if hasattr(self, '_full_inter_edge_lookup'):
            del self._full_inter_edge_lookup
        if hasattr(self, '_inter_vi_full'):
            self._inter_vi_full = None

        # Check 4b: fixed count after topology rebuild
        n_fixed_after = int(final_fixed.sum())
        print(f"[DEBUG 4] Fixed verts after  cut: {n_fixed_after}  "
              f"(delta: {n_fixed_after - n_fixed_before:+d})", flush=True)

        # Sliver filtering: cut-adjacent tets with near-zero volume have a near-singular
        # rest-shape matrix Dm, so B = Dm^-1 is huge and causes force explosions.
        # Controlled by sliver_vol_threshold param (default 0 = no filtering).
        # Only tets with at least one new vert (>= n_orig) can be cut-created slivers;
        # pure-original tets are never filtered.
        _vol_thresh   = float(self._params.get('sliver_vol_threshold', 0.0))
        _e1 = final_pos[all_tets[:, 1]] - final_pos[all_tets[:, 0]]
        _e2 = final_pos[all_tets[:, 2]] - final_pos[all_tets[:, 0]]
        _e3 = final_pos[all_tets[:, 3]] - final_pos[all_tets[:, 0]]
        _vols         = np.abs(np.einsum('ij,ij->i', _e1, np.cross(_e2, _e3))) / 6.0
        _has_new_vert = (all_tets >= n_orig).any(axis=1)
        if _vol_thresh > 0:
            _valid    = (~_has_new_vert) | (_vols > _vol_thresh)
            fem_tets  = all_tets[_valid]
        else:
            _valid    = np.ones(len(all_tets), dtype=bool)
            fem_tets  = all_tets
        _n_removed    = int((~_valid).sum())
        _n_cut_adj    = int(_has_new_vert.sum())
        _min_cut_vol  = float(_vols[_has_new_vert].min()) if _n_cut_adj > 0 else float('nan')
        print(f"[FILTER] n_orig={n_orig} total_tets={len(all_tets)} "
              f"cut_adj={_n_cut_adj} min_cut_vol={_min_cut_vol:.2e} "
              f"vol_thresh={_vol_thresh:.0e} removed={_n_removed}", flush=True)

        # Reuse the same _FEMSimulator across cuts: re-upload mesh state into
        # the existing Taichi fields so kernel SNode IDs stay stable. This is
        # the main speedup -- a fresh _FEMSimulator forces ~4s of JIT recompile
        # on the first step() after the cut.
        self._simulator.rebuild_topology(
            vertices   = final_pos,
            tetrahedra = fem_tets,
            fixed_mask = final_fixed,
            velocities = final_vel,
        )
        new_fem = self._simulator

        # Detect any verts orphaned by the filter (zero mass) and build kinematic constraints.
        # With the cut-adjacent-only filter, orig-mesh verts should retain at least one tet.
        # Only new-cut verts (>= n_orig) near a very shallow cut could become orphans here.
        n_active = new_fem.n_verts
        _masses_np  = new_fem._masses.to_numpy()[:n_active]
        _zero_mask  = _masses_np == 0.0
        _vert_remap  = {}
        _n_fallback  = 0
        if _zero_mask.any():
            _zero_idx    = np.where(_zero_mask)[0]
            _nonzero_idx = np.where(~_zero_mask)[0]
            _nz_pos      = final_pos[_nonzero_idx]

            # Intersection verts (n_orig..n_split-1) sit exactly on the cut plane and
            # have no cutting spring, so they never separate with either half.
            # Exclude them as masters so orphan verts don't get anchored to the plane.
            _not_inter = (_nonzero_idx < n_orig) | (_nonzero_idx >= n_split)

            # Master-side classification via side_label (SOFA/PhysBAM-style).
            # side_label is stamped at cut time from tet membership and SoS
            # symbolic perturbation, so every ORIG has an unambiguous +1/-1
            # tag even when its position sits exactly on the cut plane. This
            # replaces the fragile sign(dot(pos - origin, normal)) query used
            # previously, which dropped on-plane verts from both above and
            # below master pools and let orphans anchor across the plane
            # (spike-face mechanism).
            _is_dup     = _nonzero_idx >= n_split
            _is_orig    = _nonzero_idx < n_orig
            _master_side = side_label[_nonzero_idx]

            from scipy.spatial import cKDTree
            _zero_side = side_label[_zero_idx]

            _above_mask       = _is_orig & (_master_side > 0)
            _below_mask       = _is_dup | (_is_orig & (_master_side < 0))
            _above_pos        = _nz_pos[_above_mask];    _above_idx_arr    = _nonzero_idx[_above_mask]
            _below_pos        = _nz_pos[_below_mask];    _below_idx_arr    = _nonzero_idx[_below_mask]
            _notinter_pos     = _nz_pos[_not_inter];     _notinter_idx_arr = _nonzero_idx[_not_inter]

            def _resolve(group_mask, primary_pos, primary_idx_arr):
                nonlocal _n_fallback
                if not group_mask.any():
                    return
                orphan_v   = _zero_idx[group_mask]
                orphan_pos = final_pos[orphan_v]
                if len(primary_pos) > 0:
                    _, j  = cKDTree(primary_pos).query(orphan_pos)
                    masters = primary_idx_arr[j]
                else:
                    _n_fallback += int(group_mask.sum())
                    if len(_notinter_pos) > 0:
                        _, j  = cKDTree(_notinter_pos).query(orphan_pos)
                        masters = _notinter_idx_arr[j]
                    else:
                        _, j  = cKDTree(_nz_pos).query(orphan_pos)
                        masters = _nonzero_idx[j]
                for _v, _m in zip(orphan_v, masters):
                    _vert_remap[int(_v)] = int(_m)

            _resolve(_zero_side ==  1, _above_pos,    _above_idx_arr)
            _resolve(_zero_side == -1, _below_pos,    _below_idx_arr)
            _resolve(_zero_side ==  0, _notinter_pos, _notinter_idx_arr)

            # _masses is capacity-sized; modify the active slice in place via a
            # view, then re-upload the full capacity array.
            _masses_full = new_fem._masses.to_numpy()
            _active_view = _masses_full[:n_active]
            _min_m       = float(_active_view[~_zero_mask].min()) * 1e-4
            _active_view[_zero_mask] = _min_m
            new_fem._masses.from_numpy(_masses_full.astype(np.float32))
            _n_orig_orphans  = int((_zero_idx < n_orig).sum())
            _n_fixed_masters = int(sum(final_fixed[m] for m in _vert_remap.values()))
            print(f"[FEM] {len(_zero_idx)} orphans: {_n_orig_orphans} orig-mesh  "
                  f"{len(_zero_idx) - _n_orig_orphans} new-cut  "
                  f"fixed_masters={_n_fixed_masters}  fallback={_n_fallback}", flush=True)

        self._orphan_constraints = {
            orphan: (master, final_pos[orphan] - final_pos[master])
            for orphan, master in _vert_remap.items()
        }
        self._debug_orphan_verts = set(_vert_remap.keys())

        def _r(v: int) -> int:
            return _vert_remap.get(int(v), int(v))

        self._simulator       = new_fem
        self._current_tets    = all_tets
        self._n_orig_val      = n_orig
        self._n_split         = n_split
        self._cut_normal      = normal
        self._cut_origin      = origin
        self._blade_dir       = blade_dir
        self._topology_result = result  # cached for ECS surface computation

        # Cutting springs — endpoints remapped away from any zero-mass orphan verts.
        # Rest length = actual setup distance between the (possibly remapped) endpoints.
        # When neither endpoint is orphaned, this is 0 (the original assumption).
        # When orphan-remap pulled an endpoint to a non-coincident master, the rest
        # length records that initial separation so the spring is at equilibrium at
        # setup instead of immediately producing huge corrective forces.
        k_cut  = float(self._params.get('stiffness', 200.0)) * 0.5
        sa_cut = np.array([_r(v)        for v in shared_list], dtype=np.int32)
        sb_cut = np.array([_r(remap[v]) for v in shared_list], dtype=np.int32)
        sr_cut = np.linalg.norm(final_pos[sa_cut] - final_pos[sb_cut],
                                axis=1).astype(np.float32)
        sk_cut = np.full(len(shared_list), k_cut, dtype=np.float32)

        # Dedupe: orphan-remap can collapse multiple shared verts onto the same
        # (master_above, master_below) pair, producing N springs on one anchor
        # pair and N times the corrective force.  Keep the first occurrence of
        # each unordered (a, b) and deactivate the rest (k=0, r=0).  Spring
        # indices are preserved, so the blade-break logic still functions; the
        # deactivated springs just sit inert.
        _key_lo = np.minimum(sa_cut, sb_cut).astype(np.int64)
        _key_hi = np.maximum(sa_cut, sb_cut).astype(np.int64)
        _key    = (_key_lo << 32) | _key_hi
        _, _first_idx = np.unique(_key, return_index=True)
        _keep_mask = np.zeros(len(_key), dtype=bool)
        _keep_mask[_first_idx] = True
        _n_dedup = int((~_keep_mask).sum())
        sk_cut[~_keep_mask] = 0.0
        sr_cut[~_keep_mask] = 0.0
        if _n_dedup > 0:
            print(f"[Cut] Deactivated {_n_dedup} duplicate cutting springs "
                  f"({len(_keep_mask) - _n_dedup} unique anchor pairs remain)",
                  flush=True)

        new_fem.extend_springs(sa_cut, sb_cut, sr_cut, sk_cut)

        # Check 1: cutting spring endpoints vs fixed vertices
        fixed_set = set(np.where(final_fixed == 1)[0])
        springs_on_fixed = [(int(a), int(b)) for a, b in zip(sa_cut, sb_cut)
                            if a in fixed_set or b in fixed_set]
        print(f"[DEBUG 1] Cutting springs: {len(sa_cut)} total, "
              f"{len(springs_on_fixed)} touch fixed verts", flush=True)
        if springs_on_fixed:
            print(f"  Affected pairs (first 10): {springs_on_fixed[:10]}", flush=True)

        # Reset debug state for post-cut step() tracking
        self._debug_fixed_orig_pos   = final_pos[final_fixed == 1].copy()
        self._debug_fixed_mask       = final_fixed.copy()
        self._debug_frames_since_cut = 0
        self._debug_n_orig           = n_orig
        self._debug_cut_pos          = final_pos.copy()

        # Build v_above -> pristine edge map for stable seam identity across
        # rebuilds (only meaningful in progressive mode, but harmless in legacy).
        _va_to_edge_init = {int(nid): (int(min(vi, vj)), int(max(vi, vj)))
                            for nid, vi, vj, _t in _inter_data}

        seam_pairs = []
        for i, v_above in enumerate(shared_list):
            v_below     = int(remap[v_above])
            spring_idx  = i
            pos_seam    = final_pos[_r(v_above)]
            travel_dist = float(np.dot(pos_seam - origin, blade_dir))
            seam_pairs.append({
                'v_above':     _r(v_above),
                'v_below':     _r(v_below),
                'edge':        _va_to_edge_init.get(int(v_above)),
                'spring_idx':  spring_idx,
                'travel_dist': travel_dist,
                'broken':      False,
            })
        seam_pairs.sort(key=lambda p: p['travel_dist'])
        self._seam_pairs         = seam_pairs
        self._opening_ramp_queue = []
        return True

    def advance_blade(self, blade_travel: float) -> bool:
        # Progressive: grow the split mask if the blade has reached new tets,
        # rebuild topology + sim + seam_pairs before running the break loop.
        if self._progressive_cut:
            self._maybe_grow_split(blade_travel)

        # Legacy short-circuit: no seam pairs means nothing to progress and
        # the blade is done. In progressive mode we intentionally allow the
        # blade to keep advancing when seam_pairs is empty so we don't
        # short-circuit before is_cut_complete has a chance to gate on the
        # schedule.
        if not self._seam_pairs and not self._progressive_cut:
            return True

        opening_speed = float(self._params.get('opening_speed', 1.0))
        ramp_frames   = int(self._params.get('opening_ramp_frames', 1))
        normal        = self._cut_normal

        newly_broken = [p for p in self._seam_pairs
                        if not p['broken'] and p['travel_dist'] <= blade_travel]

        if newly_broken:
            for p in newly_broken:
                # spring_idx == -1 in progressive mode (no cutting springs);
                # skip the _sk write in that case.
                si = int(p['spring_idx'])
                if si >= 0:
                    self._simulator._sk[si] = 0.0
                p['broken'] = True

            if ramp_frames <= 1:
                dv = (opening_speed * normal).astype(np.float32)
                above_np = np.array([p['v_above'] for p in newly_broken], dtype=np.int32)
                below_np = np.array([p['v_below'] for p in newly_broken], dtype=np.int32)
                self._simulator._add_vel_scalar(above_np,  float(dv[0]),  float(dv[1]),  float(dv[2]))
                self._simulator._add_vel_scalar(below_np, -float(dv[0]), -float(dv[1]), -float(dv[2]))
            else:
                # Progressive mode: store pristine edge keys (stable across
                # rebuilds) instead of raw v_above/v_below indices, which
                # shift every rebuild. _process_opening_ramp resolves them
                # to the current v_above/v_below at application time.
                if self._progressive_cut:
                    self._opening_ramp_queue.append({
                        'edges':      [p.get('edge') for p in newly_broken],
                        'step_speed': opening_speed / ramp_frames,
                        'left':       ramp_frames,
                    })
                else:
                    self._opening_ramp_queue.append({
                        'above':      np.array([p['v_above'] for p in newly_broken], dtype=np.int32),
                        'below':      np.array([p['v_below'] for p in newly_broken], dtype=np.int32),
                        'step_speed': opening_speed / ramp_frames,
                        'left':       ramp_frames,
                    })

        return self.is_cut_complete(blade_travel)

    def _build_parent_slot_lookups(self) -> None:
        """
        Precompute, per pristine tet:
          - `_parent_unsplit_slot[p]`: slot ID in the FEM tet field for this
             tet's unsplit-form representation, or -1 for non-crossing tets.
          - `_parent_split_slots[start:end]` (via `_parent_split_slots_off`):
             slot IDs for the sub-tets produced by splitting p, packed CSR-style.

        Slot ID layout (must match init_superset_topology upload order):
          [0, n_full_tets)                     : full_all_tets sub-tets
          [n_full_tets, n_full_tets + n_cross) : crossing pristine tets, unsplit
        """
        n_pristine    = len(self._pristine_tets)
        full_parent   = self._full_all_tets_parent
        n_full_tets   = len(self._full_all_tets)
        crossing_idx  = np.where(self._crossing_mask)[0].astype(np.int64)
        n_cross       = len(crossing_idx)

        # Unsplit slot map: -1 for non-crossing, else n_full_tets + rank(p in crossing_idx)
        self._parent_unsplit_slot = np.full(n_pristine, -1, dtype=np.int32)
        self._parent_unsplit_slot[crossing_idx] = n_full_tets + np.arange(n_cross, dtype=np.int32)

        # Split slot map: CSR packing. For each pristine p, its sub-tet slots
        # are contiguous in `_parent_split_slots[_parent_split_slots_off[p] :
        # _parent_split_slots_off[p+1]]`.
        sort_order = np.argsort(full_parent, kind='stable')
        sorted_parents = full_parent[sort_order]
        self._parent_split_slots = sort_order.astype(np.int32)   # slots in full-block

        # Offsets: for each pristine p, find its first/last position in sorted_parents.
        ids = np.arange(n_pristine, dtype=np.int64)
        self._parent_split_slots_off = np.zeros(n_pristine + 1, dtype=np.int32)
        # np.searchsorted over sorted_parents to get boundaries
        starts = np.searchsorted(sorted_parents, ids,     side='left')
        ends   = np.searchsorted(sorted_parents, ids + 1, side='left')
        self._parent_split_slots_off[:-1] = starts.astype(np.int32)
        self._parent_split_slots_off[-1]  = ends[-1] if n_pristine > 0 else 0

    def _init_superset_layout(self) -> None:
        """
        One-time FEM init at first progressive rebuild. Uploads the FULL
        superset topology (all possible sub-tets + all unsplit crossings)
        to the Taichi fields with pristine rest-shape data, computes B/W
        for all of them, then zeros out W for split-side sub-tets (which
        start inactive because no crossing tet has flipped yet). Pins
        inactive INTER/DUP vertices. Subsequent per-frame changes are
        applied by _apply_split_mask_delta.

        Starts with "everything unsplit" (all crossing tets shown in their
        pristine 4-vert form). The caller then feeds the current new_mask
        into _apply_split_mask_delta to activate the tets that should
        already be split at this blade position.
        """
        self._build_parent_slot_lookups()

        n_full_tets = len(self._full_all_tets)
        crossing_idx = np.where(self._crossing_mask)[0].astype(np.int64)
        n_cross      = len(crossing_idx)
        n_super      = n_full_tets + n_cross
        n_full_v     = int(self._full_final_pos.shape[0])

        # Build stable-order superset tet array (must match slot IDs).
        unsplit_cross_tets = self._pristine_tets[self._crossing_mask].astype(np.int32)
        super_tets = np.vstack([self._full_all_tets.astype(np.int32),
                                unsplit_cross_tets])

        # Initial activation: full-block sub-tets active only if their parent
        # is non-crossing (they represent the parent as-is); unsplit-block
        # rows all active (all crossings start unsplit).
        parent_of_slot = np.empty(n_super, dtype=np.int64)
        parent_of_slot[:n_full_tets] = self._full_all_tets_parent
        parent_of_slot[n_full_tets:] = crossing_idx
        tet_active_mask = np.zeros(n_super, dtype=bool)
        tet_active_mask[:n_full_tets] = ~self._crossing_mask[parent_of_slot[:n_full_tets]]
        tet_active_mask[n_full_tets:] = True

        active_verts_used = np.zeros(n_full_v, dtype=bool)
        active_slots = np.where(tet_active_mask)[0]
        if len(active_slots) > 0:
            active_verts_used[super_tets[active_slots].ravel()] = True

        n_p = int(self._pristine_n_orig)
        cur_pristine_pos = self._simulator.positions.to_numpy()[:n_p]
        cur_pristine_vel = self._simulator.velocities.to_numpy()[:n_p]

        init_pos = self._full_final_pos.copy()
        init_vel = np.zeros((n_full_v, 3), dtype=np.float32)
        init_pos[:n_p] = cur_pristine_pos
        init_vel[:n_p] = cur_pristine_vel

        init_fixed = self._full_final_fixed.copy()
        init_fixed[~active_verts_used] = 1  # pin inactive INTER/DUP
        init_vel[~active_verts_used]   = 0

        self._simulator.init_superset_topology(
            vertices        = init_pos,
            super_tets      = super_tets,
            tet_active_mask = tet_active_mask,
            fixed_mask      = init_fixed,
            velocities      = init_vel,
            rest_positions  = self._full_final_pos,
        )

        # Permanently disable sliver sub-tets: zero out W_rest so
        # _activate_tets_kernel becomes a no-op for these slots when their
        # parent flips split. Also pin all orphan verts so FEM never touches
        # them (they're driven by the per-step orphan kinematics kernel).
        if hasattr(self, '_sliver_slots') and len(self._sliver_slots) > 0:
            # These slots may have been "active" in the initial mask and thus
            # contributed mass in init_tet_data. Deactivate first, then also
            # zero W_rest so they never reactivate.
            self._simulator._deactivate_tets_kernel(self._sliver_slots)
            self._simulator._zero_W_rest_kernel(self._sliver_slots)
        if hasattr(self, '_orphan_ids') and len(self._orphan_ids) > 0:
            # Pin orphan verts unconditionally (they'll be moved by the
            # orphan-kinematics kernel each step, not by FEM).
            self._simulator._pin_verts_kernel(self._orphan_ids)

        self._super_tets_ref       = super_tets
        self._active_verts_used    = active_verts_used
        self._split_mask           = np.zeros(len(self._pristine_tets), dtype=bool)
        self._superset_initialized = True

    def _apply_split_mask_delta(self, new_mask: np.ndarray) -> None:
        """
        Per-frame delta: for each pristine tet flipping unsplit->split,
        deactivate its unsplit slot and activate its split slots, then
        write initial state for any newly-touched INTER/DUP vertices.
        Assumes _init_superset_layout has been called.
        """
        old_mask = (self._split_mask if self._split_mask is not None
                    else np.zeros(len(self._pristine_tets), dtype=bool))
        newly_split = new_mask & ~old_mask
        flipped_ids = np.where(newly_split)[0]
        if len(flipped_ids) == 0:
            return

        # Slots to deactivate: parent's unsplit-block slot (one per flip).
        deact_all = self._parent_unsplit_slot[flipped_ids]
        deact_slots = deact_all[deact_all >= 0].astype(np.int32)

        # Slots to activate: concat of parent's split-block slots (variable per flip).
        offs = self._parent_split_slots_off
        act_parts = [self._parent_split_slots[offs[p]:offs[p+1]]
                     for p in flipped_ids]
        if act_parts:
            act_slots = np.concatenate(act_parts).astype(np.int32)
        else:
            act_slots = np.zeros(0, dtype=np.int32)

        # Which vertices are used by any newly-activated slot but weren't
        # active before? These need initial state written.
        if len(act_slots) > 0:
            act_verts = self._super_tets_ref[act_slots].ravel()
            act_verts_unique = np.unique(act_verts)
        else:
            act_verts_unique = np.zeros(0, dtype=np.int32)

        was_active = self._active_verts_used
        newly_active_v = act_verts_unique[~was_active[act_verts_unique]]

        if len(newly_active_v) > 0:
            n_p = int(self._pristine_n_orig)
            cur_pristine_pos = self._simulator.positions.to_numpy()[:n_p]
            cur_pristine_vel = self._simulator.velocities.to_numpy()[:n_p]

            # For each newly-active vert: interpolate from pristine ORIG endpoints
            # (INTER via inter_data lookup, DUP via shared_list->INTER->same).
            n_orig = self._full_n_orig
            n_split = self._full_n_split

            # Precomputed once at ensure_full_precompute (numpy arrays).
            nids = self._full_inter_nids
            vis  = self._full_inter_vis
            vjs  = self._full_inter_vjs
            ts   = self._full_inter_ts

            # Map each newly-active vert to (vi, vj, t). DUPs map back to their INTER.
            # Build a per-full-vert lookup for (vi, vj, t) that supports both INTER and DUP.
            if not hasattr(self, '_inter_vi_full') or self._inter_vi_full is None:
                total_v = int(self._full_final_pos.shape[0])
                vi_full = np.arange(total_v, dtype=np.int64)
                vj_full = np.arange(total_v, dtype=np.int64)
                t_full  = np.zeros(total_v, dtype=np.float32)
                # INTERs
                if len(nids) > 0:
                    vi_full[nids] = vis
                    vj_full[nids] = vjs
                    t_full[nids]  = ts
                # DUPs: mirror their INTER's (vi, vj, t)
                shared = self._full_shared_arr
                if len(shared) > 0:
                    dup_ids = self._full_shared_dup_arr
                    vi_full[dup_ids] = vi_full[shared]
                    vj_full[dup_ids] = vj_full[shared]
                    t_full[dup_ids]  = t_full[shared]
                self._inter_vi_full = vi_full
                self._inter_vj_full = vj_full
                self._inter_t_full  = t_full

            na_v = newly_active_v.astype(np.int64)
            is_new_vert = na_v >= n_orig   # skip ORIGs (already active from B-press)
            na_v = na_v[is_new_vert]

            # Orphan verts stay pinned; the per-step orphan kinematics kernel
            # drives them. Only unpin+interpolate normal (non-orphan) INTER/DUPs.
            if len(na_v) > 0 and len(getattr(self, '_orphan_ids', [])) > 0:
                orphan_arr = self._orphan_ids.astype(np.int64)
                keep_mask  = ~np.isin(na_v, orphan_arr)
                na_v = na_v[keep_mask]

            if len(na_v) > 0:
                via = self._inter_vi_full[na_v]
                vjb = self._inter_vj_full[na_v]
                t   = self._inter_t_full[na_v][:, None]
                pos_new = cur_pristine_pos[via] * (1.0 - t) + cur_pristine_pos[vjb] * t
                vel_new = cur_pristine_vel[via] * (1.0 - t) + cur_pristine_vel[vjb] * t

                self._simulator.write_vert_state(
                    indices    = na_v.astype(np.int32),
                    positions  = pos_new.astype(np.float32),
                    velocities = vel_new.astype(np.float32),
                    fixed      = np.zeros(len(na_v), dtype=np.int32),
                )

        # Flip tet activations on the FEM.
        self._simulator.flip_tets_delta(deact_slots, act_slots)

        # Update the maintained active-verts mask.
        if len(act_verts_unique) > 0:
            self._active_verts_used[act_verts_unique] = True
        # Note: we do NOT deactivate any verts. In progressive cut, once a
        # vert is used it stays used until the cut is done.

    def _maybe_grow_split(self, blade_travel: float) -> None:
        """
        Progressive per-frame topology grower.

        Computes the new split_mask from the schedule (crossing tets whose
        centroid the blade has reached). If it differs from the current
        _split_mask, re-invokes _cut_topology_physics on the pristine tets
        with the new mask, rebuilds the FEM sim topology in place, and
        rebuilds the seam_pairs list. Broken state is preserved by
        (v_above, v_below) pair keys captured before the rebuild.

        No cutting springs are created in progressive mode: INTER / DUP
        verts are supposed to be kinematically slaved to their ORIG
        endpoints (a future step). For now they are regular FEM DOFs, but
        seam breaking still just applies an opening impulse -- the tets
        holding the two halves together are the FEM tets, not springs.
        """
        if self._split_schedule is None or self._crossing_mask is None:
            return
        new_mask = self._crossing_mask & (self._split_schedule <= blade_travel)
        if self._split_mask is not None and np.array_equal(new_mask, self._split_mask):
            return

        import time as _t
        _t0 = _t.perf_counter()

        # Snapshot broken seams by pristine EDGE key. seam_pair['edge'] was
        # stashed at build time so this loop is O(active seams).
        broken_edge_keys: set = set()
        for p in self._seam_pairs:
            if p['broken']:
                stored = p.get('edge')
                if stored is not None:
                    broken_edge_keys.add(stored)

        n_new_split = int(new_mask.sum() - (self._split_mask.sum()
                                             if self._split_mask is not None else 0))
        print(f"[Progressive] blade_travel={blade_travel:.4f}  "
              f"newly-splitting {n_new_split} tets  "
              f"(cumulative {int(new_mask.sum())}/{int(self._crossing_mask.sum())} crossing tets split)",
              flush=True)

        self._ensure_full_precompute()
        _t_precomp = _t.perf_counter() - _t0

        # ---- Physics: superset init on first call, then delta ----
        _t1 = _t.perf_counter()
        if not self._superset_initialized:
            # First call: initialize with everything unsplit, then apply the
            # current mask via the same delta logic that runs each frame.
            self._init_superset_layout()
        self._apply_split_mask_delta(new_mask)
        _t_apply = _t.perf_counter() - _t1

        # Snapshot current sim state as a small numpy array for the ECS
        # to read via _topology_result. Only needed for the render-side
        # winding correction / wound centroid sort in the fast rebuild.
        _t2 = _t.perf_counter()
        n_full  = int(self._full_final_pos.shape[0])
        final_pos_snap = self._simulator.positions.to_numpy()[:n_full]
        final_vel_snap = self._simulator.velocities.to_numpy()[:n_full]

        self._split_mask         = new_mask.copy()
        self._topology_version  += 1
        # For ECS compatibility: expose the FULL superset as _current_tets so
        # downstream helpers (pick tool, etc.) can index into positions
        # correctly. Inactive tets have W=0 and contribute nothing to physics.
        self._current_tets    = self._super_tets_ref
        self._n_orig_val      = int(self._full_n_orig)
        self._n_split         = int(self._full_n_split)

        full = self._full_precompute_cache
        self._topology_result = (
            full[0],
            final_pos_snap,
            final_vel_snap,
            full[3],
            self._full_final_fixed,   # inactive verts pinned=1 (stable across frames)
            self._super_tets_ref,     # full superset (W=0 disables inactive)
            int(self._full_n_orig),
            int(self._full_n_split),
            full[8], full[9], full[10], full[11], full[12], full[13], full[14],
        )
        _t_snap = _t.perf_counter() - _t2

        # ---- Rebuild seam_pairs from the FULL shared_list; filter to active ----
        _t3 = _t.perf_counter()
        shared_arr_np = self._full_shared_arr
        dup_arr_np    = self._full_shared_dup_arr
        active_verts_used = self._active_verts_used
        seam_active_mask = (active_verts_used[shared_arr_np]
                          & active_verts_used[dup_arr_np])
        active_seam_nids = shared_arr_np[seam_active_mask]
        active_seam_dups = dup_arr_np[seam_active_mask]

        if len(active_seam_nids) > 0:
            positions_seam = self._full_final_pos[active_seam_nids]
            travel_dists   = ((positions_seam - self._cut_origin)
                              @ self._blade_dir).astype(np.float32)
            sort_idx = np.argsort(travel_dists)
            active_seam_nids = active_seam_nids[sort_idx]
            active_seam_dups = active_seam_dups[sort_idx]
            travel_dists     = travel_dists[sort_idx]

            if not hasattr(self, '_full_inter_edge_lookup'):
                n_full_v = int(self._full_final_pos.shape[0])
                self._full_inter_edge_lookup = np.full((n_full_v, 2), -1,
                                                        dtype=np.int64)
                if len(self._full_inter_nids) > 0:
                    mn = np.minimum(self._full_inter_vis, self._full_inter_vjs)
                    mx = np.maximum(self._full_inter_vis, self._full_inter_vjs)
                    self._full_inter_edge_lookup[self._full_inter_nids, 0] = mn
                    self._full_inter_edge_lookup[self._full_inter_nids, 1] = mx
            edges_active = self._full_inter_edge_lookup[active_seam_nids]

            seam_pairs = []
            for i in range(len(active_seam_nids)):
                nid = int(active_seam_nids[i])
                dup = int(active_seam_dups[i])
                e0 = int(edges_active[i, 0])
                edge = ((e0, int(edges_active[i, 1])) if e0 >= 0 else None)
                was_broken = (edge in broken_edge_keys) if edge is not None else False
                seam_pairs.append({
                    'v_above':     nid,
                    'v_below':     dup,
                    'edge':        edge,
                    'spring_idx':  -1,
                    'travel_dist': float(travel_dists[i]),
                    'broken':      was_broken,
                })
        else:
            seam_pairs = []
        self._seam_pairs = seam_pairs
        _t_seams = _t.perf_counter() - _t3
        _t_total = _t.perf_counter() - _t0
        print(f"[GrowTiming] total={_t_total*1000:.1f}ms  "
              f"precomp={_t_precomp*1000:.1f}  "
              f"apply={_t_apply*1000:.1f}  "
              f"snap={_t_snap*1000:.1f}  "
              f"seams={_t_seams*1000:.1f}  "
              f"newly_split={n_new_split}  "
              f"n_full_verts={n_full:,}",
              flush=True)

    def _ensure_full_precompute(self) -> None:
        """
        Lazily run one full-split _cut_topology_physics call at the pristine
        mesh and cache all outputs. Called on the first _maybe_grow_split
        invocation. Subsequent per-frame rebuilds subset this cache rather
        than re-running the O(pristine_tets) split logic.
        """
        if getattr(self, '_full_precompute_cache', None) is not None:
            return
        gravity = np.array(self._params.get('gravity', [0.0, 0.0, 0.0]), dtype=np.float32)
        n_p = int(self._pristine_n_orig)
        vel0 = np.zeros((n_p, 3), dtype=np.float32)
        mass0 = np.ones(n_p, dtype=np.float32)  # placeholder; not used downstream
        fixed0 = self._simulator._fixed.to_numpy()[:n_p]

        _tcu._QUIET_CUT_LOGS = True
        try:
            result = _cut_topology_physics(
                self._pristine_tets,
                self._pristine_positions,
                vel0, mass0, fixed0,
                0.0, gravity, self._cut_origin, self._cut_normal,
                build_simulator = False,
                blade_dir       = self._blade_dir,
                split_mask      = None,      # FULL split
            )
        finally:
            _tcu._QUIET_CUT_LOGS = False
        if result is None:
            raise RuntimeError("_ensure_full_precompute: cut plane misses mesh?")

        self._full_precompute_cache = result
        self._full_all_tets        = result[5]
        self._full_n_orig          = int(result[6])
        self._full_n_split         = int(result[7])
        self._full_shared_list     = list(result[8])
        self._full_remap           = result[9]
        self._full_inter_data      = list(result[10])
        self._full_final_pos       = result[1].copy()
        self._full_final_fixed     = result[4].copy()
        self._full_side_label      = result[13].copy()
        self._full_all_tets_parent = result[14]

        # Numpy views of inter_data / shared_list for vectorized use in
        # _maybe_grow_split. Building these once avoids Python-loop overhead
        # in every progressive rebuild.
        if self._full_inter_data:
            _idata = np.array(
                [(int(d[0]), int(d[1]), int(d[2]), float(d[3]))
                 for d in self._full_inter_data],
                dtype=np.float64,
            )
            self._full_inter_nids = _idata[:, 0].astype(np.int64)
            self._full_inter_vis  = _idata[:, 1].astype(np.int64)
            self._full_inter_vjs  = _idata[:, 2].astype(np.int64)
            self._full_inter_ts   = _idata[:, 3].astype(np.float32)
        else:
            self._full_inter_nids = np.zeros(0, dtype=np.int64)
            self._full_inter_vis  = np.zeros(0, dtype=np.int64)
            self._full_inter_vjs  = np.zeros(0, dtype=np.int64)
            self._full_inter_ts   = np.zeros(0, dtype=np.float32)
        self._full_shared_arr = np.asarray(self._full_shared_list, dtype=np.int64)
        if len(self._full_shared_arr) > 0:
            self._full_shared_dup_arr = (
                self._full_remap[self._full_shared_arr].astype(np.int64)
            )
        else:
            self._full_shared_dup_arr = np.zeros(0, dtype=np.int64)

        print(f"[Step4] Precomputed full split: "
              f"{len(self._full_all_tets):,} tets, "
              f"{len(self._full_final_pos):,} verts, "
              f"{len(self._full_inter_data)} INTERs, "
              f"{len(self._full_shared_list)} DUPs "
              f"(vs pristine {len(self._pristine_tets):,} tets, {n_p:,} verts)",
              flush=True)

        self._build_render_precompute_cache()
        self._build_sliver_and_orphan_data()

    def _build_sliver_and_orphan_data(self) -> None:
        """
        Identify sliver sub-tets in the full-split superset (pristine volume
        below `sliver_vol_threshold`) and build kinematic orphan constraints
        for any INTER/DUP vertex whose only sub-tets are slivers.

        Slivers stay permanently inactive: _W_rest=0 in _init_superset_layout
        so _activate_tets_kernel becomes a no-op for them. Orphan verts get
        their position and velocity slaved to a nearest well-massed master
        vertex (typically an ORIG one hop away) via a small per-step Taichi
        kernel.
        """
        _thresh = float(self._params.get('sliver_vol_threshold', 0.0))
        # Compute pristine volumes over the FULL superset (full-block + unsplit-block).
        n_full_tets = len(self._full_all_tets)
        crossing_idx = np.where(self._crossing_mask)[0].astype(np.int64)
        unsplit_cross_tets = self._pristine_tets[self._crossing_mask].astype(np.int32)
        super_tets = np.vstack([self._full_all_tets.astype(np.int32),
                                unsplit_cross_tets])
        v = self._full_final_pos
        e1 = v[super_tets[:, 1]] - v[super_tets[:, 0]]
        e2 = v[super_tets[:, 2]] - v[super_tets[:, 0]]
        e3 = v[super_tets[:, 3]] - v[super_tets[:, 0]]
        vols = np.abs(np.einsum('ij,ij->i', e1, np.cross(e2, e3))) / 6.0
        # Only cut-created tets (any vert >= n_orig) are candidates for filtering.
        # Pristine unsplit tets are already known-good.
        n_orig = self._full_n_orig
        has_new_vert = (super_tets >= n_orig).any(axis=1)
        is_sliver = has_new_vert & (vols < _thresh) if _thresh > 0.0 else np.zeros(len(super_tets), dtype=bool)
        self._sliver_slots = np.where(is_sliver)[0].astype(np.int32)

        # Orphan detection based on END-OF-CUT topology.
        # Every crossing parent tet appears in the superset TWICE: once as an
        # unsplit-block row (all ORIGs, normal volume) and once as several
        # split-block sub-tets (INTER/DUP verts, potentially slivers). At any
        # progressive-cut state only ONE of the two is active per parent, and
        # progressive cutting monotonically moves each parent from unsplit to
        # split. To predict which verts will end up orphaned we must consider
        # only the END STATE -- all crossings split. In that state:
        #   - non-crossing pristine tets contribute (full-block rows whose
        #     parent isn't crossing)
        #   - split-block sub-tets of crossing parents contribute
        #   - unsplit-block rows are all inactive (W=0)
        # Verts touched only by slivers in the end state are the true orphans.
        n_full_v = int(self._full_final_pos.shape[0])

        # End-state active mask over superset slots.
        n_full_tets = len(self._full_all_tets)
        end_state_active = np.zeros(len(super_tets), dtype=bool)
        end_state_active[:n_full_tets] = True     # full-block always active at end
        # unsplit-block rows are all inactive at end (all crossings split)
        # good_end_slots = active AND non-sliver
        good_end_slots = np.where(end_state_active & (~is_sliver))[0]

        vol_by_vert = np.zeros(n_full_v, dtype=np.float64)
        if len(good_end_slots) > 0:
            contribs = np.repeat(vols[good_end_slots], 4) / 4.0
            verts_flat = super_tets[good_end_slots].ravel()
            np.add.at(vol_by_vert, verts_flat, contribs)

        touched_by_any_end = np.zeros(n_full_v, dtype=bool)
        touched_by_any_end[super_tets[end_state_active].ravel()] = True

        # Volume floor: small fraction of median end-state good-tet volume.
        if len(good_end_slots) > 0:
            _median_vol = float(np.median(vols[good_end_slots]))
            _floor = _median_vol * 1e-4
        else:
            _floor = 0.0
        orphan_mask = touched_by_any_end & (vol_by_vert < _floor)
        orphan_ids = np.where(orphan_mask)[0].astype(np.int32)

        # Master finding: for each orphan, look at its geometric neighborhood.
        # For an INTER vert nid, we know its pristine edge (vi, vj) -- those two
        # ORIG endpoints are its natural masters. Pick the closer one by
        # pristine distance. For a DUP vert (n_split + i), resolve to its
        # INTER via _full_shared_arr then use the same rule.
        if len(orphan_ids) > 0:
            n_split = self._full_n_split
            # For each orphan, resolve to underlying INTER
            underlying_inter = orphan_ids.copy().astype(np.int64)
            is_dup = underlying_inter >= n_split
            if is_dup.any() and len(self._full_shared_arr) > 0:
                # dup_to_inter: n_split + i -> shared_list[i]
                dup_offsets = underlying_inter[is_dup] - n_split
                underlying_inter[is_dup] = self._full_shared_arr[dup_offsets]

            # Look up each INTER's (vi, vj, t). Populate INTERs first, then
            # DUPs (which mirror their shared-INTER's endpoints), so both
            # this method and _apply_split_mask_delta can index the arrays
            # by ANY [n_orig, n_full_v) vertex without a DUP-resolution step.
            if not hasattr(self, '_inter_vi_full') or self._inter_vi_full is None:
                total_v = n_full_v
                self._inter_vi_full = np.arange(total_v, dtype=np.int64)
                self._inter_vj_full = np.arange(total_v, dtype=np.int64)
                self._inter_t_full  = np.zeros(total_v, dtype=np.float32)
                if len(self._full_inter_nids) > 0:
                    self._inter_vi_full[self._full_inter_nids] = self._full_inter_vis
                    self._inter_vj_full[self._full_inter_nids] = self._full_inter_vjs
                    self._inter_t_full[self._full_inter_nids]  = self._full_inter_ts
                if len(self._full_shared_arr) > 0:
                    src = self._full_shared_arr
                    dup = self._full_shared_dup_arr
                    self._inter_vi_full[dup] = self._inter_vi_full[src]
                    self._inter_vj_full[dup] = self._inter_vj_full[src]
                    self._inter_t_full[dup]  = self._inter_t_full[src]

            vis = self._inter_vi_full[underlying_inter]
            vjs = self._inter_vj_full[underlying_inter]

            # Master picking uses side_label (not raw geometric sign):
            #   - side_label is stamped from signed_dist_split, which drives
            #     the physics (each half moves with its side_label group).
            #   - For a near-plane ORIG, raw geometric sign is float noise
            #     while side_label is stable and consistent with which half
            #     the vert actually moves with.
            #   - For the edge to have produced an INTER at all, split logic
            #     required signed_dist_split[vi] and [vj] to have opposite
            #     signs. So side_label[vi] and [vj] are always opposite and
            #     exactly one of them matches the orphan's side_label.
            side_label  = self._full_side_label
            orphan_side = side_label[orphan_ids]
            vi_side     = side_label[vis]
            vj_side     = side_label[vjs]
            vi_match    = (vi_side == orphan_side)
            vj_match    = (vj_side == orphan_side)

            # geom_side is still needed for the kd-tree fallback pool, so
            # keep computing it. Pool membership is "physically above/below
            # pristine plane" which is the natural geometric definition.
            n_p = int(self._pristine_n_orig)
            pristine_pos = self._full_final_pos[:n_p]
            geom_dot = (pristine_pos - self._cut_origin) @ self._cut_normal
            geom_side = np.sign(geom_dot).astype(np.int8)

            # Orphan-vert set for exclusion from master pools. Prevents
            # orphan-to-orphan slaving chains where an orphan A masters to
            # orphan B, and B is being slaved elsewhere. Such chains produced
            # incoherent cluster motion visible as sliver-edge spikes. This
            # also handles ORIG orphans (idx < n_orig): they self-reference
            # in the vi/vj lookup (identity), so vi_is_orphan/vj_is_orphan
            # catches them and forces the kd-tree fallback below.
            orphan_bool = np.zeros(n_full_v, dtype=bool)
            orphan_bool[orphan_ids] = True
            vi_is_orphan = orphan_bool[vis]
            vj_is_orphan = orphan_bool[vjs]
            vi_match &= ~vi_is_orphan
            vj_match &= ~vj_is_orphan

            # Case A: vi matches -> vi. Case B: vj matches -> vj. Case D:
            # neither matches -> kd-tree fallback among non-orphan same-side
            # ORIGs and non-orphan DUPs (mimics one-shot's pool).
            masters = np.where(vi_match, vis, vjs).astype(np.int64)

            neither_matches = ~vi_match & ~vj_match
            if neither_matches.any():
                # Master pool: same-side ORIGs (by pristine geometry) PLUS
                # non-orphan DUPs (side_label = -1 by construction).
                #   Above pool: ORIGs with geom_side > 0, NOT orphans.
                #   Below pool: ORIGs with geom_side < 0 + non-orphan DUPs.
                # Matches one-shot's `_is_dup | (_is_orig & side<0)` selection.
                above_pool_mask = np.zeros(n_full_v, dtype=bool)
                above_pool_mask[:n_orig] = (geom_side > 0) & (~orphan_bool[:n_orig])
                below_pool_mask = np.zeros(n_full_v, dtype=bool)
                below_pool_mask[:n_orig] = (geom_side < 0) & (~orphan_bool[:n_orig])
                # DUPs are side_label = -1 by construction (below half).
                dup_start = self._full_n_split
                dup_end   = dup_start + len(self._full_shared_arr)
                if dup_end > dup_start:
                    below_pool_mask[dup_start:dup_end] = ~orphan_bool[dup_start:dup_end]

                above_pool_idx = np.where(above_pool_mask)[0]
                below_pool_idx = np.where(below_pool_mask)[0]

                try:
                    from scipy.spatial import cKDTree
                    above_tree = (cKDTree(self._full_final_pos[above_pool_idx])
                                  if len(above_pool_idx) > 0 else None)
                    below_tree = (cKDTree(self._full_final_pos[below_pool_idx])
                                  if len(below_pool_idx) > 0 else None)
                    stray_idx = np.where(neither_matches)[0]
                    stray_orphans = orphan_ids[stray_idx]
                    stray_side    = orphan_side[stray_idx]
                    for j, o_local in enumerate(stray_idx):
                        o_global = int(stray_orphans[j])
                        o_pos    = self._full_final_pos[o_global]
                        if stray_side[j] > 0 and above_tree is not None:
                            _, k = above_tree.query(o_pos)
                            masters[o_local] = int(above_pool_idx[k])
                        elif stray_side[j] < 0 and below_tree is not None:
                            _, k = below_tree.query(o_pos)
                            masters[o_local] = int(below_pool_idx[k])
                        # else: no pool on this side -- keep vj fallback.
                except ImportError:
                    pass  # scipy not available; keep the vj fallback

            # Offset: orphan_pristine_pos - master_pristine_pos.
            offsets = (self._full_final_pos[orphan_ids]
                     - self._full_final_pos[masters]).astype(np.float32)

            self._orphan_ids     = orphan_ids
            self._orphan_masters = masters.astype(np.int32)
            self._orphan_offsets = offsets
            self._orphan_set     = set(int(v) for v in orphan_ids)
        else:
            self._orphan_ids     = np.zeros(0, dtype=np.int32)
            self._orphan_masters = np.zeros(0, dtype=np.int32)
            self._orphan_offsets = np.zeros((0, 3), dtype=np.float32)
            self._orphan_set     = set()

        print(f"[Sliver] threshold={_thresh:.2e}  slivers={len(self._sliver_slots):,} "
              f"orphans={len(self._orphan_ids):,}", flush=True)

    def _build_render_precompute_cache(self) -> None:
        """
        Build a per-face render classification cache over the union of:
          - all sub-tets produced by the full split (each present when its
            parent pristine tet's mask == True)
          - all crossing pristine tets in unsplit form (each present when its
            parent's mask == False)
          - non-crossing pristine tets are already covered by the full-split
            output (they show up in `unsplit_arr` inside _cut_topology_physics
            with parent==self and active-always).

        Everything the current per-frame render pipeline computes from vertex
        indices alone (wound/outer/drop verdict, debug color, category) is
        precomputed here. Per-frame work in the fast render path reduces to
          - masking rows by split state
          - packed-key unique+count for boundary detection
          - indexing precomputed classification arrays
          - winding correction on the outer subset only
          - normals kernel + GL upload
        """
        import time as _t
        _t0 = _t.perf_counter()

        full_all_tets  = self._full_all_tets                # (n_full, 4)
        full_parent    = self._full_all_tets_parent         # (n_full,)
        pristine_tets  = self._pristine_tets
        crossing_mask  = self._crossing_mask
        n_orig         = self._full_n_orig
        n_split        = self._full_n_split
        shared_list    = self._full_shared_list
        inter_data     = self._full_inter_data
        orig_surf_set  = self._full_precompute_cache[11]

        crossing_idx = np.where(crossing_mask)[0].astype(np.int64)
        n_full_tets  = len(full_all_tets)
        n_cross      = len(crossing_idx)

        # Superset tet table
        unsplit_cross_tets = pristine_tets[crossing_mask].astype(np.int32)
        super_tets = np.vstack([full_all_tets.astype(np.int32),
                                unsplit_cross_tets])
        tet_parent = np.concatenate([full_parent.astype(np.int64),
                                     crossing_idx])
        tet_active_when_split = np.concatenate([
            np.ones(n_full_tets, dtype=bool),
            np.zeros(n_cross,     dtype=bool),
        ])
        # Which pristine tets are crossing (per-row).
        tet_parent_is_crossing = crossing_mask[tet_parent]

        # 4 faces per tet
        FACE_COMBOS = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
        FOURTH_PER_FACE = np.array([3, 2, 1, 0], dtype=np.int32)
        n_super = len(super_tets)
        face_verts  = super_tets[:, FACE_COMBOS].reshape(-1, 3).astype(np.int64)
        face_fourth = super_tets[np.arange(n_super)[:, None],
                                 FOURTH_PER_FACE[None, :]].reshape(-1).astype(np.int64)
        face_parent = np.repeat(tet_parent, 4)
        face_parent_is_crossing = np.repeat(tet_parent_is_crossing, 4)
        face_active_when_split  = np.repeat(tet_active_when_split, 4)

        # Sorted verts + packed int64 keys
        face_sorted = np.sort(face_verts, axis=1)
        assert int(face_sorted.max()) < (1 << 21), "vertex index too large to pack"
        face_keys = ((face_sorted[:, 0] << 42)
                   | (face_sorted[:, 1] << 21)
                   |  face_sorted[:, 2])

        # -- On-plane vertex mask --
        n_v_max = max(int(face_verts.max()) + 1,
                      n_split + len(shared_list) + 1)
        on_plane = np.zeros(n_v_max, dtype=bool)
        on_plane[n_orig:] = True  # everything >= n_orig is on-plane
        if shared_list:
            seam_arr = np.array(shared_list, dtype=np.int64)
            on_plane[seam_arr] = True

        v_onplane   = on_plane[face_verts]              # (M, 3)
        all_onplane = v_onplane.all(axis=1)
        any_onplane = v_onplane.any(axis=1)
        face_wound     = all_onplane
        face_pure_orig = ~any_onplane
        face_collar    = any_onplane & ~all_onplane

        # -- orig_surf_set as packed keys + orig_surf_verts lookup --
        if orig_surf_set:
            surf_key_arr = np.array(sorted(orig_surf_set), dtype=np.int64)
            orig_surf_keys = ((surf_key_arr[:, 0] << 42)
                            | (surf_key_arr[:, 1] << 21)
                            |  surf_key_arr[:, 2])
            orig_surf_keys.sort()
            orig_surf_verts_arr = np.unique(surf_key_arr.ravel())
        else:
            orig_surf_keys      = np.array([], dtype=np.int64)
            orig_surf_verts_arr = np.array([], dtype=np.int64)
        in_surf_lookup = np.zeros(n_v_max, dtype=bool)
        if len(orig_surf_verts_arr) > 0:
            in_surf_lookup[orig_surf_verts_arr] = True

        # PhantomCollar: collar face where an off-plane orig vert is NOT in
        # orig_surf_verts (matches the first Python filter block).
        is_off_plane_orig = (face_verts < n_orig) & ~v_onplane   # (M, 3)
        in_surf_val = in_surf_lookup[face_verts]                 # (M, 3)
        bad_orig_v  = is_off_plane_orig & ~in_surf_val
        bad_orig_face = bad_orig_v.any(axis=1)
        has_off_plane_orig = is_off_plane_orig.any(axis=1)
        phantom_collar_drop = (face_collar
                               & has_off_plane_orig
                               & bad_orig_face
                               & (len(orig_surf_verts_arr) > 0))

        # Pure-orig T-junction: drop if the sorted triple is not in orig_surf_set.
        face_in_orig_surf = (np.isin(face_keys, orig_surf_keys)
                             if len(orig_surf_keys) > 0
                             else np.zeros(len(face_keys), dtype=bool))
        pure_orig_drop = face_pure_orig & ~face_in_orig_surf

        # [CollarFilter] reconstruction filter (vectorized).
        vi_above_lookup = np.arange(n_v_max, dtype=np.int64)
        vj_below_lookup = np.arange(n_v_max, dtype=np.int64)
        if inter_data:
            idata = np.array([(int(d[0]), int(d[1]), int(d[2])) for d in inter_data],
                             dtype=np.int64)
            vi_above_lookup[idata[:, 0]] = idata[:, 1]
            vj_below_lookup[idata[:, 0]] = idata[:, 2]
        dup_to_inter = np.arange(n_v_max, dtype=np.int64)
        if shared_list:
            for i, v in enumerate(shared_list):
                dup_to_inter[n_split + i] = int(v)

        fv_inter  = np.where(face_verts >= n_split,
                             dup_to_inter[face_verts], face_verts)
        is_inter_v = (fv_inter >= n_orig) & (fv_inter < n_split)
        fv_above = np.where(is_inter_v, vi_above_lookup[fv_inter], fv_inter)
        fv_below = np.where(is_inter_v, vj_below_lookup[fv_inter], fv_inter)

        recon6 = np.column_stack([fv_above[:, 0], fv_below[:, 0],
                                  fv_above[:, 1], fv_below[:, 1],
                                  fv_above[:, 2], fv_below[:, 2]])
        recon6.sort(axis=1)
        diffs = np.diff(recon6, axis=1) != 0
        recon_size = 1 + diffs.sum(axis=1)

        collar_recon_drop = np.zeros(len(face_verts), dtype=bool)
        is_three = (recon_size == 3)
        if is_three.any():
            keep_mask = np.hstack([np.ones((len(recon6), 1), dtype=bool), diffs])
            rows3         = np.where(is_three)[0]
            recon6_three  = recon6[rows3]
            keep_three    = keep_mask[rows3]
            unique_recon  = recon6_three[keep_three].reshape(-1, 3)
            three_keys = ((unique_recon[:, 0] << 42)
                        | (unique_recon[:, 1] << 21)
                        |  unique_recon[:, 2])
            three_in_surf = (np.isin(three_keys, orig_surf_keys)
                             if len(orig_surf_keys) > 0
                             else np.zeros(len(three_keys), dtype=bool))
            collar_recon_drop[rows3] = ~three_in_surf
        # Only collar faces are subject to this filter.
        collar_recon_drop &= face_collar

        # Verdict: 0 outer, 1 wound, 2 drop
        VERDICT_OUTER, VERDICT_WOUND, VERDICT_DROP = 0, 1, 2
        face_verdict = np.full(len(face_verts), VERDICT_OUTER, dtype=np.int8)
        face_verdict[face_wound]           = VERDICT_WOUND
        face_verdict[pure_orig_drop]       = VERDICT_DROP
        face_verdict[phantom_collar_drop]  = VERDICT_DROP
        face_verdict[collar_recon_drop]    = VERDICT_DROP

        # Category (see _compute_face_categories, cats 0..2 -- 3/4 are handled
        # by the drop verdicts above, so we don't need cat 3/4 here).
        face_category = np.zeros(len(face_verts), dtype=np.int8)
        is_all_orig = (face_verts < n_orig).all(axis=1)
        face_category[is_all_orig] = 0
        face_category[any_onplane & ~all_onplane] = 1
        face_category[all_onplane] = 2

        # Debug colors (see _compute_debug_face_colors).
        face_debug_color = np.tile([0.3, 0.5, 0.8],
                                   (len(face_verts), 1)).astype(np.float32)
        any_new = np.any(face_verts >= n_orig, axis=1)
        all_new = np.all(face_verts >= n_orig, axis=1)
        face_debug_color[any_new & ~all_new] = [0.3, 0.7, 0.4]  # collar
        face_debug_color[all_new]            = [0.85, 0.1, 0.1] # disc/wound

        # Map each row to a unique-key index so per-frame boundary detection
        # uses O(K) np.bincount instead of O(M log M) np.unique. The unique
        # key set is stable across frames because it depends only on vertex
        # indices; only membership (which keys are currently active) changes.
        _unique_keys, face_key_idx = np.unique(face_keys, return_inverse=True)
        n_unique_keys = int(len(_unique_keys))

        self._precomp_render_cache = {
            'face_verts':               face_verts.astype(np.uint32),
            'face_fourth':              face_fourth.astype(np.int32),
            'face_keys':                face_keys,
            'face_key_idx':             face_key_idx.astype(np.int32),
            'n_unique_keys':            n_unique_keys,
            'face_parent':              face_parent.astype(np.int32),
            'face_parent_is_crossing':  face_parent_is_crossing,
            'face_active_when_split':   face_active_when_split,
            'face_verdict':             face_verdict,
            'face_category':            face_category,
            'face_debug_color':         face_debug_color,
            'n_super_tets':             n_super,
        }

        _t_elapsed = _t.perf_counter() - _t0
        n_wound_p = int((face_verdict == VERDICT_WOUND).sum())
        n_outer_p = int((face_verdict == VERDICT_OUTER).sum())
        n_drop_p  = int((face_verdict == VERDICT_DROP).sum())
        print(f"[RenderPrecomp] built face table in {_t_elapsed*1000:.1f}ms: "
              f"{len(face_verts):,} rows  "
              f"(outer_candidates={n_outer_p:,}  wound_candidates={n_wound_p:,}  "
              f"drops={n_drop_p:,})",
              flush=True)

    def is_cut_complete(self, blade_travel: float) -> bool:
        """
        Progressive-aware override.

        Legacy path: same as the base class -- all existing seam pairs broken.
        Progressive path: the blade must first travel past the largest split
        schedule entry among crossing tets before "done" can fire. Until then,
        even an empty seam_pairs list is still "in progress" (step 3 will
        populate it incrementally as the blade passes crossing tets).
        """
        if not self._progressive_cut:
            return all(p['broken'] for p in self._seam_pairs)

        if self._split_schedule is None or self._crossing_mask is None:
            # Progressive requested but setup_cut hasn't populated the schedule
            # yet; treat as in-progress so the blade doesn't short-circuit.
            return False

        cross_sched = self._split_schedule[self._crossing_mask]
        if len(cross_sched) == 0:
            # Plane misses the mesh -- nothing to cut, trivially done.
            return True

        if blade_travel < float(cross_sched.max()):
            return False

        # Blade has passed the last scheduled tet split. Cut is done only if
        # every seam pair that DID get created is also broken. Empty list ->
        # vacuously True, which is what we want here.
        return all(p['broken'] for p in self._seam_pairs)

    @property
    def vertex_count(self) -> int:
        return self._simulator.n_verts

    @property
    def n_orig(self) -> int | None:
        return self._n_orig_val


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
