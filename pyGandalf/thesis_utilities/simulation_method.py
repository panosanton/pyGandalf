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

        # Snapshot broken seams by pristine EDGE key (min(vi,vj), max(vi,vj))
        # of the INTER's underlying edge. Edge keys are stable across
        # rebuilds; v_above / v_below index shifts every time the split_mask
        # grows, so using indices would let already-broken seams look fresh
        # again and get their opening impulse re-applied every rebuild.
        broken_edge_keys: set = set()
        old_inter_data_snapshot = None
        if self._topology_result is not None:
            old_inter_data_snapshot = self._topology_result[10]
            old_va_to_edge = {int(nid): (int(min(vi, vj)), int(max(vi, vj)))
                              for nid, vi, vj, _t in old_inter_data_snapshot}
            for p in self._seam_pairs:
                if not p['broken']:
                    continue
                stored = p.get('edge')
                if stored is not None:
                    broken_edge_keys.add(stored)
                    continue
                e = old_va_to_edge.get(int(p['v_above']))
                if e is not None:
                    broken_edge_keys.add(e)
        n_new_split = int(new_mask.sum() - (self._split_mask.sum()
                                             if self._split_mask is not None else 0))
        print(f"[Progressive] blade_travel={blade_travel:.4f}  "
              f"newly-splitting {n_new_split} tets  "
              f"(cumulative {int(new_mask.sum())}/{int(self._crossing_mask.sum())} crossing tets split)",
              flush=True)

        # --- Snapshot old state for edge-key based preservation ---
        # _cut_topology_physics renumbers INTER/DUP verts each call because
        # iteration order in the split loop can shift when new tets enter the
        # split_mask (a lower-numbered tet inserts its edges ahead of a
        # higher-numbered tet's). Without preservation, every existing INTER
        # and DUP gets teleported back to the interpolated edge-midpoint of
        # its pristine endpoints each frame, wiping the opening impulse and
        # any FEM-integrated motion.
        n_active_old = int(self._simulator.n_verts)
        old_pos = self._simulator.positions.to_numpy()[:n_active_old].copy()
        old_vel = self._simulator.velocities.to_numpy()[:n_active_old].copy()
        old_edge_to_inter: dict = {}
        old_inter_to_dup: dict  = {}
        if self._topology_result is not None:
            old_inter_data = self._topology_result[10]  # list of (nid, vi, vj, t)
            for nid, vi, vj, _t in old_inter_data:
                edge = (int(vi), int(vj)) if int(vi) < int(vj) else (int(vj), int(vi))
                old_edge_to_inter[edge] = int(nid)
            old_n_split    = int(self._topology_result[7])
            old_shared_list = self._topology_result[8]
            for i, v in enumerate(old_shared_list):
                old_inter_to_dup[int(v)] = old_n_split + i

        # ---- Step 4: precompute + subset ----
        import time as _t
        _t0 = _t.perf_counter()
        # Populate the full-split precompute cache lazily on first call.
        self._ensure_full_precompute()
        _t_precomp = _t.perf_counter() - _t0

        n_p     = int(self._pristine_n_orig)
        n_full  = int(self._full_final_pos.shape[0])
        full_all_tets   = self._full_all_tets
        full_parent     = self._full_all_tets_parent
        pristine_tets   = self._pristine_tets
        crossing_mask   = self._crossing_mask

        # Build the ACTIVE tet array by subsetting full precompute:
        # - Sub-tets whose parent pristine tet is currently split (mask=True)
        #   OR whose parent is non-crossing come from _full_all_tets.
        # - Crossing pristine tets whose mask=False are added as 4-vert rows.
        parent_active_lookup = np.ones(len(pristine_tets), dtype=bool)
        unsplit_crossing_mask = crossing_mask & ~new_mask
        parent_active_lookup[unsplit_crossing_mask] = False
        row_active = parent_active_lookup[full_parent]
        active_sub_tets = full_all_tets[row_active]
        unsplit_pristine = pristine_tets[unsplit_crossing_mask]
        if len(unsplit_pristine) > 0:
            active_tets = np.vstack([active_sub_tets, unsplit_pristine]).astype(np.int32)
        else:
            active_tets = active_sub_tets.astype(np.int32)

        # Which verts are used by at least one active tet.
        active_verts_used = np.zeros(n_full, dtype=bool)
        if len(active_tets) > 0:
            active_verts_used[active_tets.ravel()] = True

        # Snapshot old sim state (positions and velocities) up to n_full so we
        # can preserve INTER/DUP dynamics for verts that stay active across
        # this rebuild.
        old_pos = self._simulator.positions.to_numpy()[:n_full].copy()
        old_vel = self._simulator.velocities.to_numpy()[:n_full].copy()

        # Compute old active-vert mask so we know which INTER/DUPs already had
        # meaningful sim state (versus verts activating for the first time).
        old_split_mask = self._split_mask if self._split_mask is not None \
                         else np.zeros(len(pristine_tets), dtype=bool)
        old_parent_active_lookup = np.ones(len(pristine_tets), dtype=bool)
        old_unsplit_crossing = crossing_mask & ~old_split_mask
        old_parent_active_lookup[old_unsplit_crossing] = False
        old_row_active = old_parent_active_lookup[full_parent]
        old_active_sub_tets = full_all_tets[old_row_active]
        old_unsplit_pristine = pristine_tets[old_unsplit_crossing]
        was_active_old = np.zeros(n_full, dtype=bool)
        if len(old_active_sub_tets) > 0:
            was_active_old[old_active_sub_tets.ravel()] = True
        if len(old_unsplit_pristine) > 0:
            was_active_old[old_unsplit_pristine.ravel()] = True

        # ---- Assemble sim upload arrays sized at full topology ----
        # Positions: start with pristine (from full_final_pos). Overlay current
        # sim state for pristine ORIGs and for verts that were already active.
        cur_pristine_pos = self._simulator.positions.to_numpy()[:n_p]
        cur_pristine_vel = self._simulator.velocities.to_numpy()[:n_p]

        active_pos = self._full_final_pos.copy()
        active_vel = np.zeros((n_full, 3), dtype=np.float32)

        # ORIGs: always active; use current sim state.
        active_pos[:n_p] = cur_pristine_pos
        active_vel[:n_p] = cur_pristine_vel

        # INTERs / DUPs: if they were active before, preserve their sim state.
        # Otherwise leave them at pristine (from full_final_pos) with zero vel.
        preserved_mask = np.zeros(n_full, dtype=bool)
        preserved_mask[n_p:] = was_active_old[n_p:]
        if preserved_mask.any():
            idx = np.where(preserved_mask)[0]
            active_pos[idx] = old_pos[idx]
            active_vel[idx] = old_vel[idx]

        # For NEW INTERs/DUPs (active now but not before), interpolate
        # positions/velocities from current pristine ORIG positions using the
        # full inter_data t-values, so they enter the sim at the current
        # deformed edge midpoint (not the pristine edge midpoint).
        newly_active = active_verts_used & ~was_active_old
        for nid_int, vi_int, vj_int, t_float in self._full_inter_data:
            nid = int(nid_int)
            if not newly_active[nid]:
                continue
            vi = int(vi_int); vj = int(vj_int); t = float(t_float)
            active_pos[nid] = cur_pristine_pos[vi] * (1.0 - t) + cur_pristine_pos[vj] * t
            active_vel[nid] = cur_pristine_vel[vi] * (1.0 - t) + cur_pristine_vel[vj] * t
        # Newly-active DUPs mirror their INTER's fresh state.
        full_remap = self._full_remap
        for i, nid_int in enumerate(self._full_shared_list):
            dup_idx = int(full_remap[int(nid_int)])
            if not newly_active[dup_idx]:
                continue
            src = int(nid_int)
            active_pos[dup_idx] = active_pos[src]
            active_vel[dup_idx] = active_vel[src]

        # Fixed mask: original fixed for active verts; pinned (fixed=1) for
        # inactive verts so the CG solver doesn't wander with mass=0.
        active_fixed = self._full_final_fixed.copy()
        active_fixed[~active_verts_used] = 1
        active_vel[~active_verts_used]   = 0

        _t_subset = _t.perf_counter() - _t0 - _t_precomp
        # Rebuild sim topology in place. rest_positions = pristine geometry
        # (full_final_pos) so FEM strain forces try to restore the pre-cut
        # rest shape. vertices = current deformed geometry, uploaded after B
        # is computed so the integrator sees the actual state.
        _t1 = _t.perf_counter()
        self._simulator.rebuild_topology(
            vertices       = active_pos,
            tetrahedra     = active_tets,
            fixed_mask     = active_fixed,
            velocities     = active_vel,
            rest_positions = self._full_final_pos,
        )
        _t_rebuild = _t.perf_counter() - _t1

        self._current_tets    = active_tets
        self._n_orig_val      = int(self._full_n_orig)
        self._n_split         = int(self._full_n_split)
        self._split_mask      = new_mask.copy()
        self._topology_version += 1

        # Update _topology_result for the ECS to read. Substitute the ACTIVE
        # tet array in position 5 and current positions/velocities/fixed in
        # positions 1/2/4; keep the rest from the full precompute (shared_list,
        # remap, inter_data, orig_surf_set, side_label are all "full" values).
        full = self._full_precompute_cache
        self._topology_result = (
            full[0],           # new_sim (None for FEM)
            active_pos,        # final_pos (current)
            active_vel,        # final_vel (current)
            full[3],           # final_mass (unused by ECS)
            active_fixed,      # final_fixed
            active_tets,       # all_tets  (SUBSET)
            int(self._full_n_orig),
            int(self._full_n_split),
            full[8],           # shared_list  (full)
            full[9],           # remap        (full)
            full[10],          # inter_data   (full)
            full[11],          # orig_surf_set
            full[12],          # phantom_above_face_keys
            full[13],          # side_label
            full[14],          # all_tets_parent
        )

        # ---- Rebuild seam_pairs from the FULL shared_list; filter to active ----
        full_inter_data = self._full_inter_data
        inter_to_edge = {int(nid): (int(min(vi, vj)), int(max(vi, vj)))
                         for nid, vi, vj, _t in full_inter_data}
        seam_pairs = []
        for nid_int in self._full_shared_list:
            nid = int(nid_int)
            if not active_verts_used[nid]:
                continue
            dup_idx = int(full_remap[nid])
            if not active_verts_used[dup_idx]:
                continue
            edge = inter_to_edge.get(nid)
            pos_seam = self._full_final_pos[nid]
            travel_dist = float(np.dot(pos_seam - self._cut_origin, self._blade_dir))
            was_broken = (edge in broken_edge_keys) if edge is not None else False
            seam_pairs.append({
                'v_above':     nid,
                'v_below':     dup_idx,
                'edge':        edge,
                'spring_idx':  -1,
                'travel_dist': travel_dist,
                'broken':      was_broken,
            })
        seam_pairs.sort(key=lambda p: p['travel_dist'])
        self._seam_pairs = seam_pairs
        _t_total = _t.perf_counter() - _t0
        _t_seams = _t_total - _t_precomp - _t_subset - _t_rebuild
        print(f"[GrowTiming] total={_t_total*1000:.1f}ms  "
              f"precomp={_t_precomp*1000:.1f}  "
              f"subset+alloc={_t_subset*1000:.1f}  "
              f"rebuild_topo={_t_rebuild*1000:.1f}  "
              f"seams={_t_seams*1000:.1f}  "
              f"n_active_tets={len(active_tets):,}  "
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

        print(f"[Step4] Precomputed full split: "
              f"{len(self._full_all_tets):,} tets, "
              f"{len(self._full_final_pos):,} verts, "
              f"{len(self._full_inter_data)} INTERs, "
              f"{len(self._full_shared_list)} DUPs "
              f"(vs pristine {len(self._pristine_tets):,} tets, {n_p:,} verts)",
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
