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
        )
        if result is None:
            return False

        (new_sim, final_pos, _final_vel, _final_mass, _final_fixed,
         all_tets, n_orig, n_split, shared_list, remap, _inter_data, _orig_surf_set,
         _phantom_keys) = result

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

        return all(p['broken'] for p in self._seam_pairs)

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

        result = _cut_topology_physics(
            self._current_tets,
            cur_pos, cur_vel, cur_masses, cur_fixed,
            0.0,          # stiffness unused — FEM doesn't need this for topology
            gravity, origin, normal,
            build_simulator=False,  # FEM discards new_sim; skip the build cost
        )
        if result is None:
            return False

        (_, final_pos, final_vel, _, final_fixed,
         all_tets, n_orig, n_split, shared_list, remap,
         _inter_data, _orig_surf_set, _phantom_keys) = result

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

            # Per-vert centroid dot for CANDIDATE filtering only.
            # Needed because seam-dup candidates (>= n_split) are co-located with their
            # above-half counterparts at cut time, so position is ambiguous for them.
            _tet_cdots_fem = (final_pos[fem_tets].mean(axis=1) - origin) @ normal
            _sum_f  = np.zeros(len(final_pos), dtype=np.float64)
            _cnt_f  = np.zeros(len(final_pos), dtype=np.int32)
            np.add.at(_sum_f, fem_tets.ravel(), np.repeat(_tet_cdots_fem, 4))
            np.add.at(_cnt_f, fem_tets.ravel(), 1)
            _vert_cdots = _sum_f / np.where(_cnt_f > 0, _cnt_f, 1)
            _nz_cdots   = _vert_cdots[_nonzero_idx]

            # Side label per orphan: 1 = above, -1 = below, 0 = on-plane (use _not_inter only).
            # Same rules as the per-vert branch above, but vectorized across all orphans.
            from scipy.spatial import cKDTree
            _zero_side = np.zeros(len(_zero_idx), dtype=np.int8)
            _zero_side[_zero_idx >= n_split] = -1
            _zero_side[(_zero_idx >= n_orig) & (_zero_idx < n_split)] = 1
            _orig_pick = _zero_idx < n_orig
            if _orig_pick.any():
                _dot = (final_pos[_zero_idx[_orig_pick]] - origin) @ normal
                _zero_side[_orig_pick] = np.where(_dot > 0, 1,
                                          np.where(_dot < 0, -1, 0)).astype(np.int8)

            _above_mask       = _not_inter & (_nz_cdots > 0)
            _below_mask       = _not_inter & (_nz_cdots < 0)
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

        seam_pairs = []
        for i, v_above in enumerate(shared_list):
            v_below     = int(remap[v_above])
            spring_idx  = i
            pos_seam    = final_pos[_r(v_above)]
            travel_dist = float(np.dot(pos_seam - origin, blade_dir))
            seam_pairs.append({
                'v_above':     _r(v_above),
                'v_below':     _r(v_below),
                'spring_idx':  spring_idx,
                'travel_dist': travel_dist,
                'broken':      False,
            })
        seam_pairs.sort(key=lambda p: p['travel_dist'])
        self._seam_pairs         = seam_pairs
        self._opening_ramp_queue = []
        return True

    def advance_blade(self, blade_travel: float) -> bool:
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
