"""
Headless spring-mass simulation — no OpenGL, no ECS, no window.

Used to generate training trajectories for the GNN surrogate model.
All physics logic is reused directly from taichi_simulation_system.py;
this file is purely a thin driver that strips the rendering layer out.

Typical usage:
    sim = HeadlessSim(tet_mesh, cut_normal, cut_origin, blade_travel_dir)
    data = sim.run_and_record()
    # data['positions']  shape (T, N, 3)
    # data['velocities'] shape (T, N, 3)
    # ...
"""

import numpy as np

from pyGandalf.thesis_utilities.taichi_simulation_system import (
    TaichiSimulationComponent,
    _SpringMassSimulator,
    _cut_topology,
    _extract_boundary_faces,
)


class HeadlessSim:
    """
    Runs the spring-mass cutting simulation without any rendering.

    Parameters mirror TaichiSimulationComponent exactly so that headless
    runs reproduce the same physics as the interactive viewer.
    """

    def __init__(self,
                 tet_mesh,
                 cut_normal:          np.ndarray,
                 cut_origin:          np.ndarray,
                 blade_travel_dir:    np.ndarray,
                 stiffness:           float = 200.0,
                 damping:             float = 3.5,
                 total_mass:          float = 100.0,
                 time_step:           float = 0.005,
                 substeps:            int   = 4,
                 gravity:             list  = None,
                 opening_speed:       float = 4.0,
                 blade_speed:         float = 0.5,
                 v_max:               float = 4.0,
                 spring_damping:      float = 2.0,
                 opening_ramp_frames: int   = 20):

        self.comp = TaichiSimulationComponent(
            tet_mesh            = tet_mesh,
            time_step           = time_step,
            substeps            = substeps,
            stiffness           = stiffness,
            damping             = damping,
            total_mass          = total_mass,
            gravity             = gravity if gravity is not None else [0.0, 0.0, 0.0],
            opening_speed       = opening_speed,
            v_max               = v_max,
            spring_damping      = spring_damping,
            blade_travel_dir    = list(blade_travel_dir),
            blade_speed         = blade_speed,
            opening_ramp_frames = opening_ramp_frames,
        )
        self.comp.cut_plane_origin = list(cut_origin)
        self.comp.cut_plane_normal = list(cut_normal)

        self._init_simulator()

    # ------------------------------------------------------------------
    # Initialisation (replicates TaichiSimulationSystem.on_create_entity)
    # ------------------------------------------------------------------

    def _init_simulator(self):
        comp = self.comp
        tet  = comp.tet_mesh

        y             = tet.vertices[:, 1]
        threshold     = y.min() + (y.max() - y.min()) * 0.05
        fixed_mask    = (y < threshold).astype(np.int32)

        comp.fixed_mask         = fixed_mask
        comp.current_tetrahedra = tet.tetrahedra.copy()

        comp.simulator = _SpringMassSimulator(
            vertices   = tet.vertices,
            tetrahedra = tet.tetrahedra,
            fixed_mask = fixed_mask,
            stiffness  = comp.stiffness,
            total_mass = comp.total_mass,
            gravity    = np.array(comp.gravity, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Cut setup (physics-only version of _setup_progressive_cut)
    # ------------------------------------------------------------------

    def setup_cut(self) -> bool:
        """
        Split the mesh, add cutting springs, build seam-pair manifest.
        Returns False if the cut plane misses the mesh.
        """
        comp      = self.comp
        origin    = np.array(comp.cut_plane_origin, dtype=np.float32)
        normal    = np.array(comp.cut_plane_normal,  dtype=np.float32)
        normal   /= np.linalg.norm(normal)
        blade_dir = np.array(comp.blade_travel_dir,  dtype=np.float32)
        blade_dir /= np.linalg.norm(blade_dir)

        result = _cut_topology(comp, origin, normal)
        if result is None:
            return False

        (final_pos, _vel, _mass, _fixed,
         all_tets, n_orig, n_split, shared_list, remap, inter_data,
         _orig_surf_set) = result

        comp._n_orig      = n_orig
        comp._n_split     = n_split
        comp._inter_data  = inter_data
        comp._shared_list = shared_list
        comp._cut_normal  = normal

        # --- Cutting springs (one per seam pair, rest length = 0, k = k/2) ---
        n_structural = len(comp.simulator._sa)

        sa_cut = np.array([v        for v in shared_list], dtype=np.int32)
        sb_cut = np.array([remap[v] for v in shared_list], dtype=np.int32)
        sr_cut = np.zeros(len(shared_list),                dtype=np.float32)
        sk_cut = np.full(len(shared_list), comp.stiffness * 0.5, dtype=np.float32)
        comp.simulator.extend_springs(sa_cut, sb_cut, sr_cut, sk_cut)

        # --- Seam-pair manifest (sorted by travel distance) ---
        seam_pairs = []
        for i, v_above in enumerate(shared_list):
            v_below     = int(remap[v_above])
            spring_idx  = n_structural + i
            travel_dist = float(np.dot(final_pos[v_above] - origin, blade_dir))
            seam_pairs.append({
                'v_above':     v_above,
                'v_below':     v_below,
                'spring_idx':  spring_idx,
                'travel_dist': travel_dist,
                'broken':      False,
            })
        seam_pairs.sort(key=lambda p: p['travel_dist'])
        comp._seam_pairs  = seam_pairs
        comp.blade_travel = seam_pairs[0]['travel_dist'] - 1e-3 if seam_pairs else 0.0

        comp.blade_initialized = True
        comp.blade_is_active   = True

        # --- Capture post-cut graph for dataset saving ---
        self._n_structural    = n_structural
        self._final_pos_cut   = final_pos
        return True

    # ------------------------------------------------------------------
    # Per-frame update (physics only)
    # ------------------------------------------------------------------

    def _advance_blade(self, dt: float) -> bool:
        """Advance blade cursor, break springs, queue opening ramp. Returns True when done."""
        comp   = self.comp
        normal = np.array(comp.cut_plane_normal, dtype=np.float32)
        normal /= np.linalg.norm(normal)

        comp.blade_travel += comp.blade_speed * float(dt)

        newly_broken = [p for p in comp._seam_pairs
                        if not p['broken'] and p['travel_dist'] <= comp.blade_travel]
        if newly_broken:
            for p in newly_broken:
                comp.simulator._sk[p['spring_idx']] = 0.0
                p['broken'] = True

            step_speed = comp.opening_speed / comp.opening_ramp_frames
            comp._opening_ramp_queue.append({
                'above':      np.array([p['v_above'] for p in newly_broken], dtype=np.int32),
                'below':      np.array([p['v_below'] for p in newly_broken], dtype=np.int32),
                'step_speed': step_speed,
                'left':       comp.opening_ramp_frames,
            })

        return all(p['broken'] for p in comp._seam_pairs)

    def _process_opening_ramp(self):
        comp   = self.comp
        if not comp._opening_ramp_queue:
            return
        vels   = comp.simulator.velocities.to_numpy()
        normal = np.array(comp._cut_normal, dtype=np.float32)
        still_active = []
        for entry in comp._opening_ramp_queue:
            vels[entry['above']] += entry['step_speed'] * normal
            vels[entry['below']] -= entry['step_speed'] * normal
            entry['left'] -= 1
            if entry['left'] > 0:
                still_active.append(entry)
        comp.simulator.velocities.from_numpy(vels.astype(np.float32))
        comp._opening_ramp_queue = still_active

    def _physics_step(self):
        self._process_opening_ramp()
        comp   = self.comp
        sub_dt = comp.time_step / comp.substeps
        for _ in range(comp.substeps):
            comp.simulator.step(sub_dt, comp.damping, comp.spring_damping, comp.v_max)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run_and_record(self,
                       settle_steps:  int = 200,
                       record_every:  int = 1) -> dict | None:
        """
        Run the full simulation (cut → blade advance → settling) and return
        trajectory data ready to save as a dataset entry.

        Args:
            settle_steps:  extra physics frames to run after the blade finishes.
            record_every:  record state every N frames (1 = every frame).

        Returns a dict with keys:
            cut_normal       (3,)     float32
            cut_origin       (3,)     float32
            n_orig           scalar   int      original vertex count
            n_phys           scalar   int      vertex count after cut
            n_structural     scalar   int      structural springs (before cutting springs)
            edges            (E, 2)   int32    all springs [structural | cutting]
            rest_lengths     (E,)     float32
            stiffnesses_max  (E,)     float32  initial stiffness per spring
            spring_break_at  (E,)     float32  blade_progress when broken (inf = never)
            masses           (N,)     float32
            fixed            (N,)     int32
            positions        (T, N, 3) float32
            velocities       (T, N, 3) float32
            blade_progress   (T,)     float32
        """
        if not self.setup_cut():
            print("[HeadlessSim] Cut missed mesh — skipping.")
            return None

        comp = self.comp

        # --- Snapshot graph structure (fixed for the whole trajectory) ---
        sa = np.array(comp.simulator._sa, dtype=np.int32)
        sb = np.array(comp.simulator._sb, dtype=np.int32)
        sr = np.array(comp.simulator._sr, dtype=np.float32)
        sk = np.array(comp.simulator._sk, dtype=np.float32)  # initial stiffnesses

        n_springs = len(sa)
        spring_break_at = np.full(n_springs, np.inf, dtype=np.float32)
        for p in comp._seam_pairs:
            spring_break_at[p['spring_idx']] = p['travel_dist']

        masses = comp.simulator._masses.to_numpy().astype(np.float32)
        fixed  = comp.simulator._fixed.to_numpy().astype(np.int32)

        # --- Trajectory recording ---
        positions_list      = []
        velocities_list     = []
        blade_progress_list = []
        frame_dt            = comp.time_step

        def _record():
            pos = comp.simulator.positions.to_numpy().astype(np.float32)
            vel = comp.simulator.velocities.to_numpy().astype(np.float32)
            positions_list.append(pos.copy())
            velocities_list.append(vel.copy())
            blade_progress_list.append(float(comp.blade_travel))

        # Phase 1: blade advancing
        frame      = 0
        blade_done = False
        while not blade_done:
            blade_done = self._advance_blade(frame_dt)
            self._physics_step()
            if frame % record_every == 0:
                _record()
            frame += 1

        # Phase 2: settling
        for i in range(settle_steps):
            self._physics_step()
            if i % record_every == 0:
                _record()

        n_phys = comp.simulator.positions.shape[0]

        return {
            'cut_normal':      np.array(comp.cut_plane_normal, dtype=np.float32),
            'cut_origin':      np.array(comp.cut_plane_origin, dtype=np.float32),
            'n_orig':          np.int32(comp._n_orig),
            'n_phys':          np.int32(n_phys),
            'n_structural':    np.int32(self._n_structural),
            'edges':           np.stack([sa, sb], axis=1),
            'rest_lengths':    sr,
            'stiffnesses_max': sk,
            'spring_break_at': spring_break_at,
            'masses':          masses,
            'fixed':           fixed,
            'positions':       np.stack(positions_list,  axis=0),
            'velocities':      np.stack(velocities_list, axis=0),
            'blade_progress':  np.array(blade_progress_list, dtype=np.float32),
        }
