"""
Taichi Spring-Mass Simulation System

GPU-accelerated spring-mass simulator for tetrahedral meshes using explicit
Euler integration.

Controls:
  B — start / pause progressive blade cut
  F — poke the top of the mesh downward
  C — one-shot cut at the plane defined by cut_plane_origin/normal (kept for testing)

Progressive cutting (virtual-node algorithm):
  When B is pressed the mesh topology is split once along the cut plane.
  Intersection vertices on the cut plane are duplicated (one copy per half).
  Cutting springs (stiffness = k/2, rest length = 0) connect each pair of
  duplicated vertices so the mesh looks intact initially.
  Each frame, the blade cursor advances along blade_travel_dir at blade_speed
  m/s.  Cutting springs whose seam vertex lies behind the cursor are broken
  (stiffness → 0) and the wound surface is progressively revealed.

Why per-spring stiffness:
  Breaking a cutting spring only requires setting its stiffness entry to 0 in
  the _sk numpy array.  No topology rebuild is needed per frame.  The Taichi
  spring-force kernel reads _sk via an ndarray parameter that is reuploaded
  to the GPU each substep — cheap for typical spring counts (< 200 k).

Why explicit Euler:
  The mesh starts at rest, so all springs are at their natural length and net
  force is zero.  Explicit Euler is perfectly stable at zero net force.
  dt_crit = 2*sqrt(m_vertex / k_effective).  With default params the safety
  margin is ~2-3x.
"""

import taichi as ti
import numpy as np
import time

import glfw
import OpenGL.GL as gl
from pyGandalf.core.input_manager import InputManager
from pyGandalf.systems.system import System
from pyGandalf.scene.components import Component, StaticMeshComponent

ti.init(arch=ti.gpu, log_level=ti.WARN)


# ---------------------------------------------------------------------------
# Physics engine
# ---------------------------------------------------------------------------

@ti.data_oriented
class _SpringMassSimulator:
    """
    Explicit Euler spring-mass simulator with per-spring stiffness.

    Positions and velocities live in fixed-size Taichi fields (GPU resident).
    Springs are numpy arrays passed as ndarray params to the kernel so they
    can be extended or zeroed cheaply after topology / cutting events.
    """

    def __init__(self, vertices: np.ndarray, tetrahedra: np.ndarray,
                 fixed_mask: np.ndarray, stiffness: float,
                 gravity: np.ndarray,
                 total_mass: float = None,
                 per_vertex_mass: np.ndarray = None):
        N = len(vertices)
        self._stiffness = float(stiffness)
        self._gravity   = gravity.astype(np.float32)

        self.positions  = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self._forces    = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self._masses    = ti.field(dtype=ti.f32, shape=N)
        self._fixed     = ti.field(dtype=ti.i32, shape=N)

        self.positions.from_numpy(vertices.astype(np.float32))
        self.velocities.fill(0)
        self._forces.fill(0)
        if per_vertex_mass is not None:
            self._masses.from_numpy(per_vertex_mass.astype(np.float32))
        else:
            self._masses.from_numpy(np.full(N, total_mass / N, dtype=np.float32))
        self._fixed.from_numpy(fixed_mask.astype(np.int32))

        self._sa, self._sb, self._sr = _build_springs(vertices, tetrahedra)
        # Per-spring stiffness — allows cutting springs to be zeroed individually.
        self._sk = np.full(len(self._sa), self._stiffness, dtype=np.float32)

    def rebuild_springs(self, new_tetrahedra: np.ndarray):
        """Swap in a new spring network after a topology change (one-shot cut)."""
        pos = self.positions.to_numpy()
        self._sa, self._sb, self._sr = _build_springs(pos, new_tetrahedra)
        self._sk = np.full(len(self._sa), self._stiffness, dtype=np.float32)
        print(f"[Taichi] Spring network rebuilt: {len(self._sa):,} springs")

    def extend_springs(self,
                       sa_new: np.ndarray,
                       sb_new: np.ndarray,
                       sr_new: np.ndarray,
                       sk_new: np.ndarray) -> int:
        """
        Append new springs to the spring network.
        Returns the index of the first appended spring (used to track cutting
        spring positions in the array for later zeroing).
        """
        start = len(self._sa)
        self._sa = np.concatenate([self._sa, sa_new.astype(np.int32)])
        self._sb = np.concatenate([self._sb, sb_new.astype(np.int32)])
        self._sr = np.concatenate([self._sr, sr_new.astype(np.float32)])
        self._sk = np.concatenate([self._sk, sk_new.astype(np.float32)])
        return start

    def step(self, dt: float, damping: float, v_max: float = 10.0):
        self._clear_forces()
        self._spring_forces(self._sa, self._sb, self._sr, self._sk)
        self._integrate(float(dt), float(damping), self._gravity, float(v_max))

    # --- Taichi kernels ---

    @ti.kernel
    def _clear_forces(self):
        for i in self._forces:
            self._forces[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _spring_forces(self,
                       sa: ti.types.ndarray(dtype=ti.i32, ndim=1),
                       sb: ti.types.ndarray(dtype=ti.i32, ndim=1),
                       sr: ti.types.ndarray(dtype=ti.f32, ndim=1),
                       sk: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for s in range(sa.shape[0]):
            a  = sa[s]
            b  = sb[s]
            pa = self.positions[a]
            pb = self.positions[b]
            d  = pb - pa
            length = d.norm()
            if length > 1e-8:
                f = sk[s] * (length - sr[s]) / length * d
                self._forces[a] += f
                self._forces[b] -= f

    @ti.kernel
    def _integrate(self,
                   dt:       ti.f32,
                   damping:  ti.f32,
                   gravity:  ti.types.ndarray(dtype=ti.f32, ndim=1),
                   v_max:    ti.f32):
        grav = ti.Vector([gravity[0], gravity[1], gravity[2]])
        for i in self.positions:
            if self._fixed[i] == 0:
                acc = self._forces[i] / self._masses[i] + grav
                v = self.velocities[i] * (1.0 - damping * dt) + acc * dt
                # Clamp speed so vertices cannot travel far enough in one step
                # to cross over neighbours, regardless of impulse magnitude.
                speed = v.norm()
                if speed > v_max:
                    v = v * (v_max / speed)
                self.velocities[i] = v
                self.positions[i] += v * dt


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

class TaichiSimulationComponent(Component):
    """
    Stores simulation parameters and runtime state for a tetrahedral entity.

    Args:
        tet_mesh:           TetrahedralMeshInstance to simulate.
        time_step:          Integration timestep (s).
        substeps:           Sub-steps per frame for stability.
        gravity:            Gravity vector (default zero).
        stiffness:          Spring stiffness (N/m).
        damping:            Velocity damping coefficient.
        total_mass:         Total object mass (kg) distributed uniformly.
        opening_speed:      Velocity (m/s) given to wound vertices on cut.
        poke_speed:         Velocity (m/s) applied to top vertices on F.
        blade_travel_dir:   Direction the blade moves within the cut plane.
        blade_speed:        Blade advance speed (m/s).
    """

    def __init__(self, tet_mesh,
                 time_step:          float = 0.001,
                 substeps:           int   = 10,
                 gravity:            list  = None,
                 stiffness:          float = 100.0,
                 damping:            float = 1.0,
                 total_mass:         float = 1.0,
                 opening_speed:      float = 1.0,
                 poke_speed:         float = 3.0,
                 v_max:              float = 1.5,
                 blade_travel_dir:   list  = None,
                 blade_speed:        float = 0.5):
        super().__init__()
        self.tet_mesh      = tet_mesh
        self.time_step     = time_step
        self.substeps      = substeps
        self.gravity       = gravity if gravity is not None else [0.0, 0.0, 0.0]
        self.stiffness     = stiffness
        self.damping       = damping
        self.total_mass    = total_mass
        self.opening_speed = opening_speed
        self.poke_speed    = poke_speed
        self.v_max         = v_max

        # Populated by TaichiSimulationSystem.on_create_entity
        self.simulator:           _SpringMassSimulator = None
        self.surface_indices:     np.ndarray           = None
        self.current_tetrahedra:  np.ndarray           = None
        self.fixed_mask:          np.ndarray           = None
        self.poke_mask:           np.ndarray           = None

        # One-shot cut (C key) — kept for testing
        self.should_cut       = False
        self.cut_plane_origin = [0.0, 0.0, 0.0]
        self.cut_plane_normal = [0.0, 1.0, 0.0]

        # Progressive blade cut (B key)
        self.blade_travel_dir  = blade_travel_dir if blade_travel_dir is not None \
                                  else [1.0, 0.0, 0.0]
        self.blade_speed       = float(blade_speed)
        self.blade_is_active   = False
        self.blade_initialized = False
        self.blade_travel      = 0.0  # current distance traveled along blade_travel_dir

        # Internal state populated by _setup_progressive_cut
        self._seam_pairs          = []   # [{v_above,v_below,spring_idx,travel_dist,broken}]
        self._outer_faces         = None # np.ndarray (K,3) uint32 — sphere outer surface
        self._wound_faces_by_dist = []   # [(travel_dist:float, face:np.ndarray(3,))]
        self._wound_face_ptr      = 0    # pointer into _wound_faces_by_dist


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class TaichiSimulationSystem(System):
    """
    Drives a Taichi spring-mass simulation and uploads deformed vertex
    positions to the GPU each frame.

    Must be registered BEFORE OpenGLStaticMeshRenderingSystem.
    """

    def on_create_entity(self, entity, components):
        comp: TaichiSimulationComponent
        mesh_comp: StaticMeshComponent
        comp, mesh_comp = components

        tet = comp.tet_mesh

        y          = tet.vertices[:, 1]
        threshold  = y.min() + (y.max() - y.min()) * 0.05
        fixed_mask = (y < threshold).astype(np.int32)

        comp.fixed_mask         = fixed_mask
        comp.current_tetrahedra = tet.tetrahedra.copy()

        poke_threshold = y.max() - (y.max() - y.min()) * 0.05
        comp.poke_mask = (y >= poke_threshold) & (fixed_mask == 0)
        comp.surface_indices = _extract_boundary_faces(tet.tetrahedra, tet.vertices)

        comp.simulator = _SpringMassSimulator(
            vertices   = tet.vertices,
            tetrahedra = tet.tetrahedra,
            fixed_mask = fixed_mask,
            stiffness  = comp.stiffness,
            total_mass = comp.total_mass,
            gravity    = np.array(comp.gravity, dtype=np.float32),
        )

        sub_dt = comp.time_step / comp.substeps
        print("[TaichiSimulationSystem] Initialized:")
        print(f"  Vertices:        {len(tet.vertices):,}")
        print(f"  Tetrahedra:      {len(tet.tetrahedra):,}")
        print(f"  Springs:         {len(comp.simulator._sa):,}")
        print(f"  Surface faces:   {len(comp.surface_indices):,}")
        print(f"  Fixed vertices:  {int(fixed_mask.sum())}")
        print(f"  Sub-steps/frame: {comp.substeps}  (sub_dt = {sub_dt:.5f} s)")
        print("  F — poke top of mesh downward")
        print("  B — start / pause progressive blade cut")
        print("  C — one-shot cut (testing only)")

    def on_update_entity(self, ts: float, entity, components):
        comp: TaichiSimulationComponent
        mesh_comp: StaticMeshComponent
        comp, mesh_comp = components

        if comp.simulator is None:
            return
        if mesh_comp.render_pipeline is None or len(mesh_comp.buffers) < 2:
            return

        # --- C key: one-shot cut (blocked once progressive cut is initialized) ---
        c_now = InputManager().get_key_down(glfw.KEY_C)
        if c_now and not getattr(self, '_c_prev', False):
            if comp.blade_initialized:
                print("[Cut] Progressive cut already active — C-cut ignored.")
            else:
                comp.should_cut = True
        self._c_prev = c_now

        if comp.should_cut:
            comp.should_cut = False
            _perform_cut(comp, mesh_comp)

        # --- B key: progressive blade cut ---
        b_now = InputManager().get_key_down(glfw.KEY_B)
        if b_now and not getattr(self, '_b_prev', False):
            if not comp.blade_initialized:
                _setup_progressive_cut(comp, mesh_comp)
            else:
                comp.blade_is_active = not comp.blade_is_active
                print(f"[Blade] {'Resumed' if comp.blade_is_active else 'Paused'}")
        self._b_prev = b_now

        if comp.blade_is_active:
            _advance_progressive_blade(comp, mesh_comp, ts)

        # --- F key: poke ---
        f_now = InputManager().get_key_down(glfw.KEY_F)
        if f_now and not getattr(self, '_f_prev', False):
            _apply_poke(comp)
        self._f_prev = f_now

        # --- Simulation sub-steps ---
        t0 = time.perf_counter()
        sub_dt = comp.time_step / comp.substeps
        for _ in range(comp.substeps):
            comp.simulator.step(sub_dt, comp.damping, comp.v_max)
        t1 = time.perf_counter()

        new_positions = comp.simulator.positions.to_numpy()
        t2 = time.perf_counter()

        new_normals = _compute_normals(new_positions, comp.surface_indices)
        t3 = time.perf_counter()

        _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[0], new_positions)
        _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[1], new_normals)
        t4 = time.perf_counter()

        if not hasattr(self, '_fc'):
            self._fc = 0
        self._fc += 1
        if self._fc % 60 == 0:
            print(
                f"[Frame {self._fc:4d}] "
                f"sim {(t1-t0)*1000:5.2f}ms | "
                f"readback {(t2-t1)*1000:4.2f}ms | "
                f"normals {(t3-t2)*1000:4.2f}ms | "
                f"upload {(t4-t3)*1000:4.2f}ms | "
                f"total {(t4-t0)*1000:5.2f}ms"
            )


# ---------------------------------------------------------------------------
# Poke
# ---------------------------------------------------------------------------

def _apply_poke(comp: TaichiSimulationComponent):
    """Apply a downward impulse to the top 5% of vertices."""
    if comp.simulator is None:
        return
    pos = comp.simulator.positions.to_numpy()
    y = pos[:, 1]
    poke_threshold = y.max() - (y.max() - y.min()) * 0.05
    fixed = comp.simulator._fixed.to_numpy()
    poke_mask = (y >= poke_threshold) & (fixed == 0)
    vels = comp.simulator.velocities.to_numpy()
    vels[poke_mask, 1] -= comp.poke_speed
    comp.simulator.velocities.from_numpy(vels.astype(np.float32))
    print(f"[Poke] Applied {comp.poke_speed} m/s downward to {int(poke_mask.sum())} verts")


# ---------------------------------------------------------------------------
# One-shot cut  (C key — kept for testing)
# ---------------------------------------------------------------------------

def _split_crossed_tets(tetrahedra: np.ndarray,
                         positions:  np.ndarray,
                         signed_dist: np.ndarray):
    """
    Split tetrahedra that straddle the cut plane by inserting new vertices
    exactly at edge-plane intersections.

    Handles:
      1+3 (1 above, 3 below): 1 above-tet + 3-tet prism for below
      3+1 (3 above, 1 below): symmetric
      2+2 (2 above, 2 below): 3 tets on each side using quad diagonal

    Returns:
        new_positions (N_new, 3)  — original + intersection vertices
        above_tets    (M, 4)      — tets on / above the plane
        below_tets    (K, 4)      — tets on / below the plane
        inter_data    list of (new_idx, vi, vj, t)
    """
    ext_positions: list = [p for p in positions]
    edge_cache:    dict = {}
    inter_data:    list = []
    above_list:    list = []
    below_list:    list = []

    tet_dists  = signed_dist[tetrahedra]
    above_mask = np.all(tet_dists >= 0, axis=1)
    below_mask = np.all(tet_dists <= 0, axis=1)

    above_list.extend(tetrahedra[above_mask].tolist())
    below_list.extend(tetrahedra[below_mask].tolist())

    def iv(vi: int, vj: int) -> int:
        key = (min(vi, vj), max(vi, vj))
        if key not in edge_cache:
            di, dj = float(signed_dist[vi]), float(signed_dist[vj])
            t      = di / (di - dj)
            pos    = positions[vi] * (1.0 - t) + positions[vj] * t
            nid    = len(ext_positions)
            ext_positions.append(pos.astype(np.float32))
            edge_cache[key] = (nid, vi, vj, t)
            inter_data.append((nid, vi, vj, t))
        return edge_cache[key][0]

    for idx in np.where(~above_mask & ~below_mask)[0]:
        verts = tetrahedra[idx]
        dists = signed_dist[verts]

        av  = [int(verts[i]) for i in range(4) if dists[i] >= 0]
        bv  = [int(verts[i]) for i in range(4) if dists[i] <  0]
        n_a = len(av)

        if n_a == 1:
            a, b0, b1, b2 = av[0], bv[0], bv[1], bv[2]
            p0, p1, p2    = iv(a, b0), iv(a, b1), iv(a, b2)
            above_list.append([a, p0, p1, p2])
            below_list += [[b0, b1, b2, p0],
                           [b1, b2, p0, p1],
                           [b2, p0, p1, p2]]

        elif n_a == 3:
            a0, a1, a2, b = av[0], av[1], av[2], bv[0]
            p0, p1, p2    = iv(a0, b), iv(a1, b), iv(a2, b)
            below_list.append([b, p0, p1, p2])
            above_list += [[a0, a1, a2, p0],
                           [a1, a2, p0, p1],
                           [a2, p0, p1, p2]]

        else:   # 2+2
            a0, a1 = av[0], av[1]
            b0, b1 = bv[0], bv[1]
            p00, p01 = iv(a0, b0), iv(a0, b1)
            p10, p11 = iv(a1, b0), iv(a1, b1)
            above_list += [[a0, p00, p01, p11],
                           [a1, p00, p10, p11],
                           [a0, a1,  p00, p11]]
            below_list += [[b1, p00, p01, p11],
                           [b0, p00, p10, p11],
                           [b0, b1,  p00, p11]]

    new_pos   = np.array(ext_positions, dtype=np.float32)
    above_arr = (np.array(above_list, dtype=np.int32)
                 if above_list else np.zeros((0, 4), dtype=np.int32))
    below_arr = (np.array(below_list, dtype=np.int32)
                 if below_list else np.zeros((0, 4), dtype=np.int32))
    return new_pos, above_arr, below_arr, inter_data


def _cut_topology(comp: TaichiSimulationComponent,
                  origin: np.ndarray,
                  normal: np.ndarray):
    """
    Shared topology-change logic used by both the one-shot cut and the
    progressive cut setup.

    Splits crossed tets, snaps rim vertices, extends vel/mass/fixed arrays,
    duplicates seam vertices, and rebuilds the simulator.

    Returns:
        final_pos      (N_final, 3)
        final_vel      (N_final, 3)
        final_mass     (N_final,)
        final_fixed    (N_final,)
        all_tets       (T_final, 4)
        n_orig         int   — number of vertices before split
        n_split        int   — n_orig + intersection verts (before duplication)
        shared_list    list  — intersection vertex indices (above-half)
        remap          (n_split,) int32  — maps above indices to below dup indices
        inter_data     list of (new_idx, vi, vj, t)
        orig_surf_set  set of sorted (v0,v1,v2) tuples for original outer faces
    """
    current_tets = comp.current_tetrahedra
    positions    = comp.simulator.positions.to_numpy()
    velocities   = comp.simulator.velocities.to_numpy()
    masses       = comp.simulator._masses.to_numpy()
    fixed        = comp.simulator._fixed.to_numpy()

    signed_dist  = (positions - origin) @ normal

    # --- Original outer surface set (to detect interior-exposed faces later) ---
    face_combos = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
    all_orig_f  = current_tets[:, face_combos].reshape(-1, 3)
    all_orig_s  = np.sort(all_orig_f, axis=1)
    _, _inv, _cnt = np.unique(all_orig_s, axis=0,
                               return_inverse=True, return_counts=True)
    orig_surf_set = {tuple(row) for row in all_orig_s[(_cnt == 1)[_inv]]}
    n_orig = len(positions)

    # --- Tet splitting ---
    split_pos, above_tets, below_tets, inter_data = \
        _split_crossed_tets(current_tets, positions, signed_dist)

    n_split = len(split_pos)
    n_inter = n_split - n_orig

    print(f"[Cut] {len(above_tets):,} above-tets, {len(below_tets):,} below-tets, "
          f"{n_inter} intersection verts")

    if len(above_tets) == 0 or len(below_tets) == 0:
        print("[Cut] Plane doesn't divide mesh — aborting.")
        return None

    # --- Snap rim vertices to cut plane ---
    rim_set = set()
    for new_id, vi, vj, _ in inter_data:
        if vi < n_orig: rim_set.add(vi)
        if vj < n_orig: rim_set.add(vj)
    if rim_set:
        rim_arr = np.array(sorted(rim_set), dtype=np.int32)
        snap_d  = signed_dist[rim_arr]
        split_pos[rim_arr] -= snap_d[:, np.newaxis] * normal
        print(f"[Cut] Snapped {len(rim_arr)} rim vertices to cut plane")

    # --- Extend per-vertex arrays ---
    split_vel   = np.zeros((n_split, 3), dtype=np.float32)
    split_mass  = np.zeros(n_split,      dtype=np.float32)
    split_fixed = np.zeros(n_split,      dtype=np.int32)
    split_vel[:n_orig]   = velocities
    split_mass[:n_orig]  = masses
    split_fixed[:n_orig] = fixed
    for new_id, vi, vj, t in inter_data:
        split_vel[new_id]   = velocities[vi] * (1-t) + velocities[vj] * t
        split_mass[new_id]  = masses[vi]     * (1-t) + masses[vj]     * t
        split_fixed[new_id] = 0

    # --- Vertex duplication (seam) ---
    above_vert_set = set(above_tets.flatten().tolist())
    below_vert_set = set(below_tets.flatten().tolist())
    shared_list    = sorted(above_vert_set & below_vert_set)
    n_shared       = len(shared_list)
    print(f"[Cut] Duplicating {n_shared} seam vertices")

    remap = np.arange(n_split, dtype=np.int32)
    for i, v in enumerate(shared_list):
        remap[v] = n_split + i

    shared_arr  = np.array(shared_list, dtype=np.int32)
    final_pos   = np.vstack([split_pos, split_pos[shared_arr]])
    final_vel   = np.vstack([split_vel, split_vel[shared_arr]])
    final_mass  = np.concatenate([split_mass,  split_mass[shared_arr]])
    final_fixed = np.concatenate([split_fixed, split_fixed[shared_arr]])

    new_below_tets = remap[below_tets]
    all_tets       = np.vstack([above_tets, new_below_tets])

    comp.current_tetrahedra = all_tets
    comp.fixed_mask         = final_fixed

    # --- Rebuild simulator ---
    comp.simulator = _SpringMassSimulator(
        vertices        = final_pos,
        tetrahedra      = all_tets,
        fixed_mask      = final_fixed,
        stiffness       = comp.stiffness,
        gravity         = np.array(comp.gravity, dtype=np.float32),
        per_vertex_mass = final_mass,
    )
    comp.simulator.velocities.from_numpy(final_vel.astype(np.float32))
    print(f"[Cut] Simulator: {len(final_pos):,} verts, "
          f"{len(all_tets):,} tets, {len(comp.simulator._sa):,} springs")

    return (final_pos, final_vel, final_mass, final_fixed,
            all_tets, n_orig, n_split, shared_list, remap, inter_data,
            orig_surf_set)


def _filter_surface_faces(raw_surface, final_pos, normal, n_orig, n_split,
                           orig_surf_set, mesh_centroid=None):
    """
    Partition raw boundary faces into outer faces and wound faces, applying
    correct winding to each.

    Faces with all-original vertices not in orig_surf_set are dropped — they
    are interior tet faces exposed by the split.

    Winding rules:
      outer (all_orig): centroid test — normal must point away from mesh_centroid.
        Needed because split tets have intersection vertices as their 4th vertex,
        which can fool the opposite-vertex test in _extract_boundary_faces.
      wound/collar (any non-orig vertex): cut-normal test.
        above-half (no vertex >= n_split) → dot(fn, normal) < 0
        below-half (any vertex >= n_split) → dot(fn, normal) > 0
    """
    outer = []
    wound = []

    for f in raw_surface:
        v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
        all_orig = v0 < n_orig and v1 < n_orig and v2 < n_orig

        if all_orig:
            if tuple(sorted((v0, v1, v2))) in orig_surf_set:
                if mesh_centroid is not None:
                    p0 = final_pos[v0]; p1 = final_pos[v1]; p2 = final_pos[v2]
                    fn = np.cross(p1 - p0, p2 - p0)
                    to_face = (p0 + p1 + p2) / 3.0 - mesh_centroid
                    if float(np.dot(fn, to_face)) < 0:
                        f = np.array([v0, v2, v1], dtype=np.uint32)
                outer.append(f)
            # else: interior tet face exposed by split — drop
        else:
            # Wound face or collar face — enforce winding via cut normal.
            # Collar faces (original + intersection/dup vertices) also use this
            # rule; centroid winding is wrong for faces near the cut plane.
            is_below = v0 >= n_split or v1 >= n_split or v2 >= n_split
            p0 = final_pos[v0]; p1 = final_pos[v1]; p2 = final_pos[v2]
            fn  = np.cross(p1 - p0, p2 - p0)
            dot = float(np.dot(fn, normal))
            if abs(dot) < 1e-12:
                continue
            if is_below:
                face = f if dot > 0 else np.array([v0, v2, v1], dtype=np.uint32)
            else:
                face = f if dot < 0 else np.array([v0, v2, v1], dtype=np.uint32)
            wound.append(face)

    outer_arr = (np.array(outer, dtype=np.uint32)
                 if outer else np.zeros((0, 3), dtype=np.uint32))
    return outer_arr, wound


def _perform_cut(comp: TaichiSimulationComponent,
                 mesh_comp: StaticMeshComponent):
    """
    One-shot surgical cut along cut_plane_origin / cut_plane_normal  (C key).
    Kept for testing and reference.
    """
    if comp.current_tetrahedra is None or len(comp.current_tetrahedra) == 0:
        print("[Cut] No tetrahedra.")
        return

    origin = np.array(comp.cut_plane_origin, dtype=np.float32)
    normal = np.array(comp.cut_plane_normal,  dtype=np.float32)
    normal /= np.linalg.norm(normal)

    result = _cut_topology(comp, origin, normal)
    if result is None:
        return

    (final_pos, final_vel, final_mass, final_fixed,
     all_tets, n_orig, n_split, shared_list, remap, inter_data,
     orig_surf_set) = result

    # --- Opening velocity ---
    if comp.opening_speed > 0.0:
        inter_indices = np.array([d[0] for d in inter_data], dtype=np.int32)
        dup_indices   = np.array([remap[v] for v in inter_indices
                                  if remap[v] >= n_split], dtype=np.int32)
        vels = comp.simulator.velocities.to_numpy()
        if len(inter_indices) > 0:
            vels[inter_indices] += comp.opening_speed * normal
        if len(dup_indices) > 0:
            vels[dup_indices]   -= comp.opening_speed * normal
        comp.simulator.velocities.from_numpy(vels.astype(np.float32))
        print(f"[Cut] Opening velocity on {len(inter_indices)} + {len(dup_indices)} verts")

    # --- Surface ---
    raw_surface = _extract_boundary_faces(all_tets, final_pos)
    mesh_centroid = final_pos[:n_orig].mean(axis=0)
    outer_faces, wound_faces = _filter_surface_faces(
        raw_surface, final_pos, normal, n_orig, n_split, orig_surf_set, mesh_centroid)

    all_faces = (np.vstack([outer_faces,
                             np.array(wound_faces, dtype=np.uint32)])
                 if wound_faces else outer_faces)
    comp.surface_indices = all_faces
    new_normals  = _compute_normals(final_pos, all_faces)
    new_texcoords = np.zeros((len(final_pos), 2), dtype=np.float32)
    print(f"[Cut] Surface: {len(raw_surface):,} raw → {len(all_faces):,} kept")

    _realloc_gpu_buffers(mesh_comp, final_pos, new_normals, new_texcoords, all_faces)
    print("[Cut] GPU buffers updated.")


# ---------------------------------------------------------------------------
# Progressive blade cut  (B key)
# ---------------------------------------------------------------------------

def _setup_progressive_cut(comp: TaichiSimulationComponent,
                             mesh_comp: StaticMeshComponent):
    """
    One-time setup for progressive blade cutting.

    Splits the mesh topology along the cut plane (same tet-splitting logic as
    the one-shot cut), then adds cutting springs between every seam node pair.
    Cutting springs start at stiffness k/2 and rest length 0, so they hold
    the two halves together until the blade cursor passes each pair.

    The wound surface is sorted by blade travel distance so it can be revealed
    face-by-face as the blade advances.
    """
    if comp.current_tetrahedra is None or len(comp.current_tetrahedra) == 0:
        print("[Blade] No tetrahedra.")
        return

    origin = np.array(comp.cut_plane_origin, dtype=np.float32)
    normal = np.array(comp.cut_plane_normal,  dtype=np.float32)
    normal /= np.linalg.norm(normal)

    blade_dir = np.array(comp.blade_travel_dir, dtype=np.float32)
    blade_dir /= np.linalg.norm(blade_dir)

    result = _cut_topology(comp, origin, normal)
    if result is None:
        return

    (final_pos, _final_vel, _final_mass, _final_fixed,
     all_tets, n_orig, n_split, shared_list, remap, inter_data,
     orig_surf_set) = result

    # --- Add cutting springs between seam pairs ---
    # Each seam pair (v_above, v_below) gets one cutting spring.
    # The spring holds the two halves together until the blade passes it.
    n_structural = len(comp.simulator._sa)

    sa_cut = np.array([v      for v in shared_list], dtype=np.int32)
    sb_cut = np.array([remap[v] for v in shared_list], dtype=np.int32)
    sr_cut = np.zeros(len(shared_list), dtype=np.float32)   # rest length = 0
    sk_cut = np.full(len(shared_list), comp.stiffness * 0.5, dtype=np.float32)

    comp.simulator.extend_springs(sa_cut, sb_cut, sr_cut, sk_cut)

    # Build seam-pair manifest with travel distances for ordered spring breaking.
    seam_pairs = []
    for i, v_above in enumerate(shared_list):
        v_below     = int(remap[v_above])
        spring_idx  = n_structural + i
        pos_seam    = final_pos[v_above]
        travel_dist = float(np.dot(pos_seam - origin, blade_dir))
        seam_pairs.append({
            'v_above':    v_above,
            'v_below':    v_below,
            'spring_idx': spring_idx,
            'travel_dist': travel_dist,
            'broken':     False,
        })
    seam_pairs.sort(key=lambda p: p['travel_dist'])
    comp._seam_pairs = seam_pairs

    # Blade cursor starts just before the first seam pair.
    if seam_pairs:
        comp.blade_travel = seam_pairs[0]['travel_dist'] - 1e-3
    else:
        comp.blade_travel = 0.0

    # --- Build progressive wound surface ---
    raw_surface = _extract_boundary_faces(all_tets, final_pos)
    mesh_centroid = final_pos[:n_orig].mean(axis=0)
    outer_faces, wound_faces = _filter_surface_faces(
        raw_surface, final_pos, normal, n_orig, n_split, orig_surf_set, mesh_centroid)

    comp._outer_faces = outer_faces

    # Sort wound faces by their centroid's projection onto blade_dir.
    wound_by_dist = []
    for wf in wound_faces:
        centroid    = final_pos[[int(wf[0]), int(wf[1]), int(wf[2])]].mean(axis=0)
        travel_dist = float(np.dot(centroid - origin, blade_dir))
        wound_by_dist.append((travel_dist, wf))
    wound_by_dist.sort(key=lambda x: x[0])
    comp._wound_faces_by_dist = wound_by_dist
    comp._wound_face_ptr      = 0

    # Initial surface = outer faces only (wound hidden until blade reaches it).
    comp.surface_indices = outer_faces.copy()
    new_normals   = _compute_normals(final_pos, comp.surface_indices)
    new_texcoords = np.zeros((len(final_pos), 2), dtype=np.float32)
    print(f"[Blade] Surface: {len(outer_faces):,} outer + "
          f"{len(wound_faces):,} wound faces (hidden until blade passes)")

    _realloc_gpu_buffers(mesh_comp, final_pos, new_normals, new_texcoords,
                         comp.surface_indices)

    comp.blade_initialized = True
    comp.blade_is_active   = True
    print(f"[Blade] Setup complete — {len(seam_pairs):,} seam pairs, "
          f"blade starts at travel={comp.blade_travel:.4f}")
    print("[Blade] Press B to pause / resume.")


def _advance_progressive_blade(comp: TaichiSimulationComponent,
                                 mesh_comp: StaticMeshComponent,
                                 dt: float):
    """
    Advance the blade cursor one frame.

    1. Move cursor by blade_speed * dt along blade_travel_dir.
    2. Break all cutting springs whose seam vertex is now behind the cursor.
       Apply opening velocity to each newly separated pair.
    3. Reveal wound surface faces whose centroid is now behind the cursor.
       When new faces are added, reallocate the index buffer on the GPU.
    4. Stop the blade once all springs are broken.
    """
    normal    = np.array(comp.cut_plane_normal, dtype=np.float32)
    normal   /= np.linalg.norm(normal)

    comp.blade_travel += comp.blade_speed * float(dt)

    # --- Break springs behind the cursor ---
    newly_broken = [p for p in comp._seam_pairs
                    if not p['broken'] and p['travel_dist'] <= comp.blade_travel]

    if newly_broken:
        # Batch the velocity round-trip.
        vels = comp.simulator.velocities.to_numpy()
        for p in newly_broken:
            comp.simulator._sk[p['spring_idx']] = 0.0
            vels[p['v_above']] += comp.opening_speed * normal
            vels[p['v_below']] -= comp.opening_speed * normal
            p['broken'] = True
        comp.simulator.velocities.from_numpy(vels.astype(np.float32))

    # --- Reveal wound faces behind the cursor ---
    ptr = comp._wound_face_ptr
    wbd = comp._wound_faces_by_dist
    while ptr < len(wbd) and wbd[ptr][0] <= comp.blade_travel:
        ptr += 1

    faces_added = ptr > comp._wound_face_ptr
    if faces_added:
        comp._wound_face_ptr = ptr
        revealed = [f for _, f in wbd[:ptr]]
        comp.surface_indices = (
            np.vstack([comp._outer_faces,
                       np.array(revealed, dtype=np.uint32)])
            if revealed else comp._outer_faces.copy()
        )
        mesh_comp.indices = comp.surface_indices

        # Realloc index buffer (face count changed).
        flat_idx = comp.surface_indices.flatten().astype(np.uint32)
        gl.glBindVertexArray(mesh_comp.render_pipeline)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh_comp.index_buffer)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, flat_idx.nbytes, flat_idx,
                        gl.GL_DYNAMIC_DRAW)
        gl.glBindVertexArray(0)

    # --- Stop when complete ---
    if all(p['broken'] for p in comp._seam_pairs):
        # Force-reveal any remaining wound faces (floating-point edge cases).
        if comp._wound_face_ptr < len(wbd):
            comp._wound_face_ptr = len(wbd)
            revealed = [f for _, f in wbd]
            comp.surface_indices = (
                np.vstack([comp._outer_faces,
                           np.array(revealed, dtype=np.uint32)])
                if revealed else comp._outer_faces.copy()
            )
            mesh_comp.indices = comp.surface_indices
            flat_idx = comp.surface_indices.flatten().astype(np.uint32)
            gl.glBindVertexArray(mesh_comp.render_pipeline)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh_comp.index_buffer)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, flat_idx.nbytes, flat_idx,
                            gl.GL_DYNAMIC_DRAW)
            gl.glBindVertexArray(0)

        comp.blade_is_active = False
        n_broken = sum(1 for p in comp._seam_pairs if p['broken'])
        print(f"[Blade] Cut complete — {n_broken:,} seam springs broken.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _realloc_gpu_buffers(mesh_comp: StaticMeshComponent,
                          positions:  np.ndarray,
                          normals:    np.ndarray,
                          texcoords:  np.ndarray,
                          indices:    np.ndarray):
    """Reallocate all GPU buffers with new data (called after topology changes)."""
    flat_pos  = positions.flatten().astype(np.float32)
    flat_norm = normals.flatten().astype(np.float32)
    flat_tex  = texcoords.flatten().astype(np.float32)
    flat_idx  = indices.flatten().astype(np.uint32)

    gl.glBindVertexArray(mesh_comp.render_pipeline)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[0])
    gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_pos.nbytes,  flat_pos,  gl.GL_DYNAMIC_DRAW)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[1])
    gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_norm.nbytes, flat_norm, gl.GL_DYNAMIC_DRAW)

    if len(mesh_comp.buffers) > 2:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[2])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_tex.nbytes, flat_tex, gl.GL_DYNAMIC_DRAW)

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh_comp.index_buffer)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, flat_idx.nbytes, flat_idx,
                    gl.GL_DYNAMIC_DRAW)

    gl.glBindVertexArray(0)
    mesh_comp.indices = indices


def _build_springs(vertices: np.ndarray, tetrahedra: np.ndarray) -> tuple:
    """Extract unique edges and compute rest lengths."""
    edge_pairs = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    all_edges  = tetrahedra[:, edge_pairs].reshape(-1, 2)
    all_edges  = np.sort(all_edges, axis=1)
    unique_edges = np.unique(all_edges, axis=0)

    sa   = unique_edges[:, 0].astype(np.int32)
    sb   = unique_edges[:, 1].astype(np.int32)
    rest = np.linalg.norm(
        vertices[sb] - vertices[sa], axis=1
    ).astype(np.float32)
    return sa, sb, rest


def _extract_boundary_faces(tetrahedra: np.ndarray,
                             vertices:   np.ndarray = None) -> np.ndarray:
    """
    Return boundary triangles of a tetrahedral mesh (faces belonging to
    exactly one tetrahedron).  When vertices are supplied, winding is
    corrected so normals point outward.
    """
    face_combos      = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
    all_faces        = tetrahedra[:, face_combos].reshape(-1, 3)
    all_faces_sorted = np.sort(all_faces, axis=1)

    _, inverse, counts = np.unique(
        all_faces_sorted, axis=0, return_inverse=True, return_counts=True)

    boundary_mask  = (counts == 1)[inverse]
    boundary_faces = all_faces[boundary_mask].copy()

    if vertices is not None and len(boundary_faces) > 0:
        boundary_pos      = np.where(boundary_mask)[0]
        tet_indices       = boundary_pos // 4
        face_verts        = all_faces[boundary_pos]
        tet_verts         = tetrahedra[tet_indices]

        in_face = (tet_verts[:, :, np.newaxis] ==
                   face_verts[:, np.newaxis, :]).any(axis=2)
        fourth_vertex_idx = tet_verts[~in_face].reshape(-1)

        v0     = vertices[boundary_faces[:, 0]]
        v1     = vertices[boundary_faces[:, 1]]
        v2     = vertices[boundary_faces[:, 2]]
        fourth = vertices[fourth_vertex_idx]

        face_normals = np.cross(v1 - v0, v2 - v0)
        to_fourth    = fourth - (v0 + v1 + v2) / 3.0
        inward       = np.einsum('ij,ij->i', face_normals, to_fourth) > 0
        boundary_faces[inward] = boundary_faces[inward][:, [0, 2, 1]]

    return boundary_faces.astype(np.uint32)


@ti.kernel
def _accumulate_normals_kernel(vertices: ti.types.ndarray(dtype=ti.f32, ndim=2),
                                indices:  ti.types.ndarray(dtype=ti.i32, ndim=2),
                                normals:  ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for tri in range(indices.shape[0]):
        v0 = indices[tri, 0]
        v1 = indices[tri, 1]
        v2 = indices[tri, 2]
        p0 = ti.Vector([vertices[v0, 0], vertices[v0, 1], vertices[v0, 2]])
        p1 = ti.Vector([vertices[v1, 0], vertices[v1, 1], vertices[v1, 2]])
        p2 = ti.Vector([vertices[v2, 0], vertices[v2, 1], vertices[v2, 2]])
        n = (p1 - p0).cross(p2 - p0)
        for k in ti.static(range(3)):
            ti.atomic_add(normals[v0, k], n[k])
            ti.atomic_add(normals[v1, k], n[k])
            ti.atomic_add(normals[v2, k], n[k])


@ti.kernel
def _normalize_normals_kernel(normals: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i in range(normals.shape[0]):
        n = ti.Vector([normals[i, 0], normals[i, 1], normals[i, 2]])
        length = n.norm()
        if length > 1e-6:
            n = n / length
        normals[i, 0] = n[0]
        normals[i, 1] = n[1]
        normals[i, 2] = n[2]


def _compute_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals on the GPU via Taichi."""
    normals = np.zeros_like(vertices, dtype=np.float32)
    _accumulate_normals_kernel(vertices, indices.astype(np.int32), normals)
    _normalize_normals_kernel(normals)
    return normals


def _update_vbo(vao: int, vbo: int, data: np.ndarray):
    """Upload new data into an existing VBO (no reallocation)."""
    flat = data.flatten().astype(np.float32)
    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, flat.nbytes, flat)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    gl.glBindVertexArray(0)
