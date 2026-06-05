"""
Taichi Spring-Mass Simulation System

GPU-accelerated spring-mass simulator for tetrahedral meshes using explicit
Euler integration.

Controls:
  B — start / pause progressive blade cut
  F — poke the top of the mesh downward
  C — one-shot cut at the plane defined by cut_plane_origin/normal (kept for testing)
  P — pause / resume physics simulation (cut still runs)

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

    def step(self, dt: float, damping: float, spring_damp: float = 0.0, v_max: float = 10.0):
        self._clear_forces()
        self._spring_forces(self._sa, self._sb, self._sr, self._sk, float(spring_damp))
        self._integrate(float(dt), float(damping), self._gravity, float(v_max))

    # --- Taichi kernels ---

    @ti.kernel
    def _add_vel_scalar(self,
                        indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
                        dx: ti.f32, dy: ti.f32, dz: ti.f32):
        """Add (dx,dy,dz) to velocity[i] for each i in indices — no CPU sync needed."""
        dv = ti.Vector([dx, dy, dz])
        for k in range(indices.shape[0]):
            j = indices[k]
            self.velocities[j] += dv

    @ti.kernel
    def _clear_forces(self):
        for i in self._forces:
            self._forces[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _spring_forces(self,
                       sa:          ti.types.ndarray(dtype=ti.i32, ndim=1),
                       sb:          ti.types.ndarray(dtype=ti.i32, ndim=1),
                       sr:          ti.types.ndarray(dtype=ti.f32, ndim=1),
                       sk:          ti.types.ndarray(dtype=ti.f32, ndim=1),
                       spring_damp: ti.f32):
        for s in range(sa.shape[0]):
            a  = sa[s]
            b  = sb[s]
            pa = self.positions[a]
            pb = self.positions[b]
            d  = pb - pa
            length = d.norm()
            if length > 1e-8:
                spring_dir = d / length
                # Hooke's law
                f_spring = sk[s] * (length - sr[s])
                # Spring damping: opposes relative velocity along the spring axis.
                # Prevents overshoot/oscillation without slowing unrelated motion.
                v_rel = (self.velocities[b] - self.velocities[a]).dot(spring_dir)
                f = (f_spring + spring_damp * v_rel) * spring_dir
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
                 spring_damping:     float = 0.0,
                 blade_travel_dir:   list  = None,
                 blade_speed:        float = 0.5,
                 split_disc_verts:   bool  = True,
                 opening_ramp_frames: int  = 20,
                 method_instance             = None):
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
        self.spring_damping = spring_damping
        self.opening_ramp_frames = max(1, int(opening_ramp_frames))
        self._method_instance = method_instance  # optional pre-constructed SimulationMethod

        # Populated by TaichiSimulationSystem.on_create_entity
        self.method    = None          # SpringMassMethod — drives physics
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

        # Physics pause (P key) — cut and blade still advance when paused
        self.sim_paused = False

        # Rendering option
        self.split_disc_verts = split_disc_verts  # duplicate rim verts for correct disc/collar normals

        # Set after a cut — used for correct normal computation on the cut mesh.
        self._n_orig:            int        = None  # vertex count before splitting
        self._n_split:           int        = None  # n_orig + intersection vertex count
        self._inter_data:        list       = None  # [(new_id, vi_above, vj_below, t), ...]
        self._shared_list:       list       = None  # seam vertex indices (above-half copies)
        self._cut_normal:        np.ndarray = None  # normalized cut plane normal
        self._disc_split_phys_idx: np.ndarray = None  # physics indices of rendering disc duplicates

        # Unindexed (per-face) rendering — active after the first cut.
        # Each triangle gets its own 3 private vertices so face colors never blend.
        # comp.surface_indices keeps original vertex indices for normal computation.
        # The GPU index buffer is trivial ([0,1,2,3,...]) and only covers visible faces.
        self._face_expanded:    bool       = False
        self._all_render_faces: np.ndarray = None  # (N_all_faces, 3) outer+ALL wound, for expansion
        self._n_outer_faces:    int        = 0     # len(outer_faces) — constant after cut

        # Per-face debug colors (N_all_faces, 3) — stored so yellow overlay stays persistent.
        self._debug_colors: np.ndarray = None

        # Set True after a cut; consumed on the next simulated frame to print a force audit.
        self._pending_force_audit: bool = False

        # Opening velocity ramp queue — each entry applies a small velocity increment
        # per frame for opening_ramp_frames frames instead of one large impulse.
        # Format: {'above': np.ndarray, 'below': np.ndarray, 'step_speed': float, 'left': int}
        self._opening_ramp_queue: list = []


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

        _sim_params = {
            'stiffness':           comp.stiffness,
            'damping':             comp.damping,
            'spring_damping':      comp.spring_damping,
            'total_mass':          comp.total_mass,
            'gravity':             comp.gravity,
            'v_max':               comp.v_max,
            'time_step':           comp.time_step,
            'substeps':            comp.substeps,
            'opening_speed':       comp.opening_speed,
            'blade_speed':         comp.blade_speed,
            'opening_ramp_frames': comp.opening_ramp_frames,
        }
        if comp._method_instance is not None:
            comp.method = comp._method_instance
        else:
            from pyGandalf.thesis_utilities.simulation_method import SpringMassMethod
            comp.method = SpringMassMethod()
        comp.method.initialize(comp.tet_mesh, _sim_params)
        comp.simulator = comp.method._simulator

        sub_dt = comp.time_step / comp.substeps
        print("[TaichiSimulationSystem] Initialized:")
        print(f"  Vertices:        {len(tet.vertices):,}")
        print(f"  Tetrahedra:      {len(tet.tetrahedra):,}")
        if hasattr(comp.simulator, '_sa'):
            print(f"  Springs:         {len(comp.simulator._sa):,}")
        print(f"  Surface faces:   {len(comp.surface_indices):,}")
        print(f"  Fixed vertices:  {int(fixed_mask.sum())}")
        print(f"  Sub-steps/frame: {comp.substeps}  (sub_dt = {sub_dt:.5f} s)")
        print("  F — poke top of mesh downward")
        print("  B — start / pause progressive blade cut")
        print("  C — one-shot cut (testing only)")
        print("  P — pause / resume physics simulation")
        print("  X — disc parallelism check (colors non-parallel disc faces yellow)")

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

        t_blade0 = time.perf_counter()
        if comp.blade_is_active:
            _advance_progressive_blade(comp, mesh_comp, ts)
        t_blade1 = time.perf_counter()

        # --- F key: poke ---
        f_now = InputManager().get_key_down(glfw.KEY_F)
        if f_now and not getattr(self, '_f_prev', False):
            _apply_poke(comp)
        self._f_prev = f_now

        # --- P key: pause / resume physics ---
        p_now = InputManager().get_key_down(glfw.KEY_P)
        if p_now and not getattr(self, '_p_prev', False):
            comp.sim_paused = not comp.sim_paused
            print(f"[Sim] Physics {'paused' if comp.sim_paused else 'resumed'}")
        self._p_prev = p_now

        # --- X key: disc parallelism check (yellow non-parallel disc faces) ---
        x_now = InputManager().get_key_down(glfw.KEY_X)
        if x_now and not getattr(self, '_x_prev', False):
            _check_disc_parallelism(comp, mesh_comp)
        self._x_prev = x_now

        # --- O key: orphan vert overlay (purple faces touching orphaned verts, FEM only) ---
        o_now = InputManager().get_key_down(glfw.KEY_O)
        if o_now and not getattr(self, '_o_prev', False):
            _overlay_orphan_colors(comp, mesh_comp)
        self._o_prev = o_now

        # --- Simulation sub-steps (via SpringMassMethod — handles ramp internally) ---
        t0 = time.perf_counter()
        if not comp.sim_paused:
            comp.method.step(comp.time_step)

            # Force audit: spring-mass only (uses _spring_forces kernel).
            if (comp._pending_force_audit and comp._n_orig is not None
                    and hasattr(comp.simulator, '_spring_forces')):
                comp._pending_force_audit = False
                comp.simulator._clear_forces()
                comp.simulator._spring_forces(
                    comp.simulator._sa, comp.simulator._sb,
                    comp.simulator._sr, comp.simulator._sk,
                    float(comp.spring_damping))
                _print_force_audit(
                    comp.simulator._forces.to_numpy(),
                    comp._n_orig,
                    comp.simulator.positions.shape[0])
            elif comp._pending_force_audit:
                comp._pending_force_audit = False
        t1 = time.perf_counter()

        new_positions = comp.simulator.positions.to_numpy()
        t2 = time.perf_counter()

        if comp._disc_split_phys_idx is not None and len(comp._disc_split_phys_idx) > 0:
            render_positions = np.vstack(
                [new_positions, new_positions[comp._disc_split_phys_idx]])
        else:
            render_positions = new_positions

        if comp._n_orig is not None:
            new_normals = _compute_normals_post_cut(
                render_positions, comp.surface_indices,
                comp._n_orig, comp._n_split,
                comp._inter_data, comp._shared_list,
                comp._cut_normal)
        else:
            new_normals = _compute_normals(render_positions, comp.surface_indices)
        t3 = time.perf_counter()

        # In unindexed mode every face owns its 3 vertices — expand before upload.
        if comp._face_expanded and comp._all_render_faces is not None:
            flat = comp._all_render_faces.flatten()
            upload_pos  = render_positions[flat].reshape(-1, 3)
            upload_norm = new_normals[flat].reshape(-1, 3)
        else:
            upload_pos  = render_positions
            upload_norm = new_normals

        _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[0], upload_pos)
        _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[1], upload_norm)
        t4 = time.perf_counter()

        if not hasattr(self, '_fc'):
            self._fc = 0
        self._fc += 1
        if self._fc % 60 == 0:
            blade_ms = (t_blade1 - t_blade0) * 1000
            print(
                f"[Frame {self._fc:4d}] "
                f"blade {blade_ms:4.2f}ms | "
                f"sim {(t1-t0)*1000:5.2f}ms | "
                f"readback {(t2-t1)*1000:4.2f}ms | "
                f"normals {(t3-t2)*1000:4.2f}ms | "
                f"upload {(t4-t3)*1000:4.2f}ms | "
                f"total {(t4-t_blade0)*1000:5.2f}ms"
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


def _cut_topology_physics(
        current_tets: np.ndarray,
        positions:    np.ndarray,
        velocities:   np.ndarray,
        masses:       np.ndarray,
        fixed:        np.ndarray,
        stiffness:    float,
        gravity:      np.ndarray,
        origin:       np.ndarray,
        normal:       np.ndarray):
    """
    Pure topology-change computation — no Component dependency.

    Splits tetrahedra along the cut plane, duplicates seam vertices, and
    rebuilds the spring-mass simulator.  Used by both the ECS wrapper
    (_cut_topology) and SpringMassMethod.setup_cut().

    Returns
    -------
    (new_sim, final_pos, final_vel, final_mass, final_fixed,
     all_tets, n_orig, n_split, shared_list, remap, inter_data, orig_surf_set)
    or None if the cut plane does not divide the mesh.
    """
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

    # --- Snap near-plane rim vertices to cut plane ---
    # Only snap original vertices that are nearly on the cut plane already
    # (floating-point tolerance fix).  Do NOT snap vertices that are genuinely
    # far from the plane: those are structural surface vertices whose positions
    # define the mesh shape, and moving them to the cut plane creates large
    # distorted triangles (visible as the spike/flap artifact on off-center cuts).
    # Intersection vertices from iv() already land exactly on the plane by
    # construction, so no snapping is needed for them.
    mesh_scale = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    snap_eps   = mesh_scale * 1e-4
    rim_set = set()
    for new_id, vi, vj, _ in inter_data:
        if vi < n_orig: rim_set.add(vi)
        if vj < n_orig: rim_set.add(vj)
    if rim_set:
        rim_arr  = np.array(sorted(rim_set), dtype=np.int32)
        snap_d   = signed_dist[rim_arr]
        close    = np.abs(snap_d) < snap_eps
        if close.any():
            split_pos[rim_arr[close]] -= snap_d[close, np.newaxis] * normal
            print(f"[Cut] Snapped {int(close.sum())} near-plane rim vertices "
                  f"(threshold {snap_eps:.2e})")

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
    # Only duplicate intersection vertices (idx >= n_orig).  Original vertices
    # at signed_dist ≈ 0 appear in both sets but must NOT be duplicated — they
    # sit on the cut plane and should stay shared between both halves.
    shared_list    = sorted(v for v in (above_vert_set & below_vert_set) if v >= n_orig)
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

    # --- Rebuild simulator ---
    new_sim = _SpringMassSimulator(
        vertices        = final_pos,
        tetrahedra      = all_tets,
        fixed_mask      = final_fixed,
        stiffness       = stiffness,
        gravity         = gravity,
        per_vertex_mass = final_mass,
    )
    new_sim.velocities.from_numpy(final_vel.astype(np.float32))

    # No spring zeroing: orig→new springs are structural tet edges that are
    # needed for mesh connectivity.  Zeroing them disconnects the disc from
    # the hemispheres (Fix 3 mistake — verified by force audit).
    # Collar deformation from opening velocity is physically correct elastic
    # behaviour; reduce opening_speed if it looks too violent.

    print(f"[Cut] Simulator: {len(final_pos):,} verts, "
          f"{len(all_tets):,} tets, {len(new_sim._sa):,} springs")

    _print_spring_audit(new_sim._sa, new_sim._sb, new_sim._sr, new_sim._sk, n_orig)

    return (new_sim, final_pos, final_vel, final_mass, final_fixed,
            all_tets, n_orig, n_split, shared_list, remap, inter_data, orig_surf_set)


def _cut_topology(comp: TaichiSimulationComponent,
                  origin: np.ndarray,
                  normal: np.ndarray):
    """
    ECS wrapper for _cut_topology_physics.

    Reads arrays from comp, runs the pure computation, and writes the new
    simulator / tetrahedra / fixed_mask back to comp.
    Returns the same 11-tuple callers expect, or None on failure.
    """
    result = _cut_topology_physics(
        comp.current_tetrahedra,
        comp.simulator.positions.to_numpy(),
        comp.simulator.velocities.to_numpy(),
        comp.simulator._masses.to_numpy(),
        comp.simulator._fixed.to_numpy(),
        comp.stiffness,
        np.array(comp.gravity, dtype=np.float32),
        origin, normal,
    )
    if result is None:
        return None

    (new_sim, final_pos, final_vel, final_mass, final_fixed,
     all_tets, n_orig, n_split, shared_list, remap, inter_data, orig_surf_set) = result

    comp.simulator          = new_sim
    comp.current_tetrahedra = all_tets
    comp.fixed_mask         = final_fixed
    # Only sync method._simulator when types match — avoids replacing a
    # _FEMSimulator with the _SpringMassSimulator _cut_topology_physics returns.
    if comp.method is not None and type(comp.method._simulator) is type(new_sim):
        comp.method._simulator = new_sim

    return (final_pos, final_vel, final_mass, final_fixed,
            all_tets, n_orig, n_split, shared_list, remap, inter_data, orig_surf_set)


def _filter_surface_faces(raw_surface, final_pos, normal, n_orig, n_split,
                           orig_surf_set, mesh_centroid=None,
                           inter_data=None, shared_list=None):
    """
    Partition raw boundary faces (from _extract_boundary_faces) into outer
    faces and wound faces.

    Winding is NOT re-applied here. _extract_boundary_faces already uses the
    robust opposite-vertex test which works for any mesh geometry, including
    non-convex meshes.  The old centroid and orig_surf_set approaches were
    sphere-only heuristics that dropped valid faces on complex meshes.

    Classification (index-only, no geometry needed):
      wound face: every vertex is "on the cut plane", meaning:
        - an intersection vertex (n_orig <= v < n_split), or
        - a dup seam vertex (v >= n_split), or
        - an original seam vertex (v < n_orig and v in shared_list).
      outer face: any face with at least one off-plane original vertex.
    """
    seam_set = set(int(v) for v in shared_list) if shared_list else set()

    def on_plane(v):
        return v >= n_orig or v in seam_set

    outer = []
    wound = []

    for f in raw_surface:
        v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
        if on_plane(v0) and on_plane(v1) and on_plane(v2):
            wound.append(f)
        else:
            outer.append(f)

    outer_arr = (np.array(outer, dtype=np.uint32)
                 if outer else np.zeros((0, 3), dtype=np.uint32))
    return outer_arr, wound


def _split_disc_verts_for_rendering(outer_faces, wound_faces, n_phys, n_orig):
    """
    Create rendering-only duplicate vertices for intersection vertices that appear
    in both disc (wound) faces and collar (outer) faces.

    Without duplication a single vertex index cannot simultaneously carry a
    sphere-surface normal (for collar shading) and a ±cut_normal (for disc
    shading).  This function remaps the disc face index buffer so those shared
    vertices use a fresh index backed by a duplicate position entry.

    Returns:
        remapped_wound  — list of wound face arrays; disc-boundary vertices
                          replaced with new rendering-only indices >= n_phys.
        split_phys_idx  — list of physics vertex indices for the duplicates
                          (duplicate i lives at rendering index n_phys + i).
    """
    disc_verts = set()
    for wf in wound_faces:
        if all(int(v) >= n_orig for v in wf):
            disc_verts.update(int(v) for v in wf)

    if not disc_verts:
        return wound_faces, []

    collar_verts = set()
    for f in outer_faces:
        collar_verts.update(int(v) for v in f)

    split_verts = sorted(disc_verts & collar_verts)
    if not split_verts:
        return wound_faces, []

    remap = {v: n_phys + i for i, v in enumerate(split_verts)}
    remapped = [np.array([remap.get(int(v), int(v)) for v in wf], dtype=np.uint32)
                for wf in wound_faces]
    return remapped, split_verts


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
    comp._n_orig      = n_orig
    comp._n_split     = n_split
    comp._inter_data  = inter_data
    comp._shared_list = shared_list
    comp._cut_normal  = normal

    # --- Opening velocity (ramped over opening_ramp_frames frames) ---
    if comp.opening_speed > 0.0:
        inter_indices = np.array([d[0] for d in inter_data], dtype=np.int32)
        dup_indices   = np.array([remap[v] for v in inter_indices
                                  if remap[v] >= n_split], dtype=np.int32)
        step_speed = comp.opening_speed / comp.opening_ramp_frames
        comp._opening_ramp_queue.append({
            'above': inter_indices,
            'below': dup_indices,
            'step_speed': step_speed,
            'left': comp.opening_ramp_frames,
        })
        print(f"[Cut] Opening ramp queued: {len(inter_indices)} + {len(dup_indices)} verts "
              f"over {comp.opening_ramp_frames} frames ({step_speed:.4f} m/s/frame)")
    comp._pending_force_audit = True

    # --- Surface ---
    raw_surface = _extract_boundary_faces(all_tets, final_pos)
    mesh_centroid = final_pos[:n_orig].mean(axis=0)
    outer_faces, wound_faces = _filter_surface_faces(
        raw_surface, final_pos, normal, n_orig, n_split, orig_surf_set, mesh_centroid,
        inter_data=inter_data, shared_list=shared_list)

    if comp.split_disc_verts:
        remapped_wound, split_phys = _split_disc_verts_for_rendering(
            outer_faces, wound_faces, len(final_pos), n_orig)
        comp._disc_split_phys_idx = np.array(split_phys, dtype=np.int32) if split_phys else None
        render_pos = (np.vstack([final_pos, final_pos[split_phys]])
                      if split_phys else final_pos)
        disc_faces = remapped_wound
    else:
        comp._disc_split_phys_idx = None
        render_pos = final_pos
        disc_faces = wound_faces

    all_faces = (np.vstack([outer_faces, np.array(disc_faces, dtype=np.uint32)])
                 if disc_faces else outer_faces)
    comp.surface_indices    = all_faces   # kept as original indices for normal computation
    comp._all_render_faces  = all_faces   # one-shot cut: all faces visible immediately
    comp._n_outer_faces     = len(outer_faces)
    comp._face_expanded     = True

    new_normals       = _compute_normals_post_cut(render_pos, all_faces,
                                                  n_orig, n_split, inter_data, shared_list, normal)
    face_colors       = _compute_debug_face_colors(all_faces, n_orig)
    comp._debug_colors = face_colors.copy()

    # Build expanded (unindexed) arrays — each face owns its 3 private vertices.
    flat              = all_faces.flatten()
    exp_pos           = render_pos[flat].reshape(-1, 3)
    exp_norm          = new_normals[flat].reshape(-1, 3)
    exp_tex           = np.zeros((len(all_faces) * 3, 2), dtype=np.float32)
    exp_colors        = np.repeat(face_colors, 3, axis=0)
    trivial_idx       = np.arange(len(all_faces) * 3, dtype=np.uint32).reshape(-1, 3)

    print(f"[Cut] Surface: {len(raw_surface):,} raw → {len(all_faces):,} kept "
          f"({len(exp_pos):,} expanded verts)")

    _realloc_gpu_buffers(mesh_comp, exp_pos, exp_norm, exp_tex, trivial_idx,
                         colors=exp_colors)
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

    # Delegate topology split, simulator rebuild, and cutting springs to the method.
    # Each backend (SpringMassMethod, FEMMethod) handles its own physics correctly.
    if not comp.method.setup_cut(normal, origin, blade_dir):
        print("[Blade] Cut plane does not intersect the mesh.")
        return

    # Sync ECS-level aliases — method holds the authoritative new simulator.
    comp.simulator          = comp.method._simulator
    comp.current_tetrahedra = comp.method._current_tets

    # Extract topology data from the cached result tuple for surface computation.
    (_new_sim, final_pos, _final_vel, _final_mass, _final_fixed,
     all_tets, n_orig, n_split, shared_list, remap, inter_data,
     orig_surf_set) = comp.method._topology_result

    comp._n_orig      = n_orig
    comp._n_split     = n_split
    comp._inter_data  = inter_data
    comp._shared_list = shared_list
    comp._cut_normal  = normal

    # Share the seam_pairs list so _advance_progressive_blade can observe broken flags.
    comp._seam_pairs = comp.method._seam_pairs
    seam_pairs       = comp._seam_pairs

    # Blade cursor starts just before the first seam pair.
    if seam_pairs:
        comp.blade_travel = seam_pairs[0]['travel_dist'] - 1e-3
    else:
        comp.blade_travel = 0.0

    # --- Build progressive wound surface ---
    raw_surface = _extract_boundary_faces(all_tets, final_pos)
    mesh_centroid = final_pos[:n_orig].mean(axis=0)
    outer_faces, wound_faces = _filter_surface_faces(
        raw_surface, final_pos, normal, n_orig, n_split, orig_surf_set, mesh_centroid,
        inter_data=inter_data, shared_list=shared_list)

    comp._outer_faces = outer_faces

    if comp.split_disc_verts:
        remapped_wound, split_phys = _split_disc_verts_for_rendering(
            outer_faces, wound_faces, len(final_pos), n_orig)
        comp._disc_split_phys_idx = np.array(split_phys, dtype=np.int32) if split_phys else None
        render_pos = (np.vstack([final_pos, final_pos[split_phys]])
                      if split_phys else final_pos)
        wound_for_render = remapped_wound
    else:
        comp._disc_split_phys_idx = None
        render_pos = final_pos
        wound_for_render = wound_faces

    # Sort wound faces by centroid distance along blade_dir.
    # Centroid uses original wound_faces for correct position lookup;
    # the stored face uses the (possibly remapped) rendering version.
    wound_by_dist = []
    for orig_wf, rend_wf in zip(wound_faces, wound_for_render):
        centroid    = final_pos[[int(orig_wf[0]), int(orig_wf[1]), int(orig_wf[2])]].mean(axis=0)
        travel_dist = float(np.dot(centroid - origin, blade_dir))
        wound_by_dist.append((travel_dist, rend_wf))
    wound_by_dist.sort(key=lambda x: x[0])
    comp._wound_faces_by_dist = wound_by_dist
    comp._wound_face_ptr      = 0

    # Build the full face array (outer first, then ALL wound in travel order).
    # This is the expansion template: outer faces occupy slots [0..n_outer-1],
    # wound faces occupy [n_outer..n_outer+n_wound-1] in the same sorted order
    # as _wound_faces_by_dist, so the trivial index buffer can simply grow.
    wound_arr = (np.array([f for _, f in wound_by_dist], dtype=np.uint32)
                 if wound_by_dist else np.zeros((0, 3), dtype=np.uint32))
    all_render_faces = (np.vstack([outer_faces, wound_arr])
                        if len(wound_arr) > 0 else outer_faces.copy())

    comp.surface_indices    = outer_faces.copy()   # original indices for normal computation
    comp._all_render_faces  = all_render_faces      # full set for per-frame expansion
    comp._n_outer_faces     = len(outer_faces)
    comp._face_expanded     = True

    # Per-face debug colors for the full set (wound faces pre-colored even while hidden).
    face_colors        = _compute_debug_face_colors(all_render_faces, n_orig)
    comp._debug_colors = face_colors.copy()

    new_normals = _compute_normals_post_cut(render_pos, comp.surface_indices,
                                            n_orig, n_split, inter_data, shared_list, normal)

    # Build expanded (unindexed) arrays for the full face set.
    flat_all   = all_render_faces.flatten()
    exp_pos    = render_pos[flat_all].reshape(-1, 3)
    exp_norm   = new_normals[flat_all].reshape(-1, 3)   # hidden wound normals = zero initially (OK)
    exp_tex    = np.zeros((len(all_render_faces) * 3, 2), dtype=np.float32)
    exp_colors = np.repeat(face_colors, 3, axis=0)

    # Initially only outer faces visible — trivial index buffer covers first n_outer*3 verts.
    trivial_outer = np.arange(len(outer_faces) * 3, dtype=np.uint32).reshape(-1, 3)

    print(f"[Blade] Surface: {len(outer_faces):,} outer + "
          f"{len(wound_faces):,} wound faces (hidden until blade passes) | "
          f"{len(all_render_faces):,} total expanded faces")

    _realloc_gpu_buffers(mesh_comp, exp_pos, exp_norm, exp_tex, trivial_outer,
                         colors=exp_colors)

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
    # Check first_break before advance_blade() updates the broken flags.
    newly_to_break = [p for p in comp._seam_pairs
                      if not p['broken'] and p['travel_dist'] <= comp.blade_travel]
    if newly_to_break and not any(p['broken'] for p in comp._seam_pairs):
        comp._pending_force_audit = True
    comp.method.advance_blade(comp.blade_travel)
    # comp._seam_pairs is the same list as comp.method._seam_pairs — broken flags updated in-place.

    # --- Reveal wound faces behind the cursor ---
    ptr = comp._wound_face_ptr
    wbd = comp._wound_faces_by_dist
    while ptr < len(wbd) and wbd[ptr][0] <= comp.blade_travel:
        ptr += 1

    faces_added = ptr > comp._wound_face_ptr
    if faces_added:
        comp._wound_face_ptr = ptr
        revealed = [f for _, f in wbd[:ptr]]
        comp.surface_indices = (                           # original indices for normals
            np.vstack([comp._outer_faces,
                       np.array(revealed, dtype=np.uint32)])
            if revealed else comp._outer_faces.copy()
        )
        mesh_comp.indices = comp.surface_indices

        # Trivial index buffer grows to cover outer + revealed wound face slots.
        n_vis    = comp._n_outer_faces + ptr
        flat_idx = np.arange(n_vis * 3, dtype=np.uint32)
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
            n_vis    = comp._n_outer_faces + len(wbd)
            flat_idx = np.arange(n_vis * 3, dtype=np.uint32)
            gl.glBindVertexArray(mesh_comp.render_pipeline)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh_comp.index_buffer)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, flat_idx.nbytes, flat_idx,
                            gl.GL_DYNAMIC_DRAW)
            gl.glBindVertexArray(0)

        comp.blade_is_active = False
        n_broken = sum(1 for p in comp._seam_pairs if p['broken'])
        print(f"[Blade] Cut complete — {n_broken:,} seam springs broken.")


# ---------------------------------------------------------------------------
# Diagnostic helpers (Analysis 1 + 2)
# ---------------------------------------------------------------------------

def _print_spring_audit(sa: np.ndarray, sb: np.ndarray,
                        sr: np.ndarray, sk: np.ndarray,
                        n_orig: int):
    """
    Analysis 2 — rest-length distribution of orig→new springs after Fix 2.

    Prints how many such springs exist, their rest-length histogram, and how
    many were left active (not zeroed by Fix 2).  Run immediately after
    _cut_topology so the state reflects the fix that was applied.
    """
    orig_to_new = ((sa < n_orig) & (sb >= n_orig)) | ((sb < n_orig) & (sa >= n_orig))
    count = int(orig_to_new.sum())
    print(f"[SpringAudit] orig→new springs total: {count}")
    if count == 0:
        return

    rls      = sr[orig_to_new]
    active   = sk[orig_to_new] != 0.0
    n_active = int(active.sum())
    n_zeroed = count - n_active

    print(f"[SpringAudit] rest_len  min={rls.min():.4f}  mean={rls.mean():.4f}  "
          f"max={rls.max():.4f}  std={rls.std():.4f}")
    print(f"[SpringAudit] zeroed by Fix 2: {n_zeroed}   still active: {n_active}")

    bins   = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, float('inf')]
    labels = ['<0.01', '0.01-0.02', '0.02-0.05', '0.05-0.1',
              '0.1-0.2', '0.2-0.5', '>0.5']
    for lo, hi, label in zip(bins[:-1], bins[1:], labels):
        n_all = int(((rls >= lo) & (rls < hi)).sum())
        n_act = int(((rls[active] >= lo) & (rls[active] < hi)).sum()) if n_active > 0 else 0
        print(f"  {label:>12s}: {n_all:4d} total  {n_act:4d} active")


def _print_force_audit(forces: np.ndarray, n_orig: int, n_total: int):
    """
    Analysis 1 — net force magnitudes on the first simulated frame after a cut.

    Prints per-category statistics (original vertices vs new vertices) and the
    top-10 vertices by net force magnitude.  Called once after the first batch
    of substeps runs, so opening velocity has already been applied and spring
    forces from the compressed/stretched orig→new springs are visible.
    """
    mags = np.linalg.norm(forces, axis=1)
    orig_mags = mags[:n_orig]
    new_mags  = mags[n_orig:n_total] if n_total > n_orig else np.zeros(0)

    print(f"[ForceAudit] Net forces on frame 1 after cut:")
    top5_orig = np.sort(orig_mags)[::-1][:5]
    print(f"  Original verts ({n_orig}):  "
          f"max={orig_mags.max():.4f}  mean={orig_mags.mean():.5f}  "
          f"top5={top5_orig.round(4).tolist()}")
    if len(new_mags) > 0:
        top5_new = np.sort(new_mags)[::-1][:5]
        print(f"  New verts     ({len(new_mags)}):  "
              f"max={new_mags.max():.4f}  mean={new_mags.mean():.5f}  "
              f"top5={top5_new.round(4).tolist()}")

    top10_idx = np.argsort(mags)[::-1][:10]
    print(f"[ForceAudit] Top 10 vertices by |F|:")
    for rank, idx in enumerate(top10_idx):
        cat = "orig" if idx < n_orig else "new "
        print(f"  #{rank+1:2d}: vertex {idx:5d} ({cat})  |F|={mags[idx]:.5f}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _check_disc_parallelism(comp: TaichiSimulationComponent,
                             mesh_comp: StaticMeshComponent):
    """
    X key — for each disc half (above and below), pick the face nearest to the
    group's centroid as a reference, then color yellow every disc face whose
    normal is not parallel to that reference (|dot| < 0.95).

    Uses the FULL wound-face set (all seam pairs, even if not yet revealed by
    the blade) so the check works at any point during or after cutting.
    Separates above/below halves by dotting each face's geometric normal with
    the stored cut_normal: above-half faces point in -cut_normal direction,
    below-half faces point in +cut_normal direction.
    """
    if comp._n_orig is None or not comp._wound_faces_by_dist:
        print("[DiscCheck] No cut data available — press B first.")
        return

    n_orig   = comp._n_orig
    cut_n    = comp._cut_normal

    # Current render positions (physics + rendering disc duplicates)
    sim_pos = comp.simulator.positions.to_numpy()
    if comp._disc_split_phys_idx is not None and len(comp._disc_split_phys_idx) > 0:
        render_pos = np.vstack([sim_pos, sim_pos[comp._disc_split_phys_idx]])
    else:
        render_pos = sim_pos

    # Full disc face set (rendering indices, both halves, all faces regardless of blade progress)
    all_disc = np.array([f for _, f in comp._wound_faces_by_dist], dtype=np.uint32)
    if len(all_disc) == 0:
        print("[DiscCheck] No disc faces found.")
        return

    # Compute normalised face normals via cross product
    p0 = render_pos[all_disc[:, 0]]
    p1 = render_pos[all_disc[:, 1]]
    p2 = render_pos[all_disc[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(fn, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-10, 1.0, lengths)
    fn = fn / lengths  # (K, 3)

    # Separate halves: above (fn · cut_n < 0) vs below (fn · cut_n > 0)
    dots_cut   = fn @ cut_n
    above_mask = dots_cut < 0.0
    below_mask = ~above_mask

    PARALLEL_THRESHOLD = 0.95  # |dot| below this ≈ more than ~18° off-plane

    # Overlay yellow on non-parallel faces in the per-face color array.
    all_render_faces = comp._all_render_faces
    if comp._debug_colors is not None and all_render_faces is not None:
        face_colors = comp._debug_colors.copy()   # (N_all_faces, 3)
    else:
        all_faces   = (np.vstack([comp._outer_faces, all_disc])
                       if comp._outer_faces is not None and len(comp._outer_faces) > 0
                       else all_disc)
        face_colors = _compute_debug_face_colors(all_faces, n_orig)

    YELLOW = np.array([0.95, 0.85, 0.1], dtype=np.float32)
    # Map each non-parallel disc face back to its index in _all_render_faces.
    # all_disc comes from _wound_faces_by_dist; wound faces start at _n_outer_faces in all_render_faces.
    n_outer   = comp._n_outer_faces
    disc_face_indices_in_all = np.where(
        np.all(all_render_faces >= n_orig, axis=1)
    )[0]   # indices of disc faces within all_render_faces

    for label, gmask in [("above", above_mask), ("below", below_mask)]:
        gfaces   = all_disc[gmask]
        gnormals = fn[gmask]
        if len(gfaces) < 2:
            print(f"[DiscCheck] {label}-half: only {len(gfaces)} face(s) — skipping.")
            continue
        centroids = render_pos[gfaces].mean(axis=1)
        group_cen = centroids.mean(axis=0)
        ref_idx   = int(np.argmin(np.linalg.norm(centroids - group_cen, axis=1)))
        ref_n     = gnormals[ref_idx]
        abs_dots  = np.abs(gnormals @ ref_n)
        bad       = abs_dots < PARALLEL_THRESHOLD
        n_bad     = int(bad.sum())
        print(f"[DiscCheck] {label}-half: {len(gfaces)} faces, "
              f"ref centroid {centroids[ref_idx].round(3)}, "
              f"{n_bad} non-parallel (|dot|<{PARALLEL_THRESHOLD})")

        # Find global face indices in all_render_faces for the bad faces in this half.
        half_disc_global = disc_face_indices_in_all[gmask]
        for fi in np.where(bad)[0]:
            face_colors[half_disc_global[fi]] = YELLOW

    n_yellow = int(np.all(face_colors == YELLOW, axis=1).sum())

    if len(mesh_comp.buffers) > 3 and all_render_faces is not None:
        exp_colors = np.repeat(face_colors, 3, axis=0).astype(np.float32)
        flat_col   = exp_colors.flatten()
        gl.glBindVertexArray(mesh_comp.render_pipeline)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[3])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_col.nbytes, flat_col, gl.GL_DYNAMIC_DRAW)
        gl.glBindVertexArray(0)

    if n_yellow > 0:
        print(f"[DiscCheck] {n_yellow} non-parallel disc faces colored yellow.")
    else:
        print("[DiscCheck] All disc faces are parallel to their reference — no artifacts detected.")


def _compute_debug_face_colors(faces: np.ndarray, n_orig: int) -> np.ndarray:
    """
    Assign one color per face (not per vertex) for crisp unblended debug rendering.
      green (0.3, 0.7, 0.4)  — regular outer face  (all vertices < n_orig)
      blue  (0.15, 0.35, 0.9) — collar face         (mixed: some vertices >= n_orig)
      red   (0.85, 0.1, 0.1)  — disc / wound face   (all vertices >= n_orig)
    Returns (N_faces, 3) float32.
    """
    n = len(faces)
    colors = np.tile([0.3, 0.7, 0.4], (n, 1)).astype(np.float32)

    v = faces  # (N, 3)
    all_new = np.all(v >= n_orig, axis=1)
    any_new = np.any(v >= n_orig, axis=1)

    colors[any_new & ~all_new] = [0.15, 0.35, 0.9]   # blue — collar
    colors[all_new]             = [0.85, 0.1,  0.1]   # red  — disc

    n_blue = int((any_new & ~all_new).sum())
    n_red  = int(all_new.sum())
    print(f"[Cut] Debug colors: {n_blue} collar faces (blue), {n_red} disc faces (red), "
          f"{n - n_blue - n_red} outer faces (green)")
    return colors


def _overlay_orphan_colors(comp: TaichiSimulationComponent,
                            mesh_comp: StaticMeshComponent) -> None:
    """
    O key: color purple every surface face that touches an orphaned vert.
    Orphan verts are new-cut verts whose tets were all filtered out (FEM only).
    Use this to confirm whether the visual artifact region matches the orphan region.
    """
    if not hasattr(comp.method, '_debug_orphan_verts') or not comp.method._debug_orphan_verts:
        print("[Orphan] No orphan verts recorded (not an FEM cut, or no cut yet).")
        return

    all_render_faces = comp._all_render_faces
    if all_render_faces is None or comp._debug_colors is None:
        print("[Orphan] No face data yet — cut has not run.")
        return

    ov_arr      = np.array(sorted(comp.method._debug_orphan_verts), dtype=np.int32)
    face_colors = comp._debug_colors.copy()
    PURPLE      = np.array([0.6, 0.0, 0.8], dtype=np.float32)
    hits        = np.isin(all_render_faces, ov_arr).any(axis=1)
    face_colors[hits] = PURPLE
    comp._debug_colors = face_colors.copy()

    n_purple = int(hits.sum())
    print(f"[Orphan] Purple overlay: {n_purple} faces touch an orphan vert "
          f"({len(comp.method._debug_orphan_verts)} orphans total).", flush=True)

    if len(mesh_comp.buffers) > 3:
        exp_colors = np.repeat(face_colors, 3, axis=0).astype(np.float32)
        flat_col   = exp_colors.flatten()
        gl.glBindVertexArray(mesh_comp.render_pipeline)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[3])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_col.nbytes, flat_col, gl.GL_DYNAMIC_DRAW)
        gl.glBindVertexArray(0)


def _realloc_gpu_buffers(mesh_comp: StaticMeshComponent,
                          positions:  np.ndarray,
                          normals:    np.ndarray,
                          texcoords:  np.ndarray,
                          indices:    np.ndarray,
                          colors:     np.ndarray = None):
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

    if colors is not None and len(mesh_comp.buffers) > 3:
        flat_col = colors.flatten().astype(np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[3])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_col.nbytes, flat_col, gl.GL_DYNAMIC_DRAW)

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


def _compute_normals_post_cut(vertices:    np.ndarray,
                               indices:     np.ndarray,
                               n_orig:      int,
                               n_split:     int,
                               inter_data:  list,
                               shared_list: list,
                               cut_normal:  np.ndarray) -> np.ndarray:
    """
    Normal computation for a cut mesh.  Fully vectorized -- no Python loops.
    """
    normals = np.zeros_like(vertices, dtype=np.float32)
    idx_i32 = indices.astype(np.int32)
    _accumulate_normals_kernel(vertices, idx_i32, normals)
    _normalize_normals_kernel(normals)

    cut_n = (cut_normal / np.linalg.norm(cut_normal)).astype(np.float32)

    # Disc vertices: appear in any face where all three indices >= n_orig.
    disc_mask = np.all(idx_i32 >= n_orig, axis=1)
    if disc_mask.any():
        disc_arr = np.unique(idx_i32[disc_mask])
    else:
        disc_arr = np.empty(0, dtype=np.int32)

    # Lerp normals for collar-only intersection vertices (vectorized).
    if inter_data:
        inter_np  = np.array([(d[0], d[1], d[2], d[3]) for d in inter_data],
                              dtype=np.float64)
        new_ids   = inter_np[:, 0].astype(np.int32)
        vis       = inter_np[:, 1].astype(np.int32)
        vjs       = inter_np[:, 2].astype(np.int32)
        ts        = inter_np[:, 3].astype(np.float32)[:, np.newaxis]

        is_disc   = np.isin(new_ids, disc_arr)
        collar    = ~is_disc
        if collar.any():
            cids     = new_ids[collar]
            n_interp = normals[vis[collar]] * (1.0 - ts[collar]) + \
                       normals[vjs[collar]] * ts[collar]
            lengths  = np.linalg.norm(n_interp, axis=1, keepdims=True)
            valid    = (lengths.squeeze(axis=1) > 1e-6)
            n_interp[valid]  /= lengths[valid]
            n_interp[~valid]  = normals[vis[collar]][~valid]
            normals[cids]     = n_interp

    # Snap disc vertex normals to ±cut_n (vectorized).
    if len(disc_arr) > 0:
        dots = normals[disc_arr] @ cut_n          # (D,)
        nonzero = np.abs(dots) > 1e-6
        normals[disc_arr[nonzero]] = np.where(
            dots[nonzero, np.newaxis] < 0.0, -cut_n, cut_n)

    # Non-disc dup vertices: copy normal from seam partner (vectorized).
    if shared_list is not None and len(shared_list) > 0:
        seam_arr = np.asarray(shared_list, dtype=np.int32)
        dup_arr  = np.arange(n_split, n_split + len(seam_arr), dtype=np.int32)
        valid    = dup_arr < len(normals)
        if valid.any() and len(disc_arr) > 0:
            not_disc = ~np.isin(dup_arr[valid], disc_arr)
            tgt = dup_arr[valid][not_disc]
            src = seam_arr[valid][not_disc]
            normals[tgt] = normals[src]
        elif valid.any():
            normals[dup_arr[valid]] = normals[seam_arr[valid]]

    return normals


def _update_vbo(vao: int, vbo: int, data: np.ndarray):
    """Upload new data into an existing VBO (no reallocation)."""
    flat = data.flatten().astype(np.float32)
    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, flat.nbytes, flat)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    gl.glBindVertexArray(0)
