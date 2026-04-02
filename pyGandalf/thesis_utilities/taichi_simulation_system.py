"""
Taichi Spring-Mass Simulation System

GPU-accelerated spring-mass simulator for tetrahedral meshes using explicit
Euler integration.  Replaces the SOFA backend entirely.

Why explicit Euler here:
  - The mesh starts at rest with gravity=0, so all springs are at their
    natural length and net force is zero.  Explicit Euler is perfectly
    stable when the net force is zero or small.
  - After a cut the wound springs are released, creating transient forces
    that open the wound.  The critical timestep for stability is
    dt_crit = 2 * sqrt(m_vertex / k).  With default params (total_mass=1,
    ~3000 verts, stiffness=100) dt_crit ≈ 0.026 s, so dt=0.001 has a
    26x safety margin.
  - Topology changes (cuts) are just numpy array operations — no external
    framework API required.

Press C to cut along the plane defined in TaichiSimulationComponent.
After the cut, wound-boundary vertices receive an opening velocity that
simulates tissue pre-stress releasing.
"""

import taichi as ti
import numpy as np
import ctypes
import time

import glfw
import OpenGL.GL as gl
from pyGandalf.core.input_manager import InputManager
from pyGandalf.systems.system import System
from pyGandalf.scene.components import Component, StaticMeshComponent

# Initialise Taichi once at module import.  Uses the best available GPU
# backend (CUDA on NVIDIA, Vulkan on AMD/Intel).  Change to ti.cpu if
# no compatible GPU is present.
ti.init(arch=ti.gpu, log_level=ti.WARN)


# ---------------------------------------------------------------------------
# Physics engine
# ---------------------------------------------------------------------------

@ti.data_oriented
class _SpringMassSimulator:
    """
    Explicit Euler spring-mass simulator.

    Positions and velocities live in fixed-size Taichi fields (GPU resident).
    Springs are stored as numpy arrays so they can be swapped cheaply after
    topology changes without recreating the vertex fields.
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

    def rebuild_springs(self, new_tetrahedra: np.ndarray):
        """Swap in a new spring network after a topology change (cut)."""
        pos = self.positions.to_numpy()
        self._sa, self._sb, self._sr = _build_springs(pos, new_tetrahedra)
        print(f"[Taichi] Spring network rebuilt: {len(self._sa):,} springs")

    def step(self, dt: float, damping: float):
        self._clear_forces()
        self._spring_forces(self._sa, self._sb, self._sr,
                            float(self._stiffness))
        self._integrate(float(dt), float(damping), self._gravity)

    # --- Taichi kernels ---

    @ti.kernel
    def _clear_forces(self):
        for i in self._forces:
            self._forces[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _spring_forces(self,
                       sa:        ti.types.ndarray(dtype=ti.i32, ndim=1),
                       sb:        ti.types.ndarray(dtype=ti.i32, ndim=1),
                       sr:        ti.types.ndarray(dtype=ti.f32, ndim=1),
                       stiffness: ti.f32):
        for s in range(sa.shape[0]):
            a  = sa[s]
            b  = sb[s]
            pa = self.positions[a]
            pb = self.positions[b]
            d  = pb - pa
            length = d.norm()
            if length > 1e-8:
                # Hooke's law along the edge direction.
                # Taichi inserts atomics automatically for indirect field writes.
                f = stiffness * (length - sr[s]) / length * d
                self._forces[a] += f
                self._forces[b] -= f

    @ti.kernel
    def _integrate(self,
                   dt:      ti.f32,
                   damping: ti.f32,
                   gravity: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        grav = ti.Vector([gravity[0], gravity[1], gravity[2]])
        for i in self.positions:
            if self._fixed[i] == 0:
                acc = self._forces[i] / self._masses[i] + grav
                # Simple velocity damping: v *= (1 - c*dt).
                self.velocities[i] = (
                    self.velocities[i] * (1.0 - damping * dt) + acc * dt
                )
                self.positions[i] += self.velocities[i] * dt


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

class TaichiSimulationComponent(Component):
    """
    Stores simulation parameters and runtime state for a tetrahedral entity.

    Args:
        tet_mesh:      TetrahedralMeshInstance to simulate.
        time_step:     Integration timestep (s).  Must satisfy
                       dt < 2*sqrt(m_vertex/k) for stability.
        gravity:       Gravity vector (default [0,0,0] — no gravity).
        stiffness:     Spring stiffness (N/m).
        damping:       Velocity damping coefficient.  Fraction of velocity
                       lost per second: v *= (1 - damping*dt).
        total_mass:    Total object mass (kg) distributed uniformly.
        opening_speed: Velocity (m/s) given to wound-boundary vertices on cut
                       to simulate pre-stress release.
    """

    def __init__(self, tet_mesh,
                 time_step:     float = 0.001,
                 substeps:      int   = 10,
                 gravity:       list  = None,
                 stiffness:     float = 100.0,
                 damping:       float = 1.0,
                 total_mass:    float = 1.0,
                 opening_speed: float = 1.0,
                 poke_speed:    float = 3.0):
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

        # Populated by TaichiSimulationSystem.on_create_entity
        self.simulator:           _SpringMassSimulator = None
        self.surface_indices:     np.ndarray           = None
        self.current_tetrahedra:  np.ndarray           = None
        self.fixed_mask:          np.ndarray           = None
        self.poke_mask:           np.ndarray           = None  # top 20% of vertices

        # Cutting
        self.should_cut       = False
        self.cut_plane_origin = [0.0, 0.0, 0.0]
        self.cut_plane_normal = [0.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class TaichiSimulationSystem(System):
    """
    Drives a Taichi spring-mass simulation and uploads deformed vertex
    positions to the GPU each frame.

    Must be registered BEFORE OpenGLStaticMeshRenderingSystem.
    The paired StaticMeshComponent must be initialised with attributes and
    indices directly (load_from_file=False) using the tet mesh vertices.
    """

    def on_create_entity(self, entity, components):
        comp: TaichiSimulationComponent
        mesh_comp: StaticMeshComponent
        comp, mesh_comp = components

        tet = comp.tet_mesh

        # Fix the bottom 5 % of vertices so the object doesn't free-fall.
        y         = tet.vertices[:, 1]
        threshold = y.min() + (y.max() - y.min()) * 0.05
        fixed_mask = (y < threshold).astype(np.int32)

        comp.fixed_mask         = fixed_mask
        comp.current_tetrahedra = tet.tetrahedra.copy()

        # Poke target: top 5 % of vertices, excluding any that are fixed.
        # Keeping the area small means the poked vertices are surrounded by
        # unperturbed mesh on all sides, which pulls them back naturally.
        poke_threshold  = y.max() - (y.max() - y.min()) * 0.05
        comp.poke_mask  = (y >= poke_threshold) & (fixed_mask == 0)
        comp.surface_indices    = _extract_boundary_faces(
            tet.tetrahedra, tet.vertices)

        comp.simulator = _SpringMassSimulator(
            vertices    = tet.vertices,
            tetrahedra  = tet.tetrahedra,
            fixed_mask  = fixed_mask,
            stiffness   = comp.stiffness,
            total_mass  = comp.total_mass,
            gravity     = np.array(comp.gravity, dtype=np.float32),
        )

        sub_dt = comp.time_step / comp.substeps
        print("[TaichiSimulationSystem] Initialized:")
        print(f"  Vertices:       {len(tet.vertices):,}")
        print(f"  Tetrahedra:     {len(tet.tetrahedra):,}")
        print(f"  Springs:        {len(comp.simulator._sa):,}")
        print(f"  Surface faces:  {len(comp.surface_indices):,}")
        print(f"  Fixed vertices: {int(fixed_mask.sum())}")
        print(f"  Poke vertices:  {int(comp.poke_mask.sum())}  (top 20%)")
        print(f"  Sub-steps/frame: {comp.substeps}  (sub_dt = {sub_dt:.5f} s)")
        print("  F — poke top of mesh downward")
        print("  C — cut along the plane defined by cut_plane_origin/normal")

    def on_update_entity(self, ts: float, entity, components):
        comp: TaichiSimulationComponent
        mesh_comp: StaticMeshComponent
        comp, mesh_comp = components

        if comp.simulator is None:
            return
        # GPU buffers are created by the rendering system on first frame.
        if mesh_comp.render_pipeline is None or len(mesh_comp.buffers) < 2:
            return

        # --- One-shot C key: cut ---
        c_now = InputManager().get_key_down(glfw.KEY_C)
        if c_now and not getattr(self, '_c_prev', False):
            comp.should_cut = True
        self._c_prev = c_now

        if comp.should_cut:
            comp.should_cut = False
            _perform_cut(comp, mesh_comp)

        # --- One-shot F key: poke ---
        f_now = InputManager().get_key_down(glfw.KEY_F)
        if f_now and not getattr(self, '_f_prev', False):
            _apply_poke(comp)
        self._f_prev = f_now

        # --- Simulation sub-steps ---
        # Run multiple small steps per frame to stay below dt_crit.
        # dt_crit = 2*sqrt(m_vertex / k_effective), where k_effective is the
        # sum of stiffnesses at the most-connected vertex (~25*k for tet meshes).
        t0 = time.perf_counter()
        sub_dt = comp.time_step / comp.substeps
        for _ in range(comp.substeps):
            comp.simulator.step(sub_dt, comp.damping)
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
    """Apply a downward impulse to the top 20 % of vertices."""
    if comp.poke_mask is None or comp.simulator is None:
        return
    vels = comp.simulator.velocities.to_numpy()
    vels[comp.poke_mask, 1] -= comp.poke_speed   # negative Y = downward
    comp.simulator.velocities.from_numpy(vels.astype(np.float32))
    n = int(comp.poke_mask.sum())
    print(f"[Poke] Applied {comp.poke_speed} m/s downward to {n} vertices")


# ---------------------------------------------------------------------------
# Cut
# ---------------------------------------------------------------------------

def _split_crossed_tets(tetrahedra: np.ndarray,
                         positions:  np.ndarray,
                         signed_dist: np.ndarray):
    """
    Split tetrahedra that straddle the cut plane by inserting new vertices
    exactly at edge-plane intersections.

    Handles two crossing configurations:
      1+3 (1 vertex above, 3 below): 1 above-tet + 3 below-tets (prism split)
      3+1 (3 above, 1 below):        3 above-tets + 1 below-tet
      2+2 (2 above, 2 below):        tet removed (creates a thin gap; rare)

    Returns:
        new_positions (N_new, 3)   — original verts extended with intersection verts
        above_tets    (M, 4) i32  — tets on/above the plane
        below_tets    (K, 4) i32  — tets on/below the plane
        inter_data    list of (new_idx, vi, vj, t) for each new intersection vert
    """
    ext_positions: list = [p for p in positions]   # grow as we add new verts
    edge_cache: dict    = {}                        # (min_v,max_v) -> (new_idx,vi,vj,t)
    inter_data: list    = []

    above_list: list = []
    below_list:  list = []

    tet_dists  = signed_dist[tetrahedra]
    above_mask = np.all(tet_dists >= 0, axis=1)
    below_mask = np.all(tet_dists <= 0, axis=1)

    above_list.extend(tetrahedra[above_mask].tolist())
    below_list.extend(tetrahedra[below_mask].tolist())

    def iv(vi: int, vj: int) -> int:
        """Return index of intersection vertex on edge (vi, vj)."""
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

        av = [int(verts[i]) for i in range(4) if dists[i] >= 0]
        bv = [int(verts[i]) for i in range(4) if dists[i] <  0]
        n_a = len(av)

        if n_a == 1:
            # 1 above (a), 3 below (b0,b1,b2)
            # Above: single tet  [a, p0, p1, p2]
            # Below: prism {b0,b1,b2,p0,p1,p2} → 3 tets
            #   The only face entirely on the cut plane is (p0,p1,p2),
            #   which appears exactly once (in tet [b2,p0,p1,p2]).
            #   After vertex duplication it becomes a wound boundary face.
            a, b0, b1, b2 = av[0], bv[0], bv[1], bv[2]
            p0, p1, p2    = iv(a, b0), iv(a, b1), iv(a, b2)

            above_list.append([a,  p0, p1, p2])
            below_list += [[b0, b1, b2, p0],
                           [b1, b2, p0, p1],
                           [b2, p0, p1, p2]]

        elif n_a == 3:
            # 3 above (a0,a1,a2), 1 below (b)  — symmetric to 1+3
            a0, a1, a2, b = av[0], av[1], av[2], bv[0]
            p0, p1, p2    = iv(a0, b), iv(a1, b), iv(a2, b)

            below_list.append([b,  p0, p1, p2])
            above_list += [[a0, a1, a2, p0],
                           [a1, a2, p0, p1],
                           [a2, p0, p1, p2]]

        else:
            # n_a == 2: 2+2 case — 2 vertices above (a0,a1), 2 below (b0,b1).
            # 4 edges cross the plane, creating a quad cut surface.
            # We pick the diagonal p00-p11 to split the quad into 2 triangles,
            # then decompose the above and below frustums into 3 tets each.
            a0, a1 = av[0], av[1]
            b0, b1 = bv[0], bv[1]
            p00, p01 = iv(a0, b0), iv(a0, b1)
            p10, p11 = iv(a1, b0), iv(a1, b1)

            above_list += [[a0, p00, p01, p11],
                           [a1, p00, p10, p11],
                           [a0, a1, p00, p11]]

            below_list += [[b1, p00, p01, p11],
                           [b0, p00, p10, p11],
                           [b0, b1, p00, p11]]

    new_pos    = np.array(ext_positions, dtype=np.float32)
    above_arr  = (np.array(above_list, dtype=np.int32)
                  if above_list else np.zeros((0, 4), dtype=np.int32))
    below_arr  = (np.array(below_list, dtype=np.int32)
                  if below_list else np.zeros((0, 4), dtype=np.int32))
    return new_pos, above_arr, below_arr, inter_data


def _perform_cut(comp: TaichiSimulationComponent,
                 mesh_comp: StaticMeshComponent):
    """
    Surgical incision cut using tet splitting for a smooth flat wound surface.

    Steps:
    1. Split each crossed tet at the cut plane (1+3 and 3+1 cases).
       New vertices are placed EXACTLY on the cut plane → wound surface is flat.
    2. Extend velocities/masses for new intersection verts (linearly interpolated).
    3. Duplicate intersection verts so each half owns its own seam copy.
    4. Recreate the simulator with the extended vertex/tet arrays.
    5. Apply wound-opening velocity to seam vertices.
    6. Re-extract boundary faces → cut surface consists of exact-plane triangles.
    7. Reallocate all GPU buffers to the new vertex/face count.
    """
    current_tets = comp.current_tetrahedra
    if current_tets is None or len(current_tets) == 0:
        print("[Cut] No tetrahedra remaining.")
        return

    positions  = comp.simulator.positions.to_numpy()
    velocities = comp.simulator.velocities.to_numpy()
    masses     = comp.simulator._masses.to_numpy()
    fixed      = comp.simulator._fixed.to_numpy()

    origin = np.array(comp.cut_plane_origin, dtype=np.float32)
    normal = np.array(comp.cut_plane_normal,  dtype=np.float32)
    normal /= np.linalg.norm(normal)

    signed_dist = (positions - origin) @ normal

    # --- Precompute original outer-surface topology ---
    # After splitting, boundary-face extraction will also return interior faces
    # that got exposed because their neighbouring crossed tet was replaced.
    # We remove those by keeping only faces that either:
    #   (a) were on the original outer surface, OR
    #   (b) contain at least one new vertex (intersection / duplicate).
    face_combos = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
    all_orig_f  = current_tets[:, face_combos].reshape(-1, 3)
    all_orig_s  = np.sort(all_orig_f, axis=1)
    _, _inv, _cnt = np.unique(all_orig_s, axis=0,
                               return_inverse=True, return_counts=True)
    orig_surf_set = {
        tuple(row) for row in all_orig_s[(_cnt == 1)[_inv]]
    }
    n_orig = len(positions)   # indices < n_orig are original vertices

    # --- Tet splitting ---
    split_pos, above_tets, below_tets, inter_data = \
        _split_crossed_tets(current_tets, positions, signed_dist)

    n_split = len(split_pos)   # n_orig + number of intersection verts
    n_inter = n_split - n_orig

    print(f"[Cut] {len(above_tets):,} above-tets, {len(below_tets):,} below-tets, "
          f"{n_inter} intersection verts created")

    if len(above_tets) == 0 or len(below_tets) == 0:
        print("[Cut] Plane doesn't divide mesh into two non-empty halves.")
        return

    # Snap rim vertices (original mesh vertices that were endpoints of crossing
    # edges) to the cut plane.  Without this, the collar faces connecting the
    # outer sphere shell to the wound surface create internal triangular fins
    # visible from inside the hemisphere, and removing them leaves a visible gap.
    # After snapping, those vertices sit exactly on the plane: the collar faces
    # become coplanar with the wound surface, and the rim is seamless.
    rim_set = set()
    for new_id, vi, vj, t in inter_data:
        if vi < n_orig:
            rim_set.add(vi)
        if vj < n_orig:
            rim_set.add(vj)
    if rim_set:
        rim_arr = np.array(sorted(rim_set), dtype=np.int32)
        snap_d  = signed_dist[rim_arr]
        split_pos[rim_arr] -= snap_d[:, np.newaxis] * normal
        print(f"[Cut] Snapped {len(rim_arr)} rim vertices to cut plane")

    # Extend velocity/mass/fixed arrays to match new vertex count.
    split_vel   = np.zeros((n_split, 3), dtype=np.float32)
    split_mass  = np.zeros(n_split,      dtype=np.float32)
    split_fixed = np.zeros(n_split,      dtype=np.int32)
    split_vel[:n_orig]   = velocities
    split_mass[:n_orig]  = masses
    split_fixed[:n_orig] = fixed

    for new_id, vi, vj, t in inter_data:
        split_vel[new_id]   = velocities[vi] * (1.0 - t) + velocities[vj] * t
        split_mass[new_id]  = masses[vi]     * (1.0 - t) + masses[vj]     * t
        split_fixed[new_id] = 0   # seam verts always free to move

    # --- Vertex duplication ---
    # Intersection verts appear in both above and below sub-tets → duplicate them.
    # (After proper splitting, only intersection verts are shared.)
    above_vert_set = set(above_tets.flatten().tolist())
    below_vert_set = set(below_tets.flatten().tolist())
    shared_list    = sorted(above_vert_set & below_vert_set)
    n_shared       = len(shared_list)
    print(f"[Cut] Duplicating {n_shared} seam vertices")

    # Build remap: shared verts in below tets → new duplicate indices.
    remap = np.arange(n_split, dtype=np.int32)
    for i, v in enumerate(shared_list):
        remap[v] = n_split + i

    shared_arr     = np.array(shared_list, dtype=np.int32)
    final_pos      = np.vstack([split_pos, split_pos[shared_arr]])
    final_vel      = np.vstack([split_vel, split_vel[shared_arr]])
    final_mass     = np.concatenate([split_mass,  split_mass[shared_arr]])
    final_fixed    = np.concatenate([split_fixed, split_fixed[shared_arr]])

    new_below_tets = remap[below_tets]
    all_tets       = np.vstack([above_tets, new_below_tets])
    comp.current_tetrahedra = all_tets
    comp.fixed_mask         = final_fixed

    # --- Wound-opening velocity ---
    if comp.opening_speed > 0.0 and len(comp.simulator._sr) > 0:
        thresh = float(np.mean(comp.simulator._sr)) * 2.0
        # Intersection verts are the seam — they're all near the plane by construction.
        inter_indices  = np.array([d[0] for d in inter_data], dtype=np.int32)
        dup_indices    = np.array([remap[v] for v in inter_indices
                                   if remap[v] >= n_split], dtype=np.int32)

        if len(inter_indices) > 0:
            final_vel[inter_indices] += comp.opening_speed * normal
        if len(dup_indices) > 0:
            final_vel[dup_indices]   -= comp.opening_speed * normal
        print(f"[Cut] Opening velocity applied to {len(inter_indices)} + "
              f"{len(dup_indices)} seam verts")

    # --- Recreate simulator ---
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

    # --- Re-extract boundary faces ---
    # The wound surface now consists of intersection triangles that lie
    # exactly on the cut plane → smooth, flat incision.
    raw_surface = _extract_boundary_faces(all_tets, final_pos)

    # Filter faces into three categories:
    #   1. All-orig faces that are in orig_surf_set → original outer surface, keep as-is.
    #   2. All-orig faces NOT in orig_surf_set → interior faces exposed by split, discard.
    #   3. Any face with at least one new/dup vert → wound surface or collar face.
    #      Collar faces close the seam gap between the outer surface and the wound disc.
    #      Their winding is enforced explicitly (above half faces down, below half faces up).
    # n_split is the vertex count BEFORE duplication, i.e. n_orig + n_inter.
    # [n_orig, n_split)  → above-half intersection vertices (original indices)
    # [n_split, ...)     → below-half duplicate intersection vertices
    # For the above half the wound surface faces DOWNWARD (normal dot < 0).
    # For the below half the wound surface faces UPWARD   (normal dot > 0).
    keep = []
    for f in raw_surface:
        v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
        all_orig = v0 < n_orig and v1 < n_orig and v2 < n_orig

        if all_orig:
            if tuple(sorted((v0, v1, v2))) in orig_surf_set:
                keep.append(f)   # original outer sphere surface face
            # else: interior all-orig face → discard
        else:
            # wound surface face OR collar face (mixed orig+new)
            # Determine which half: any dup vert (>= n_split) → below half.
            is_below = v0 >= n_split or v1 >= n_split or v2 >= n_split

            p0 = final_pos[v0]; p1 = final_pos[v1]; p2 = final_pos[v2]
            fn = np.cross(p1 - p0, p2 - p0)
            dot = float(np.dot(fn, normal))

            if abs(dot) < 1e-12:
                continue   # degenerate face (zero area) — skip

            if is_below:
                # Below half wound surface faces upward (dot > 0).
                if dot < 0:
                    keep.append([v0, v2, v1])   # flip
                else:
                    keep.append(f)
            else:
                # Above half wound surface faces downward (dot < 0).
                if dot > 0:
                    keep.append([v0, v2, v1])   # flip
                else:
                    keep.append(f)

    new_surface = (np.array(keep, dtype=np.uint32)
                   if keep else np.zeros((0, 3), dtype=np.uint32))
    comp.surface_indices = new_surface
    new_normals          = _compute_normals(final_pos, new_surface)
    new_texcoords        = np.zeros((len(final_pos), 2), dtype=np.float32)
    print(f"[Cut] Surface triangles: {len(raw_surface):,} raw → "
          f"{len(new_surface):,} after filtering")

    # --- Reallocate GPU buffers ---
    flat_pos  = final_pos.flatten().astype(np.float32)
    flat_norm = new_normals.flatten().astype(np.float32)
    flat_tex  = new_texcoords.flatten().astype(np.float32)
    flat_idx  = new_surface.flatten().astype(np.uint32)

    gl.glBindVertexArray(mesh_comp.render_pipeline)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[0])
    gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_pos.nbytes,  flat_pos,  gl.GL_DYNAMIC_DRAW)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[1])
    gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_norm.nbytes, flat_norm, gl.GL_DYNAMIC_DRAW)

    if len(mesh_comp.buffers) > 2:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[2])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_tex.nbytes, flat_tex, gl.GL_DYNAMIC_DRAW)

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh_comp.index_buffer)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, flat_idx.nbytes, flat_idx, gl.GL_DYNAMIC_DRAW)

    gl.glBindVertexArray(0)

    mesh_comp.indices = new_surface
    print("[Cut] GPU buffers updated.")


# ---------------------------------------------------------------------------
# Shared helpers  (also imported by test files)
# ---------------------------------------------------------------------------

def _build_springs(vertices: np.ndarray,
                   tetrahedra: np.ndarray) -> tuple:
    """
    Extract unique edges from a tetrahedral mesh and compute rest lengths.

    Returns (spring_a, spring_b, rest_lengths) as int32/float32 numpy arrays.
    """
    # Each tet has 6 edges: all pairs from its 4 vertex indices.
    edge_pairs = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    # tetrahedra[:, edge_pairs] → (N_tets, 6, 2), reshape → (N_tets*6, 2)
    all_edges = tetrahedra[:, edge_pairs].reshape(-1, 2)
    all_edges = np.sort(all_edges, axis=1)          # canonical (low, high) order
    unique_edges = np.unique(all_edges, axis=0)     # deduplicate shared edges

    sa   = unique_edges[:, 0].astype(np.int32)
    sb   = unique_edges[:, 1].astype(np.int32)
    rest = np.linalg.norm(
        vertices[sb] - vertices[sa], axis=1
    ).astype(np.float32)
    return sa, sb, rest


def _extract_boundary_faces(tetrahedra: np.ndarray,
                             vertices: np.ndarray = None) -> np.ndarray:
    """
    Return the boundary (outer surface) triangles of a tetrahedral mesh.

    A face is on the boundary iff it belongs to exactly one tetrahedron.
    When `vertices` is supplied, face winding is corrected so normals
    point outward (required for correct backface culling).

    Returns np.ndarray of shape (N_faces, 3), dtype uint32.
    """
    face_combos   = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
    all_faces     = tetrahedra[:, face_combos].reshape(-1, 3)
    all_faces_sorted = np.sort(all_faces, axis=1)

    _, inverse, counts = np.unique(
        all_faces_sorted, axis=0, return_inverse=True, return_counts=True)

    boundary_mask  = (counts == 1)[inverse]
    boundary_faces = all_faces[boundary_mask].copy()

    if vertices is not None and len(boundary_faces) > 0:
        boundary_pos  = np.where(boundary_mask)[0]
        tet_indices   = boundary_pos // 4

        face_verts = all_faces[boundary_pos]
        tet_verts  = tetrahedra[tet_indices]

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


def _compute_normals(vertices: np.ndarray,
                     indices:  np.ndarray) -> np.ndarray:
    """
    Compute per-vertex normals on the GPU via Taichi.

    Replaces the previous np.add.at implementation which was the single
    biggest CPU bottleneck (~50 ms on the bunny mesh).
    """
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
