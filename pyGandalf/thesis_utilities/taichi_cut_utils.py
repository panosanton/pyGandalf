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
  (stiffness -> 0) and the wound surface is progressively revealed.

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

import os
import sys

import taichi as ti
import numpy as np

import OpenGL.GL as gl

# kernel_profiler adds non-trivial per-launch overhead. Gate behind the
# --profile-kernels CLI flag so interactive runs aren't slowed down by default.
# ti.init runs at import time, before argparse, so we sniff sys.argv here.
_PROFILE_KERNELS = "--profile-kernels" in sys.argv
ti.init(arch=ti.gpu, log_level=ti.WARN,
        kernel_profiler=_PROFILE_KERNELS, offline_cache=True)


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
# One-shot cut helpers
# ---------------------------------------------------------------------------

def _split_crossed_tets(tetrahedra: np.ndarray,
                         positions:  np.ndarray,
                         signed_dist: np.ndarray,
                         surface_tet_indices: set = None):
    """
    Split tetrahedra that straddle the cut plane by inserting new vertices
    exactly at edge-plane intersections.

    Handles:
      1+3 (1 above, 3 below): 1 above-tet + 3-tet prism for below
      3+1 (3 above, 1 below): symmetric
      2+2 (2 above, 2 below): 3 tets on each side using quad diagonal

    surface_tet_indices: set of original tet indices that have at least one
      face in the original outer surface.  When provided, collar faces from
      crossing tets NOT in this set are collected in phantom_above_face_keys
      (sorted 3-tuples) so the caller can mark them for debugging / filtering.

    Returns:
        new_positions            (N_new, 3)
        above_tets               (M, 4)
        below_tets               (K, 4)
        inter_data               list of (new_idx, vi, vj, t)
        phantom_above_face_keys  set of sorted int-triple keys for collar faces
                                 originating from non-surface crossing tets
    """
    ext_positions: list = [p for p in positions]
    edge_cache:    dict = {}
    inter_data:    list = []
    above_list:    list = []   # only crossing-tet sub-tets; non-crossing stitched in below
    below_list:    list = []
    phantom_above_face_keys: set = set()

    tet_dists  = signed_dist[tetrahedra]
    above_mask = np.all(tet_dists >= 0, axis=1)
    below_mask = np.all(tet_dists <= 0, axis=1)

    # Keep non-crossing tets as numpy arrays. The crossing-tet split loop below
    # appends to above_list/below_list; we concatenate at the end. Skipping the
    # tolist() round trip on these (often 100k+ rows) avoids ~100ms.
    above_nocross = tetrahedra[above_mask]
    below_nocross = tetrahedra[below_mask]

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

    def _add_phantom_keys(faces):
        for f in faces:
            phantom_above_face_keys.add(tuple(sorted(f)))

    for idx in np.where(~above_mask & ~below_mask)[0]:
        verts = tetrahedra[idx]
        dists = signed_dist[verts]

        av  = [int(verts[i]) for i in range(4) if dists[i] >= 0]
        bv  = [int(verts[i]) for i in range(4) if dists[i] <  0]
        n_a = len(av)
        is_surface = surface_tet_indices is None or int(idx) in surface_tet_indices

        if n_a == 1:
            a, b0, b1, b2 = av[0], bv[0], bv[1], bv[2]
            p0, p1, p2    = iv(a, b0), iv(a, b1), iv(a, b2)
            above_list.append([a, p0, p1, p2])
            below_list += [[b0, b1, b2, p0],
                           [b1, b2, p0, p1],
                           [b2, p0, p1, p2]]
            if not is_surface:
                _add_phantom_keys([[a,p0,p1],[a,p0,p2],[a,p1,p2]])

        elif n_a == 3:
            a0, a1, a2, b = av[0], av[1], av[2], bv[0]
            p0, p1, p2    = iv(a0, b), iv(a1, b), iv(a2, b)
            below_list.append([b, p0, p1, p2])
            above_list += [[a0, a1, a2, p0],
                           [a1, a2, p0, p1],
                           [a2, p0, p1, p2]]
            if not is_surface:
                _add_phantom_keys([
                    [a0,a1,p0],[a0,a2,p0],
                    [a1,a2,p1],[a1,p0,p1],
                    [a2,p0,p2],[a2,p1,p2],
                ])

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
            if not is_surface:
                _add_phantom_keys([
                    [a0,p00,p01],[a0,p01,p11],
                    [a1,p00,p10],[a1,p10,p11],
                    [a0,a1,p00],[a0,a1,p11],
                ])

    new_pos       = np.array(ext_positions, dtype=np.float32)
    above_cross   = (np.array(above_list, dtype=np.int32)
                     if above_list else np.zeros((0, 4), dtype=np.int32))
    below_cross   = (np.array(below_list, dtype=np.int32)
                     if below_list else np.zeros((0, 4), dtype=np.int32))
    above_arr     = np.vstack([above_nocross.astype(np.int32), above_cross])
    below_arr     = np.vstack([below_nocross.astype(np.int32), below_cross])
    return new_pos, above_arr, below_arr, inter_data, phantom_above_face_keys


def _cut_topology_physics(
        current_tets:    np.ndarray,
        positions:       np.ndarray,
        velocities:      np.ndarray,
        masses:          np.ndarray,
        fixed:           np.ndarray,
        stiffness:       float,
        gravity:         np.ndarray,
        origin:          np.ndarray,
        normal:          np.ndarray,
        build_simulator: bool = True):
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
    # Pack each (a, b, c) into one int64 (21 bits per slot -- safe for any mesh
    # with <2M verts) so unique runs on a 1D array. axis=0 on 2D is ~10x slower.
    assert int(all_orig_s.max()) < (1 << 21), "vertex index too large to pack"
    _keys_o = ((all_orig_s[:, 0].astype(np.int64) << 42)
             | (all_orig_s[:, 1].astype(np.int64) << 21)
             |  all_orig_s[:, 2].astype(np.int64))
    _, _inv, _cnt = np.unique(_keys_o, return_inverse=True, return_counts=True)
    orig_surf_set = {tuple(row) for row in all_orig_s[(_cnt == 1)[_inv]]}
    n_orig = len(positions)

    # --- Surface tet set: original tets with at least one face in orig_surf_set ---
    # Used by _split_crossed_tets to identify phantom collar faces from interior tets.
    # Vectorized: each row of all_orig_s comes from tet (i // 4), so the
    # boundary-face mask directly gives the set of tets with a surface face.
    _bnd_mask        = (_cnt == 1)[_inv]
    surface_tet_set  = set(int(t) for t in np.unique(np.where(_bnd_mask)[0] // 4))
    print(f"[Cut] {len(surface_tet_set)} surface tets, "
          f"{len(current_tets) - len(surface_tet_set)} interior tets")

    # --- Pre-snap near-plane above verts to below ---
    # A vert with 0 < dist < snap_eps is effectively on the cut plane.  If left
    # as above, it becomes the lone above vert in a 1+3 split, producing a
    # near-degenerate above-half tet [vi, p0, p1, p2] where p0~p1~p2~vi.
    # Those near-zero-area outer faces accumulate zero normals for vi, causing
    # the collar lerp in _compute_normals_post_cut to assign zero normals to the
    # intersection verts (Bug 1 saw-tooth artifact).  Snapping dist to just
    # below sends the whole tet to the below-half intact, preserving vi's
    # original surface faces and their normals.
    mesh_scale = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    snap_eps   = mesh_scale * 1e-4
    near_above = (signed_dist > 0) & (signed_dist < snap_eps)
    if near_above.any():
        signed_dist_split = signed_dist.copy()
        signed_dist_split[near_above] = -snap_eps
        print(f"[Cut] Pre-snap {int(near_above.sum())} near-plane above verts to below "
              f"(prevents degenerate 1+3 splits, threshold {snap_eps:.2e})")
    else:
        signed_dist_split = signed_dist

    # --- Tet splitting ---
    split_pos, above_tets, below_tets, inter_data, phantom_above_face_keys = \
        _split_crossed_tets(current_tets, positions, signed_dist_split,
                            surface_tet_indices=surface_tet_set)

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
    # mesh_scale and snap_eps already computed above.
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
    # Standard case: INTER verts (idx >= n_orig) appearing in both halves.
    # ORIG verts get classified into exactly one half by pre-snap / signed_dist,
    # so they never appear in the intersection here -- their below-cap variant
    # is handled by the "extra case" block below.
    _shared_all   = np.intersect1d(above_tets.ravel(), below_tets.ravel(),
                                   assume_unique=False)
    shared_list   = [int(v) for v in _shared_all[_shared_all >= n_orig]]

    # Extra case: ORIG verts that the cut plane hits dead on (|signed_dist|
    # within tolerance).  These are the only ORIGs that cause the "spike face"
    # bug -- the plane passes through them, so a below-half boundary face
    # touching them has its "on-plane corner" anchored to a near-plane master
    # that doesn't move with the below half.  Give them an explicit below-half
    # DUP so the face uses that instead.  Deep-below or deep-above ORIGs are
    # unaffected: their masters are deep on their own half and move with it.
    _plane_hit_eps = snap_eps * 10.0
    _plane_hit     = np.where(np.abs(signed_dist) < _plane_hit_eps)[0]
    if len(_plane_hit) > 0:
        _below_verts = np.unique(below_tets.ravel())
        _above_verts = np.unique(above_tets.ravel())
        # Only need dup for plane-hit ORIGs that ended up in below_tets but
        # not above_tets (a below-half DUP is the remap target). Above-only
        # cases don't get corrupted by remap and don't produce below-cap
        # spikes.
        _hit_in_below = _plane_hit[np.isin(_plane_hit, _below_verts)
                                    & ~np.isin(_plane_hit, _above_verts)]
        _extra = _hit_in_below[~np.isin(_hit_in_below, _shared_all)]
        if len(_extra) > 0:
            shared_list = shared_list + [int(v) for v in _extra]
            print(f"[Cut] Extra seam dup: {len(_extra)} plane-hit ORIG verts "
                  f"(|dist| < {_plane_hit_eps:.2e})")
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

    # --- Rebuild simulator (skipped when caller is FEM and only wants topology) ---
    if build_simulator:
        new_sim = _SpringMassSimulator(
            vertices        = final_pos,
            tetrahedra      = all_tets,
            fixed_mask      = final_fixed,
            stiffness       = stiffness,
            gravity         = gravity,
            per_vertex_mass = final_mass,
        )
        new_sim.velocities.from_numpy(final_vel.astype(np.float32))

        # No spring zeroing: orig->new springs are structural tet edges that are
        # needed for mesh connectivity.  Zeroing them disconnects the disc from
        # the hemispheres (Fix 3 mistake — verified by force audit).
        # Collar deformation from opening velocity is physically correct elastic
        # behaviour; reduce opening_speed if it looks too violent.

        print(f"[Cut] Simulator: {len(final_pos):,} verts, "
              f"{len(all_tets):,} tets, {len(new_sim._sa):,} springs")

        _print_spring_audit(new_sim._sa, new_sim._sb, new_sim._sr, new_sim._sk, n_orig)
    else:
        new_sim = None
        print(f"[Cut] Topology only: {len(final_pos):,} verts, "
              f"{len(all_tets):,} tets (no spring-mass simulator built)")

    return (new_sim, final_pos, final_vel, final_mass, final_fixed,
            all_tets, n_orig, n_split, shared_list, remap, inter_data, orig_surf_set,
            phantom_above_face_keys)


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

      With the plane-hit ORIG augmentation in _cut_topology_physics,
      shared_list may include ORIG indices whose position was on the cut
      plane at cut time. Their DUPs also sit on the plane initially, so
      treating them as on-plane matches the geometry.
    """
    seam_set = set(int(v) for v in shared_list) if shared_list else set()

    orig_surf_verts = set()
    if orig_surf_set:
        for face_key in orig_surf_set:
            orig_surf_verts.update(face_key)

    def on_plane(v):
        return v >= n_orig or v in seam_set

    outer = []
    wound = []
    n_phantom = 0

    for f in raw_surface:
        v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
        if on_plane(v0) and on_plane(v1) and on_plane(v2):
            wound.append(f)
        else:
            orig_vs = [v for v in (v0, v1, v2) if v < n_orig and v not in seam_set]
            new_vs  = [v for v in (v0, v1, v2) if on_plane(v)]
            if new_vs and orig_vs and orig_surf_verts:
                if any(v not in orig_surf_verts for v in orig_vs):
                    n_phantom += 1
                    continue   # discard phantom collar face (interior orig vert)
            outer.append(f)

    print(f"[PhantomCollar] discarded {n_phantom} phantom collar faces")

    # [CollarFilter] Reconstruction-based phantom removal.
    # The virtual-node split algorithm inherently produces extra interior faces when
    # adjacent crossing tets land in different split cases and triangulate their shared
    # face differently. We detect them by reconstructing each collar face's pre-cut
    # parent and checking against orig_surf_set.
    # Collar flavours:
    #   above-side: face contains an inter vert in [n_orig, n_split).
    #   below-side: face contains a seam-dup vert in [n_split, n_split + len(shared_list))
    #               (the post-split remap rewrote shared inter verts to dup indices).
    # Map dup vert back to its original inter vert so reconstruction works for both halves.
    if inter_data is not None and orig_surf_set:
        ivert_above  = {int(nid): int(vi) for nid, vi, vj, t in inter_data}
        ivert_below  = {int(nid): int(vj) for nid, vi, vj, t in inter_data}
        dup_to_inter = ({n_split + i: int(shared_list[i]) for i in range(len(shared_list))}
                        if shared_list else {})
        filtered = []
        n_removed_a = 0
        n_removed_b = 0
        for f in outer:
            v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
            has_inter = any(n_orig <= v < n_split for v in (v0, v1, v2))
            has_dup   = any(v >= n_split          for v in (v0, v1, v2))
            if not (has_inter or has_dup):
                filtered.append(f)
                continue
            recon = set()
            for v in (v0, v1, v2):
                if v >= n_split:
                    inter = dup_to_inter.get(v, v)
                    recon.add(ivert_above.get(inter, inter))
                    recon.add(ivert_below.get(inter, inter))
                elif n_orig <= v < n_split:
                    recon.add(ivert_above.get(v, v))
                    recon.add(ivert_below.get(v, v))
                else:
                    recon.add(v)
            if len(recon) != 3 or tuple(sorted(recon)) in orig_surf_set:
                filtered.append(f)
            else:
                if has_dup:
                    n_removed_b += 1
                else:
                    n_removed_a += 1
        print(f"[CollarFilter] removed {n_removed_a} above-collar + "
              f"{n_removed_b} below-collar = {n_removed_a + n_removed_b} extra faces; "
              f"{len(filtered)} outer faces remaining")
        outer = filtered

    outer_arr = (np.array(outer, dtype=np.uint32)
                 if outer else np.zeros((0, 3), dtype=np.uint32))
    return outer_arr, wound


def _split_disc_verts_for_rendering(outer_faces, wound_faces, n_phys, n_orig):
    """
    Create rendering-only duplicate vertices for intersection vertices that appear
    in both disc (wound) faces and collar (outer) faces.

    Without duplication a single vertex index cannot simultaneously carry a
    sphere-surface normal (for collar shading) and a +-cut_normal (for disc
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


# ---------------------------------------------------------------------------
# Diagnostic helpers (Analysis 1 + 2)
# ---------------------------------------------------------------------------

def _print_spring_audit(sa: np.ndarray, sb: np.ndarray,
                        sr: np.ndarray, sk: np.ndarray,
                        n_orig: int):
    """
    Analysis 2 — rest-length distribution of orig->new springs after Fix 2.

    Prints how many such springs exist, their rest-length histogram, and how
    many were left active (not zeroed by Fix 2).  Run immediately after
    _cut_topology so the state reflects the fix that was applied.
    """
    orig_to_new = ((sa < n_orig) & (sb >= n_orig)) | ((sb < n_orig) & (sa >= n_orig))
    count = int(orig_to_new.sum())
    print(f"[SpringAudit] orig->new springs total: {count}")
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
    forces from the compressed/stretched orig->new springs are visible.
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

    # Pack each (a, b, c) into one int64 (21 bits per slot) so unique runs on a
    # 1D array. axis=0 on 2D is ~10x slower (its argsort dominates the profile).
    assert int(all_faces_sorted.max()) < (1 << 21), "vertex index too large to pack"
    _keys = ((all_faces_sorted[:, 0].astype(np.int64) << 42)
           | (all_faces_sorted[:, 1].astype(np.int64) << 21)
           |  all_faces_sorted[:, 2].astype(np.int64))
    _, inverse, counts = np.unique(_keys, return_inverse=True, return_counts=True)

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

        # [WindingDiag] Re-check after correction: any face still pointing toward
        # its fourth vertex has genuinely wrong winding (not centroid-test noise).
        v0c     = vertices[boundary_faces[:, 0]]
        v1c     = vertices[boundary_faces[:, 1]]
        v2c     = vertices[boundary_faces[:, 2]]
        fn_post = np.cross(v1c - v0c, v2c - v0c)
        still_inward = np.einsum('ij,ij->i', fn_post, to_fourth) > 0
        n_still = int(still_inward.sum())
        if n_still:
            print(f"[WindingDiag] {n_still} / {len(boundary_faces)} boundary faces still "
                  f"pointing toward their fourth vertex after winding correction")
        else:
            print(f"[WindingDiag] 0 / {len(boundary_faces)} boundary faces have residual "
                  f"winding errors after correction")

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
        # Near-plane intersection verts (t~0 or t~1) were reclassified from
        # wound to collar.  They appear in all-intersection-vert collar faces,
        # which triggers disc_mask=True.  Exclude them so they go through the
        # collar lerp path and get surface normals instead of +-cut_n.
        if inter_data:
            near_ids = np.array([d[0] for d in inter_data
                                 if d[3] < 0.01 or d[3] > 0.99], dtype=np.int32)
            if len(near_ids) > 0:
                disc_arr = disc_arr[~np.isin(disc_arr, near_ids)]
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

    # Snap disc vertex normals to +-cut_n (vectorized).
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


def _finalize_wound_slot_normals(exp_norm:      np.ndarray,
                                  exp_pos:       np.ndarray,
                                  n_wound_faces: int,
                                  n_outer_faces: int,
                                  cut_normal:    np.ndarray) -> np.ndarray:
    """
    Override normals for wound-face slots with the face-oriented +-cut_n.

    The per-vertex normal pipeline (`_compute_normals_post_cut`) cannot give
    correct shading for a vertex that belongs to both a wound face and a
    collar face -- one per-vertex slot is broadcast to every face-slot it
    appears in. This finalizer runs in the exploded (per-slot) frame and
    stamps `sign * cut_n` into every wound-face slot without touching
    collar/surface slots.

    exp_norm       : (F*3, 3) already-broadcast per-slot normal buffer.
    exp_pos        : (F*3, 3) matching per-slot position buffer.
    n_wound_faces  : number of wound faces (they sit after n_outer_faces).
    n_outer_faces  : number of outer faces at the start of the buffer.
    cut_normal     : (3,) unit-ish cut-plane normal.

    Modifies exp_norm in-place and returns it.
    """
    if n_wound_faces <= 0:
        return exp_norm
    cut_n = np.asarray(cut_normal, dtype=np.float32)
    cn_norm = float(np.linalg.norm(cut_n))
    if cn_norm < 1e-12:
        return exp_norm
    cut_n = cut_n / cn_norm

    # Wound-face slot IDs: 3 slots per wound face, contiguous after outer.
    start = n_outer_faces * 3
    stop  = start + n_wound_faces * 3
    if stop > len(exp_norm):
        return exp_norm  # buffer smaller than expected -- bail safely.

    # Face-oriented sign per wound face from geometric cross product.
    p0 = exp_pos[start    : stop : 3]
    p1 = exp_pos[start + 1: stop : 3]
    p2 = exp_pos[start + 2: stop : 3]
    face_n = np.cross(p1 - p0, p2 - p0)          # (n_wound, 3)
    dots   = face_n @ cut_n                       # (n_wound,)
    signs  = np.where(dots >= 0.0, 1.0, -1.0).astype(np.float32)
    signed = signs[:, np.newaxis] * cut_n[np.newaxis, :]  # (n_wound, 3)

    # Broadcast each face's signed cut_n across its 3 slots.
    exp_norm[start:stop] = np.repeat(signed, 3, axis=0)
    return exp_norm


def _update_vbo(vao: int, vbo: int, data: np.ndarray):
    """Upload new data into an existing VBO (no reallocation)."""
    flat = data.flatten().astype(np.float32)
    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, flat.nbytes, flat)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    gl.glBindVertexArray(0)


def _compute_debug_face_colors(faces: np.ndarray, n_orig: int) -> np.ndarray:
    """
    Assign one color per face (not per vertex) for crisp unblended debug rendering.
      green (0.3, 0.7, 0.4)  — regular outer face  (all vertices < n_orig)
      blue  (0.15, 0.35, 0.9) — collar face         (mixed: some vertices >= n_orig)
      red   (0.85, 0.1, 0.1)  — disc / wound face   (all vertices >= n_orig)
    Returns (N_faces, 3) float32.
    """
    n = len(faces)
    colors = np.tile([0.3, 0.5, 0.8], (n, 1)).astype(np.float32)   # blue — original surface

    v = faces  # (N, 3)
    all_new = np.all(v >= n_orig, axis=1)
    any_new = np.any(v >= n_orig, axis=1)

    colors[any_new & ~all_new] = [0.3, 0.7, 0.4]    # green — collar
    colors[all_new]             = [0.85, 0.1, 0.1]   # red   — disc

    n_green = int((any_new & ~all_new).sum())
    n_red   = int(all_new.sum())
    print(f"[Cut] Debug colors: {n_green} collar faces (green), {n_red} disc faces (red), "
          f"{n - n_green - n_red} outer faces (blue)")
    return colors


# ---------------------------------------------------------------------------
# Face category color cycling  (V key, requires --debug-colors shader)
# ---------------------------------------------------------------------------

_CAT_COLORS = np.array([
    [1.0, 1.0, 1.0],    # 0 white  — pure surface (all original verts)
    [1.0, 0.55, 0.0],   # 1 orange — valid collar (mixed orig+inter, outward-facing)
    [1.0, 1.0, 0.0],    # 2 yellow — all-intersection outer (reclassified near-plane)
    [1.0, 0.0, 0.5],    # 3 pink   — Cat3 reconstruction phantom (over-filters 1+3 collar)
    [0.0, 1.0, 0.5],    # 4 cyan   — tet-provenance phantom (parent crossing tet is interior)
], dtype=np.float32)

_CAT_NAMES = [
    "pure-surface outer (white)  — all-original verts",
    "valid collar outer (orange) — reconstructed original face is in orig_surf_set",
    "all-inter outer    (yellow) — all intersection verts",
    "phantom collar     (pink)   — reconstructed original face not in orig_surf_set (over-filters)",
    "tet-provenance phantom (cyan) — parent crossing tet has no face in orig_surf_set",
]


def _compute_face_categories(all_render_faces: np.ndarray, n_outer: int,
                              n_orig: int, n_split: int,
                              inter_data=None, orig_surf_set=None,
                              phantom_keys=None) -> np.ndarray:
    """
    Classify each outer face into one of 4 categories.
    Wound faces (beyond n_outer) are left as 0 -- they are hidden.

    Cat 3 (phantom): a collar face whose reconstructed original tet face is not
    in orig_surf_set.  Reconstruction: union of (face orig verts) + (both
    endpoints of every intersection edge in the face), keeping only orig verts
    (< n_orig).  If the result is not exactly 3 orig verts or those 3 are not
    in orig_surf_set, the face came from an interior tet face, not a surface
    one -- it is a phantom exposed by a non-conforming split.
    """
    cats = np.zeros(len(all_render_faces), dtype=np.int32)
    arr  = all_render_faces.astype(np.int64)    # (N, 3)
    vmax = arr.max(axis=1)
    vmin = arr.min(axis=1)

    outer_mask = np.zeros(len(all_render_faces), dtype=bool)
    outer_mask[:n_outer] = True

    cats[outer_mask & (vmax < n_orig)] = 0
    cats[outer_mask & (vmax >= n_orig) & (vmin < n_orig)] = 1
    cats[outer_mask & (vmin >= n_orig)] = 2

    # Cat 3: phantom collar -- cat 1 face whose reconstructed original tet face
    # is not in orig_surf_set.
    if inter_data is not None and orig_surf_set is not None:
        inter_edge = {int(d[0]): (int(d[1]), int(d[2])) for d in inter_data}
        cat1_idx   = np.where(cats == 1)[0]
        for idx in cat1_idx:
            f        = all_render_faces[idx]
            all_orig = set()
            for v in f:
                v = int(v)
                if v < n_orig:
                    all_orig.add(v)
                elif v < n_split:
                    va, vb = inter_edge.get(v, (None, None))
                    if va is not None:
                        all_orig.add(va)
                        all_orig.add(vb)
            # Only keep original verts (below verts from inter edges are orig too)
            all_orig = {v for v in all_orig if v < n_orig}
            if len(all_orig) != 3 or tuple(sorted(all_orig)) not in orig_surf_set:
                cats[idx] = 3

    # Cat 4: tet-provenance phantom — parent crossing tet is an interior tet.
    if phantom_keys is not None:
        for idx in range(n_outer):
            f   = all_render_faces[idx]
            key = tuple(sorted([int(f[0]), int(f[1]), int(f[2])]))
            if key in phantom_keys:
                cats[idx] = 4

    return cats


def _build_springs(vertices: np.ndarray, tetrahedra: np.ndarray) -> tuple:
    """Extract unique edges and compute rest lengths."""
    edge_pairs = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    all_edges  = tetrahedra[:, edge_pairs].reshape(-1, 2)
    all_edges  = np.sort(all_edges, axis=1)

    # Pack each (lo, hi) pair into a single int64 so np.unique runs on a 1D
    # array. axis=0 on 2D is a known slow path; the packed form is 5-10x faster.
    lo = all_edges[:, 0].astype(np.int64)
    hi = all_edges[:, 1].astype(np.int64)
    keys = (lo << 32) | hi
    unique_keys = np.unique(keys)
    sa = (unique_keys >> 32).astype(np.int32)
    sb = (unique_keys & 0xFFFFFFFF).astype(np.int32)

    rest = np.linalg.norm(
        vertices[sb] - vertices[sa], axis=1
    ).astype(np.float32)
    return sa, sb, rest
