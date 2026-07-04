"""
Ctrl+click debug picker for the Taichi simulation.

Ray-casts from the cursor into the scene, finds the closest rendered
triangle, then dumps everything we know about that face, the tets that
share those 3 verts, and each vertex's live state (position, velocity,
mass, fixed/orphan flags, cutting springs). Also highlights the picked
face and the rest of its parent tet(s) in the mesh's per-face color
buffer if --debug-colors is enabled.

Usage:
  Ctrl + Left click : pick nearest triangle, print + highlight
  Ctrl + Right click: clear highlight

The picker reads live positions from comp.method.get_positions() and
walks numpy arrays; it never touches Taichi kernels.
"""

import numpy as np
import glm
import OpenGL.GL as gl


# ---------------------------------------------------------------------------
# Ray-triangle math
# ---------------------------------------------------------------------------

def _build_ray(view, projection, cam_pos, cursor_x, cursor_y, win_w, win_h):
    """Build a world-space ray from a cursor pixel.

    view, projection : glm.mat4
    cam_pos          : np.ndarray shape (3,)
    cursor_x, cursor_y : float, pixel coords (GLFW convention: (0,0) top-left)
    win_w, win_h     : int

    Returns (origin, direction) as np.ndarray shape (3,), direction normalised.
    """
    ndc_x = (2.0 * cursor_x / win_w) - 1.0
    ndc_y = 1.0 - (2.0 * cursor_y / win_h)   # GL flips Y

    inv_proj = glm.inverse(projection)
    inv_view = glm.inverse(view)

    # ray in view space: pick a point on the near plane (z = -1 in NDC)
    clip_p = glm.vec4(ndc_x, ndc_y, -1.0, 1.0)
    eye_p  = inv_proj * clip_p
    eye_dir = glm.vec4(eye_p.x, eye_p.y, -1.0, 0.0)  # forward in view space

    world_dir = inv_view * eye_dir
    dir_np = np.array([world_dir.x, world_dir.y, world_dir.z], dtype=np.float32)
    dir_np /= np.linalg.norm(dir_np)

    return cam_pos.astype(np.float32), dir_np


def _ray_intersect_all(origin, direction, positions, faces):
    """Vectorised Moller-Trumbore across all triangles.

    origin    : (3,) float32
    direction : (3,) float32
    positions : (N, 3) float32
    faces     : (F, 3) int (uint32/int32/int64)

    Returns (best_face_idx, best_t) or (None, +inf).
    """
    if len(faces) == 0:
        return None, float('inf')

    v0 = positions[faces[:, 0]]
    v1 = positions[faces[:, 1]]
    v2 = positions[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0
    h  = np.cross(direction, e2)
    a  = np.einsum('ij,ij->i', e1, h)

    valid = np.abs(a) > 1e-8
    f     = np.where(valid, 1.0 / np.where(valid, a, 1.0), 0.0)
    s     = origin - v0
    u     = f * np.einsum('ij,ij->i', s, h)
    q     = np.cross(s, e1)
    v     = f * np.einsum('j,ij->i', direction, q)
    t     = f * np.einsum('ij,ij->i', e2, q)

    hit = valid & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-4)
    if not hit.any():
        return None, float('inf')

    t_masked = np.where(hit, t, np.inf)
    best     = int(np.argmin(t_masked))
    return best, float(t_masked[best])


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def _classify_vert(v, n_orig, n_split, n_phys):
    if n_orig is not None and v < n_orig:
        return "ORIG"
    if n_split is not None and v < n_split:
        return "INTER"
    if n_phys is None or v < n_phys:
        return "DUP"
    return "RENDER_DUP"


def _find_tets_containing(face_verts, tets):
    """Return tet indices whose 4 verts include all 3 face verts (unordered)."""
    fv = set(int(v) for v in face_verts)
    if len(fv) != 3:
        return []
    hits = []
    for i, t in enumerate(tets):
        tv = set(int(v) for v in t)
        if fv.issubset(tv):
            hits.append(int(i))
    return hits


def _find_tets_containing_vectorised(face_verts, tets_np):
    """Fast path when tets is a (T, 4) numpy array."""
    fv0, fv1, fv2 = int(face_verts[0]), int(face_verts[1]), int(face_verts[2])
    has0 = (tets_np == fv0).any(axis=1)
    has1 = (tets_np == fv1).any(axis=1)
    has2 = (tets_np == fv2).any(axis=1)
    idx  = np.where(has0 & has1 & has2)[0]
    return [int(i) for i in idx]


def _tet_volume(vert_positions_4x3):
    p = vert_positions_4x3
    return float(abs(np.linalg.det(np.stack(
        [p[1] - p[0], p[2] - p[0], p[3] - p[0]]))) / 6.0)


# ---------------------------------------------------------------------------
# Main pick + dump
# ---------------------------------------------------------------------------

def pick_and_dump(comp, mesh_comp, camera_comp, camera_transform,
                  cursor_x, cursor_y, win_w, win_h):
    """Run the whole pipeline. Prints to stdout. Returns hit face info or None."""
    if comp.method is None or comp.simulator is None:
        print("[Pick] no simulator yet")
        return None

    # ---- Camera ray ---------------------------------------------------------
    cam_pos = np.array([camera_transform.translation.x,
                        camera_transform.translation.y,
                        camera_transform.translation.z], dtype=np.float32)
    origin, direction = _build_ray(
        camera_comp.view, camera_comp.projection,
        cam_pos, cursor_x, cursor_y, win_w, win_h)

    # ---- Rendered vertex positions (live) -----------------------------------
    pos_active = comp.method.get_positions()  # (n_active, 3)
    if (comp._disc_split_phys_idx is not None
            and len(comp._disc_split_phys_idx) > 0):
        render_pos = np.vstack([pos_active,
                                 pos_active[comp._disc_split_phys_idx]])
    else:
        render_pos = pos_active

    # ---- Currently-rendered faces (respect wound reveal + T-key state) -----
    faces_to_test = _current_rendered_faces(comp)
    if faces_to_test is None or len(faces_to_test) == 0:
        print("[Pick] no rendered faces")
        return None

    # ---- Ray-triangle -------------------------------------------------------
    face_row, t_hit = _ray_intersect_all(origin, direction, render_pos,
                                          faces_to_test)
    if face_row is None:
        print("[Pick] ray missed all rendered faces")
        return None
    face_verts = faces_to_test[face_row]
    hit_point  = origin + direction * t_hit
    face_bucket, face_bucket_idx = _classify_face_bucket(comp, face_row)

    # ---- Face info ----------------------------------------------------------
    n_orig  = comp._n_orig
    n_split = comp._n_split
    n_phys  = pos_active.shape[0]
    v_cls = [_classify_vert(int(v), n_orig, n_split, n_phys) for v in face_verts]
    print(f"\n[Pick] ------------------------------------------------------")
    print(f"[Pick] cursor=({cursor_x:.0f}, {cursor_y:.0f})  "
          f"hit at t={t_hit:.4f}  world=({hit_point[0]:+.4f}, "
          f"{hit_point[1]:+.4f}, {hit_point[2]:+.4f})")
    print(f"[Pick] Face: bucket={face_bucket} local_idx={face_bucket_idx} "
          f"verts={[int(v) for v in face_verts]} classes={v_cls}")

    # dist_to_plane per vertex (if cut is active)
    if comp._cut_normal is not None:
        cut_origin = np.array(comp.cut_plane_origin, dtype=np.float32)
        cut_normal = np.array(comp._cut_normal,        dtype=np.float32)
        for v, cls in zip(face_verts, v_cls):
            p = render_pos[int(v)]
            d = float(np.dot(p - cut_origin, cut_normal))
            print(f"[Pick]   vert {int(v):>6d} [{cls:5}]  "
                  f"pos=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})  "
                  f"dist_to_plane={d:+.4f}")

    # ---- Parent tets --------------------------------------------------------
    tets_np = comp.simulator._tets.to_numpy()[:comp.simulator.n_tets]
    parent_tets = _find_tets_containing_vectorised(face_verts, tets_np)
    print(f"[Pick] Parent tets ({len(parent_tets)}): {parent_tets}")
    for ti in parent_tets:
        tv = tets_np[ti]
        tp = pos_active[tv]
        vol = _tet_volume(tp)
        tv_cls = [_classify_vert(int(v), n_orig, n_split, n_phys) for v in tv]
        print(f"[Pick]   tet #{ti}: verts={tv.tolist()} classes={tv_cls} vol={vol:.6e}")

    # ---- Per-vertex live state ---------------------------------------------
    _dump_vertex_states(comp, list(set(int(v) for v in face_verts)
                                    | {int(v) for ti in parent_tets
                                       for v in tets_np[ti]}),
                        n_orig=n_orig, n_split=n_split)

    # ---- Highlight if debug-colors are enabled ------------------------------
    _apply_highlight(comp, mesh_comp, face_row, face_verts, parent_tets, tets_np)

    return {
        'face_row':  face_row,
        'face_verts': [int(v) for v in face_verts],
        'parent_tets': parent_tets,
        'hit_point': hit_point.tolist(),
    }


# ---------------------------------------------------------------------------
# Sub-helpers
# ---------------------------------------------------------------------------

def _current_rendered_faces(comp):
    """Faces currently visible: outer + revealed wound (respecting hide_wound_faces
    and _wound_face_ptr). Returned as a (F, 3) int32 array. Falls back to the
    pre-cut surface when no cut has happened yet."""
    if comp._outer_faces is None:
        if getattr(comp, 'surface_indices', None) is not None:
            return np.asarray(comp.surface_indices, dtype=np.int64)
        return None
    outer = np.asarray(comp._outer_faces, dtype=np.int64)
    if getattr(comp, 'hide_wound_faces', False):
        return outer
    n_revealed = int(getattr(comp, '_wound_face_ptr', 0))
    if n_revealed <= 0 or getattr(comp, '_all_render_faces', None) is None:
        return outer
    # _all_render_faces holds outer followed by wound (in reveal order).
    all_faces = np.asarray(comp._all_render_faces, dtype=np.int64)
    return all_faces[: len(outer) + n_revealed]


def _classify_face_bucket(comp, face_row):
    """Return ('OUTER', local_idx) or ('WOUND', local_idx) based on face_row."""
    n_outer = int(getattr(comp, '_n_outer_faces', 0))
    if n_outer == 0 and comp._outer_faces is not None:
        n_outer = len(comp._outer_faces)
    if face_row < n_outer:
        return "OUTER", int(face_row)
    return "WOUND", int(face_row - n_outer)


def _dump_vertex_states(comp, vert_indices, n_orig=None, n_split=None):
    sim = comp.simulator
    pos = comp.method.get_positions()
    vel = comp.method.get_velocities()
    cut_normal = None
    cut_origin = None
    if comp._cut_normal is not None:
        cut_normal = np.asarray(comp._cut_normal, dtype=np.float32)
        cut_origin = np.asarray(comp.cut_plane_origin, dtype=np.float32)
    try:
        mass = sim._masses.to_numpy()[:sim.n_verts]
    except Exception:
        mass = None
    try:
        fixed = sim._fixed.to_numpy()[:sim.n_verts]
    except Exception:
        fixed = None

    # Springs: dict from vert -> list of (spring_idx, partner, k, r, current_len)
    sa = getattr(sim, '_sa', None)
    sb = getattr(sim, '_sb', None)
    sr = getattr(sim, '_sr', None)
    sk = getattr(sim, '_sk', None)

    orphan_map = getattr(comp.method, '_orphan_constraints', {}) or {}

    print(f"[Pick] Per-vertex state:")
    for v in sorted(vert_indices):
        if v >= sim.n_verts:
            print(f"[Pick]   vert {v}: outside active range (render-only dup?)")
            continue
        p = pos[v]; vv = vel[v]
        m = float(mass[v]) if mass is not None else float('nan')
        fx = int(fixed[v]) if fixed is not None else -1
        is_orphan = v in orphan_map
        orphan_str = ""
        if is_orphan:
            master, offset = orphan_map[v]
            orphan_str = f" ORPHAN(master={master}, offset=({offset[0]:+.4f},{offset[1]:+.4f},{offset[2]:+.4f}))"
        print(f"[Pick]   vert {v:>6d}: "
              f"pos=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}) "
              f"vel=({vv[0]:+.4f}, {vv[1]:+.4f}, {vv[2]:+.4f}) "
              f"mass={m:.4e} fixed={fx}{orphan_str}")

        # Half-consistency diagnostic for orphans
        if is_orphan and cut_normal is not None and n_orig is not None and n_split is not None:
            master, _ = orphan_map[v]
            d_orph = float(np.dot(p - cut_origin, cut_normal))
            mp = pos[master] if master < sim.n_verts else None
            if mp is None:
                print(f"[Pick]     >> master {master} out of range")
            else:
                d_mast = float(np.dot(mp - cut_origin, cut_normal))
                # Setup-time label per matcher's rule (INTER=+1, DUP=-1, ORIG=sign(pos.n))
                if n_orig <= v < n_split:
                    orph_label = +1
                    label_reason = "INTER"
                elif v >= n_split:
                    orph_label = -1
                    label_reason = "DUP"
                else:
                    orph_label = int(np.sign(d_orph)) if abs(d_orph) > 1e-6 else 0
                    label_reason = f"ORIG(sign(pos.n)={d_orph:+.4f})"
                # Master's current-side (live position)
                mast_side = int(np.sign(d_mast)) if abs(d_mast) > 1e-6 else 0
                # Verdict
                if orph_label == 0:
                    verdict = "ON-PLANE (fallback)"
                elif mast_side == 0:
                    verdict = "MASTER-ON-PLANE"
                elif orph_label == mast_side:
                    verdict = "AGREE"
                else:
                    verdict = "*** DISAGREE ***"
                print(f"[Pick]     >> half-check: orph_label={orph_label:+d} "
                      f"({label_reason})  master_side={mast_side:+d} "
                      f"(d_mast={d_mast:+.4f})  d_orph={d_orph:+.4f}  {verdict}")

        # Cutting springs involving this vert
        if sa is not None and len(sa) > 0:
            hits = np.where((sa == v) | (sb == v))[0]
            for si in hits:
                a = int(sa[si]); b = int(sb[si])
                other = b if a == v else a
                op = pos[other] if other < sim.n_verts else None
                if op is not None:
                    cur = float(np.linalg.norm(p - op))
                    print(f"[Pick]     spring #{int(si)}: partner={other} "
                          f"k={float(sk[si]):.3f} rest={float(sr[si]):.6f} "
                          f"current_len={cur:.6f}")


# ---------------------------------------------------------------------------
# Highlight (per-face color override)
# ---------------------------------------------------------------------------

_PICK_COLOR_FACE = np.array([1.0, 0.15, 0.9], dtype=np.float32)   # magenta
_PICK_COLOR_TET  = np.array([1.0, 0.85, 0.0], dtype=np.float32)   # yellow


def _apply_highlight(comp, mesh_comp, face_row, face_verts, parent_tets, tets_np):
    """Override the color VBO to highlight picked face + parent tet faces.
    Only works when the mesh has a color attribute (--debug-colors mode)."""
    if not getattr(comp, '_face_expanded', False):
        return
    if len(mesh_comp.buffers) <= 3:
        return  # no color buffer
    if comp._debug_colors is None:
        return

    # Rebuild the per-face color array from the debug baseline and stamp the
    # picked face + parent-tet faces.
    face_colors = comp._debug_colors.copy()

    # Highlight the exact picked face
    if face_row < len(face_colors):
        face_colors[face_row] = _PICK_COLOR_FACE

    # Highlight all outer faces that share a full 3-vert subset with the parent tets.
    if parent_tets and comp._outer_faces is not None:
        parent_vert_sets = [set(int(v) for v in tets_np[ti]) for ti in parent_tets]
        outer = np.asarray(comp._outer_faces, dtype=np.int64)
        for i, f in enumerate(outer):
            fv = set(int(v) for v in f)
            for pv in parent_vert_sets:
                if fv.issubset(pv) and i != face_row:
                    face_colors[i] = _PICK_COLOR_TET
                    break

    n_render = min(len(face_colors) * 3,
                    _current_expanded_vertex_count(comp, mesh_comp))
    exp_colors = np.repeat(face_colors, 3, axis=0).astype(np.float32)
    gl.glBindVertexArray(mesh_comp.render_pipeline)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[3])
    gl.glBufferData(gl.GL_ARRAY_BUFFER, exp_colors.nbytes, exp_colors,
                    gl.GL_DYNAMIC_DRAW)
    gl.glBindVertexArray(0)
    comp._pick_highlight_active = True


def _current_expanded_vertex_count(comp, mesh_comp):
    if comp._debug_colors is None:
        return 0
    return len(comp._debug_colors) * 3


def clear_highlight(comp, mesh_comp):
    """Reset colors to the debug baseline."""
    if not getattr(comp, '_pick_highlight_active', False):
        return
    if not getattr(comp, '_face_expanded', False):
        return
    if len(mesh_comp.buffers) <= 3:
        return
    if comp._debug_colors is None:
        return
    face_colors = comp._debug_colors.copy()
    exp_colors  = np.repeat(face_colors, 3, axis=0).astype(np.float32)
    gl.glBindVertexArray(mesh_comp.render_pipeline)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[3])
    gl.glBufferData(gl.GL_ARRAY_BUFFER, exp_colors.nbytes, exp_colors,
                    gl.GL_DYNAMIC_DRAW)
    gl.glBindVertexArray(0)
    comp._pick_highlight_active = False
    print("[Pick] highlight cleared")
