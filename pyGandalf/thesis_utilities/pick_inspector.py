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

    # Piece membership (connected component of the end-state tets, computed in
    # _build_sliver_and_orphan_data). A drawn face whose verts sit on different
    # pieces is a render-side bridge: no tet can span pieces.
    _plab = getattr(comp.method, '_piece_lab', None)
    if _plab is not None:
        _n_act = pos_active.shape[0]
        _dsi = comp._disc_split_phys_idx
        _phys = []
        for v in face_verts:
            v = int(v)
            if v < _n_act:
                _phys.append(v)
            elif _dsi is not None and v - _n_act < len(_dsi):
                _phys.append(int(_dsi[v - _n_act]))
            else:
                _phys.append(-1)
        _labs = [int(_plab[p]) if 0 <= p < len(_plab) else -1 for p in _phys]
        _span = len(set(_labs)) > 1
        _map = "" if _phys == [int(v) for v in face_verts] else f"  render->phys {_phys}"
        print(f"[Piece] face verts on pieces {_labs}{_map}"
              + ("  <-- SPANS PIECES (render-side bridge)" if _span else "  (single piece)"))
        _glab = getattr(comp.method, '_piece_lab_good', None)
        if _glab is not None:
            _gl = [int(_glab[p]) if 0 <= p < len(_glab) else -1 for p in _phys]
            _gset = set(x for x in _gl if x >= 0)
            print(f"[Piece] good-graph (active tets only) pieces {_gl}  (-1 = no active tet)"
                  + ("  <-- verts in DIFFERENT good pieces: slivers lump them" if len(_gset) > 1 else ""))
        # Orphan masters of these verts: same piece?
        _oid = getattr(comp.method, '_orphan_ids', None)
        _oma = getattr(comp.method, '_orphan_masters', None)
        if _oid is not None and len(_oid) > 0:
            _om = {int(_oid[i]): int(_oma[i]) for i in range(len(_oid))}
            for p in _phys:
                if p in _om:
                    m = _om[p]
                    print(f"[Piece]   vert {p} ORPHAN -> master {m}: pieces {int(_plab[p])} -> {int(_plab[m])}"
                          + ("  <-- CROSS-PIECE MASTER" if _plab[p] != _plab[m] else ""))

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

    # Per-face-vertex normals (from the VBO — what the shader actually samples).
    _dump_picked_face_normals(comp, mesh_comp, face_row, face_verts)

    # ---- Parent tets --------------------------------------------------------
    tets_np = comp.simulator._tets.to_numpy()[:comp.simulator.n_tets]
    parent_tets = _find_tets_containing_vectorised(face_verts, tets_np)
    print(f"[Pick] Parent tets ({len(parent_tets)}): {parent_tets}")

    # FEM activation + sliver bookkeeping for each parent slot.
    method  = comp.method
    W_np      = comp.simulator._W.to_numpy()[:comp.simulator.n_tets]
    Wrest_np  = comp.simulator._W_rest.to_numpy()[:comp.simulator.n_tets]
    sliver_set = set()
    _ss = getattr(method, '_sliver_slots', None)
    if _ss is not None and len(_ss) > 0:
        sliver_set = set(int(s) for s in _ss)
    full_parent   = getattr(method, '_full_all_tets_parent', None)
    crossing_mask = getattr(method, '_crossing_mask', None)
    n_full_tets   = (len(method._full_all_tets)
                     if getattr(method, '_full_all_tets', None) is not None else None)
    crossing_idx  = (np.where(crossing_mask)[0]
                     if crossing_mask is not None else None)

    for ti in parent_tets:
        tv = tets_np[ti]
        tp = pos_active[tv]
        vol = _tet_volume(tp)
        tv_cls = [_classify_vert(int(v), n_orig, n_split, n_phys) for v in tv]
        w  = float(W_np[ti])
        wr = float(Wrest_np[ti])
        if w > 0.0:
            state = "ACTIVE"
        elif wr > 0.0:
            state = "INERT(activatable)"
        else:
            state = "INERT(permanent)"
        flag_str = " [SLIVER]" if ti in sliver_set else ""
        parent_str = ""
        if n_full_tets is not None:
            if ti < n_full_tets and full_parent is not None:
                parent_str = (f" parent_pristine={int(full_parent[ti])}"
                              f"(split-block)")
            elif crossing_idx is not None and ti - n_full_tets < len(crossing_idx):
                parent_str = (f" parent_pristine={int(crossing_idx[ti - n_full_tets])}"
                              f"(unsplit-block)")
        print(f"[Pick]   tet #{ti}: verts={tv.tolist()} classes={tv_cls} "
              f"vol={vol:.6e} W={w:.3e} W_rest={wr:.3e} {state}{flag_str}{parent_str}")

    # ---- Bake history (why is this face in/out of the carried set?) --------
    _dump_bake_history(method, face_verts, n_orig)

    # ---- Old-wound flat-shade flag (why is this face still smooth?) ---------
    # Direct readout: is THIS face flagged as previous-cut wound (so the render
    # would flat-shade it under --flat-old-wound)? Checks both the carried set
    # (baked-index triples) and the render cache's face_is_old_wound mask.
    try:
        _tri = tuple(sorted(int(v) for v in face_verts))
        _cws = getattr(method, '_carried_wound_set', None)
        _in_carried = (_cws is not None) and (_tri in _cws)
        _flag_on = bool(getattr(method, '_flat_old_wound', False))
        _cache = getattr(method, '_precomp_render_cache', None)
        _in_cache_mask = None
        if _cache is not None and _cache.get('face_is_old_wound') is not None:
            _fv_cache = _cache['face_verts']
            _fkeys = np.sort(_fv_cache, axis=1)
            _hit = np.where((_fkeys[:, 0] == _tri[0]) & (_fkeys[:, 1] == _tri[1])
                            & (_fkeys[:, 2] == _tri[2]))[0]
            if len(_hit) > 0:
                _in_cache_mask = bool(_cache['face_is_old_wound'][_hit].any())
            else:
                _in_cache_mask = "face-not-in-cache-table"
        print(f"[OldWound] flat_old_wound_flag={_flag_on}  "
              f"in_carried_wound_set={_in_carried}  "
              f"face_is_old_wound(cache)={_in_cache_mask}  "
              f"carried_set_size={0 if _cws is None else len(_cws):,}")
    except Exception as _e:
        print(f"[OldWound] check skipped ({_e})")

    # ---- Per-vertex live state ---------------------------------------------
    _dump_vertex_states(comp, list(set(int(v) for v in face_verts)
                                    | {int(v) for ti in parent_tets
                                       for v in tets_np[ti]}),
                        n_orig=n_orig, n_split=n_split)

    # ---- Genealogy: current ids -> ids in every earlier epoch ---------------
    try:
        _dump_genealogy(method, face_verts)
    except Exception as _e:
        print(f"[Genealogy] skipped ({_e})")

    # ---- Needle faces: drawn faces near the pick that are FLAT on the reference
    # plane (all verts within 5 mm) yet have a geometric normal far from it.
    # Only a degenerate (needle) triangle can do that; its normal is set by its
    # shortest edge and swings with sub-mm vertex noise -> dark/bright streaks.
    try:
        _dump_needle_faces(comp, face_verts)
    except Exception as _e:
        print(f"[Needle] skipped ({_e})")

    # ---- Crater diagnostics ([Flat] / [Walls] / [TopCover] / [Support]) are
    # verbose and only meaningful with --watch-ref; off unless comp._pick_crater.
    if getattr(comp, '_pick_crater', False):
        try:
            _dump_wound_flatness(comp, face_verts, tets_np, W_np)
        except Exception as _e:
            print(f"[Flat] skipped ({_e})")
        try:
            _dump_nearby_offplane_faces(comp, face_verts, tets_np, W_np, cam_pos=cam_pos)
        except Exception as _e:
            print(f"[Walls] skipped ({_e})")

    # ---- Pocket depth (outer-surface reference; misleading for wound faces,
    # which legitimately sit below the outer tangent plane). Off unless asked.
    if getattr(comp, '_pick_pocket_depth', False):
        try:
            _dump_pocket_depth(comp, face_verts, tets_np, W_np)
        except Exception as _e:
            print(f"[Pocket] skipped ({_e})")

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


def _dump_picked_face_normals(comp, mesh_comp, face_row, face_verts):
    """Print the 3 face-vertex normals the shader will sample for this face.

    Exploded (unindexed) layout: slots [face_row*3, face_row*3+1, face_row*3+2].
    Indexed layout: slots [face_verts[0], face_verts[1], face_verts[2]].
    Reads directly from the normal VBO (mesh_comp.buffers[1]) so we see the same
    data the shader sees.
    """
    if len(mesh_comp.buffers) < 2:
        print(f"[Pick] no normal VBO (buffers={len(mesh_comp.buffers)})")
        return
    exploded = bool(getattr(comp, '_face_expanded', False))
    if exploded:
        slots = [face_row * 3 + i for i in range(3)]
    else:
        slots = [int(v) for v in face_verts]

    try:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[1])
        normals = np.empty((3, 3), dtype=np.float32)
        stride = 3 * 4  # 3 float32
        for k, s in enumerate(slots):
            data = gl.glGetBufferSubData(gl.GL_ARRAY_BUFFER, s * stride, stride)
            normals[k] = np.frombuffer(data, dtype=np.float32).copy()
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    except Exception as e:
        print(f"[Pick] normal VBO read failed: {e}")
        return

    label = "exploded" if exploded else "indexed"
    print(f"[Pick] Face-vertex normals ({label} VBO, buffer[1]):")
    for k, (s, v) in enumerate(zip(slots, face_verts)):
        n = normals[k]
        mag = float(np.linalg.norm(n))
        print(f"[Pick]   slot {s:>7d}  vert {int(v):>6d}  "
              f"n=({n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f})  |n|={mag:.4f}")


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
        force = sim._forces.to_numpy()[:sim.n_verts]
    except Exception:
        force = None
    try:
        fixed = sim._fixed.to_numpy()[:sim.n_verts]
    except Exception:
        fixed = None

    # Springs: dict from vert -> list of (spring_idx, partner, k, r, current_len)
    sa = getattr(sim, '_sa', None)
    sb = getattr(sim, '_sb', None)
    sr = getattr(sim, '_sr', None)
    sk = getattr(sim, '_sk', None)

    # Orphan slaving: the OLD (spring-mass) path stores a dict in
    # _orphan_constraints; the PROGRESSIVE FEM path stores parallel arrays
    # (_orphan_ids / _orphan_masters / _orphan_offsets). Merge both so the
    # half-check below fires regardless of path.
    orphan_map = dict(getattr(comp.method, '_orphan_constraints', {}) or {})
    _oids = getattr(comp.method, '_orphan_ids', None)
    _omas = getattr(comp.method, '_orphan_masters', None)
    _offs = getattr(comp.method, '_orphan_offsets', None)
    if _oids is not None and len(_oids) > 0 and _omas is not None:
        for i in range(len(_oids)):
            oid = int(_oids[i])
            if oid in orphan_map:
                continue
            off = (_offs[i] if _offs is not None and i < len(_offs)
                   else np.zeros(3, dtype=np.float32))
            orphan_map[oid] = (int(_omas[i]), off)

    # Reference plane from --watch-ref (a known-good face): per-vertex height
    # above it. Ids live on the method and are remapped at every bake.
    ref_c = ref_n = None
    _rid = getattr(comp.method, '_watch_ref_ids', None)
    if _rid is not None and len(_rid) >= 3 and (_rid >= 0).all() and _rid.max() < sim.n_verts:
        B = pos[_rid]
        ref_n = np.cross(B[1] - B[0], B[2] - B[0])
        ref_n = ref_n / (float(np.linalg.norm(ref_n)) or 1.0)
        # Orient outward: away from the mesh centroid.
        if float(np.dot(ref_n, B.mean(axis=0) - pos.mean(axis=0))) < 0:
            ref_n = -ref_n
        ref_c = B.mean(axis=0)
        print(f"[Pick] reference plane (--watch-ref {_rid.tolist()}): "
              f"n=({ref_n[0]:+.3f},{ref_n[1]:+.3f},{ref_n[2]:+.3f})  "
              f"h_ref>0 = above the good face (outward)")

    # Who holds each vertex: incident tets with W, and how far each one is
    # stretched now vs its rest shape. A free vertex far from its neighbours
    # is held by whatever active tets it has; if those are stretched, they are
    # dragging it (or being dragged).
    def _holders(v):
        try:
            _tets = sim._tets.to_numpy()[:sim.n_tets]
            _W    = sim._W.to_numpy()[:sim.n_tets]
            _rest = getattr(comp.method, '_full_final_pos', None)
            inc = np.where((_tets == v).any(axis=1))[0]
            if len(inc) == 0:
                print(f"[Holders]   vert {v}: NO tets at all")
                return
            act = inc[_W[inc] > 0.0]
            rows = []
            for t in inc:
                tv = _tets[t]
                if tv.max() >= len(pos):
                    continue
                cur = pos[tv]
                ce = max(float(np.linalg.norm(cur[i] - cur[j]))
                         for i in range(4) for j in range(i + 1, 4))
                if _rest is not None and tv.max() < len(_rest):
                    rp = np.asarray(_rest[tv], dtype=np.float64)
                    re = max(float(np.linalg.norm(rp[i] - rp[j]))
                             for i in range(4) for j in range(i + 1, 4))
                else:
                    re = float('nan')
                rows.append((int(t), float(_W[t]), ce, re, ce / re if re > 1e-9 else float('inf')))
            rows.sort(key=lambda r: -r[4])
            print(f"[Holders]   vert {v}: {len(inc)} tets ({len(act)} active, "
                  f"{len(inc) - len(act)} inert); most stretched first:")
            for t, w, ce, re, s in rows[:4]:
                print(f"[Holders]     tet #{t:<7d} W={w:.3e} {'ACTIVE' if w > 0 else 'inert '} "
                      f"max_edge cur={ce:.4f} rest={re:.4f} stretch={s:.2f}x "
                      f"verts={_tets[t].tolist()}")
        except Exception as _e:
            print(f"[Holders]   vert {v}: skipped ({_e})")

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
            _ocl = getattr(comp.method, '_orphan_class', None)
            _cname = {1: 'sliver-only', 2: 'island', 3: 'weak-support'}
            _src = ('legacy-zero-mass' if v in (getattr(comp.method, '_orphan_constraints', {}) or {})
                    else 'progressive')
            _ctxt = (_cname.get(int(_ocl[v]), 'unknown') if _ocl is not None and v < len(_ocl) else '?')
            orphan_str = (f" ORPHAN(master={master}, offset=({offset[0]:+.4f},{offset[1]:+.4f},{offset[2]:+.4f}), "
                          f"class={_ctxt}, via={_src})")
            if ref_n is not None and master < sim.n_verts:
                orphan_str += f" h_ref(master)={float(np.dot(pos[master] - ref_c, ref_n)):+.4f}"
        h_str = (f" h_ref={float(np.dot(p - ref_c, ref_n)):+.4f}"
                 if ref_n is not None else "")
        _f_str = ""
        if force is not None and v < len(force):
            _fm = float(np.linalg.norm(force[v]))
            _f_str = f" |F|={_fm:.3e}" + (f" a={_fm / m:.2e}" if m > 0 else "")
        print(f"[Pick]   vert {v:>6d}: "
              f"pos=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}) "
              f"vel=({vv[0]:+.4f}, {vv[1]:+.4f}, {vv[2]:+.4f}) "
              f"mass={m:.4e} fixed={fx}{_f_str}{h_str}{orphan_str}")

        # Rest (creation-time) vs current -- tells caved-from-rest (dynamic) apart
        # from rest-already-caved (baked). radial<0 => moved toward mesh centroid.
        _rest = getattr(comp.method, '_full_final_pos', None)
        if _rest is not None and v < len(_rest):
            rp  = np.asarray(_rest[v], dtype=np.float32)
            d   = p - rp
            cen = pos.mean(axis=0)
            outward = rp - cen
            outn = outward / (float(np.linalg.norm(outward)) or 1.0)
            radial = float(np.dot(d, outn))
            tag = "INWARD" if radial < -1e-3 else "outward" if radial > 1e-3 else "flush"
            print(f"[Pick]           rest=({rp[0]:+.4f}, {rp[1]:+.4f}, {rp[2]:+.4f}) "
                  f"|cur-rest|={float(np.linalg.norm(d)):.4f}  radial={radial:+.4f} ({tag})")
        _holders(v)

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


_DROP_REASON_NAMES = {0: "kept", 1: "PURE_ORIG_DROP", 2: "PHANTOM_COLLAR_DROP",
                      3: "COLLAR_RECON_DROP"}
_VERDICT_NAMES = {0: "OUTER", 1: "WOUND", 2: "DROP"}


def _dump_needle_faces(comp, face_verts, radius=0.2, flat_tol=0.005, angle_deg=10.0, max_rows=15):
    method = comp.method
    rid = getattr(method, '_watch_ref_ids', None)
    faces = getattr(comp, '_all_render_faces', None)
    if rid is None or len(rid) < 3 or faces is None:
        print("[Needle] needs --watch-ref and rendered faces")
        return
    faces = np.asarray(faces, dtype=np.int64)
    pos = np.asarray(method.get_positions(), dtype=np.float64)
    n_act = len(pos)
    dsi = getattr(comp, '_disc_split_phys_idx', None)
    phys_of = (np.concatenate([np.arange(n_act), np.asarray(dsi, dtype=np.int64)])
               if dsi is not None and len(dsi) > 0 else np.arange(n_act))
    ok = (faces < len(phys_of)).all(axis=1)
    pf = np.where(ok[:, None], phys_of[np.where(ok[:, None], faces, 0)], 0)
    B = pos[rid]
    n = np.cross(B[1] - B[0], B[2] - B[0]); n /= (np.linalg.norm(n) or 1.0)
    if float(np.dot(n, B.mean(axis=0) - pos.mean(axis=0))) < 0:
        n = -n
    c = B.mean(axis=0)
    fv = np.asarray([int(v) for v in face_verts]); pick_c = pos[fv].mean(axis=0)
    P = pos[pf]                                                     # (F,3,3)
    cen = P.mean(axis=1)
    rows_all = np.arange(len(faces))
    n_out = int(getattr(comp, '_n_outer_faces', 0) or (len(comp._outer_faces) if comp._outer_faces is not None else 0))
    wv = 0 if getattr(comp, 'hide_wound_faces', False) else int(getattr(comp, '_wound_face_ptr', 0))
    drawn = rows_all < n_out + wv
    near = ok & drawn & (np.linalg.norm(cen - pick_c, axis=1) < radius)
    h = (P - c) @ n                                                 # (F,3)
    flat = np.abs(h).max(axis=1) < flat_tol
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area2 = np.linalg.norm(fn, axis=1)
    fn = fn / (area2[:, None] + 1e-30)
    ang = np.degrees(np.arccos(np.clip(np.abs(fn @ n), -1.0, 1.0)))
    e = np.stack([np.linalg.norm(P[:, 1] - P[:, 0], axis=1), np.linalg.norm(P[:, 2] - P[:, 1], axis=1),
                  np.linalg.norm(P[:, 0] - P[:, 2], axis=1)], axis=1)
    emin, emax = e.min(axis=1), e.max(axis=1)
    sel = near & flat
    bad = sel & (ang > angle_deg)
    print(f"[Needle] drawn faces within {radius:.2f} of the pick that are FLAT (all verts within "
          f"{flat_tol*1000:.0f}mm of the plane): {int(sel.sum())}; of those with normal > {angle_deg:.0f} deg "
          f"off the plane: {int(bad.sum())}")
    if sel.any():
        good = sel & ~bad
        print(f"[Needle]   shortest edge: tilted p50={np.median(emin[bad])*1000 if bad.any() else float('nan'):.1f}mm  "
              f"vs untilted p50={np.median(emin[good])*1000 if good.any() else float('nan'):.1f}mm; "
              f"tilted faces' aspect (max/min edge) p50={np.median(emax[bad]/np.maximum(emin[bad],1e-9)) if bad.any() else float('nan'):.0f}")
    oid = getattr(method, '_orphan_ids', None)
    orph = set(int(o) for o in oid) if oid is not None and len(oid) > 0 else set()
    for r in np.where(bad)[0][np.argsort(-ang[bad])][:max_rows]:
        r = int(r); v = pf[r]
        tags = ",".join(("O" if int(x) in orph else "-") for x in v)
        print(f"[Needle]   row {r:>6d} verts={v.tolist()} orphan[{tags}] angle={ang[r]:5.1f}deg "
              f"min_edge={emin[r]*1000:5.1f}mm max_edge={emax[r]*1000:6.1f}mm "
              f"h=({h[r,0]*1000:+.1f},{h[r,1]*1000:+.1f},{h[r,2]*1000:+.1f})mm")


def _drawn_row(comp, row, n_faces):
    """Is render-face row `row` inside the current IBO draw range?
    Layout: [outer | wound(revealed by _wound_face_ptr) | collar(T-key)]."""
    n_outer = int(getattr(comp, '_n_outer_faces', 0) or
                  (len(comp._outer_faces) if comp._outer_faces is not None else 0))
    wound_vis = 0 if getattr(comp, 'hide_wound_faces', False) else int(getattr(comp, '_wound_face_ptr', 0))
    c_off = int(getattr(comp, '_collar_tet_face_offset', n_faces))
    n_col = int(getattr(comp, '_n_collar_tet_faces', 0))
    show_col = bool(getattr(comp, '_show_collar_tets', False))
    if row < n_outer + wound_vis:
        return True
    return show_col and c_off <= row < c_off + n_col


def _dump_wound_flatness(comp, face_verts, tets_np, W_np, radius=0.3, n_out=10):
    """Is the picked face a dent in an otherwise flat wound?

    Takes every drawn old-wound face whose centroid lies within radius_edges
    of the pick, fits a least-squares plane (SVD) to the union of their
    vertices, and prints the residual distribution (the noise floor) plus the
    picked verts' residuals and the largest outliers in the region with their
    physics state. Outliers well beyond p90 are real dents/bumps.
    """
    method = comp.method
    faces = getattr(comp, '_all_render_faces', None)
    cws = getattr(method, '_carried_wound_set', None) or set()
    if faces is None or not cws:
        print("[Flat] needs rendered faces and a carried wound set (after a bake)")
        return
    faces = np.asarray(faces, dtype=np.int64)
    pos = np.asarray(method.get_positions(), dtype=np.float32)
    n_active = len(pos)
    dsi = getattr(comp, '_disc_split_phys_idx', None)
    if dsi is not None and len(dsi) > 0:
        phys_of = np.concatenate([np.arange(n_active), np.asarray(dsi, dtype=np.int64)])
    else:
        phys_of = np.arange(n_active)
    ok = (faces < len(phys_of)).all(axis=1)
    pf_all = np.where(ok[:, None], phys_of[np.where(ok[:, None], faces, 0)], -1)

    fv = np.asarray([int(v) for v in face_verts], dtype=np.int64)
    pick_c = pos[fv].mean(axis=0)
    # Fixed absolute radius: the tet mesh is very non-uniform (face edges from
    # ~0.08 to ~0.6), so scaling by the picked face's edge makes picks
    # incomparable and can sweep the whole (bent) wound.
    R = float(radius)

    keys = [tuple(sorted(int(v) for v in f)) if o else None for f, o in zip(pf_all, ok)]
    is_ow = np.array([k is not None and k in cws for k in keys])
    cen = np.where(ok[:, None], pos[np.where(ok[:, None], pf_all, 0)].mean(axis=1), np.inf)
    near = is_ow & (np.linalg.norm(cen - pick_c, axis=1) < R)
    # Same sheet only: drop old-wound faces (other cuts' wounds) more than
    # 2.5 cm off the picked face's own plane. The cut-2 sheet under the wedge
    # starts ~5 cm down and is legitimate geometry, not a dent.
    n0 = np.cross(pos[fv[1]] - pos[fv[0]], pos[fv[2]] - pos[fv[0]])
    n0 = n0 / (float(np.linalg.norm(n0)) or 1.0)
    cand = np.where(near)[0]
    h0 = (pos[pf_all[cand]] - pick_c) @ n0                    # (k,3)
    same_sheet = np.abs(h0).max(axis=1) < 0.025
    n_other = int((~same_sheet).sum())
    ow_rows = cand[same_sheet]
    if len(ow_rows) < 4:
        print(f"[Flat] only {len(ow_rows)} same-sheet old-wound faces within R={R:.4f}")
        return
    verts = np.unique(pf_all[ow_rows].ravel())
    fit_verts = verts[~np.isin(verts, fv)]
    # Reference = the PICKED face's own plane (pick a face you know is correct;
    # a large one gives a stiffer reference). No fitting, so nothing is pulled
    # toward the plane and every correct neighbour should read ~0.
    c = pick_c.astype(np.float64)
    n = n0.astype(np.float64)
    if float(np.dot(n, c - pos.mean(axis=0))) < 0:
        n = -n
    P = pos[fit_verts].astype(np.float64)
    res = (P - c) @ n
    a = np.abs(res)
    p50, p90, p99 = np.percentile(a, [50, 90, 99])
    edge = float(np.mean([np.linalg.norm(pos[fv[i]] - pos[fv[(i + 1) % 3]]) for i in range(3)]))
    print(f"[Flat] reference = picked face plane (edge~{edge:.3f}m, "
          f"n=({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f})); old-wound patch: {len(ow_rows)} same-sheet "
          f"faces / {len(verts)} verts within R={R:.2f}, {n_other} other-sheet faces skipped")
    print(f"[Flat]   |h| of the other {len(fit_verts)} patch verts: p50={p50*1000:.1f}mm  "
          f"p90={p90*1000:.1f}mm  p99={p99*1000:.1f}mm  max={a.max()*1000:.1f}mm  "
          f"(mean signed={res.mean()*1000:+.1f}mm; a large |mean| = the picked face is tilted)")
    try:
        mass = comp.simulator._masses.to_numpy()[:n_active]
        fixed = comp.simulator._fixed.to_numpy()[:n_active]
    except Exception:
        mass = fixed = None
    oid = getattr(method, '_orphan_ids', None)
    omap = {}
    if oid is not None and len(oid) > 0:
        for i in range(len(oid)):
            omap[int(oid[i])] = int(method._orphan_masters[i])
    # Rest positions of the current topology: same plane fit on the rest coords
    # of the same verts -> h_rest says whether the dent existed at the bake.
    rest = getattr(method, '_full_final_pos', None)
    c_r = n_r = None
    if rest is not None and len(rest) >= n_active:
        rest = np.asarray(rest, dtype=np.float64)
        c_r = rest[fv].mean(axis=0)
        n_r = np.cross(rest[fv[1]] - rest[fv[0]], rest[fv[2]] - rest[fv[0]])
        n_r = n_r / (float(np.linalg.norm(n_r)) or 1.0)
        if float(np.dot(n_r, n)) < 0:
            n_r = -n_r
    vel = np.asarray(method.get_velocities(), dtype=np.float32)
    # Seam pairs (current cut) and cutting springs touching a vert.
    seam = {}
    for p in (getattr(method, '_seam_pairs', None) or []):
        seam[int(p['v_above'])] = ('above', int(p['v_below']), bool(p['broken']))
        seam[int(p['v_below'])] = ('below', int(p['v_above']), bool(p['broken']))
    sim = comp.simulator
    sa = getattr(sim, '_sa', None); sb = getattr(sim, '_sb', None); sk = getattr(sim, '_sk', None)
    chain_g = getattr(method, '_bake_used_chain', None) or []
    def _born(v):
        cur = int(v); epoch = len(chain_g) + 1
        for k in range(len(chain_g) - 1, -1, -1):
            if cur >= len(chain_g[k]):
                return epoch
            cur = int(chain_g[k][cur]); epoch -= 1
        return epoch
    def _state(v):
        inc = np.where((tets_np == v).any(axis=1))[0]
        n_act = int((W_np[inc] > 0.0).sum())
        m = f"{float(mass[v]):.2e}" if mass is not None else "?"
        fx = int(fixed[v]) if fixed is not None else -1
        s = f"mass={m} fixed={fx} tets(active={n_act},inert={len(inc) - n_act}) born=cut{_born(v)}"
        if v in omap:
            mv = omap[v]
            s += f" ORPHAN(master={mv} h_master={float(np.dot(pos[mv] - c, n))*1000:+.1f}mm)"
        if c_r is not None:
            s += f" h_rest={float(np.dot(rest[v] - c_r, n_r))*1000:+.1f}mm"
        s += f" |vel|={float(np.linalg.norm(vel[v])):.3f}"
        if v in seam:
            side, partner, broken = seam[v]
            s += (f" SEAM({side}, partner={partner} h_partner="
                  f"{float(np.dot(pos[partner] - c, n))*1000:+.1f}mm, broken={broken})")
        if sa is not None and len(sa) > 0:
            hit = np.where((sa == v) | (sb == v))[0]
            live = [int(i) for i in hit if sk[i] > 0.0]
            if len(hit):
                s += f" springs={len(hit)}(live={len(live)})"
        return s
    print(f"[Flat]   picked (reference) verts:")
    for v in fv:
        print(f"[Flat]     v{int(v):>6d}  {_state(int(v))}")
    order = fit_verts[np.argsort(-a)][:n_out]
    print(f"[Flat]   largest |h| among the other {len(fit_verts)} patch verts:")
    for v in order:
        h = float(np.dot(pos[v] - c, n))
        print(f"[Flat]     v{int(v):>6d} h={h*1000:+7.1f}mm  {_state(int(v))}")
    # [TopCover] Is the cut-1 wound top rendered where cut 2 subdivided it?
    # Cut-2-born verts lying ON this plane (|h| < 1 cm) are the trace of the
    # cut-2 plane across this wound. Every such vert should be in at least one
    # rendered face that lies on this plane (the subdivided top). Verts on the
    # plane with NO on-plane rendered face = the top is missing there = hole.
    try:
        born_arr = {}
        all_near = np.unique(pf_all[np.where(ok & (np.linalg.norm(cen - pick_c, axis=1) < R))[0]].ravel())
        all_near = all_near[all_near >= 0]
        hall = (pos[all_near] - c) @ n
        on_plane = all_near[np.abs(hall) < 0.01]
        # on-plane rendered faces (all 3 verts within 1 cm), any bucket, drawn
        _rows_all = np.arange(len(faces))
        _n_out = int(getattr(comp, '_n_outer_faces', 0) or
                     (len(comp._outer_faces) if comp._outer_faces is not None else 0))
        _wv = 0 if getattr(comp, 'hide_wound_faces', False) else int(getattr(comp, '_wound_face_ptr', 0))
        drawn_mask = _rows_all < _n_out + _wv
        if getattr(comp, '_show_collar_tets', False) and int(getattr(comp, '_n_collar_tet_faces', 0)) > 0:
            _co = int(comp._collar_tet_face_offset)
            drawn_mask |= (_rows_all >= _co) & (_rows_all < _co + int(comp._n_collar_tet_faces))
        f_on = ok & drawn_mask & (np.abs((pos[np.where(ok[:, None], pf_all, 0)] - c) @ n) < 0.01).all(axis=1)
        on_rows = np.where(f_on)[0]
        has_face = np.zeros(int(pf_all.max()) + 2, dtype=bool)
        has_face[np.unique(pf_all[on_rows].ravel())] = True
        c2 = [int(v) for v in on_plane if _born(int(v)) >= 2]
        c1 = [int(v) for v in on_plane if _born(int(v)) < 2]
        c2_bare = [v for v in c2 if not has_face[v]]
        c1_bare = [v for v in c1 if not has_face[v]]
        print(f"[TopCover] verts ON this plane within R: cut1-born={len(c1)} (without any on-plane "
              f"rendered face: {len(c1_bare)}), cut2-born={len(c2)} (without: {len(c2_bare)})")
        if c2_bare:
            print(f"[TopCover]   cut2-born on-plane verts with NO on-plane face (hole along cut-2 trace): "
                  f"{c2_bare[:20]}")
        if c1_bare:
            print(f"[TopCover]   cut1-born on-plane verts with NO on-plane face: {c1_bare[:20]}")
        # carried old-wound triples near the pick that are NOT rendered any more
        near_set = set(int(v) for v in all_near)
        render_keys = set(k for k in keys if k is not None)
        missing = [t for t in cws if all(v in near_set for v in t) and t not in render_keys]
        print(f"[TopCover] carried old-wound triples near the pick that are not in the render set: "
              f"{len(missing)}")
        for t in missing[:10]:
            hs = [float(np.dot(pos[v] - c, n)) * 1000 for v in t]
            print(f"[TopCover]   {t} h=({hs[0]:+.1f},{hs[1]:+.1f},{hs[2]:+.1f})mm")
    except Exception as _e:
        print(f"[TopCover] skipped ({_e})")

    # [Support] for the top outliers: each ACTIVE tet holding the vertex, and
    # the position that tet alone predicts for it (rigid Kabsch fit of the
    # tet's REST shape onto the current positions of its other 3 verts).
    # predicted h ~ actual h => the dent is that tet's equilibrium.
    top2 = [int(v) for v in order[:2]]
    if rest is not None and top2:
        print(f"[Support] active tets holding {top2} and the position each predicts alone:")
        for v in top2:
            inc = np.where((tets_np == v).any(axis=1))[0]
            act = [int(t) for t in inc if W_np[t] > 0.0]
            for t in act:
                tv = [int(x) for x in tets_np[t]]
                others = [x for x in tv if x != v]
                X = rest[others].astype(np.float64); Y = pos[others].astype(np.float64)
                xc, yc = X.mean(axis=0), Y.mean(axis=0)
                H = (X - xc).T @ (Y - yc)
                U, _, Vt = np.linalg.svd(H)
                d = np.sign(np.linalg.det(Vt.T @ U.T))
                Dm = np.diag([1.0, 1.0, d])
                Rm = Vt.T @ Dm @ U.T
                pred = yc + Rm @ (rest[v].astype(np.float64) - xc)
                h_pred = float(np.dot(pred - c, n)) * 1000
                h_act = float(np.dot(pos[v] - c, n)) * 1000
                hs = ", ".join(f"{x}:{float(np.dot(pos[x] - c, n))*1000:+.1f}" for x in others)
                # rest-shape sanity: rest edge lengths vs current for the anchors
                dr = np.linalg.norm(X[:, None] - X[None], axis=2); dc = np.linalg.norm(Y[:, None] - Y[None], axis=2)
                strain = float(np.abs(dc - dr)[np.triu_indices(3, 1)].max()) * 1000
                print(f"[Support]   v{v} tet #{t} W={float(W_np[t]):.2e} others(h mm)=[{hs}] "
                      f"anchor edge change max={strain:.1f}mm  predicted h={h_pred:+.1f}mm  "
                      f"actual h={h_act:+.1f}mm  |pred-actual|={float(np.linalg.norm(pred - pos[v]))*1000:.1f}mm")
            if not act:
                print(f"[Support]   v{v}: NO active tet")
    if top2:
        print(f"[Flat]   wound faces touching {top2}:")
        for r in ow_rows:
            f = pf_all[r]
            if any(int(v) in top2 for v in f):
                hs = [float(np.dot(pos[int(v)] - c, n)) * 1000 for v in f]
                print(f"[Flat]     row {int(r):>6d} verts={f.tolist()} "
                      f"h=({hs[0]:+.1f},{hs[1]:+.1f},{hs[2]:+.1f})mm")


def _dump_nearby_offplane_faces(comp, face_verts, tets_np, W_np, cam_pos=None,
                                radius_edges=3.0, off_eps=0.02, max_rows=20):
    """List rendered triangles near the picked face whose vertices leave the
    --watch-ref plane. These are the only candidates for visible crater walls
    when every picked face is flat."""
    method = comp.method
    rid = getattr(method, '_watch_ref_ids', None)
    if rid is None or len(rid) < 3:
        print("[Walls] needs --watch-ref a,b,c (a known-good face) to define the plane")
        return
    faces = getattr(comp, '_all_render_faces', None)
    if faces is None:
        print("[Walls] no _all_render_faces")
        return
    faces = np.asarray(faces, dtype=np.int64)
    pos = np.asarray(method.get_positions(), dtype=np.float32)
    n_active = len(pos)
    if (rid < 0).any() or rid.max() >= n_active:
        print("[Walls] reference ids out of range")
        return
    dsi = getattr(comp, '_disc_split_phys_idx', None)
    if dsi is not None and len(dsi) > 0:
        render_pos = np.vstack([pos, pos[np.asarray(dsi, dtype=np.int64)]])
        phys_of = np.concatenate([np.arange(n_active), np.asarray(dsi, dtype=np.int64)])
    else:
        render_pos = pos
        phys_of = np.arange(n_active)
    ok = (faces < len(render_pos)).all(axis=1)

    B = pos[rid]
    n = np.cross(B[1] - B[0], B[2] - B[0]); n = n / (float(np.linalg.norm(n)) or 1.0)
    if float(np.dot(n, B.mean(axis=0) - pos.mean(axis=0))) < 0:
        n = -n
    c = B.mean(axis=0)

    fv = np.asarray([int(v) for v in face_verts], dtype=np.int64)
    pf = pos[fv]
    pick_c = pf.mean(axis=0)
    edge = float(np.mean([np.linalg.norm(pf[i] - pf[(i + 1) % 3]) for i in range(3)]))
    R = radius_edges * max(edge, 1e-6)
    cws = getattr(method, '_carried_wound_set', None) or set()

    # Draw range: [outer | wound(revealed) | collar(T-key)].
    n_outer = int(getattr(comp, '_n_outer_faces', 0) or
                  (len(comp._outer_faces) if comp._outer_faces is not None else 0))
    wound_vis = 0 if getattr(comp, 'hide_wound_faces', False) else int(getattr(comp, '_wound_face_ptr', 0))
    c_off = int(getattr(comp, '_collar_tet_face_offset', len(faces)))
    n_col = int(getattr(comp, '_n_collar_tet_faces', 0))
    show_col = bool(getattr(comp, '_show_collar_tets', False))
    def _drawn(row):
        if row < n_outer:
            return True
        if row < n_outer + wound_vis:
            return True
        if show_col and c_off <= row < c_off + n_col:
            return True
        return False
    def _bucket(row):
        if row < n_outer:
            return "OUTER"
        if n_col > 0 and row >= c_off:
            return "COLLAR"
        return "WOUND"

    rp = render_pos[np.where(ok[:, None], faces, 0)]            # (F,3,3)
    near = ok & (np.linalg.norm(rp.mean(axis=1) - pick_c, axis=1) < R)
    rows = np.where(near)[0]
    if len(rows) == 0:
        print(f"[Walls] no rendered faces within R={R:.4f}")
        return
    h = (rp[rows] - c) @ n                                      # (k,3) heights
    hmax = np.abs(h).max(axis=1)
    off = hmax > off_eps
    print(f"[Walls] {len(rows)} rendered faces within R={R:.4f} ({radius_edges:.0f} edges) "
          f"of the pick; {int(off.sum())} leave the reference plane by >{off_eps} "
          f"(flat ones: {int((~off).sum())}); draw range: outer<{n_outer}, "
          f"wound revealed={wound_vis}, collar shown={show_col}")
    # Coverage: is each off-plane face hidden under a flat face (projected onto
    # the plane), or exposed through a hole in the flat sheet?
    u = B[1] - B[0]; u = u - np.dot(u, n) * n; u = u / (float(np.linalg.norm(u)) or 1.0)
    w = np.cross(n, u)
    def _proj(P):                       # (...,3) -> (...,2)
        d = P - c
        return np.stack([d @ u, d @ w], axis=-1)
    # Flat candidates from a wider radius so the edge of the search sphere does
    # not create fake holes.
    wide = ok & (np.linalg.norm(rp.mean(axis=1) - pick_c, axis=1) < 1.5 * R)
    wide_rows = np.where(wide)[0]
    wide_h = np.abs((rp[wide_rows] - c) @ n).max(axis=1)
    flat_rows = wide_rows[wide_h <= off_eps]
    flat_tri = _proj(rp[flat_rows]) if len(flat_rows) else np.zeros((0, 3, 2))
    # Only flat faces actually in the draw range can hide anything.
    if len(flat_rows):
        fd = np.array([_drawn(int(r)) for r in flat_rows])
        flat_tri = flat_tri[fd]
    def _covered(pt):
        if len(flat_tri) == 0:
            return False
        a, b2, c2 = flat_tri[:, 0], flat_tri[:, 1], flat_tri[:, 2]
        def _s(p1, p2, p3):
            return (p1[:, 0] - p3[:, 0]) * (p2[:, 1] - p3[:, 1]) - (p2[:, 0] - p3[:, 0]) * (p1[:, 1] - p3[:, 1])
        P = np.broadcast_to(pt, a.shape)
        d1 = _s(P, a, b2); d2 = _s(P, b2, c2); d3 = _s(P, c2, a)
        neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        pos_ = (d1 > 0) | (d2 > 0) | (d3 > 0)
        return bool((~(neg & pos_)).any())
    off_rows = rows[off]
    cen2 = _proj(rp[off_rows].mean(axis=1))
    cov = np.array([_covered(p) for p in cen2], dtype=bool)
    print(f"[Walls] coverage of the {len(off_rows)} off-plane faces by DRAWN flat faces: "
          f"covered={int(cov.sum())}  EXPOSED(hole)={int((~cov).sum())}")
    # Which cut created the off-plane sheet's vertices?
    chain = getattr(method, '_bake_used_chain', None) or []
    if chain and len(off_rows):
        sample = np.unique(phys_of[faces[off_rows]].ravel())
        n_cuts = len(chain) + 1
        born = {}
        for v in sample:
            cur = int(v); epoch = n_cuts
            for k in range(len(chain) - 1, -1, -1):
                if cur >= len(chain[k]):
                    break
                cur = int(chain[k][cur]); epoch -= 1
            born[epoch] = born.get(epoch, 0) + 1
        print(f"[Walls] off-plane sheet verts by creating cut: "
              + ", ".join(f"cut{e}:{cnt}" for e, cnt in sorted(born.items())))
    cov_of_row = {int(r): bool(cv) for r, cv in zip(off_rows, cov)}

    # Camera occlusion: from each off-plane face centroid toward the camera,
    # does ANY drawn triangle (any height) block the view? This is the honest
    # visibility test; the flat-sheet coverage above ignores curved outer faces.
    vis_of_row = {}
    if cam_pos is not None:
        all_drawn = np.array([r for r in range(len(faces)) if _drawn(r) and ok[r]], dtype=np.int64)
        tri_all = faces[all_drawn]
        for r in off_rows:
            r = int(r)
            C = rp[r].mean(axis=0)
            d = cam_pos - C
            L = float(np.linalg.norm(d)) or 1.0
            d = d / L
            # start slightly toward the camera so the face itself is not hit
            hit_row, t_hit = _ray_intersect_all(C + d * 1e-3, d, render_pos, tri_all)
            vis_of_row[r] = not (hit_row is not None and t_hit < L - 1e-3)
        n_vis = sum(1 for v in vis_of_row.values() if v)
        print(f"[Walls] camera occlusion test: {n_vis} of {len(off_rows)} off-plane faces "
              f"are VISIBLE from the camera (nothing drawn in front of them), "
              f"{len(off_rows) - n_vis} occluded")

    # Open-edge test over the WHOLE drawn surface (phys ids, so the collar
    # vertex-split seam does not count as open). A closed surface has every
    # edge in exactly 2 drawn faces; count==1 edges near the pick = a hole.
    drawn_rows = np.array([r for r in range(len(faces)) if _drawn(r) and ok[r]], dtype=np.int64)
    pf_all = phys_of[faces[drawn_rows]]                         # (D,3) phys ids
    e = np.concatenate([pf_all[:, [0, 1]], pf_all[:, [1, 2]], pf_all[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    ekey = e[:, 0].astype(np.int64) * (1 << 32) + e[:, 1].astype(np.int64)
    ukey, cnt = np.unique(ekey, return_counts=True)
    open_keys = set(ukey[cnt == 1].tolist())
    def _open_edges(row):
        pv = phys_of[faces[row]]
        out = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            lo, hi = sorted((int(pv[a]), int(pv[b])))
            if lo * (1 << 32) + hi in open_keys:
                out.append((lo, hi))
        return out
    region_rows = [int(r) for r in rows if _drawn(int(r))]
    open_by_row = {r: _open_edges(r) for r in region_rows}
    n_open_flat = sum(len(open_by_row[int(r)]) for r in flat_rows if int(r) in open_by_row)
    n_open_off  = sum(len(open_by_row[int(r)]) for r in off_rows if int(r) in open_by_row)
    print(f"[Walls] open edges (used by ONE drawn face) near the pick: "
          f"on flat faces={n_open_flat}  on off-plane faces={n_open_off}  "
          f"(whole drawn surface has {len(open_keys)} open edges total)")
    shown = 0
    hole_verts = set()
    for r in region_rows:
        for lo, hi in open_by_row[r]:
            hl = float(np.dot(pos[lo] - c, n)); hh_ = float(np.dot(pos[hi] - c, n))
            if abs(hl) <= off_eps and abs(hh_) <= off_eps:
                hole_verts.update((lo, hi))
            if shown < 15:
                print(f"[Walls]     OPEN edge {lo}-{hi} on row {r} ({_bucket(r)}) "
                      f"h=({hl:+.3f},{hh_:+.3f})")
                shown += 1

    # VISIBLE lower faces: where does each one's camera ray cross the top
    # plane, and how far is that crossing from the top-sheet hole (the open
    # edge loop at plane height)?
    vis_rows = [r for r, v in vis_of_row.items() if v]

    # Original-skin test: map a current vert to its cut-1 id through the bake
    # chain; if all three are original verts, look the triple up in the UNCUT
    # mesh's boundary faces. ORIG-SKIN = the bunny's own surface (legit).
    # EXPOSED-INTERIOR = an interior face of the uncut mesh now drawn, i.e. a
    # tet is missing next to it.
    chain_g = getattr(method, '_bake_used_chain', None) or []
    orig_n = getattr(method, '_orig_n_verts', None)
    orig_keys = method.orig_surface_keys() if hasattr(method, 'orig_surface_keys') else None
    def _cut1_id(v):
        cur = int(v)
        for k in range(len(chain_g) - 1, -1, -1):
            if cur >= len(chain_g[k]):
                return None
            cur = int(chain_g[k][cur])
        return cur
    def _skin_tag(pv):
        if orig_keys is None or orig_n is None:
            return ""
        ids = [_cut1_id(v) for v in pv]
        if any(i is None or i >= orig_n for i in ids):
            return " [has cut verts]"
        a, b2, c2 = sorted(ids)
        key = (a << 42) | (b2 << 21) | c2
        return " [ORIG-SKIN]" if key in orig_keys else " [EXPOSED-INTERIOR]"
    if vis_rows:
        n_skin = n_int = n_cut = 0
        for r in vis_rows:
            tg = _skin_tag(phys_of[faces[r]])
            n_skin += tg == " [ORIG-SKIN]"; n_int += tg == " [EXPOSED-INTERIOR]"; n_cut += tg == " [has cut verts]"
        print(f"[Walls] visible faces by origin: ORIG-SKIN={n_skin}  EXPOSED-INTERIOR={n_int}  "
              f"with cut verts={n_cut}")
        hole_c = pos[list(hole_verts)].mean(axis=0) if hole_verts else None
        if hole_c is not None:
            print(f"[Walls] top-sheet hole: verts {sorted(hole_verts)} "
                  f"center=({hole_c[0]:+.4f},{hole_c[1]:+.4f},{hole_c[2]:+.4f})")
        print(f"[Walls] the {len(vis_rows)} VISIBLE lower faces (camera ray crossing of the top plane):")
        for r in sorted(vis_rows, key=lambda r: -np.abs((rp[r] - c) @ n).max()):
            C = rp[r].mean(axis=0)
            d = cam_pos - C
            dn = float(np.dot(d, n))
            hC = float(np.dot(C - c, n))
            cross_s = ""
            if abs(dn) > 1e-9:
                X = C + d * (-hC / dn)             # point on the plane h=0
                cross_s = f" plane-crossing=({X[0]:+.4f},{X[1]:+.4f},{X[2]:+.4f})"
                if hole_c is not None:
                    cross_s += f" dist_to_hole={float(np.linalg.norm(X - hole_c)):.4f}"
                cross_s += f" dist_to_pick={float(np.linalg.norm(X - pick_c)):.4f}"
            f = faces[r]; pv = phys_of[f]
            hh = (render_pos[f] - c) @ n
            parents = _find_tets_containing_vectorised(pv, tets_np)
            pstr = ",".join(f"#{t}({'A' if W_np[t] > 0 else 'inert'})" for t in parents[:3])
            print(f"[Walls]     row {r:>6d} verts={pv.tolist()} h=({hh[0]:+.3f},{hh[1]:+.3f},{hh[2]:+.3f}) "
                  f"parents={pstr or 'none'}{_skin_tag(pv)}{cross_s}")

        # Pit walls: visible faces spanning from the plane (|h|<=eps) down to an
        # apex below it. For each apex: which current tets still touch it, and
        # the bake history of the wall triangle (which pre-bake tets owned it,
        # kept or dropped). Needs --debug-show-dropped for the history.
        apex_done = set()
        face_rows = None
        n_orig_cur = getattr(comp, '_n_orig', None)
        if getattr(method, '_bake_debug', None) is None:
            print("[Walls] (no bake history recorded yet: no bake has happened)")
        for r in vis_rows:
            f = faces[r]; pv = phys_of[f]
            hh = (render_pos[f] - c) @ n
            on_plane = np.abs(hh) <= off_eps
            below = hh < -off_eps
            if on_plane.sum() != 2 or below.sum() != 1:
                continue
            apex = int(pv[np.where(below)[0][0]])
            if apex in apex_done:
                continue
            apex_done.add(apex)
            inc = np.where((tets_np == apex).any(axis=1))[0]
            n_act = int((W_np[inc] > 0.0).sum())
            # Are there ANY current tets (active or inert) containing both plane verts
            # AND the apex other than the wall's own parent? => is the cap really gone.
            pl = [int(v) for v in pv[on_plane]]
            both = ((tets_np == pl[0]).any(axis=1) & (tets_np == pl[1]).any(axis=1))
            owners = np.where(both & (tets_np == apex).any(axis=1))[0]
            # Skin or interior? Count drawn faces touching the apex that are NOT
            # old-wound (bunny skin) vs old-wound.
            touch = np.where((pf_all == apex).any(axis=1))[0]
            n_skin = n_ow = 0
            for tr in touch:
                k3 = tuple(sorted(int(v) for v in pf_all[tr]))
                if k3 in cws:
                    n_ow += 1
                else:
                    n_skin += 1
            print(f"[Walls] PIT apex vert {apex} h={float(np.dot(pos[apex] - c, n)):+.4f}: "
                  f"touched by {len(inc)} current tets ({n_act} active, {len(inc) - n_act} inert); "
                  f"drawn faces touching it: skin(non-wound)={n_skin} old-wound={n_ow} "
                  f"{'=> INTERIOR vertex, walls are a pit' if n_skin == 0 else '=> on the skin, walls may be rim'}")
            print(f"[Walls]     wall edge {pl[0]}-{pl[1]} + apex owned by current tets "
                  f"{[(int(t), 'A' if W_np[t] > 0 else 'inert') for t in owners]}")
            # The owner tet(s): all 4 faces, are they in the render set / drawn /
            # how many tets own each. A missing top face (3 plane verts) over the
            # apex is the hole the pit is seen through.
            if face_rows is None:
                face_rows = {}
                for rr in range(len(faces)):
                    if ok[rr]:
                        face_rows.setdefault(tuple(sorted(int(v) for v in phys_of[faces[rr]])), []).append(rr)
            for t in owners:
                t = int(t)
                tv = [int(v) for v in tets_np[t]]
                print(f"[Walls]     owner tet #{t} ({'ACTIVE' if W_np[t] > 0 else 'inert'}) verts={tv} "
                      f"h=({', '.join(f'{float(np.dot(pos[v] - c, n)):+.3f}' for v in tv)})")
                for skip in range(4):
                    tri = tuple(sorted(tv[i] for i in range(4) if i != skip))
                    rr_list = face_rows.get(tri, [])
                    n_own = int(((tets_np == tri[0]).any(axis=1) & (tets_np == tri[1]).any(axis=1)
                                 & (tets_np == tri[2]).any(axis=1)).sum())
                    htri = [float(np.dot(pos[v] - c, n)) for v in tri]
                    kind = "TOP(on plane)" if all(abs(x) <= off_eps for x in htri) else "side"
                    print(f"[Walls]       face {tri} {kind:14s} owners={n_own} "
                          f"render rows={rr_list} drawn={[_drawn(x) for x in rr_list]} "
                          f"old_wound={tri in cws}")
            try:
                _dump_bake_history(method, pv, n_orig_cur)
            except Exception as _e:
                print(f"[BakeHist] wall face skipped ({_e})")
            if len(apex_done) >= 3:
                break

    order = rows[off][np.argsort(-hmax[off])][:max_rows]
    active_tets = tets_np[W_np > 0.0]
    for r in order:
        r = int(r)
        f = faces[r]
        pv = phys_of[f]
        hh = (render_pos[f] - c) @ n
        key = tuple(sorted(int(v) for v in pv))
        has = ((active_tets == pv[0]).any(axis=1) & (active_tets == pv[1]).any(axis=1)
               & (active_tets == pv[2]).any(axis=1))
        parents = _find_tets_containing_vectorised(pv, tets_np)
        pstr = ",".join(f"#{t}({'A' if W_np[t] > 0 else 'inert'})" for t in parents[:4])
        dup = "" if (f == pv).all() else f" render-dups->{pv.tolist()}"
        vis = vis_of_row.get(r)
        vis_s = "" if vis is None else ("VISIBLE " if vis else "occluded ")
        print(f"[Walls]   row {r:>6d} {_bucket(r):6s} drawn={str(_drawn(r)):5s} {vis_s}"
              f"{'no-flat-above' if not cov_of_row.get(r, True) else 'flat-above'} "
              f"verts={f.tolist()}{dup} h=({hh[0]:+.3f},{hh[1]:+.3f},{hh[2]:+.3f}) "
              f"old_wound={key in cws} active_backed={bool(has.any())} parents={pstr or 'none'}")


def _dump_genealogy(method, face_verts):
    """Walk each picked vert's id back through every bake renumbering.

    _bake_used_chain[k] maps baked idx -> pre-bake idx for bake k (oldest
    first). A vert with id >= len(chain[k]) was created AFTER that bake (it is
    an INTER/DUP of the later cut), so its lineage stops there. The oldest id
    printed is in the CUT-1 superset space: that is what --watch-verts takes.
    """
    chain = getattr(method, '_bake_used_chain', None)
    info  = getattr(method, '_bake_epoch_info', None) or []
    if not chain:
        print("[Genealogy] no bakes yet: current ids ARE cut-1 ids")
        return
    n_cuts = len(chain) + 1
    print(f"[Genealogy] {len(chain)} bake(s); cut-1 ids are what --watch-verts takes")
    cut1_ids = []
    for v in face_verts:
        v = int(v)
        hops = [f"cut{n_cuts}={v}"]
        cur = v
        epoch = n_cuts
        ok = True
        for k in range(len(chain) - 1, -1, -1):
            used = chain[k]
            if cur >= len(used):
                hops.append(f"(created in cut{epoch}: INTER/DUP, no earlier id)")
                ok = False
                break
            cur = int(used[cur])
            epoch -= 1
            cls = ""
            if k < len(info):
                e = info[k]
                if cur >= e['n_split']:
                    cls = "DUP"
                elif cur >= e['n_orig']:
                    cls = "INTER"
                else:
                    cls = "ORIG"
            hops.append(f"cut{epoch}={cur}{('[' + cls + ']') if cls else ''}")
        if ok:
            cut1_ids.append(cur)
        print(f"[Genealogy]   v{v}: " + " -> ".join(hops))
    if cut1_ids:
        ids = ','.join(str(i) for i in cut1_ids)
        print(f"[Genealogy] cut-1 ids of this face: {ids}   "
              f"(use as --watch-verts {ids} for the bad face, "
              f"or --watch-ref {ids} for the good reference face)")


def _dump_pocket_depth(comp, face_verts, tets_np, W_np, radius_edges=4.0):
    """Signed depth of the picked verts below the surrounding LIVE surface.

    Reference surface = outer faces near the picked face that (a) are backed by
    an ACTIVE tet (W>0) and (b) are not old-wound faces. A plane is fitted to
    them (area-weighted normal, vertex centroid) in CURRENT coords and again in
    REST coords (_full_final_pos, the baked rest of the current topology).
    depth < 0 means the vert sits inside the live surface.

    depth_rest ~ depth_cur  -> the pocket is frozen into the rest state (it was
                               already there when the last bake copied live
                               positions into rest); the FEM is at equilibrium.
    depth_rest ~ 0, cur < 0 -> the pocket formed after the bake; something is
                               actively pulling the verts in.
    """
    method = comp.method
    pos = np.asarray(method.get_positions(), dtype=np.float32)
    n_active = len(pos)
    rest = getattr(method, '_full_final_pos', None)
    if rest is None:
        print("[Pocket] no rest positions (_full_final_pos missing)")
        return
    rest = np.asarray(rest, dtype=np.float32)
    if comp._outer_faces is None:
        print("[Pocket] no outer faces yet")
        return
    outer = np.asarray(comp._outer_faces, dtype=np.int64)
    # Map render-only disc dups back to physical verts.
    outer_p = outer.copy()
    hi = outer_p >= n_active
    if hi.any():
        dsi = getattr(comp, '_disc_split_phys_idx', None)
        if dsi is None:
            outer_p = outer_p[~hi.any(axis=1)]
        else:
            outer_p[hi] = np.asarray(dsi, dtype=np.int64)[outer_p[hi] - n_active]
    outer_p = outer_p[(outer_p < n_active).all(axis=1)]

    fv = [int(v) for v in face_verts]
    pf = pos[fv]
    pick_c = pf.mean(axis=0)
    edge = float(np.mean([np.linalg.norm(pf[i] - pf[(i + 1) % 3]) for i in range(3)]))
    R = radius_edges * max(edge, 1e-6)

    cen = pos[outer_p].mean(axis=1)
    near = np.where(np.linalg.norm(cen - pick_c, axis=1) < R)[0]
    if len(near) == 0:
        print(f"[Pocket] no outer faces within R={R:.4f}")
        return

    cws = getattr(method, '_carried_wound_set', None) or set()
    active_tets = tets_np[W_np > 0.0]
    ref_rows = []
    n_old = n_inert = 0
    for r in near:
        f = outer_p[r]
        if tuple(sorted(int(v) for v in f)) in cws:
            n_old += 1
            continue
        has = ((active_tets == f[0]).any(axis=1)
               & (active_tets == f[1]).any(axis=1)
               & (active_tets == f[2]).any(axis=1))
        if not has.any():
            n_inert += 1
            continue
        ref_rows.append(r)
    if len(ref_rows) < 3:
        print(f"[Pocket] only {len(ref_rows)} live reference faces within R={R:.4f} "
              f"(old-wound skipped={n_old}, inert-backed skipped={n_inert})")
        return
    ref = outer_p[ref_rows]
    ref_verts = np.unique(ref.ravel())
    ref_verts = ref_verts[~np.isin(ref_verts, fv)]

    def _plane(P):
        a = P[ref[:, 0]]; b = P[ref[:, 1]]; c = P[ref[:, 2]]
        n = np.cross(b - a, c - a).sum(axis=0)          # area-weighted
        n = n / (float(np.linalg.norm(n)) or 1.0)
        return P[ref_verts].mean(axis=0), n

    c_cur, n_cur = _plane(pos)
    ok_rest = len(rest) >= n_active
    if ok_rest:
        c_rest, n_rest = _plane(rest)
    print(f"[Pocket] reference: {len(ref_rows)} live outer faces / {len(ref_verts)} verts "
          f"within R={R:.4f} ({radius_edges:.0f} edges); skipped old-wound={n_old}, "
          f"inert-backed={n_inert}")
    print(f"[Pocket]   plane_cur : n=({n_cur[0]:+.3f},{n_cur[1]:+.3f},{n_cur[2]:+.3f}) "
          f"c=({c_cur[0]:+.4f},{c_cur[1]:+.4f},{c_cur[2]:+.4f})")
    if ok_rest:
        print(f"[Pocket]   plane_rest: n=({n_rest[0]:+.3f},{n_rest[1]:+.3f},{n_rest[2]:+.3f}) "
              f"c=({c_rest[0]:+.4f},{c_rest[1]:+.4f},{c_rest[2]:+.4f})")
    else:
        print(f"[Pocket]   rest array too short ({len(rest)} < {n_active}); rest depth skipped")
    # Spread of the reference verts about their own plane = noise floor.
    spread = float(np.abs((pos[ref_verts] - c_cur) @ n_cur).std())
    print(f"[Pocket]   reference-vert scatter about plane_cur: std={spread:.4f} "
          f"(depths well beyond this are real)")
    for v in fv:
        d_cur = float(np.dot(pos[v] - c_cur, n_cur))
        line = f"[Pocket]   vert {v:>6d}: depth_cur={d_cur:+.4f}"
        if ok_rest:
            d_rest = float(np.dot(rest[v] - c_rest, n_rest))
            line += f"  depth_rest={d_rest:+.4f}  delta={d_cur - d_rest:+.4f}"
            if d_cur < -2 * spread:
                line += ("  <- BAKED-IN pocket" if d_rest < -2 * spread
                         else "  <- formed AFTER bake")
        print(line)


def _dump_bake_history(method, face_verts, n_orig):
    """
    For an all-ORIG (baked) face after a multi-cut bake, explain its
    carried-surface status using the bake-time evidence stashed by
    _bake_current_state under method._bake_debug (debug_show_dropped runs
    only). Prints: carried-set membership, the cut-N face-table rows that
    matched this triple (verdict / drop reason / bake-time activity), and
    the cut-N superset tets that owned it (W, sliver flag, kept status).
    """
    bd = getattr(method, '_bake_debug', None)
    if bd is None:
        return
    fv = [int(v) for v in face_verts]
    if n_orig is None or any(v >= int(n_orig) for v in fv):
        print("[BakeHist] face has non-ORIG verts (cut-N+1 INTER/DUP); "
              "no bake history for it")
        return

    tri_baked = tuple(sorted(fv))
    carried = getattr(method, '_carried_surf_set', None)
    in_carried = (carried is not None) and (tri_baked in carried)

    used = bd['used_verts']
    old3 = tuple(sorted(int(used[v]) for v in fv))
    key  = (old3[0] << 42) | (old3[1] << 21) | old3[2]

    print(f"[BakeHist] baked triple={tri_baked}  cut-N triple={old3}  "
          f"in_carried_set={in_carried}")

    cache = bd['cache_old']
    fk    = cache['face_keys']
    rows  = np.where(fk == key)[0]
    fp    = cache['face_parent']
    fic   = cache['face_parent_is_crossing']
    faws  = cache['face_active_when_split']
    fverd = cache['face_verdict']
    freas = cache.get('face_dropped_reason')
    smask = bd['split_mask_old']

    n_active_rows = 0
    print(f"[BakeHist] cut-N face-table rows matching: {len(rows)}")
    for r in rows:
        r = int(r)
        parent = int(fp[r])
        active = (not bool(fic[r])) or (bool(smask[parent]) == bool(faws[r]))
        n_active_rows += int(active)
        reason_str = ""
        if freas is not None:
            reason_str = f"  reason={_DROP_REASON_NAMES.get(int(freas[r]), '?')}"
        print(f"[BakeHist]   row={r}  parent_pristine={parent}  "
              f"active_at_bake={active}  "
              f"verdict={_VERDICT_NAMES.get(int(fverd[r]), '?')}{reason_str}")
    print(f"[BakeHist] active rows at bake = {n_active_rows}  "
          f"(count==1 needed for drawable)")

    tets_old = bd['tets_old']
    W_old    = bd['W_old']
    sliver_set = set(int(s) for s in bd['sliver_slots'])
    has = ((tets_old == old3[0]).any(axis=1)
           & (tets_old == old3[1]).any(axis=1)
           & (tets_old == old3[2]).any(axis=1))
    owners = np.where(has)[0]
    print(f"[BakeHist] cut-N superset tets owning the triple: {len(owners)}")
    for t in owners:
        t = int(t)
        w = float(W_old[t])
        tags = []
        tags.append("kept" if w > 0.0 else "dropped-at-bake")
        if t in sliver_set:
            tags.append("SLIVER")
        print(f"[BakeHist]   tet #{t}  verts={tets_old[t].tolist()}  "
              f"W={w:.3e}  [{', '.join(tags)}]")


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


# ---------------------------------------------------------------------------
# Island drag tool (plain left-click + drag). Grabs the tet under the cursor
# and moves it to follow the cursor on a camera-facing plane; the connected
# island comes along -- ELASTIC via the FEM solve (pin the tet, physics pulls
# the rest) or RIGID (translate the whole connected tet-component). Because it
# follows tet CONNECTIVITY, it separates islands that overlap in space but are
# topologically distinct -- which position-welding could not do.
# ---------------------------------------------------------------------------

def _phys_vert(comp, v, n_active):
    """Map a render-space vertex (which may be a disc-split render dup) back to
    its physical simulator vertex."""
    if v < n_active:
        return int(v)
    dsi = comp._disc_split_phys_idx
    return int(dsi[v - n_active])


def _project_cursor_to_plane(camera_comp, cam_pos, cursor_x, cursor_y,
                             win_w, win_h, plane_pt, plane_n):
    """Intersect the cursor ray with the drag plane (plane_pt, plane_n)."""
    _, d = _build_ray(camera_comp.view, camera_comp.projection, cam_pos,
                      cursor_x, cursor_y, win_w, win_h)
    o = cam_pos.astype(np.float32)
    denom = float(np.dot(d, plane_n))
    if abs(denom) < 1e-8:
        return None
    t = float(np.dot(plane_pt - o, plane_n)) / denom
    if t <= 0:
        return None
    return (o + t * d).astype(np.float32)


def _connected_component_verts(tets_np, W, seed_verts):
    """All vertices in the connected component(s) (over ACTIVE tets, W>0) that
    contain seed_verts. Two verts are connected if an active tet shares them."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    active = tets_np[W > 0.0]
    if len(active) == 0:
        return np.unique(seed_verts).astype(np.int64)
    nv = int(tets_np.max()) + 1
    a = np.concatenate([active[:, 0], active[:, 0], active[:, 0],
                        active[:, 1], active[:, 1], active[:, 2]])
    b = np.concatenate([active[:, 1], active[:, 2], active[:, 3],
                        active[:, 2], active[:, 3], active[:, 3]])
    g = csr_matrix((np.ones(len(a)), (a, b)), shape=(nv, nv))
    _, lab = connected_components(g, directed=False)
    seed_labels = np.unique(lab[np.asarray(seed_verts, dtype=np.int64)])
    return np.where(np.isin(lab, seed_labels))[0].astype(np.int64)


def drag_begin(comp, mesh_comp, camera_comp, camera_transform,
               cursor_x, cursor_y, win_w, win_h):
    """Grab the tet under the cursor. Returns True if a drag started."""
    if comp.method is None or comp.simulator is None:
        return False
    cam_pos = np.array([camera_transform.translation.x,
                        camera_transform.translation.y,
                        camera_transform.translation.z], dtype=np.float32)
    origin, direction = _build_ray(camera_comp.view, camera_comp.projection,
                                   cam_pos, cursor_x, cursor_y, win_w, win_h)
    pos_active = comp.method.get_positions()
    n_active   = pos_active.shape[0]
    if (comp._disc_split_phys_idx is not None
            and len(comp._disc_split_phys_idx) > 0):
        render_pos = np.vstack([pos_active, pos_active[comp._disc_split_phys_idx]])
    else:
        render_pos = pos_active
    faces = _current_rendered_faces(comp)
    if faces is None or len(faces) == 0:
        return False
    face_row, t_hit = _ray_intersect_all(origin, direction, render_pos, faces)
    if face_row is None:
        return False
    hit_world  = origin + direction * t_hit
    face_verts = faces[face_row]
    phys = np.array([_phys_vert(comp, int(v), n_active) for v in face_verts],
                    dtype=np.int64)

    sim     = comp.simulator
    n_tets  = sim.n_tets
    tets_np = sim._tets.to_numpy()[:n_tets]
    parents = _find_tets_containing_vectorised(phys, tets_np)
    if not parents:
        print("[Drag] no parent tet under cursor")
        return False
    tet0      = int(parents[0])
    tet_verts = tets_np[tet0].astype(np.int64)

    mode = getattr(comp, '_drag_mode', 'elastic')
    W = sim._W.to_numpy()[:n_tets]
    if mode == 'rigid':
        ctrl_verts = _connected_component_verts(tets_np, W, tet_verts)
    else:
        # elastic: grab the clicked tet PLUS its immediate (1-ring) neighbours --
        # any ACTIVE tet sharing a vertex with it -- so the grabbed patch is
        # stiffer and translates a local chunk instead of stretching one tet.
        nbr = np.isin(tets_np, tet_verts).any(axis=1) & (W > 0.0)
        ctrl_verts = np.unique(tets_np[nbr].ravel())

    # Pin the controlled verts so the solve treats them as kinematic and our
    # per-frame position write is respected.
    fixed = sim._fixed.to_numpy()
    saved_fixed = fixed[ctrl_verts].copy()
    fixed[ctrl_verts] = 1
    sim._fixed.from_numpy(fixed)

    # Suspend orphan slaving for the controlled verts (else the per-step orphan
    # kinematics overrides our drag each frame). Saved + restored on release.
    saved_orphans = None
    oid = getattr(comp.method, '_orphan_ids', None)
    if oid is not None and len(oid) > 0:
        oid = np.asarray(oid)
        keep = ~np.isin(oid, ctrl_verts)
        if not keep.all():
            saved_orphans = (comp.method._orphan_ids,
                             comp.method._orphan_masters,
                             comp.method._orphan_offsets)
            comp.method._orphan_ids     = oid[keep].astype(np.int32)
            comp.method._orphan_masters = np.asarray(comp.method._orphan_masters)[keep]
            comp.method._orphan_offsets = np.asarray(comp.method._orphan_offsets)[keep]

    start_pos = sim.positions.to_numpy()
    comp._drag_state = {
        'mode':          mode,
        'verts':         ctrl_verts,
        'start_pos':     start_pos[ctrl_verts].copy(),
        'plane_pt':      hit_world.astype(np.float32),
        'plane_n':       (direction / (np.linalg.norm(direction) or 1.0)).astype(np.float32),
        'grab_world':    hit_world.astype(np.float32),
        'saved_fixed':   saved_fixed,
        'saved_orphans': saved_orphans,
    }
    print(f"[Drag] begin ({mode}): tet {tet0}, controlling "
          f"{len(ctrl_verts)} verts. Drag to move, release to drop.")
    return True


def drag_update(comp, camera_comp, camera_transform,
                cursor_x, cursor_y, win_w, win_h):
    """Move the controlled verts to follow the cursor on the drag plane."""
    st = getattr(comp, '_drag_state', None)
    if st is None:
        return
    cam_pos = np.array([camera_transform.translation.x,
                        camera_transform.translation.y,
                        camera_transform.translation.z], dtype=np.float32)
    target = _project_cursor_to_plane(camera_comp, cam_pos, cursor_x, cursor_y,
                                      win_w, win_h, st['plane_pt'], st['plane_n'])
    if target is None:
        return
    offset = (target - st['grab_world']).astype(np.float32)
    sim = comp.simulator
    pos = sim.positions.to_numpy()
    pos[st['verts']] = st['start_pos'] + offset
    sim.positions.from_numpy(pos)
    vel = sim.velocities.to_numpy()
    vel[st['verts']] = 0.0
    sim.velocities.from_numpy(vel)


def drag_end(comp):
    """Restore pinned/orphan state; leave the island where it was dragged."""
    st = getattr(comp, '_drag_state', None)
    if st is None:
        return
    sim = comp.simulator
    fixed = sim._fixed.to_numpy()
    fixed[st['verts']] = st['saved_fixed']
    sim._fixed.from_numpy(fixed)
    vel = sim.velocities.to_numpy()
    vel[st['verts']] = 0.0
    sim.velocities.from_numpy(vel)
    if st['saved_orphans'] is not None:
        (comp.method._orphan_ids,
         comp.method._orphan_masters,
         comp.method._orphan_offsets) = st['saved_orphans']
    comp._drag_state = None
    print(f"[Drag] released ({st['mode']}): {len(st['verts'])} verts dropped.")
