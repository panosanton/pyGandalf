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


_DROP_REASON_NAMES = {0: "kept", 1: "PURE_ORIG_DROP", 2: "PHANTOM_COLLAR_DROP",
                      3: "COLLAR_RECON_DROP"}
_VERDICT_NAMES = {0: "OUTER", 1: "WOUND", 2: "DROP"}


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
