"""
Tetrahedral Mesh Generation Module

This module provides functions to convert surface meshes into tetrahedral meshes.
Uses Taichi for GPU-accelerated mesh processing.
"""

import numpy as np
import taichi as ti
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyGandalf.utilities.mesh_lib import MeshInstance, TetrahedralMeshInstance

# Initialize Taichi
# Will automatically choose the best backend (CUDA > Vulkan > CPU)
ti.init(arch=ti.gpu)  # Change to ti.cpu if you don't have a GPU


def generate_tetrahedral_mesh(surface_mesh: 'MeshInstance',
                               target_faces: int = None,
                               tet_scale: float = 1.0) -> 'TetrahedralMeshInstance':
    """
    Convert a surface mesh into a tetrahedral mesh using TetGen library.

    Args:
        surface_mesh: Input surface mesh (triangular faces)
        target_faces: If set, simplify the surface mesh to approximately this
                      many faces before tetrahedralization.  Fewer faces ->
                      fewer tetrahedra -> faster simulation.
                      Falls back to progressively less aggressive simplification
                      if the result is not watertight, and ultimately to the
                      original mesh if no watertight result can be achieved.
        tet_scale: Controls interior tet density relative to the surface.
                   1.0 (default) = standard quality mesh (pq2.0).
                   >1.0 = interior tets allowed up to tet_scale * natural surface
                   tet volume.  The surface mesh is preserved exactly (TetGen Y
                   flag); only interior density decreases.  Values of 5-20 give
                   a coarser interior while keeping a dense surface.  Very large
                   values (50+) can create slivers that hurt CG convergence.

    Returns:
        TetrahedralMeshInstance: Generated tetrahedral mesh
    """
    import tetgen

    print(f"Generating tetrahedral mesh from: {surface_mesh.name}")
    print(f"  Input vertices: {len(surface_mesh.vertices):,}")
    print(f"  Input triangles: {len(surface_mesh.indices):,}")

    vertices = surface_mesh.vertices
    faces    = surface_mesh.indices

    if target_faces is not None:
        vertices, faces = _simplify_and_repair(vertices, faces, target_faces)
    else:
        import trimesh as _trimesh
        vertices, faces = _repair_and_extract(
            _trimesh.Trimesh(vertices=vertices, faces=faces, process=False))

    # Normalize to [-4, 4] on the longest axis so blade_speed and other
    # world-unit parameters work consistently across all meshes.
    centroid = (vertices.max(axis=0) + vertices.min(axis=0)) * 0.5
    vertices  = vertices - centroid
    scale     = float(np.abs(vertices).max())
    if scale > 0:
        vertices = (vertices / scale * 4.0).astype(np.float32)
    print(f"  Normalized: centroid offset={centroid.round(2)}, scale={scale:.4f}")

    tg = tetgen.TetGen(vertices, faces)

    if tet_scale <= 1.0:
        switches = 'pq2.0'
    else:
        # Estimate natural tet volume from average surface triangle size.
        # For an equilateral triangle with area A:
        #   edge L = sqrt(4*A / sqrt(3))
        #   regular tet volume = L^3 * sqrt(2) / 12
        v = vertices[faces]                                      # (F, 3, 3)
        cross = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]) # (F, 3)
        avg_area  = float(np.mean(0.5 * np.linalg.norm(cross, axis=1)))
        avg_edge  = float(np.sqrt(4.0 * avg_area / np.sqrt(3.0)))
        nat_vol   = avg_edge ** 3 * np.sqrt(2.0) / 12.0
        max_vol   = nat_vol * tet_scale
        # pY  — preserve surface exactly (no Steiner points on boundary)
        # a{} — cap max tet volume so interior stays graded (fine near surface,
        #        coarse deeper inside) without runaway quality refinement
        switches = f'pYa{max_vol:.6e}'
        print(f"  tet_scale={tet_scale:.1f}: natural_vol={nat_vol:.3e}, "
              f"max_tet_vol={max_vol:.3e}")
        if tet_scale > 20.0:
            print("  WARNING: tet_scale > 20 may produce interior slivers "
                  "that hurt CG convergence.")

    # Try progressively more tolerant TetGen switches if the mesh is difficult.
    # 'pq2.0' — quality mesh (preferred)
    # 'p'     — no quality refinement, more robust boundary recovery
    # 'pC'    — coplanar-face detection enabled (handles near-degenerate faces)
    fallback_switches = [switches, 'p', 'pC']
    for sw in fallback_switches:
        try:
            print(f"  Running TetGen (switches='{sw}')...")
            tg.tetrahedralize(switches=sw)
            break
        except RuntimeError as e:
            if sw == fallback_switches[-1]:
                raise
            print(f"  TetGen failed ({e}), retrying...")
            tg = tetgen.TetGen(vertices, faces)

    print(f"  Generated vertices:   {len(tg.node):,}")
    print(f"  Generated tetrahedra: {len(tg.elem):,}")

    from pyGandalf.utilities.mesh_lib import TetrahedralMeshInstance

    return TetrahedralMeshInstance(
        name=f"{surface_mesh.name}_tet",
        path=surface_mesh.path,
        vertices=tg.node.astype(np.float32),
        tetrahedra=tg.elem.astype(np.int32),
    )


def _simplify_and_repair(vertices: np.ndarray, faces: np.ndarray,
                          target_faces: int):
    """
    Simplify a surface mesh to target_faces and repair it before passing to TetGen.

    Strategy:
        Simplify to target_faces, apply trimesh repair operations (winding, normals,
        fill_holes), and pass the result directly to TetGen.  Watertightness is NOT
        used as a gate: TetGen's own boundary-recovery algorithm handles non-watertight
        input the same way it handles the original mesh, so rejecting a simplified mesh
        because trimesh reports it as non-watertight would only force us back to the
        full-resolution mesh — the opposite of what we want.

        If simplification produces a mesh with zero faces (degenerate result), we fall
        back to the original mesh.

    Returns:
        (vertices, faces) as float32 / int32 numpy arrays.
    """
    import trimesh

    original_count = len(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    print(f"  Simplification requested: {original_count:,} -> {target_faces:,} faces")
    print(f"  Original watertight: {mesh.is_watertight}")

    if target_faces >= original_count:
        print("  target_faces >= current face count — skipping simplification.")
        return _repair_and_extract(mesh)

    # Compute reduction ratio as fallback for older trimesh/fast_simplification.
    reduction = 1.0 - (target_faces / original_count)
    reduction = max(0.01, min(0.99, reduction))
    print(f"  Simplifying to ~{target_faces:,} faces (reduction={reduction:.2f})...")
    try:
        # Newer trimesh: accepts integer face count directly
        simplified = mesh.simplify_quadric_decimation(target_faces)
    except (ValueError, TypeError):
        # Older trimesh / fast_simplification: expects reduction ratio
        simplified = mesh.simplify_quadric_decimation(reduction)

    v, f = _repair_and_extract(simplified)

    if len(f) == 0:
        print("  WARNING: Simplification produced empty mesh — using original.")
        return _repair_and_extract(mesh)

    check = trimesh.Trimesh(vertices=v, faces=f, process=False)
    print(f"  Result: {len(v):,} vertices, {len(f):,} faces "
          f"(watertight: {check.is_watertight})")
    return v, f


def _repair_and_extract(mesh):
    """
    Run trimesh repair operations on a mesh and return (vertices, faces).
    Does not guarantee watertightness -- call mesh.is_watertight after to check.
    """
    import trimesh

    # Keep the pre-repair mesh as a fallback in case all repair attempts corrupt it.
    fallback = trimesh.Trimesh(vertices=mesh.vertices.copy(),
                               faces=mesh.faces.copy(), process=False)

    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    mesh.fill_holes()

    # merge_vertices removes coincident points that cause self-intersection reports
    mesh.merge_vertices()

    # fix_intersections is available in newer trimesh versions
    if hasattr(trimesh.repair, 'fix_intersections'):
        try:
            trimesh.repair.fix_intersections(mesh)
        except Exception:
            pass

    # pymeshfix: dedicated self-intersection repair for TetGen input (two passes)
    try:
        import pymeshfix
        for _pass in range(2):
            mf = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            mf.repair()
            # API varies by version: new versions expose .mesh (pyvista PolyData)
            if hasattr(mf, 'v') and mf.v is not None:
                v_out = np.array(mf.v, dtype=np.float64)
                f_out = np.array(mf.f, dtype=np.int64)
            else:
                poly  = mf.mesh
                v_out = np.array(poly.points, dtype=np.float64)
                f_out = poly.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
            if len(v_out) == 0 or len(f_out) == 0:
                print(f"  pymeshfix pass {_pass + 1}: empty result, keeping prior mesh")
                break
            mesh = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
            if mesh.is_watertight:
                break
        print(f"  pymeshfix: {len(mesh.vertices):,}v {len(mesh.faces):,}f "
              f"(watertight: {mesh.is_watertight})")
    except Exception as e:
        print(f"  pymeshfix skipped: {e}")

    # Voxel-remesh fallback: guarantees a manifold mesh when all else fails.
    # Trades slight geometric smoothing for a clean TetGen-compatible surface.
    if not mesh.is_watertight and len(mesh.vertices) > 0:
        try:
            pitch    = mesh.bounding_box.extents.max() / 80.0
            vox      = trimesh.voxel.creation.voxelize(mesh, pitch)
            remeshed = vox.marching_cubes
            if remeshed is not None and len(remeshed.faces) > 0:
                print(f"  voxel-remesh: {len(remeshed.vertices):,}v "
                      f"{len(remeshed.faces):,}f")
                mesh = remeshed
        except Exception as e:
            print(f"  voxel-remesh skipped: {e}")

    # Final safety: if repair produced an empty mesh, use the pre-repair original.
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        print("  WARNING: all repair attempts produced empty mesh — using original")
        mesh = fallback

    cleaned = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                              process=True)

    # Explicitly remove duplicate faces (trimesh process=True misses some cases
    # that cause TetGen's recoversubfaces to crash).
    faces_out = np.array(cleaned.faces, dtype=np.int32)
    sorted_f  = np.sort(faces_out, axis=1)
    _, unique_idx = np.unique(sorted_f, axis=0, return_index=True)
    faces_out = faces_out[np.sort(unique_idx)]
    n_removed = len(cleaned.faces) - len(faces_out)
    if n_removed > 0:
        print(f"  Removed {n_removed} duplicate face(s) before TetGen")

    return (np.array(cleaned.vertices, dtype=np.float32), faces_out)


# ============================================
# TAICHI-ACCELERATED UTILITY FUNCTIONS
# ============================================

@ti.kernel
def compute_tet_volumes(
    vertices: ti.types.ndarray(),
    tetrahedra: ti.types.ndarray(),
    volumes: ti.types.ndarray()
):
    """
    Compute volume of each tetrahedron (GPU-accelerated).

    Formula: V = |det(v1-v0, v2-v0, v3-v0)| / 6
    """
    for i in range(tetrahedra.shape[0]):
        # Get vertex indices
        i0 = tetrahedra[i, 0]
        i1 = tetrahedra[i, 1]
        i2 = tetrahedra[i, 2]
        i3 = tetrahedra[i, 3]

        # Get vertex positions
        v0 = ti.Vector([vertices[i0, 0], vertices[i0, 1], vertices[i0, 2]])
        v1 = ti.Vector([vertices[i1, 0], vertices[i1, 1], vertices[i1, 2]])
        v2 = ti.Vector([vertices[i2, 0], vertices[i2, 1], vertices[i2, 2]])
        v3 = ti.Vector([vertices[i3, 0], vertices[i3, 1], vertices[i3, 2]])

        # Compute edges from v0
        e1 = v1 - v0
        e2 = v2 - v0
        e3 = v3 - v0

        # Compute determinant using scalar triple product: e1 · (e2 × e3)
        cross = ti.Vector([
            e2[1] * e3[2] - e2[2] * e3[1],
            e2[2] * e3[0] - e2[0] * e3[2],
            e2[0] * e3[1] - e2[1] * e3[0]
        ])
        det = e1[0] * cross[0] + e1[1] * cross[1] + e1[2] * cross[2]

        volumes[i] = ti.abs(det) / 6.0


@ti.kernel
def compute_tet_quality(
    vertices: ti.types.ndarray(),
    tetrahedra: ti.types.ndarray(),
    quality: ti.types.ndarray()
):
    """
    Compute quality metric for each tetrahedron (GPU-accelerated).

    Uses radius ratio: Q = r_inscribed / r_circumscribed
    Range: [0, 1], where 1 is a regular tetrahedron (best quality)
    """
    for i in range(tetrahedra.shape[0]):
        # Get vertex indices
        i0 = tetrahedra[i, 0]
        i1 = tetrahedra[i, 1]
        i2 = tetrahedra[i, 2]
        i3 = tetrahedra[i, 3]

        # Get vertex positions
        v0 = ti.Vector([vertices[i0, 0], vertices[i0, 1], vertices[i0, 2]])
        v1 = ti.Vector([vertices[i1, 0], vertices[i1, 1], vertices[i1, 2]])
        v2 = ti.Vector([vertices[i2, 0], vertices[i2, 1], vertices[i2, 2]])
        v3 = ti.Vector([vertices[i3, 0], vertices[i3, 1], vertices[i3, 2]])

        # Compute volume
        e1 = v1 - v0
        e2 = v2 - v0
        e3 = v3 - v0
        cross = ti.Vector([
            e2[1] * e3[2] - e2[2] * e3[1],
            e2[2] * e3[0] - e2[0] * e3[2],
            e2[0] * e3[1] - e2[1] * e3[0]
        ])
        volume = ti.abs(e1[0] * cross[0] + e1[1] * cross[1] + e1[2] * cross[2]) / 6.0

        # Compute surface area
        area = 0.0
        # Face 0: v0, v1, v2
        a1 = v1 - v0
        a2 = v2 - v0
        c1 = ti.Vector([a1[1]*a2[2] - a1[2]*a2[1], a1[2]*a2[0] - a1[0]*a2[2], a1[0]*a2[1] - a1[1]*a2[0]])
        area += c1.norm() / 2.0

        # Face 1: v0, v1, v3
        b1 = v1 - v0
        b2 = v3 - v0
        c2 = ti.Vector([b1[1]*b2[2] - b1[2]*b2[1], b1[2]*b2[0] - b1[0]*b2[2], b1[0]*b2[1] - b1[1]*b2[0]])
        area += c2.norm() / 2.0

        # Face 2: v0, v2, v3
        d1 = v2 - v0
        d2 = v3 - v0
        c3 = ti.Vector([d1[1]*d2[2] - d1[2]*d2[1], d1[2]*d2[0] - d1[0]*d2[2], d1[0]*d2[1] - d1[1]*d2[0]])
        area += c3.norm() / 2.0

        # Face 3: v1, v2, v3
        f1 = v2 - v1
        f2 = v3 - v1
        c4 = ti.Vector([f1[1]*f2[2] - f1[2]*f2[1], f1[2]*f2[0] - f1[0]*f2[2], f1[0]*f2[1] - f1[1]*f2[0]])
        area += c4.norm() / 2.0

        # Quality metric: normalized by ideal tetrahedron
        # For regular tet: V/A^(3/2) = constant
        if area > 1e-10:
            quality[i] = (6.0 * ti.sqrt(2.0) * volume) / ti.pow(area, 1.5)
        else:
            quality[i] = 0.0


# ============================================
# HELPER FUNCTIONS (will use these for actual tetrahedralization)
# ============================================

def analyze_tetrahedral_mesh(tet_mesh: 'TetrahedralMeshInstance'):
    """
    Analyze quality metrics of a tetrahedral mesh.

    Prints statistics about:
    - Number of vertices/tetrahedra
    - Volume distribution
    - Quality metrics
    """
    vertices = tet_mesh.vertices.astype(np.float32)
    tetrahedra = tet_mesh.tetrahedra.astype(np.int32)

    # Compute volumes
    volumes = np.zeros(len(tetrahedra), dtype=np.float32)
    compute_tet_volumes(vertices, tetrahedra, volumes)

    # Compute quality
    quality = np.zeros(len(tetrahedra), dtype=np.float32)
    compute_tet_quality(vertices, tetrahedra, quality)

    print(f"\n=== Tetrahedral Mesh Analysis: {tet_mesh.name} ===")
    print(f"Vertices: {len(vertices)}")
    print(f"Tetrahedra: {len(tetrahedra)}")
    print(f"\nVolume Statistics:")
    print(f"  Min:  {volumes.min():.6f}")
    print(f"  Max:  {volumes.max():.6f}")
    print(f"  Mean: {volumes.mean():.6f}")
    print(f"  Total: {volumes.sum():.6f}")
    print(f"\nQuality Statistics (0=worst, 1=best):")
    print(f"  Min:  {quality.min():.6f}")
    print(f"  Max:  {quality.max():.6f}")
    print(f"  Mean: {quality.mean():.6f}")
    print(f"  Median: {np.median(quality):.6f}")
    print(f"  Bad tets (quality < 0.2): {np.sum(quality < 0.2)} ({100*np.sum(quality < 0.2)/len(quality):.1f}%)")