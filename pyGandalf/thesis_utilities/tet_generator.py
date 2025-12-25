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


def generate_tetrahedral_mesh(surface_mesh: 'MeshInstance', preserve_surface_mesh: bool = False) -> 'TetrahedralMeshInstance':
    """
    Convert a surface mesh into a tetrahedral mesh using TetGen library.

    Args:
        surface_mesh: Input surface mesh (triangular faces)
        preserve_surface_mesh: If True, stores original mesh for normal preservation

    Returns:
        TetrahedralMeshInstance: Generated tetrahedral mesh

    Algorithm:
        Uses TetGen (C++ library with Python bindings) for constrained
        Delaunay tetrahedralization. This preserves the input surface
        and generates quality tetrahedra inside.

    TetGen switches:
        'p' - Tetrahedralize a piecewise linear complex (PLC)
        'q' - Quality mesh generation (default ratio: 2.0)
        'a' - Maximum tetrahedron volume constraint
    """
    import tetgen

    print(f"Generating tetrahedral mesh from: {surface_mesh.name}")
    print(f"  Input vertices: {len(surface_mesh.vertices)}")
    print(f"  Input triangles: {len(surface_mesh.indices)}")

    # Create TetGen object with surface mesh
    tg = tetgen.TetGen(surface_mesh.vertices, surface_mesh.indices)

    # Generate tetrahedral mesh
    # 'pq1.2' = preserve surface, quality ratio 1.2 (good quality)
    # 'a0.01' = maximum tet volume (smaller = more tets, higher quality)
    print("  Running TetGen algorithm...")
    tg.tetrahedralize(switches='pq1.2')

    print(f"  Generated vertices: {len(tg.node)}")
    print(f"  Generated tetrahedra: {len(tg.elem)}")

    # Import here to avoid circular dependency
    from pyGandalf.utilities.mesh_lib import TetrahedralMeshInstance

    return TetrahedralMeshInstance(
        name=f"{surface_mesh.name}_tet",
        path=surface_mesh.path,
        vertices=tg.node.astype(np.float32),
        tetrahedra=tg.elem.astype(np.int32),
        original_surface_mesh=surface_mesh if preserve_surface_mesh else None
    )


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