"""
Tetrahedral Mesh Exploder for Visualization

Converts tetrahedral meshes to exploded surface meshes for visualization in pyGandalf.
Each tetrahedron is moved away from the center to create an "exploded view".
"""

import numpy as np
import taichi as ti
from pyGandalf.utilities.mesh_lib import MeshInstance, TetrahedralMeshInstance
from pathlib import Path

# Initialize Taichi for GPU acceleration
ti.init(arch=ti.gpu, default_fp=ti.f32)


def explode_tetrahedral_mesh(tet_mesh: TetrahedralMeshInstance, explosion_factor: float = 0.3) -> MeshInstance:
    """
    Create an exploded view of a tetrahedral mesh for visualization.

    Each tetrahedron is moved away from the model center, and all 4 faces
    of each tetrahedron are extracted as triangles for rendering.

    Args:
        tet_mesh: Input tetrahedral mesh
        explosion_factor: How far to move tetrahedra (0.0 = no movement, 1.0 = full explosion)

    Returns:
        MeshInstance with all tetrahedral faces as triangles
    """

    # Compute model center
    model_center = np.mean(tet_mesh.vertices, axis=0)

    # Each tetrahedron has 4 triangular faces
    # Face indices for a tetrahedron with vertices [v0, v1, v2, v3]:
    # Face 0: [v0, v2, v1]  (base, CCW from outside)
    # Face 1: [v0, v1, v3]  (side)
    # Face 2: [v0, v3, v2]  (side)
    # Face 3: [v1, v2, v3]  (top)
    tet_face_indices = np.array([
        [0, 2, 1],  # Base
        [0, 1, 3],  # Side 1
        [0, 3, 2],  # Side 2
        [1, 2, 3]   # Top
    ])

    num_tets = len(tet_mesh.tetrahedra)
    num_triangles = num_tets * 4

    # Pre-allocate arrays
    all_vertices = []
    all_triangles = []
    vertex_offset = 0

    print(f"Exploding {num_tets:,} tetrahedra...")

    for tet_idx, tet in enumerate(tet_mesh.tetrahedra):
        if tet_idx % 50000 == 0:
            print(f"  Processing tetrahedron {tet_idx:,}/{num_tets:,}...")

        # Get the 4 vertices of this tetrahedron
        tet_vertices = tet_mesh.vertices[tet]  # Shape: (4, 3)

        # Compute tetrahedron center
        tet_center = np.mean(tet_vertices, axis=0)

        # Compute explosion direction (from model center to tet center)
        explosion_direction = tet_center - model_center
        explosion_distance = np.linalg.norm(explosion_direction)

        if explosion_distance > 1e-6:  # Avoid division by zero
            explosion_direction = explosion_direction / explosion_distance
        else:
            explosion_direction = np.array([0, 0, 0])

        # Move vertices away from center
        explosion_offset = explosion_direction * explosion_distance * explosion_factor
        exploded_vertices = tet_vertices + explosion_offset

        # Add vertices for this tetrahedron
        all_vertices.append(exploded_vertices)

        # Add the 4 faces as triangles
        for face in tet_face_indices:
            triangle = [vertex_offset + face[0], vertex_offset + face[1], vertex_offset + face[2]]
            all_triangles.append(triangle)

        vertex_offset += 4

    # Combine all vertices and triangles
    vertices = np.vstack(all_vertices).astype(np.float32)
    indices = np.array(all_triangles, dtype=np.uint32)

    print(f"Created exploded mesh: {len(vertices):,} vertices, {len(indices):,} triangles")

    # Compute normals using Taichi (GPU-accelerated)
    print("Computing normals on GPU...")
    normals = _compute_normals_taichi(vertices, indices)
    print("Normals computed!")

    # Create MeshInstance for pyGandalf rendering
    exploded_mesh = MeshInstance(
        name=f"{tet_mesh.name}_exploded",
        path=tet_mesh.path,
        vertices=vertices,
        indices=indices,
        normals=normals,
        texcoords=None
    )

    return exploded_mesh


@ti.kernel
def _compute_normals_kernel(vertices: ti.types.ndarray(), indices: ti.types.ndarray(), normals: ti.types.ndarray()):
    """
    Taichi kernel to compute face normals in parallel on GPU

    Each triangle computes its normal and assigns it to all 3 vertices
    """
    for tri_idx in range(indices.shape[0]):
        # Get vertex indices
        v0 = indices[tri_idx, 0]
        v1 = indices[tri_idx, 1]
        v2 = indices[tri_idx, 2]

        # Get vertex positions
        p0 = ti.Vector([vertices[v0, 0], vertices[v0, 1], vertices[v0, 2]])
        p1 = ti.Vector([vertices[v1, 0], vertices[v1, 1], vertices[v1, 2]])
        p2 = ti.Vector([vertices[v2, 0], vertices[v2, 1], vertices[v2, 2]])

        # Compute edges
        edge1 = p1 - p0
        edge2 = p2 - p0

        # Compute cross product (face normal)
        normal = edge1.cross(edge2)

        # Normalize
        length = normal.norm()
        if length > 1e-6:
            normal = normal / length

        # Assign to all 3 vertices
        normals[v0, 0] = normal[0]
        normals[v0, 1] = normal[1]
        normals[v0, 2] = normal[2]

        normals[v1, 0] = normal[0]
        normals[v1, 1] = normal[1]
        normals[v1, 2] = normal[2]

        normals[v2, 0] = normal[0]
        normals[v2, 1] = normal[1]
        normals[v2, 2] = normal[2]


def _compute_normals_taichi(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Compute normals using Taichi GPU acceleration

    Args:
        vertices: (N, 3) array of vertex positions
        indices: (M, 3) array of triangle indices

    Returns:
        (N, 3) array of vertex normals
    """
    normals = np.zeros_like(vertices, dtype=np.float32)

    # Call Taichi kernel (runs on GPU)
    _compute_normals_kernel(vertices, indices, normals)

    return normals
