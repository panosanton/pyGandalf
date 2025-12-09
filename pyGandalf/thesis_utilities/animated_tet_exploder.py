"""
Animated Tetrahedral Explosion System

Creates an animated explosion effect where tetrahedra continuously move
away from and back towards the center in a loop.
"""

import numpy as np
import math
import time
import taichi as ti
from pyGandalf.utilities.mesh_lib import MeshInstance, TetrahedralMeshInstance

# Initialize Taichi for GPU acceleration
ti.init(arch=ti.gpu, default_fp=ti.f32)


class AnimatedTetExplosion:
    """
    Stores data for animated tetrahedral explosion effect
    """
    def __init__(self, tet_mesh: TetrahedralMeshInstance, max_explosion: float = 0.5, speed: float = 1.0):
        """
        Args:
            tet_mesh: Original tetrahedral mesh
            max_explosion: Maximum explosion distance
            speed: Animation speed (higher = faster)
        """
        self.tet_mesh = tet_mesh
        self.max_explosion = max_explosion
        self.speed = speed
        self.time = 0.0

        # Performance tracking
        self.frame_count = 0
        self.total_update_time = 0.0

        # Pre-compute static data
        self.model_center = np.mean(tet_mesh.vertices, axis=0)
        self.num_tets = len(tet_mesh.tetrahedra)

        # Pre-compute tetrahedron centers and directions
        print(f"Pre-computing explosion data for {self.num_tets:,} tetrahedra...")
        self.tet_centers = []
        self.explosion_directions = []
        self.explosion_distances = []

        for tet in tet_mesh.tetrahedra:
            tet_vertices = tet_mesh.vertices[tet]
            tet_center = np.mean(tet_vertices, axis=0)

            explosion_direction = tet_center - self.model_center
            explosion_distance = np.linalg.norm(explosion_direction)

            if explosion_distance > 1e-6:
                explosion_direction = explosion_direction / explosion_distance
            else:
                explosion_direction = np.array([0, 0, 0])

            self.tet_centers.append(tet_center)
            self.explosion_directions.append(explosion_direction)
            self.explosion_distances.append(explosion_distance)

        self.tet_centers = np.array(self.tet_centers)
        self.explosion_directions = np.array(self.explosion_directions)
        self.explosion_distances = np.array(self.explosion_distances)

        # Face indices for tetrahedra
        self.tet_face_indices = np.array([
            [0, 2, 1],  # Base
            [0, 1, 3],  # Side 1
            [0, 3, 2],  # Side 2
            [1, 2, 3]   # Top
        ])

        # Pre-allocate vertex array
        self.base_vertices = []
        self.vertex_to_tet = []  # Maps each vertex to its tetrahedron index

        for tet_idx, tet in enumerate(tet_mesh.tetrahedra):
            tet_vertices = tet_mesh.vertices[tet]
            self.base_vertices.extend(tet_vertices)
            self.vertex_to_tet.extend([tet_idx] * 4)

        self.base_vertices = np.array(self.base_vertices, dtype=np.float32)
        self.vertex_to_tet = np.array(self.vertex_to_tet)

        print(f"Pre-computation complete: {len(self.base_vertices):,} vertices")

    def update(self, delta_time: float) -> np.ndarray:
        """
        Update animation and return current vertex positions

        Args:
            delta_time: Time since last frame in seconds

        Returns:
            Updated vertex positions
        """
        start_time = time.perf_counter()

        self.time += delta_time * self.speed

        # Use sine wave for smooth back-and-forth animation
        # Range: 0 to max_explosion
        current_factor = (math.sin(self.time) + 1.0) * 0.5 * self.max_explosion

        # Compute explosion offsets for each tetrahedron
        explosion_offsets = (self.explosion_directions.T * self.explosion_distances * current_factor).T

        # Apply offsets to vertices using NumPy fancy indexing (fast vectorized operation)
        # vertex_to_tet maps each vertex to its tetrahedron, so explosion_offsets[vertex_to_tet]
        # gives us the correct offset for each vertex
        updated_vertices = self.base_vertices + explosion_offsets[self.vertex_to_tet]

        # Track performance
        elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
        self.total_update_time += elapsed
        self.frame_count += 1

        # Print stats every 100 frames
        if self.frame_count % 100 == 0:
            avg_time = self.total_update_time / self.frame_count
            print(f"[AnimatedTetExplosion] Avg update time: {avg_time:.2f}ms ({self.frame_count} frames)")

        return updated_vertices

    def create_initial_mesh(self) -> MeshInstance:
        """
        Create the initial mesh with all tetrahedral faces

        Returns:
            MeshInstance ready for rendering
        """
        # Create triangle indices
        all_triangles = []
        vertex_offset = 0

        for _ in range(self.num_tets):
            for face in self.tet_face_indices:
                triangle = [vertex_offset + face[0], vertex_offset + face[1], vertex_offset + face[2]]
                all_triangles.append(triangle)
            vertex_offset += 4

        indices = np.array(all_triangles, dtype=np.uint32)

        # Compute initial normals using Taichi (GPU-accelerated)
        normals = _compute_normals_taichi(self.base_vertices, indices)

        mesh = MeshInstance(
            name=f"{self.tet_mesh.name}_animated_exploded",
            path=self.tet_mesh.path,
            vertices=self.base_vertices.copy(),
            indices=indices,
            normals=normals,
            texcoords=None
        )

        return mesh


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
