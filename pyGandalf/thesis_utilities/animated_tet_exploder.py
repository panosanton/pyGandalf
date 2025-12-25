"""
Animated Tetrahedral Explosion System

Creates an animated explosion effect where tetrahedra continuously move
away from and back towards the center in a loop.

Uses hybrid normals: smooth for boundary faces, flat for interior faces.
"""

import numpy as np
import math
import time
import taichi as ti
from pyGandalf.utilities.mesh_lib import MeshInstance, TetrahedralMeshInstance

# Initialize Taichi for GPU acceleration
ti.init(arch=ti.gpu, default_fp=ti.f32)


def _identify_boundary_faces(tetrahedra: np.ndarray) -> set:
    """
    Identify which faces are on the boundary (appear in only one tetrahedron).

    Returns:
        Set of boundary faces (each face is a sorted tuple of 3 vertex indices)
    """
    # Face index combinations
    face_indices = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ])

    # Generate all faces using vectorization
    all_faces = tetrahedra[:, face_indices].reshape(-1, 3)
    all_faces_sorted = np.sort(all_faces, axis=1)

    # Count occurrences
    unique_faces, counts = np.unique(all_faces_sorted, axis=0, return_counts=True)

    # Boundary faces appear exactly once
    boundary_faces = unique_faces[counts == 1]

    # Convert to set of tuples for fast lookup
    return set(tuple(face) for face in boundary_faces)


class AnimatedTetExplosion:
    """
    Stores data for animated tetrahedral explosion effect with hybrid normals
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

        # Face indices for tetrahedra
        self.tet_face_indices = np.array([
            [0, 2, 1],  # Base
            [0, 1, 3],  # Side 1
            [0, 3, 2],  # Side 2
            [1, 2, 3]   # Top
        ])

        # Identify boundary faces
        print(f"Identifying boundary faces for {self.num_tets:,} tetrahedra...")
        self.boundary_faces_set = _identify_boundary_faces(tet_mesh.tetrahedra)
        print(f"  Found {len(self.boundary_faces_set):,} boundary faces")

        # Build separate geometry for interior and boundary faces
        print(f"Building separate geometry...")
        self._build_geometry()

        print(f"Pre-computation complete:")
        print(f"  Interior: {len(self.tet_base_vertices):,} vertices, {len(self.tet_indices)//3:,} triangles")
        print(f"  Boundary: {len(self.boundary_base_vertices):,} vertices, {len(self.boundary_indices)//3:,} triangles")

    def _build_geometry(self):
        """Build separated interior and boundary geometry"""
        # Interior tetrahedral geometry
        tet_vertices_list = []
        tet_triangles_list = []
        tet_vertex_offset = 0

        # Boundary surface geometry (separate, duplicated vertices)
        boundary_vertices_list = []
        boundary_triangles_list = []
        boundary_vertex_offset = 0
        boundary_vertex_to_original = []

        # Track which tetrahedron each vertex/boundary-vertex belongs to
        self.tet_vertex_to_tet_idx = []
        self.boundary_vertex_to_tet_idx = []

        for tet_idx, tet in enumerate(self.tet_mesh.tetrahedra):
            tet_vertices = self.tet_mesh.vertices[tet]  # Shape: (4, 3)

            # Add vertices for this tetrahedron (for interior faces)
            tet_vertices_list.append(tet_vertices)

            # Track which tet each vertex belongs to (for explosion animation)
            self.tet_vertex_to_tet_idx.extend([tet_idx] * 4)

            # Process the 4 faces
            for face_local_indices in self.tet_face_indices:
                # Check if this face is a boundary face
                original_face = tuple(sorted([tet[face_local_indices[0]],
                                             tet[face_local_indices[1]],
                                             tet[face_local_indices[2]]]))
                is_boundary = original_face in self.boundary_faces_set

                if is_boundary:
                    # Boundary face: create separate vertices for this face
                    face_verts = tet_vertices[face_local_indices]  # Shape: (3, 3)
                    boundary_vertices_list.append(face_verts)

                    # Track which tet these boundary vertices belong to
                    self.boundary_vertex_to_tet_idx.extend([tet_idx] * 3)

                    # Track which original tet vertices these came from
                    for local_idx in face_local_indices:
                        original_tet_vertex_idx = tet[local_idx]
                        boundary_vertex_to_original.append(original_tet_vertex_idx)

                    # Add triangle using separate boundary vertex indices
                    triangle = [boundary_vertex_offset, boundary_vertex_offset + 1, boundary_vertex_offset + 2]
                    boundary_triangles_list.append(triangle)
                    boundary_vertex_offset += 3
                else:
                    # Interior face: use shared tetrahedron vertices
                    triangle = [tet_vertex_offset + face_local_indices[0],
                               tet_vertex_offset + face_local_indices[1],
                               tet_vertex_offset + face_local_indices[2]]
                    tet_triangles_list.append(triangle)

            tet_vertex_offset += 4

        # Convert to numpy arrays
        self.tet_base_vertices = np.vstack(tet_vertices_list).astype(np.float32)
        self.tet_indices = np.array(tet_triangles_list, dtype=np.uint32)
        self.tet_vertex_to_tet_idx = np.array(self.tet_vertex_to_tet_idx, dtype=np.int32)

        self.boundary_base_vertices = np.vstack(boundary_vertices_list).astype(np.float32)
        self.boundary_indices = np.array(boundary_triangles_list, dtype=np.uint32)
        self.boundary_vertex_to_tet_idx = np.array(self.boundary_vertex_to_tet_idx, dtype=np.int32)
        self.boundary_vertex_to_original = np.array(boundary_vertex_to_original, dtype=np.int32)

        # Pre-compute explosion directions for each tetrahedron
        self.explosion_directions = []
        self.explosion_distances = []

        for tet in self.tet_mesh.tetrahedra:
            tet_vertices = self.tet_mesh.vertices[tet]
            tet_center = np.mean(tet_vertices, axis=0)

            explosion_direction = tet_center - self.model_center
            explosion_distance = np.linalg.norm(explosion_direction)

            if explosion_distance > 1e-6:
                explosion_direction = explosion_direction / explosion_distance
            else:
                explosion_direction = np.array([0, 0, 0], dtype=np.float32)

            self.explosion_directions.append(explosion_direction)
            self.explosion_distances.append(explosion_distance)

        self.explosion_directions = np.array(self.explosion_directions, dtype=np.float32)
        self.explosion_distances = np.array(self.explosion_distances, dtype=np.float32)

    def update(self, delta_time: float) -> np.ndarray:
        """
        Update animation and return current vertex positions for BOTH interior and boundary geometry

        Args:
            delta_time: Time since last frame in seconds

        Returns:
            Combined updated vertex positions (interior vertices followed by boundary vertices)
        """
        start_time = time.perf_counter()

        self.time += delta_time * self.speed

        # Use sine wave for smooth back-and-forth animation
        # Range: 0 to max_explosion
        current_factor = (math.sin(self.time) + 1.0) * 0.5 * self.max_explosion

        # Compute explosion offsets for each tetrahedron
        explosion_offsets = (self.explosion_directions.T * self.explosion_distances * current_factor).T

        # Apply offsets to tet vertices (interior faces)
        updated_tet_vertices = self.tet_base_vertices + explosion_offsets[self.tet_vertex_to_tet_idx]

        # Apply offsets to boundary vertices (boundary faces)
        updated_boundary_vertices = self.boundary_base_vertices + explosion_offsets[self.boundary_vertex_to_tet_idx]

        # Combine for rendering: interior vertices first, then boundary vertices
        combined_vertices = np.vstack([updated_tet_vertices, updated_boundary_vertices])

        # Track performance
        elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
        self.total_update_time += elapsed
        self.frame_count += 1

        # Print stats every 100 frames
        if self.frame_count % 100 == 0:
            avg_time = self.total_update_time / self.frame_count
            print(f"[AnimatedTetExplosion] Avg update time: {avg_time:.2f}ms ({self.frame_count} frames)")

        return combined_vertices

    def create_initial_mesh(self) -> MeshInstance:
        """
        Create the initial mesh with hybrid normals (flat for interior, smooth for boundary)

        Returns:
            MeshInstance ready for rendering with combined geometry
        """
        print(f"Creating initial mesh with hybrid normals...")

        # Combine indices: adjust boundary indices by number of tet vertices
        num_tet_vertices = len(self.tet_base_vertices)
        adjusted_boundary_indices = self.boundary_indices + num_tet_vertices
        combined_indices = np.vstack([self.tet_indices.reshape(-1, 3), adjusted_boundary_indices.reshape(-1, 3)])

        # Combine vertices
        combined_vertices = np.vstack([self.tet_base_vertices, self.boundary_base_vertices])

        print(f"  Computing flat normals for {len(self.tet_indices)//3:,} interior triangles...")
        # Compute flat normals for interior faces (Taichi GPU-accelerated)
        tet_normals = _compute_flat_normals_taichi(self.tet_base_vertices, self.tet_indices.reshape(-1, 3))

        print(f"  Computing smooth normals for {len(self.boundary_indices)//3:,} boundary triangles...")
        # Compute smooth normals for boundary faces (Taichi GPU-accelerated)
        boundary_normals = _compute_smooth_normals_taichi(self.boundary_base_vertices, self.boundary_indices.reshape(-1, 3))

        # Combine normals
        combined_normals = np.vstack([tet_normals, boundary_normals])

        print(f"  Creating mesh instance...")
        mesh = MeshInstance(
            name=f"{self.tet_mesh.name}_animated_exploded",
            path=self.tet_mesh.path,
            vertices=combined_vertices.copy(),
            indices=combined_indices,
            normals=combined_normals,
            texcoords=None
        )

        print(f"  Done! Total: {len(combined_vertices):,} vertices, {len(combined_indices):,} triangles")
        return mesh


@ti.kernel
def _compute_flat_normals_kernel(vertices: ti.types.ndarray(), indices: ti.types.ndarray(), normals: ti.types.ndarray()):
    """
    Taichi kernel to compute flat normals (one normal per triangle, duplicated for all 3 vertices)
    Parallel on GPU
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

        # Compute face normal and normalize
        normal = edge1.cross(edge2)
        length = normal.norm()
        if length > 1e-6:
            normal = normal / length

        # Set the same normal for all 3 vertices of this triangle
        normals[v0, 0] = normal[0]
        normals[v0, 1] = normal[1]
        normals[v0, 2] = normal[2]

        normals[v1, 0] = normal[0]
        normals[v1, 1] = normal[1]
        normals[v1, 2] = normal[2]

        normals[v2, 0] = normal[0]
        normals[v2, 1] = normal[1]
        normals[v2, 2] = normal[2]


def _compute_flat_normals_taichi(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Compute flat normals using Taichi GPU acceleration

    Each triangle gets its own face normal, duplicated for all 3 vertices.

    Args:
        vertices: (N, 3) array of vertex positions
        indices: (M, 3) array of triangle indices

    Returns:
        (N, 3) array of flat normals
    """
    normals = np.zeros_like(vertices, dtype=np.float32)
    _compute_flat_normals_kernel(vertices, indices, normals)
    return normals


@ti.kernel
def _accumulate_smooth_normals_kernel(vertices: ti.types.ndarray(), indices: ti.types.ndarray(), normals: ti.types.ndarray()):
    """
    Taichi kernel to accumulate face normals for each vertex (for smooth normals)
    Parallel on GPU
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

        # Compute face normal (not normalized yet)
        normal = edge1.cross(edge2)

        # Accumulate to all 3 vertices using atomic operations
        ti.atomic_add(normals[v0, 0], normal[0])
        ti.atomic_add(normals[v0, 1], normal[1])
        ti.atomic_add(normals[v0, 2], normal[2])

        ti.atomic_add(normals[v1, 0], normal[0])
        ti.atomic_add(normals[v1, 1], normal[1])
        ti.atomic_add(normals[v1, 2], normal[2])

        ti.atomic_add(normals[v2, 0], normal[0])
        ti.atomic_add(normals[v2, 1], normal[1])
        ti.atomic_add(normals[v2, 2], normal[2])


@ti.kernel
def _normalize_normals_kernel(normals: ti.types.ndarray()):
    """
    Taichi kernel to normalize accumulated normals
    Parallel on GPU
    """
    for i in range(normals.shape[0]):
        normal = ti.Vector([normals[i, 0], normals[i, 1], normals[i, 2]])
        length = normal.norm()
        if length > 1e-6:
            normal = normal / length
            normals[i, 0] = normal[0]
            normals[i, 1] = normal[1]
            normals[i, 2] = normal[2]


def _compute_smooth_normals_taichi(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Compute smooth vertex normals using Taichi GPU acceleration

    Averages face normals for all triangles sharing each vertex.

    Args:
        vertices: (N, 3) array of vertex positions
        indices: (M, 3) array of triangle indices

    Returns:
        (N, 3) array of smooth vertex normals
    """
    normals = np.zeros_like(vertices, dtype=np.float32)

    # Step 1: Accumulate face normals for each vertex (GPU parallel)
    _accumulate_smooth_normals_kernel(vertices, indices, normals)

    # Step 2: Normalize the accumulated normals (GPU parallel)
    _normalize_normals_kernel(normals)

    return normals
