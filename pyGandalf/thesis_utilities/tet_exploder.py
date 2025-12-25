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


def explode_tetrahedral_mesh(tet_mesh: TetrahedralMeshInstance, explosion_factor: float = 0.3, use_hybrid_normals: bool = True) -> MeshInstance:
    """
    Create an exploded view of a tetrahedral mesh for visualization.

    Each tetrahedron is moved away from the model center, and all 4 faces
    of each tetrahedron are extracted as triangles for rendering.

    Args:
        tet_mesh: Input tetrahedral mesh
        explosion_factor: How far to move tetrahedra (0.0 = no movement, 1.0 = full explosion)
        use_hybrid_normals: If True and original surface mesh available, uses smooth normals
                           for boundary faces and flat normals for interior faces

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

    # Step 1: Identify boundary faces if hybrid normals requested
    boundary_faces_set = None
    if use_hybrid_normals and tet_mesh.original_surface_mesh is not None:
        print("Identifying boundary faces for hybrid normals...")
        boundary_faces_set = _identify_boundary_faces(tet_mesh.tetrahedra)
        print(f"  Found {len(boundary_faces_set):,} boundary faces")

    # Pre-allocate arrays for tetrahedral interior faces
    tet_vertices_list = []
    tet_triangles_list = []
    tet_vertex_offset = 0

    # Pre-allocate arrays for boundary surface faces (separate geometry)
    boundary_vertices_list = []
    boundary_triangles_list = []
    boundary_vertex_offset = 0
    boundary_vertex_to_original = []  # Map boundary vertex index -> original tet vertex index

    exploded_to_original_vertex_map = []  # Map exploded vertex index -> original tet vertex index

    # Pre-compute all exploded vertex positions using GPU
    print(f"Exploding {num_tets:,} tetrahedra on GPU...")
    exploded_all_tets = np.zeros((num_tets, 4, 3), dtype=np.float32)

    # Convert model_center to Taichi vector
    model_center_ti = [model_center[0], model_center[1], model_center[2]]

    # Call Taichi kernel to explode all tetrahedra in parallel
    _explode_tetrahedra_kernel(tet_mesh.vertices.astype(np.float32),
                               tet_mesh.tetrahedra.astype(np.int32),
                               model_center_ti,
                               explosion_factor,
                               exploded_all_tets)

    print(f"Building face lists for {num_tets:,} tetrahedra...")

    for tet_idx, tet in enumerate(tet_mesh.tetrahedra):
        if tet_idx % 100000 == 0 and tet_idx > 0:
            print(f"  Processing tetrahedron {tet_idx:,}/{num_tets:,}...")

        # Get pre-computed exploded vertices for this tetrahedron
        exploded_vertices = exploded_all_tets[tet_idx]

        # Add vertices for this tetrahedron (for interior faces)
        tet_vertices_list.append(exploded_vertices)

        # Track mapping from exploded vertex indices to original tet vertex indices
        # tet[0], tet[1], tet[2], tet[3] are the original vertex indices in the tetrahedral mesh
        for local_idx in range(4):
            exploded_to_original_vertex_map.append(tet[local_idx])

        # Process the 4 faces
        for face_local_indices in tet_face_indices:
            # Check if this face is a boundary face
            is_boundary = False
            if boundary_faces_set is not None:
                # Get original vertex indices for this face
                original_face = tuple(sorted([tet[face_local_indices[0]],
                                             tet[face_local_indices[1]],
                                             tet[face_local_indices[2]]]))
                is_boundary = original_face in boundary_faces_set

            if use_hybrid_normals and boundary_faces_set is not None:
                if is_boundary:
                    # Boundary face: create separate vertices for this face
                    face_verts = exploded_vertices[face_local_indices]  # Shape: (3, 3)
                    boundary_vertices_list.append(face_verts)

                    # Track which original tet vertices these boundary vertices came from
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
            else:
                # Not using hybrid normals - add all faces to tet geometry
                triangle = [tet_vertex_offset + face_local_indices[0],
                           tet_vertex_offset + face_local_indices[1],
                           tet_vertex_offset + face_local_indices[2]]
                tet_triangles_list.append(triangle)

        tet_vertex_offset += 4

    # Combine tetrahedral and boundary geometry
    if use_hybrid_normals and boundary_faces_set is not None and len(boundary_vertices_list) > 0:
        print("Combining tetrahedral interior and boundary surface geometry...")

        # Combine tet vertices
        tet_vertices = np.vstack(tet_vertices_list).astype(np.float32)
        tet_indices = np.array(tet_triangles_list, dtype=np.uint32)

        # Combine boundary vertices
        boundary_vertices = np.vstack(boundary_vertices_list).astype(np.float32)
        boundary_indices = np.array(boundary_triangles_list, dtype=np.uint32)

        # Merge both geometries: boundary vertices come after tet vertices
        # Offset boundary indices to account for tet vertices
        boundary_indices_offset = boundary_indices + len(tet_vertices)

        vertices = np.vstack([tet_vertices, boundary_vertices]).astype(np.float32)
        indices = np.vstack([tet_indices.reshape(-1, 3), boundary_indices_offset.reshape(-1, 3)]).astype(np.uint32)

        print(f"Created exploded mesh: {len(vertices):,} vertices, {len(indices):,} triangles")
        print(f"  Tetrahedral interior: {len(tet_vertices):,} vertices, {len(tet_indices)//3:,} triangles")
        print(f"  Boundary surface: {len(boundary_vertices):,} vertices, {len(boundary_indices):,} triangles")

        # Compute normals
        print("Computing hybrid normals (smooth boundary + flat interior)...")
        normals = _compute_hybrid_normals_separate(tet_vertices, tet_indices,
                                                   boundary_vertices, boundary_indices_offset,
                                                   tet_mesh.original_surface_mesh,
                                                   boundary_vertex_to_original)
    else:
        # Standard approach: all faces together
        vertices = np.vstack(tet_vertices_list).astype(np.float32)
        indices = np.array(tet_triangles_list, dtype=np.uint32)

        print(f"Created exploded mesh: {len(vertices):,} vertices, {len(indices):,} triangles")

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


@ti.kernel
def _explode_tetrahedra_kernel(vertices: ti.types.ndarray(),
                               tetrahedra: ti.types.ndarray(),
                               model_center: ti.types.vector(3, ti.f32),
                               explosion_factor: ti.f32,
                               exploded_tet_vertices: ti.types.ndarray()):
    """
    Taichi kernel to compute exploded vertex positions for all tetrahedra.
    Each tetrahedron's 4 vertices are moved away from the model center.

    Output: exploded_tet_vertices[tet_idx, local_vertex_idx, xyz]
    """
    num_tets = tetrahedra.shape[0]

    for tet_idx in range(num_tets):
        # Get the 4 vertex indices for this tetrahedron
        v0_idx = tetrahedra[tet_idx, 0]
        v1_idx = tetrahedra[tet_idx, 1]
        v2_idx = tetrahedra[tet_idx, 2]
        v3_idx = tetrahedra[tet_idx, 3]

        # Get the 4 vertex positions
        v0 = ti.Vector([vertices[v0_idx, 0], vertices[v0_idx, 1], vertices[v0_idx, 2]])
        v1 = ti.Vector([vertices[v1_idx, 0], vertices[v1_idx, 1], vertices[v1_idx, 2]])
        v2 = ti.Vector([vertices[v2_idx, 0], vertices[v2_idx, 1], vertices[v2_idx, 2]])
        v3 = ti.Vector([vertices[v3_idx, 0], vertices[v3_idx, 1], vertices[v3_idx, 2]])

        # Compute tetrahedron center
        tet_center = (v0 + v1 + v2 + v3) * 0.25

        # Compute explosion direction (from model center to tet center)
        explosion_direction = tet_center - model_center
        explosion_distance = explosion_direction.norm()

        if explosion_distance > 1e-6:
            explosion_direction = explosion_direction / explosion_distance
        else:
            explosion_direction = ti.Vector([0.0, 0.0, 0.0])

        # Compute explosion offset
        explosion_offset = explosion_direction * explosion_distance * explosion_factor

        # Apply explosion to all 4 vertices and store
        exploded_v0 = v0 + explosion_offset
        exploded_v1 = v1 + explosion_offset
        exploded_v2 = v2 + explosion_offset
        exploded_v3 = v3 + explosion_offset

        # Store exploded vertices
        exploded_tet_vertices[tet_idx, 0, 0] = exploded_v0[0]
        exploded_tet_vertices[tet_idx, 0, 1] = exploded_v0[1]
        exploded_tet_vertices[tet_idx, 0, 2] = exploded_v0[2]

        exploded_tet_vertices[tet_idx, 1, 0] = exploded_v1[0]
        exploded_tet_vertices[tet_idx, 1, 1] = exploded_v1[1]
        exploded_tet_vertices[tet_idx, 1, 2] = exploded_v1[2]

        exploded_tet_vertices[tet_idx, 2, 0] = exploded_v2[0]
        exploded_tet_vertices[tet_idx, 2, 1] = exploded_v2[1]
        exploded_tet_vertices[tet_idx, 2, 2] = exploded_v2[2]

        exploded_tet_vertices[tet_idx, 3, 0] = exploded_v3[0]
        exploded_tet_vertices[tet_idx, 3, 1] = exploded_v3[1]
        exploded_tet_vertices[tet_idx, 3, 2] = exploded_v3[2]


@ti.kernel
def _compute_flat_normals_kernel(vertices: ti.types.ndarray(),
                                 indices: ti.types.ndarray(),
                                 normals: ti.types.ndarray()):
    """
    Taichi kernel to compute flat normals for triangular faces.
    Each triangle gets a face normal assigned to all 3 vertices.
    """
    num_triangles = indices.shape[0]

    for tri_idx in range(num_triangles):
        # Get vertex indices
        v0 = indices[tri_idx, 0]
        v1 = indices[tri_idx, 1]
        v2 = indices[tri_idx, 2]

        # Get vertex positions
        p0 = ti.Vector([vertices[v0, 0], vertices[v0, 1], vertices[v0, 2]])
        p1 = ti.Vector([vertices[v1, 0], vertices[v1, 1], vertices[v1, 2]])
        p2 = ti.Vector([vertices[v2, 0], vertices[v2, 1], vertices[v2, 2]])

        # Compute face normal
        edge1 = p1 - p0
        edge2 = p2 - p0
        face_normal = edge1.cross(edge2)

        # Normalize
        length = face_normal.norm()
        if length > 1e-6:
            face_normal = face_normal / length

        # Assign flat normal to all 3 vertices
        normals[v0, 0] = face_normal[0]
        normals[v0, 1] = face_normal[1]
        normals[v0, 2] = face_normal[2]

        normals[v1, 0] = face_normal[0]
        normals[v1, 1] = face_normal[1]
        normals[v1, 2] = face_normal[2]

        normals[v2, 0] = face_normal[0]
        normals[v2, 1] = face_normal[1]
        normals[v2, 2] = face_normal[2]


@ti.kernel
def _assign_smooth_normals_kernel(original_normals: ti.types.ndarray(),
                                  boundary_vertex_to_original: ti.types.ndarray(),
                                  normals: ti.types.ndarray(),
                                  num_surface_vertices: ti.i32,
                                  normals_offset: ti.i32):
    """
    Taichi kernel to assign smooth normals from original surface to boundary vertices.
    For vertices from original surface: copy the original normal.
    For Steiner points: normal will be computed separately (fallback).
    """
    num_boundary_verts = boundary_vertex_to_original.shape[0]

    for i in range(num_boundary_verts):
        original_idx = boundary_vertex_to_original[i]

        # If this vertex is from the original surface, use its smooth normal
        if original_idx < num_surface_vertices:
            normals[normals_offset + i, 0] = original_normals[original_idx, 0]
            normals[normals_offset + i, 1] = original_normals[original_idx, 1]
            normals[normals_offset + i, 2] = original_normals[original_idx, 2]


def _compute_hybrid_normals_separate(tet_vertices: np.ndarray, tet_indices: np.ndarray,
                                    boundary_vertices: np.ndarray, boundary_indices: np.ndarray,
                                    original_surface_mesh,
                                    boundary_vertex_to_original: list) -> np.ndarray:
    """
    Compute hybrid normals for separated tetrahedral and boundary geometry.

    Args:
        tet_vertices: Vertices for tetrahedral interior faces
        tet_indices: Triangle indices for tetrahedral interior faces
        boundary_vertices: Vertices for boundary surface faces (separate, duplicated)
        boundary_indices: Triangle indices for boundary surface faces
        original_surface_mesh: Original surface mesh with normals
        boundary_vertex_to_original: Maps boundary vertex index -> original tet vertex index

    Returns:
        Normal vectors for all vertices (tet vertices + boundary vertices)
    """
    total_vertices = len(tet_vertices) + len(boundary_vertices)
    normals = np.zeros((total_vertices, 3), dtype=np.float32)

    # Compute flat normals for all tetrahedral interior faces using Taichi (GPU accelerated)
    num_interior_faces = len(tet_indices) // 3
    print(f"  Computing flat normals for {num_interior_faces:,} interior faces on GPU...")

    # Reshape indices to 2D array (num_triangles, 3)
    tet_indices_2d = tet_indices.reshape(-1, 3).astype(np.int32)

    # Call Taichi kernel
    _compute_flat_normals_kernel(tet_vertices, tet_indices_2d, normals)

    # Compute smooth normals for boundary surface faces using original surface mesh
    num_boundary_verts = len(boundary_vertices)
    print(f"  Computing smooth normals for {num_boundary_verts:,} boundary vertices on GPU...")
    num_surface_vertices = len(original_surface_mesh.vertices)

    # Convert boundary_vertex_to_original to numpy array for Taichi
    boundary_vertex_to_original_np = np.array(boundary_vertex_to_original, dtype=np.int32)

    # Call Taichi kernel to assign smooth normals from original surface
    normals_offset = len(tet_vertices)  # Boundary vertices start after tet vertices
    _assign_smooth_normals_kernel(original_surface_mesh.normals,
                                  boundary_vertex_to_original_np,
                                  normals,
                                  num_surface_vertices,
                                  normals_offset)

    # Handle Steiner points (vertices not in original surface) with flat normals
    # These are rare, so we can handle them in a CPU loop
    print(f"  Handling Steiner points (if any)...")
    steiner_count = 0
    for i in range(len(boundary_vertex_to_original)):
        original_idx = boundary_vertex_to_original[i]
        if original_idx >= num_surface_vertices:
            # This is a Steiner point - compute flat normal for its triangle
            tri_idx = i // 3
            v0_raw = tri_idx * 3
            v1_raw = tri_idx * 3 + 1
            v2_raw = tri_idx * 3 + 2

            p0 = boundary_vertices[v0_raw]
            p1 = boundary_vertices[v1_raw]
            p2 = boundary_vertices[v2_raw]

            edge1 = p1 - p0
            edge2 = p2 - p0
            face_normal = np.cross(edge1, edge2)
            face_normal_length = np.linalg.norm(face_normal)
            if face_normal_length > 1e-6:
                face_normal = face_normal / face_normal_length

            # Assign to this vertex in the normals array
            normals[normals_offset + i] = face_normal
            steiner_count += 1

    if steiner_count > 0:
        print(f"    Processed {steiner_count} Steiner points with flat normals")

    return normals


@ti.kernel
def _accumulate_normals_kernel(vertices: ti.types.ndarray(), indices: ti.types.ndarray(), normals: ti.types.ndarray()):
    """
    Taichi kernel to accumulate face normals for each vertex (parallel on GPU)
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

        # Compute cross product (face normal, not normalized yet)
        normal = edge1.cross(edge2)

        # Accumulate (add) to all 3 vertices using atomic operations
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
    Taichi kernel to normalize accumulated normals (parallel on GPU)
    """
    for i in range(normals.shape[0]):
        normal = ti.Vector([normals[i, 0], normals[i, 1], normals[i, 2]])
        length = normal.norm()
        if length > 1e-6:
            normal = normal / length
            normals[i, 0] = normal[0]
            normals[i, 1] = normal[1]
            normals[i, 2] = normal[2]


def _compute_normals_taichi(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Compute smooth vertex normals using Taichi GPU acceleration

    Averages face normals for all triangles sharing each vertex.

    Args:
        vertices: (N, 3) array of vertex positions
        indices: (M, 3) array of triangle indices

    Returns:
        (N, 3) array of averaged vertex normals
    """
    normals = np.zeros_like(vertices, dtype=np.float32)

    # Step 1: Accumulate face normals for each vertex (GPU parallel)
    _accumulate_normals_kernel(vertices, indices, normals)

    # Step 2: Normalize the accumulated normals (GPU parallel)
    _normalize_normals_kernel(normals)

    return normals
