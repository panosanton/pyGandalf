from pyGandalf.utilities.logger import logger
from pyGandalf.utilities.definitions import MODELS_PATH

import numpy as np
import trimesh
from pxr import Usd, UsdGeom
import taichi as ti

import os
from pathlib import Path

# Initialize Taichi for GPU acceleration
ti.init(arch=ti.gpu, default_fp=ti.f32, kernel_profiler=True, offline_cache=True)

class MeshInstance:
    def __init__(self, name, path, vertices, indices, normals, texcoords):
        self.name = name
        self.path = path
        self.vertices = vertices
        self.indices = indices
        self.normals = normals
        self.texcoords = texcoords
        
class TetrahedralMeshInstance:
    def __init__(self, name, path, vertices, tetrahedra):
        self.name = name
        self.path = path
        self.vertices = vertices        # Nx3 array of vertex positions
        self.tetrahedra = tetrahedra    # Mx4 array of tet indices

    def extract_surface(self) -> 'MeshInstance':
        """
        Extract only the outer surface triangles from the tetrahedral mesh.

        Boundary faces are those that appear in exactly one tetrahedron.
        Each face's winding order is corrected so normals point outward:
        the face normal is checked against the opposite vertex of its tet —
        if it points toward the interior, the winding is flipped.

        Returns:
            MeshInstance with surface triangles and smooth vertex normals
        """
        print(f"Extracting surface from {len(self.tetrahedra):,} tetrahedra...")

        # face_combos[i] are the 3 local indices making up face i of a tet.
        # opposite[i] is the local index of the vertex NOT in face i.
        face_combos = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
        opposite    = np.array([3, 2, 1, 0], dtype=np.int32)

        # all_faces[i]    — global vertex indices of face i  (N_tets*4, 3)
        # all_opposite[i] — global index of the opposite vertex for face i  (N_tets*4,)
        all_faces    = self.tetrahedra[:, face_combos].reshape(-1, 3)
        all_opposite = self.tetrahedra[:, opposite].reshape(-1)

        # Find boundary faces (appear in exactly one tet)
        all_faces_sorted = np.sort(all_faces, axis=1)
        _, inverse_indices, counts = np.unique(
            all_faces_sorted, axis=0, return_inverse=True, return_counts=True
        )
        is_boundary = counts[inverse_indices] == 1

        surface_faces    = all_faces[is_boundary].copy()     # (M, 3)
        surface_opposite = all_opposite[is_boundary]         # (M,)

        # Fix winding order: normal must point AWAY from the opposite vertex.
        v0  = self.vertices[surface_faces[:, 0]]
        v1  = self.vertices[surface_faces[:, 1]]
        v2  = self.vertices[surface_faces[:, 2]]
        opp = self.vertices[surface_opposite]

        face_normal = np.cross(v1 - v0, v2 - v0)           # (M, 3), unnormalised
        to_interior = opp - v0                               # points into the tet

        # If dot > 0 the normal faces inward — swap two vertices to flip it
        inward = np.sum(face_normal * to_interior, axis=1) > 0
        surface_faces[inward] = surface_faces[inward][:, [0, 2, 1]]

        surface_triangles = surface_faces.astype(np.uint32)

        print(f"  Surface faces: {len(surface_triangles):,} ({inward.sum()} windings corrected)")
        print(f"  Computing normals...")
        normals = _compute_normals_taichi(self.vertices, surface_triangles)

        return MeshInstance(
            name=f"{self.name}_surface",
            path=self.path,
            vertices=self.vertices,
            indices=surface_triangles,
            normals=normals,
            texcoords=np.zeros((len(self.vertices), 2), dtype=np.float32)
        )

    # ------------------------------------------------------------------
    # LEGACY — kept for reference only.
    # These methods were written to visualise interior tetrahedra by
    # duplicating surface vertices so each region could have independent
    # normals.  The duplication made TetGen report non-manifold input
    # whenever those meshes were reused.  Current rendering uses plain
    # boundary-face extraction (extract_surface above) with no vertex
    # duplication.
    # ------------------------------------------------------------------

    def extract_all_faces(self) -> 'MeshInstance':
        """
        Extract all triangular faces from tetrahedra using shared vertices.

        Efficient method for static visualization - uses existing vertex array
        and creates triangle indices pointing to those vertices.
        Normals are averaged across all triangles sharing a vertex.

        Returns:
            MeshInstance with all tetrahedral faces as triangles
        """
        print(f"Extracting faces from {len(self.tetrahedra):,} tetrahedra...")

        # Each tetrahedron has 4 triangular faces
        all_triangles = []

        for tet in self.tetrahedra:
            v0, v1, v2, v3 = tet

            # Extract 4 triangular faces (indices into original vertex array)
            faces = [
                [v0, v1, v2],
                [v0, v1, v3],
                [v0, v2, v3],
                [v1, v2, v3],
            ]
            all_triangles.extend(faces)

        indices = np.array(all_triangles, dtype=np.uint32)

        # Compute normals using Taichi GPU acceleration
        print(f"Computing normals for {len(indices):,} triangles...")
        normals = _compute_normals_taichi(self.vertices, indices)

        # Create MeshInstance using original vertices (no duplication)
        mesh = MeshInstance(
            name=f"{self.name}_all_faces",
            path=self.path,
            vertices=self.vertices,
            indices=indices,
            normals=normals,
            texcoords=np.zeros((len(self.vertices), 2), dtype=np.float32)
        )

        print(f"Created mesh: {len(self.vertices):,} vertices, {len(indices):,} triangles")
        return mesh

    def extract_interior_and_surface_hybrid(self) -> 'MeshInstance':
        """
        Extract tetrahedral faces with hybrid approach for optimal lighting.

        This method creates two separate parts:
        1. Interior faces: Uses original vertices, excludes boundary faces
        2. Surface shell: Duplicated vertices with smooth surface normals

        This avoids normal blending between surface and interior while being
        more efficient than full vertex duplication.

        Uses NumPy vectorization for optimal performance.

        Returns:
            MeshInstance with interior + surface shell (no boundary face overlap)
        """
        if self.original_surface_mesh is None:
            print("WARNING: No original surface mesh available. Falling back to standard extraction.")
            return self.extract_all_faces()

        print(f"Extracting hybrid interior+surface mesh from {len(self.tetrahedra):,} tetrahedra...")

        # Step 1: Generate all faces from all tetrahedra using NumPy vectorization
        tetrahedra = self.tetrahedra  # Shape: (N, 4)

        # Face index combinations for a tetrahedron [v0, v1, v2, v3]
        face_indices = np.array([
            [0, 1, 2],  # Face 0
            [0, 1, 3],  # Face 1
            [0, 2, 3],  # Face 2
            [1, 2, 3]   # Face 3
        ], dtype=np.int32)

        # Extract all faces: (N_tets, 4, 3) -> (N_tets * 4, 3)
        # For each tet, get its 4 faces
        all_faces = tetrahedra[:, face_indices].reshape(-1, 3)

        # Sort each face for consistent comparison
        all_faces_sorted = np.sort(all_faces, axis=1)

        # Step 2: Find unique faces and count occurrences
        print("  Identifying boundary and interior faces...")
        unique_faces, inverse_indices, counts = np.unique(
            all_faces_sorted, axis=0, return_inverse=True, return_counts=True
        )

        # Boundary faces appear once, interior faces appear twice
        boundary_mask = counts == 1
        interior_mask = counts == 2

        # Map back to original face list
        is_boundary = boundary_mask[inverse_indices]
        is_interior = interior_mask[inverse_indices]

        # Extract interior faces (use original unsorted order)
        interior_triangles = all_faces[is_interior]

        num_boundary = is_boundary.sum()
        num_interior = is_interior.sum()

        print(f"  Boundary faces: {num_boundary:,}")
        print(f"  Interior faces: {num_interior:,}")

        # Step 3: Create duplicated surface vertices and triangles
        num_original_verts = len(self.vertices)
        num_surface_verts = len(self.original_surface_mesh.vertices)

        surface_vertices = self.original_surface_mesh.vertices.copy()
        surface_normals = self.original_surface_mesh.normals.copy()

        # Offset surface triangle indices
        vertex_offset = num_original_verts
        surface_triangles = self.original_surface_mesh.indices + vertex_offset

        # Step 4: Combine everything using NumPy vstack/concatenate
        all_vertices = np.vstack([self.vertices, surface_vertices]).astype(np.float32)
        all_indices = np.vstack([interior_triangles, surface_triangles]).astype(np.uint32)

        # Step 5: Compute normals
        print(f"  Computing normals for interior vertices...")
        normals = np.zeros((len(all_vertices), 3), dtype=np.float32)

        # Compute normals for interior triangles only
        if len(interior_triangles) > 0:
            interior_normals = _compute_normals_taichi(self.vertices, interior_triangles.astype(np.uint32))
            normals[:num_original_verts] = interior_normals

        # Copy smooth normals for duplicated surface vertices
        normals[vertex_offset:] = surface_normals

        # Create final mesh
        mesh = MeshInstance(
            name=f"{self.name}_hybrid",
            path=self.path,
            vertices=all_vertices,
            indices=all_indices,
            normals=normals,
            texcoords=np.zeros((len(all_vertices), 2), dtype=np.float32)
        )

        print(f"Created hybrid mesh:")
        print(f"  Total vertices: {len(all_vertices):,} (original: {num_original_verts:,}, duplicated surface: {num_surface_verts:,})")
        print(f"  Total triangles: {len(all_indices):,} (interior: {len(interior_triangles):,}, surface: {len(surface_triangles):,})")

        return mesh

class MeshLib(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(MeshLib, cls).__new__(cls)
            cls.instance.meshes: dict[str, MeshInstance] = {} # type: ignore
            cls.instance.meshes_names: dict[str, str] = {} # type: ignore
        return cls.instance
    
    def build(cls, name: str, path: Path):
        filename = str(path)
        if cls.instance.meshes.get(filename) != None:
            cls.instance.meshes_names[name] = filename
            return cls.instance.meshes[filename]

        mesh = None
        vertices = None
        indices = None
        normals = None
        texcoords = None

        if '.usd' in path.name:
            if True:
                stage = Usd.Stage.Open(filename)
                flattened_stage = stage.Flatten().ExportToString()
                logger.debug(flattened_stage)

                meshes, face_vertex_count = cls.instance._parse_usd(name, filename)
                mesh: MeshInstance = meshes[0]

                vertices = np.asarray(mesh.vertices, dtype=np.float32)

                if face_vertex_count == 3:
                    indices = np.asarray(mesh.indices, dtype=np.uint32).reshape(-1, 3)
                elif face_vertex_count == 4:
                    result = []
                    indices = np.asarray(mesh.indices, dtype=np.uint32)
                    for i in range(0, len(indices), 4):
                        sub_array = indices[i:i+4]
                        extracted_elements = np.array([
                            [sub_array[0], sub_array[1], sub_array[2]],
                            [sub_array[2], sub_array[3], sub_array[0]]
                        ])
                        result.extend(extracted_elements)
                    indices = np.array(result)

                # Compute normals if missing or invalid (like trimesh does for OBJ)
                if mesh.normals is None or len(mesh.normals) != len(vertices):
                    # Use trimesh to compute proper vertex normals
                    temp_mesh = trimesh.Trimesh(vertices=vertices, faces=indices)
                    normals = np.asarray(temp_mesh.vertex_normals, dtype=np.float32)
                else:
                    normals = np.asarray(mesh.normals, dtype=np.float32)

                texcoords = np.asarray(mesh.texcoords, dtype=np.float32) if mesh.texcoords is not None else None
        else:
            mesh: trimesh.Trimesh = trimesh.load(filename, force='mesh')
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            indices = np.asarray(mesh.faces, dtype=np.uint32)
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

            if hasattr(mesh.visual, 'uv'):
                texcoords = np.asarray(mesh.visual.uv, dtype=np.float32)

        rel_path = Path(os.path.relpath(path, MODELS_PATH))

        cls.instance.meshes_names[name] = filename
        cls.instance.meshes[filename] = MeshInstance(name, rel_path, vertices, indices, normals, texcoords)

        return cls.instance.meshes[filename]

    def get(cls, name: str) -> MeshInstance | None:
        if name not in cls.instance.meshes_names.keys():
            return None
        
        filename = cls.instance.meshes_names[name]

        if filename not in cls.instance.meshes.keys():
            return None
        
        return cls.instance.meshes[filename]
    
    def get_meshes(cls) -> dict[str, MeshInstance]:
        return cls.instance.meshes
    
    def _parse_usd(cls, name, file_path):
        logger.debug(file_path)
        stage = Usd.Stage.Open(file_path)

        submeshes: list[MeshInstance] = []

        # Iterate over all prims in the stage
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)

                # Get vertices
                points_attr = mesh.GetPointsAttr()
                vertices = np.array(points_attr.Get(), dtype=np.float32)

                face_vertex_count_attr = mesh.GetFaceVertexCountsAttr()
                face_vertex_count = np.array(face_vertex_count_attr.Get(), dtype=np.uint32)[0] if face_vertex_count_attr else 0

                # Get vertex indices (faces) if available
                indices_attr = mesh.GetFaceVertexIndicesAttr()
                indices = np.array(indices_attr.Get(), dtype=np.uint32) if indices_attr else None

                # Get normals if available
                normals_attr = mesh.GetNormalsAttr()
                normals = np.array(normals_attr.Get(), dtype=np.float32) if normals_attr else None

                # Get UVs if available
                uvs = None
                primvar_names = prim.GetAttributes()
                for primvar in primvar_names:
                    if primvar.GetTypeName() == 'texCoord2f[]':
                        uvs = np.array(primvar.Get(), dtype=np.float32)

                submeshes.append(MeshInstance(name, file_path, vertices, indices, normals, uvs))
    
        return submeshes, face_vertex_count
    
    def build_tetrahedral(cls, name: str, surface_mesh_path: Path,
                          target_faces: int = None, tet_scale: float = 1.0):

        from pyGandalf.thesis_utilities.tet_generator import generate_tetrahedral_mesh

        # 1. Load surface mesh using existing build()
        surface_mesh = cls.build(name, surface_mesh_path)

        # 2. Generate tetrahedral mesh (optionally simplified first)
        tet_mesh = generate_tetrahedral_mesh(surface_mesh,
                                             target_faces=target_faces,
                                             tet_scale=tet_scale)

        # 3. Store and return
        return tet_mesh


# Taichi GPU-accelerated normal computation
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
        # This averages normals across all triangles sharing a vertex
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

    Averages face normals for all triangles sharing each vertex,
    resulting in smooth shading.

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