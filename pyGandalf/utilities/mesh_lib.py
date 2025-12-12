from pyGandalf.utilities.logger import logger
from pyGandalf.utilities.definitions import MODELS_PATH

import numpy as np
import trimesh
from pxr import Usd, UsdGeom
import taichi as ti

import os
from pathlib import Path

# Initialize Taichi for GPU acceleration
ti.init(arch=ti.gpu, default_fp=ti.f32)

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
        # Could add more later: boundary faces, vertex markers, etc.

    def extract_all_faces(self) -> 'MeshInstance':
        """
        Extract all triangular faces from tetrahedra using shared vertices.

        Efficient method for static visualization - uses existing vertex array
        and creates triangle indices pointing to those vertices.

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
    
    def build_tetrahedral(cls, name: str, surface_mesh_path: Path):

        from pyGandalf.thesis_utilities.tet_generator import generate_tetrahedral_mesh

        # 1. Load surface mesh using existing build()
        surface_mesh = cls.build(name, surface_mesh_path)

        # 2. Generate tetrahedral mesh
        tet_mesh = generate_tetrahedral_mesh(surface_mesh)

        # 3. Store and return
        return tet_mesh


# Taichi GPU-accelerated normal computation
@ti.kernel
def _compute_normals_kernel(vertices: ti.types.ndarray(), indices: ti.types.ndarray(), normals: ti.types.ndarray()):
    """
    Taichi kernel to compute face normals in parallel on GPU
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
    Compute normals using Taichi GPU acceleration (wrapper function)

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