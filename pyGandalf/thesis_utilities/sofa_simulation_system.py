"""
SOFA Simulation System

Integrates SOFA physics simulation with pyGandalf's rendering pipeline.
Each frame: SOFA steps the FEM simulation, new vertex positions are read back
and uploaded to the GPU via glBufferSubData.
"""

import OpenGL.GL as gl
import numpy as np

from pyGandalf.systems.system import System
from pyGandalf.scene.components import Component, StaticMeshComponent


class SofaSimulationComponent(Component):
    """
    Stores SOFA simulation state for a tetrahedral mesh entity.

    Holds the tet mesh data, SOFA scene handles (set by the system on creation),
    material parameters, and cutting control flags.
    """
    def __init__(self, tet_mesh, time_step: float = 0.01,
                 gravity: list = None, young_modulus: float = 5000.0,
                 poisson_ratio: float = 0.45, total_mass: float = 1.0):
        """
        Args:
            tet_mesh:        TetrahedralMeshInstance to simulate.
            time_step:       SOFA simulation timestep in seconds.
            gravity:         Gravity vector, defaults to [0, -9.81, 0].
            young_modulus:   Stiffness of the material (Pa).
            poisson_ratio:   Incompressibility of the material (0-0.5).
            total_mass:      Total mass of the object (kg).
        """
        super().__init__()
        self.tet_mesh = tet_mesh
        self.time_step = time_step
        self.gravity = gravity if gravity is not None else [0, -9.81, 0]
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.total_mass = total_mass

        # Populated by SofaSimulationSystem.on_create_entity
        self.sofa_root = None
        self.mechanical_object = None  # Reference to SOFA MechanicalObject (source of positions)
        self.surface_indices = None    # uint32 (N_faces, 3) — used for normal recomputation

        # Cutting control (used in Task 2)
        self.should_cut = False
        self.cut_plane_origin = [0.0, 0.0, 0.0]
        self.cut_plane_normal = [1.0, 0.0, 0.0]


class SofaSimulationSystem(System):
    """
    Drives a SOFA physics simulation and uploads deformed vertex positions
    to the GPU each frame.

    Requires entities with both SofaSimulationComponent and StaticMeshComponent.
    The StaticMeshComponent must be initialised with attributes and indices directly
    (load_from_file=False), using the tet mesh vertices as the position buffer.
    OpenGLStaticMeshRenderingSystem must be registered AFTER this system so that
    GPU buffers exist by the time on_update_entity first runs.
    """

    def on_create_entity(self, entity, components):
        sofa_comp: SofaSimulationComponent
        mesh_comp: StaticMeshComponent
        sofa_comp, mesh_comp = components

        import Sofa
        import Sofa.Core

        tet_mesh = sofa_comp.tet_mesh

        # Store surface indices on the component so on_update_entity can
        # recompute normals without re-extracting them every frame.
        sofa_comp.surface_indices = _extract_boundary_faces(tet_mesh.tetrahedra)

        # Fix the bottom 5% of vertices so the object doesn't fall indefinitely.
        y = tet_mesh.vertices[:, 1]
        threshold = y.min() + (y.max() - y.min()) * 0.05
        fixed_indices = np.where(y < threshold)[0].tolist()

        # Build the SOFA scene graph.
        root = Sofa.Core.Node("root")
        root.gravity.value = sofa_comp.gravity
        root.dt.value = sofa_comp.time_step

        # Since SOFA v22.06, components were reorganised into separate plugins
        # that must be explicitly loaded before they can be used.
        root.addObject('RequiredPlugin', name='Sofa.Component.ODESolver.Backward')
        root.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Iterative')
        root.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Dynamic')
        root.addObject('RequiredPlugin', name='Sofa.Component.StateContainer')
        root.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.FEM.Elastic')
        root.addObject('RequiredPlugin', name='Sofa.Component.Mass')
        root.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Projective')
        root.addObject('RequiredPlugin', name='Sofa.Component.AnimationLoop')

        root.addObject('DefaultAnimationLoop')

        obj = root.addChild('Object')
        obj.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)
        obj.addObject('CGLinearSolver', iterations=25, tolerance=1e-5, threshold=1e-5)

        obj.addObject('TetrahedronSetTopologyContainer',
                      points=tet_mesh.vertices.tolist(),
                      tetrahedra=tet_mesh.tetrahedra.tolist())
        obj.addObject('TetrahedronSetTopologyModifier')

        mech = obj.addObject('MechanicalObject', name='dofs', template='Vec3d')

        obj.addObject('TetrahedronFEMForceField',
                      youngModulus=sofa_comp.young_modulus,
                      poissonRatio=sofa_comp.poisson_ratio,
                      method='large')

        obj.addObject('MeshMatrixMass', totalMass=sofa_comp.total_mass)

        if fixed_indices:
            obj.addObject('FixedProjectiveConstraint', indices=fixed_indices)

        Sofa.Simulation.init(root)

        sofa_comp.sofa_root = root
        sofa_comp.mechanical_object = mech

        print(f"[SofaSimulationSystem] Scene initialized:")
        print(f"  Vertices:       {len(tet_mesh.vertices):,}")
        print(f"  Tetrahedra:     {len(tet_mesh.tetrahedra):,}")
        print(f"  Surface faces:  {len(sofa_comp.surface_indices):,}")
        print(f"  Fixed vertices: {len(fixed_indices)}")

    def on_update_entity(self, ts: float, entity, components):
        sofa_comp: SofaSimulationComponent
        mesh_comp: StaticMeshComponent
        sofa_comp, mesh_comp = components

        if sofa_comp.sofa_root is None:
            return

        # GPU buffers are set up by OpenGLStaticMeshRenderingSystem.on_create_entity.
        # Guard until that has happened (first frame).
        if mesh_comp.render_pipeline is None or len(mesh_comp.buffers) < 2:
            return

        import Sofa

        # Advance the simulation by one fixed SOFA timestep.
        Sofa.Simulation.animate(sofa_comp.sofa_root, sofa_comp.time_step)

        # Read deformed positions from SOFA (Vec3d → float64) and cast to float32.
        new_positions = np.array(sofa_comp.mechanical_object.position.value, dtype=np.float32)

        # Recompute per-vertex normals from the new surface geometry.
        new_normals = _compute_normals(new_positions, sofa_comp.surface_indices)

        # Upload both buffers to the GPU.
        _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[0], new_positions)
        _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[1], new_normals)


# ---------------------------------------------------------------------------
# Module-level helpers (also imported by test files for mesh preparation)
# ---------------------------------------------------------------------------

def _extract_boundary_faces(tetrahedra: np.ndarray) -> np.ndarray:
    """
    Return the boundary (outer surface) triangles of a tetrahedral mesh.

    A face is on the boundary if it belongs to exactly one tetrahedron.
    The returned indices reference the original tet mesh vertex array directly,
    so they can be used with the tet vertex VBO without remapping.

    Returns:
        np.ndarray of shape (N_boundary_faces, 3), dtype uint32.
    """
    face_combos = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    all_faces = tetrahedra[:, face_combos].reshape(-1, 3)  # (N_tets * 4, 3)
    all_faces_sorted = np.sort(all_faces, axis=1)

    _, inverse, counts = np.unique(
        all_faces_sorted, axis=0, return_inverse=True, return_counts=True
    )

    is_boundary = (counts == 1)[inverse]
    return all_faces[is_boundary].astype(np.uint32)


def _compute_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Compute per-vertex normals by accumulating face normals at each vertex.

    Args:
        vertices: float32 (N, 3) vertex positions.
        indices:  uint32  (M, 3) triangle indices into vertices.

    Returns:
        float32 (N, 3) normalised per-vertex normals.
    """
    normals = np.zeros_like(vertices, dtype=np.float32)
    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)

    np.add.at(normals, indices[:, 0], face_normals)
    np.add.at(normals, indices[:, 1], face_normals)
    np.add.at(normals, indices[:, 2], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths == 0.0, 1.0, lengths)
    return (normals / lengths).astype(np.float32)


def _update_vbo(vao: int, vbo: int, data: np.ndarray):
    """Upload new data to an existing VBO using glBufferSubData (no reallocation)."""
    flat = data.flatten().astype(np.float32)
    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, flat.nbytes, flat)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    gl.glBindVertexArray(0)
