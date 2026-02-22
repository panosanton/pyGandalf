"""
SOFA Simulation System

Integrates SOFA physics simulation with pyGandalf's rendering pipeline.
Each frame: SOFA steps the spring-mass simulation, new vertex positions are read back
and uploaded to the GPU via glBufferSubData.
"""

import OpenGL.GL as gl
import numpy as np
import time

from pyGandalf.systems.system import System
from pyGandalf.scene.components import Component, StaticMeshComponent


class SofaSimulationComponent(Component):
    """
    Stores SOFA simulation state for a tetrahedral mesh entity.

    Holds the tet mesh data, SOFA scene handles (set by the system on creation),
    material parameters, and cutting control flags.
    """
    def __init__(self, tet_mesh, time_step: float = 0.001,
                 gravity: list = None, stiffness: float = 500.0,
                 damping: float = 5.0, total_mass: float = 1.0):
        """
        Args:
            tet_mesh:    TetrahedralMeshInstance to simulate.
            time_step:   SOFA simulation timestep in seconds.
                         Explicit integration requires small steps (~0.001) for stability.
            gravity:     Gravity vector, defaults to [0, -9.81, 0].
            stiffness:   Spring stiffness along tetrahedral edges.
            damping:     Spring damping (higher = less oscillation).
            total_mass:  Total mass of the object (kg).
        """
        super().__init__()
        self.tet_mesh = tet_mesh
        self.time_step = time_step
        self.gravity = gravity if gravity is not None else [0, -9.81, 0]
        self.stiffness = stiffness
        self.damping = damping
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
    Drives a SOFA spring-mass simulation and uploads deformed vertex positions
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
        # Pass vertices so winding order can be corrected (outward normals).
        sofa_comp.surface_indices = _extract_boundary_faces(tet_mesh.tetrahedra, tet_mesh.vertices)

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
        # Spring model uses EulerExplicitSolver — no linear solver needed.
        root.addObject('RequiredPlugin', name='Sofa.Component.ODESolver.Forward')
        root.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Dynamic')
        root.addObject('RequiredPlugin', name='Sofa.Component.StateContainer')
        root.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.Spring')
        root.addObject('RequiredPlugin', name='Sofa.Component.Mass')
        root.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Projective')
        root.addObject('RequiredPlugin', name='Sofa.Component.AnimationLoop')

        root.addObject('DefaultAnimationLoop')

        obj = root.addChild('Object')
        # Explicit integration: no linear system to solve — much faster than FEM.
        # Requires a small timestep (~0.001s) to remain stable.
        obj.addObject('EulerExplicitSolver')

        obj.addObject('TetrahedronSetTopologyContainer',
                      points=tet_mesh.vertices.tolist(),
                      tetrahedra=tet_mesh.tetrahedra.tolist())
        obj.addObject('TetrahedronSetTopologyModifier')

        # Explicitly pass position= so SOFA initialises the DOFs from the tet
        # mesh vertices rather than relying on implicit topology inheritance
        # (which is version-dependent and not guaranteed).
        mech = obj.addObject('MechanicalObject', name='dofs', template='Vec3d',
                             position=tet_mesh.vertices.tolist())

        # Spring forces computed per-edge — O(edges) instead of O(N^1.5) for FEM solve.
        obj.addObject('MeshSpringForceField',
                      stiffness=sofa_comp.stiffness,
                      damping=sofa_comp.damping)

        obj.addObject('UniformMass', totalMass=sofa_comp.total_mass)

        if fixed_indices:
            obj.addObject('FixedProjectiveConstraint', indices=fixed_indices)

        Sofa.Simulation.init(root)

        sofa_comp.sofa_root = root
        sofa_comp.mechanical_object = mech

        # Verify SOFA received the correct positions after init.
        sofa_pos = np.array(mech.position.value, dtype=np.float32)
        orig = tet_mesh.vertices
        print(f"[SofaSimulationSystem] Scene initialized:")
        print(f"  Vertices:       {len(orig):,}")
        print(f"  Tetrahedra:     {len(tet_mesh.tetrahedra):,}")
        print(f"  Surface faces:  {len(sofa_comp.surface_indices):,}")
        print(f"  Fixed vertices: {len(fixed_indices)}")
        print(f"  Original bbox   X[{orig[:,0].min():.4f}, {orig[:,0].max():.4f}]"
              f"  Y[{orig[:,1].min():.4f}, {orig[:,1].max():.4f}]"
              f"  Z[{orig[:,2].min():.4f}, {orig[:,2].max():.4f}]")
        if len(sofa_pos) == 0:
            print("  [WARNING] SOFA MechanicalObject returned ZERO positions after init!")
        else:
            print(f"  SOFA init bbox  X[{sofa_pos[:,0].min():.4f}, {sofa_pos[:,0].max():.4f}]"
                  f"  Y[{sofa_pos[:,1].min():.4f}, {sofa_pos[:,1].max():.4f}]"
                  f"  Z[{sofa_pos[:,2].min():.4f}, {sofa_pos[:,2].max():.4f}]")

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

        # --- SOFA simulation step ---
        t0 = time.perf_counter()
        Sofa.Simulation.animate(sofa_comp.sofa_root, sofa_comp.time_step)
        t1 = time.perf_counter()

        # --- Read positions back from SOFA (Vec3d → float64 → float32) ---
        new_positions = np.array(sofa_comp.mechanical_object.position.value, dtype=np.float32)
        t2 = time.perf_counter()

        # --- Recompute per-vertex normals ---
        new_normals = _compute_normals(new_positions, sofa_comp.surface_indices)
        t3 = time.perf_counter()

        # --- Upload both buffers to the GPU ---
        _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[0], new_positions)
        _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[1], new_normals)
        t4 = time.perf_counter()

        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1

        # On frame 1, dump the position bbox to confirm SOFA is returning good values.
        if self._frame_count == 1:
            print(f"[Frame 1] SOFA positions bbox:"
                  f"  X[{new_positions[:,0].min():.4f}, {new_positions[:,0].max():.4f}]"
                  f"  Y[{new_positions[:,1].min():.4f}, {new_positions[:,1].max():.4f}]"
                  f"  Z[{new_positions[:,2].min():.4f}, {new_positions[:,2].max():.4f}]")

        if self._frame_count % 20 == 0:
            print(
                f"[Frame {self._frame_count:4d}] "
                f"SOFA: {(t1-t0)*1000:7.1f}ms | "
                f"readback: {(t2-t1)*1000:5.1f}ms | "
                f"normals: {(t3-t2)*1000:5.1f}ms | "
                f"upload: {(t4-t3)*1000:5.1f}ms | "
                f"total: {(t4-t0)*1000:7.1f}ms"
            )


# ---------------------------------------------------------------------------
# Module-level helpers (also imported by test files for mesh preparation)
# ---------------------------------------------------------------------------

def _extract_boundary_faces(tetrahedra: np.ndarray,
                            vertices: np.ndarray = None) -> np.ndarray:
    """
    Return the boundary (outer surface) triangles of a tetrahedral mesh.

    A face is on the boundary if it belongs to exactly one tetrahedron.
    The returned indices reference the original tet mesh vertex array directly,
    so they can be used with the tet vertex VBO without remapping.

    When `vertices` is supplied the winding order of every face is corrected so
    that its normal points away from the mesh centroid (outward).  This is
    required for correct backface-culling: TetGen does not guarantee a consistent
    outward orientation for boundary faces extracted this way.

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
    boundary_faces = all_faces[is_boundary].copy()

    if vertices is not None:
        # For each boundary face use the 4th vertex of its own tetrahedron as
        # the interior reference — this vertex is always on the inside regardless
        # of mesh concavity (ears, underside, etc.).  The centroid approach fails
        # for concave regions and causes holes due to incorrect backface culling.
        boundary_positions = np.where(is_boundary)[0]   # index into all_faces
        tet_indices        = boundary_positions // 4     # which tet owns this face

        face_verts = all_faces[boundary_positions]       # (M, 3) vertex indices
        tet_verts  = tetrahedra[tet_indices]             # (M, 4) vertex indices

        # Find the one vertex in each tet that is NOT in the face.
        in_face = (tet_verts[:, :, np.newaxis] == face_verts[:, np.newaxis, :]).any(axis=2)
        fourth_vertex_idx = tet_verts[~in_face].reshape(-1)   # (M,)

        v0     = vertices[boundary_faces[:, 0]]
        v1     = vertices[boundary_faces[:, 1]]
        v2     = vertices[boundary_faces[:, 2]]
        fourth = vertices[fourth_vertex_idx]

        face_normals = np.cross(v1 - v0, v2 - v0)
        to_fourth    = fourth - (v0 + v1 + v2) / 3.0   # face-centre → 4th vertex (inward)
        dot          = np.einsum('ij,ij->i', face_normals, to_fourth)
        inward       = dot > 0   # normal points toward inside → flip
        boundary_faces[inward] = boundary_faces[inward][:, [0, 2, 1]]
        print(f"[_extract_boundary_faces] Flipped {inward.sum():,} / {len(boundary_faces):,} "
              f"faces to ensure outward winding.")

    return boundary_faces.astype(np.uint32)


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
