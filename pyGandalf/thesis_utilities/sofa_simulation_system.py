"""
SOFA Simulation System

Integrates SOFA physics simulation with pyGandalf's rendering pipeline.
Each frame: SOFA steps the spring-mass simulation, new vertex positions are read back
and uploaded to the GPU via glBufferSubData.

Press C to perform a cut along the plane defined in SofaSimulationComponent.
"""

import OpenGL.GL as gl
import numpy as np
import ctypes
import time

import glfw
from pyGandalf.core.input_manager import InputManager

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
            gravity:     Gravity vector, defaults to [0, 0, 0] (no gravity).
            stiffness:   Spring stiffness along tetrahedral edges.
            damping:     Spring damping (higher = less oscillation).
            total_mass:  Total mass of the object (kg).
        """
        super().__init__()
        self.tet_mesh = tet_mesh
        self.time_step = time_step
        self.gravity = gravity if gravity is not None else [0, 0, 0]
        self.stiffness = stiffness
        self.damping = damping
        self.total_mass = total_mass

        # Populated by SofaSimulationSystem.on_create_entity
        self.sofa_root = None
        self.mechanical_object = None   # MechanicalObject (source of positions)
        self.surface_indices = None     # uint32 (N_faces, 3) — boundary faces
        self.current_tetrahedra = None  # Python-side tet array, shrinks on each cut

        # Cutting control: set should_cut=True to trigger a cut next frame.
        # cut_plane_origin / cut_plane_normal define the cutting plane.
        self.should_cut = False
        self.cut_plane_origin = [0.0, 0.0, 0.0]
        self.cut_plane_normal = [0.0, 1.0, 0.0]   # horizontal cut by default


class SofaSimulationSystem(System):
    """
    Drives a SOFA spring-mass simulation and uploads deformed vertex positions
    to the GPU each frame.

    Press C to perform a cut along SofaSimulationComponent.cut_plane_*.

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
        sofa_comp.surface_indices = _extract_boundary_faces(tet_mesh.tetrahedra, tet_mesh.vertices)

        # Fix the bottom 5% of vertices so the object doesn't fall indefinitely.
        y = tet_mesh.vertices[:, 1]
        threshold = y.min() + (y.max() - y.min()) * 0.05
        fixed_indices = np.where(y < threshold)[0].tolist()

        # Build the SOFA scene graph.
        root = Sofa.Core.Node("root")
        root.gravity.value = sofa_comp.gravity
        root.dt.value = sofa_comp.time_step

        root.addObject('RequiredPlugin', name='Sofa.Component.ODESolver.Backward')
        root.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Iterative')
        root.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Dynamic')
        root.addObject('RequiredPlugin', name='Sofa.Component.StateContainer')
        root.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.Spring')
        root.addObject('RequiredPlugin', name='Sofa.Component.Mass')
        root.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Projective')
        root.addObject('RequiredPlugin', name='Sofa.Component.AnimationLoop')

        root.addObject('DefaultAnimationLoop')

        obj = root.addChild('Object')
        obj.addObject('EulerImplicitSolver', rayleighStiffness=0.01, rayleighMass=0.01)
        obj.addObject('CGLinearSolver', iterations=25, tolerance=1e-9, threshold=1e-9)

        obj.addObject('TetrahedronSetTopologyContainer',
                      points=tet_mesh.vertices.tolist(),
                      tetrahedra=tet_mesh.tetrahedra.tolist())
        obj.addObject('TetrahedronSetTopologyModifier')

        mech = obj.addObject('MechanicalObject', name='dofs', template='Vec3d',
                             position=tet_mesh.vertices.tolist())

        obj.addObject('MeshSpringForceField',
                      stiffness=sofa_comp.stiffness,
                      damping=sofa_comp.damping)

        obj.addObject('UniformMass', totalMass=sofa_comp.total_mass)

        if fixed_indices:
            obj.addObject('FixedProjectiveConstraint', indices=fixed_indices)

        Sofa.Simulation.init(root)

        sofa_comp.sofa_root = root
        sofa_comp.mechanical_object = mech
        sofa_comp.current_tetrahedra = tet_mesh.tetrahedra.copy()

        sofa_pos = np.array(mech.position.value, dtype=np.float32)
        orig = tet_mesh.vertices
        print(f"[SofaSimulationSystem] Scene initialized:")
        print(f"  Vertices:       {len(orig):,}")
        print(f"  Tetrahedra:     {len(tet_mesh.tetrahedra):,}")
        print(f"  Surface faces:  {len(sofa_comp.surface_indices):,}")
        print(f"  Fixed vertices: {len(fixed_indices)}")
        print(f"  Press C to cut along the horizontal plane at y=0")
        if len(sofa_pos) == 0:
            print("  [WARNING] SOFA MechanicalObject returned ZERO positions after init!")

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

        # --- One-shot cut detection (C key, rising edge only) ---
        c_now = InputManager().get_key_down(glfw.KEY_C)
        if c_now and not getattr(self, '_c_was_pressed', False):
            sofa_comp.should_cut = True
        self._c_was_pressed = c_now

        if sofa_comp.should_cut:
            sofa_comp.should_cut = False
            _perform_cut(sofa_comp, mesh_comp)

        # --- SOFA simulation step ---
        t0 = time.perf_counter()
        Sofa.Simulation.animate(sofa_comp.sofa_root, sofa_comp.time_step)
        t1 = time.perf_counter()

        # --- Read positions back from SOFA ---
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

        if self._frame_count % 60 == 0:
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

def _perform_cut(sofa_comp: SofaSimulationComponent, mesh_comp: StaticMeshComponent):
    """
    Remove all tetrahedra that straddle or sit above the cutting plane.

    The cut is performed entirely in Python against sofa_comp.current_tetrahedra
    (SOFA's Python bindings do not expose topology modifier methods directly).
    SOFA continues simulating the full mesh for dynamics; only the rendered
    surface is updated here.
    """
    current_tets = sofa_comp.current_tetrahedra
    if current_tets is None or len(current_tets) == 0:
        print("[Cut] No tetrahedra remaining.")
        return

    positions = np.array(sofa_comp.mechanical_object.position.value, dtype=np.float32)

    origin = np.array(sofa_comp.cut_plane_origin, dtype=np.float32)
    normal = np.array(sofa_comp.cut_plane_normal, dtype=np.float32)
    normal /= np.linalg.norm(normal)

    # Signed distance of each vertex from the plane (positive = above).
    signed_dist = (positions - origin) @ normal  # (N_verts,)

    # For each tet collect the signed distances of its 4 vertices.
    tet_dists = signed_dist[current_tets]  # (N_tets, 4)

    any_above = np.any(tet_dists > 0, axis=1)
    any_below = np.any(tet_dists < 0, axis=1)

    # Keep only tets with NO vertex above the plane.
    keep_mask = ~any_above

    n_removed = int((~keep_mask).sum())
    if n_removed == 0:
        print("[Cut] No tetrahedra are above the cutting plane.")
        return

    print(f"[Cut] Removing {n_removed:,} / {len(current_tets):,} tetrahedra...")
    new_tets = current_tets[keep_mask]
    sofa_comp.current_tetrahedra = new_tets
    print(f"[Cut] Remaining tetrahedra: {len(new_tets):,}")

    if len(new_tets) == 0:
        print("[Cut] All tetrahedra removed — nothing to render.")
        return

    # Re-extract boundary faces from the surviving tetrahedra.
    new_surface = _extract_boundary_faces(new_tets, positions)
    sofa_comp.surface_indices = new_surface
    print(f"[Cut] New surface triangles: {len(new_surface):,}")

    new_normals = _compute_normals(positions, new_surface)

    # Update the EBO — size changes so glBufferData (full reallocation) is required.
    flat_indices = new_surface.flatten().astype(np.uint32)
    mesh_comp.indices = new_surface  # .size drives the draw_indexed call count

    gl.glBindVertexArray(mesh_comp.render_pipeline)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh_comp.index_buffer)
    gl.glBufferData(
        gl.GL_ELEMENT_ARRAY_BUFFER,
        flat_indices.nbytes,
        flat_indices.ctypes.data_as(ctypes.POINTER(gl.GLuint)),
        gl.GL_DYNAMIC_DRAW,
    )
    gl.glBindVertexArray(0)

    _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[1], new_normals)
    print("[Cut] GPU buffers updated.")


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
        boundary_positions = np.where(is_boundary)[0]
        tet_indices        = boundary_positions // 4

        face_verts = all_faces[boundary_positions]
        tet_verts  = tetrahedra[tet_indices]

        in_face = (tet_verts[:, :, np.newaxis] == face_verts[:, np.newaxis, :]).any(axis=2)
        fourth_vertex_idx = tet_verts[~in_face].reshape(-1)

        v0     = vertices[boundary_faces[:, 0]]
        v1     = vertices[boundary_faces[:, 1]]
        v2     = vertices[boundary_faces[:, 2]]
        fourth = vertices[fourth_vertex_idx]

        face_normals = np.cross(v1 - v0, v2 - v0)
        to_fourth    = fourth - (v0 + v1 + v2) / 3.0
        dot          = np.einsum('ij,ij->i', face_normals, to_fourth)
        inward       = dot > 0
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
