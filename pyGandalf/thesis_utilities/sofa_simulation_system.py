"""
SOFA Simulation System

Uses SOFA's corotational Finite Element Method for physically accurate
deformation, and SOFA's own topology modifier for proper surgical cuts.

Physics:
    TetrahedralCorotationalFEMForceField — corotational FEM preserves volume
    correctly under large deformations.  Parameterised by Young's modulus (Pa)
    and Poisson ratio rather than per-spring stiffness.

Solver:
    EulerImplicitSolver + CGLinearSolver — unconditionally stable implicit
    integration.  A timestep of 0.01 s is fine for most stiffness values;
    no substep hacks are needed.

Cutting (C key):
    Attempts removeTetrahedra() on TetrahedronSetTopologyModifier.  If the
    method is not exposed in the installed Python bindings, falls back to
    re-initialising the full SOFA scene with the reduced topology while
    restoring current positions and velocities so motion is continuous.

GPU acceleration:
    Set use_cuda=True to attempt loading SofaCUDA.  If the plugin is not
    installed the system silently falls back to CPU components.

Poke (F key):
    Applies a downward velocity impulse to the top 5 % of vertices.
"""

import OpenGL.GL as gl
import numpy as np
import ctypes
import time

import glfw
from pyGandalf.core.input_manager import InputManager
from pyGandalf.systems.system import System
from pyGandalf.scene.components import Component, StaticMeshComponent


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

class SofaSimulationComponent(Component):
    """
    Stores SOFA simulation state for a tetrahedral mesh entity.
    """

    def __init__(self, tet_mesh,
                 time_step:     float = 0.01,
                 gravity:       list  = None,
                 young_modulus: float = 5000.0,
                 poisson_ratio: float = 0.3,
                 total_mass:    float = 1.0,
                 use_cuda:      bool  = True,
                 poke_speed:    float = 2.0):
        """
        Args:
            tet_mesh:      TetrahedralMeshInstance to simulate.
            time_step:     SOFA timestep (s).  Implicit solver is stable at 0.01.
            gravity:       Gravity vector.  Default [0,0,0] (no gravity).
            young_modulus: FEM material stiffness (Pa).
                           Soft tissue range: 1 000–10 000 Pa.
                           Higher = stiffer, less visible deformation.
            poisson_ratio: Volume preservation (0–0.49).  0.3 is a good default.
                           Values near 0.5 (incompressible) are expensive to solve.
            total_mass:    Total object mass (kg).
            use_cuda:      Attempt SofaCUDA GPU acceleration, fall back to CPU.
            poke_speed:    Velocity (m/s) applied downward to top 5 % on F key.
        """
        super().__init__()
        self.tet_mesh      = tet_mesh
        self.time_step     = time_step
        self.gravity       = gravity if gravity is not None else [0, 0, 0]
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.total_mass    = total_mass
        self.use_cuda      = use_cuda
        self.poke_speed    = poke_speed

        # Populated by SofaSimulationSystem.on_create_entity
        self.sofa_root          = None
        self.mechanical_object  = None
        self.topology_modifier  = None   # TetrahedronSetTopologyModifier
        self.surface_indices    = None   # (N_faces, 3) uint32, boundary faces
        self.fixed_indices      = None   # list of int, bottom 5 % vertices
        self.poke_mask          = None   # bool array, top 5 % vertices
        self._cuda_active       = False  # actual CUDA state after init attempt

        # Cut control
        self.should_cut       = False
        self.cut_plane_origin = [0.0, 0.0, 0.0]
        self.cut_plane_normal = [0.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class SofaSimulationSystem(System):
    """
    Drives a SOFA FEM simulation and uploads deformed vertex positions to the
    GPU each frame.

    Must be registered BEFORE OpenGLStaticMeshRenderingSystem.
    The paired StaticMeshComponent must be initialised with attributes/indices
    directly (load_from_file=False) using the tet mesh vertices as the VBO.

    Controls:
        C — cut along cut_plane_origin / cut_plane_normal
        F — poke top 5 % of vertices downward
    """

    def on_create_entity(self, entity, components):
        sofa_comp: SofaSimulationComponent
        mesh_comp: StaticMeshComponent
        sofa_comp, mesh_comp = components

        tet = sofa_comp.tet_mesh

        sofa_comp.surface_indices = _extract_boundary_faces(
            tet.tetrahedra, tet.vertices)

        # Bottom 5 % of vertices — fixed so the object doesn't free-fall.
        y         = tet.vertices[:, 1]
        threshold = y.min() + (y.max() - y.min()) * 0.05
        sofa_comp.fixed_indices = np.where(y < threshold)[0].tolist()

        # Top 5 % — poke target.
        poke_thr          = y.max() - (y.max() - y.min()) * 0.05
        sofa_comp.poke_mask = (y >= poke_thr)

        root, mech, modifier, cuda_ok = _build_sofa_scene(
            sofa_comp, tet.vertices, tet.tetrahedra)

        sofa_comp.sofa_root         = root
        sofa_comp.mechanical_object = mech
        sofa_comp.topology_modifier = modifier
        sofa_comp._cuda_active      = cuda_ok

        print("[SofaSimulationSystem] Initialized:")
        print(f"  Backend:        {'SofaCUDA (GPU)' if cuda_ok else 'CPU'}")
        print(f"  Vertices:       {len(tet.vertices):,}")
        print(f"  Tetrahedra:     {len(tet.tetrahedra):,}")
        print(f"  Surface faces:  {len(sofa_comp.surface_indices):,}")
        print(f"  Fixed vertices: {len(sofa_comp.fixed_indices)}")
        print(f"  Young modulus:  {sofa_comp.young_modulus} Pa")
        print(f"  Poisson ratio:  {sofa_comp.poisson_ratio}")
        print("  F — poke top of mesh | C — cut along cut_plane")

    def on_update_entity(self, ts: float, entity, components):
        sofa_comp: SofaSimulationComponent
        mesh_comp: StaticMeshComponent
        sofa_comp, mesh_comp = components

        if sofa_comp.sofa_root is None:
            return
        # GPU buffers are created by OpenGLStaticMeshRenderingSystem on first frame.
        if mesh_comp.render_pipeline is None or len(mesh_comp.buffers) < 2:
            return

        import Sofa

        # --- One-shot C key: cut ---
        c_now = InputManager().get_key_down(glfw.KEY_C)
        if c_now and not getattr(self, '_c_prev', False):
            sofa_comp.should_cut = True
        self._c_prev = c_now

        if sofa_comp.should_cut:
            sofa_comp.should_cut = False
            _perform_cut(sofa_comp, mesh_comp)

        # --- One-shot F key: poke ---
        f_now = InputManager().get_key_down(glfw.KEY_F)
        if f_now and not getattr(self, '_f_prev', False):
            _apply_poke(sofa_comp)
        self._f_prev = f_now

        # --- SOFA simulation step ---
        t0 = time.perf_counter()
        Sofa.Simulation.animate(sofa_comp.sofa_root, sofa_comp.time_step)
        t1 = time.perf_counter()

        # --- Read back positions ---
        new_positions = np.array(
            sofa_comp.mechanical_object.position.value, dtype=np.float32)
        t2 = time.perf_counter()

        # --- Recompute normals ---
        new_normals = _compute_normals(new_positions, sofa_comp.surface_indices)
        t3 = time.perf_counter()

        # --- Upload to GPU ---
        _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[0], new_positions)
        _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[1], new_normals)
        t4 = time.perf_counter()

        if not hasattr(self, '_fc'):
            self._fc = 0
        self._fc += 1
        if self._fc % 60 == 0:
            print(
                f"[Frame {self._fc:4d}] "
                f"SOFA: {(t1-t0)*1000:7.1f}ms | "
                f"readback: {(t2-t1)*1000:5.1f}ms | "
                f"normals: {(t3-t2)*1000:5.1f}ms | "
                f"upload: {(t4-t3)*1000:5.1f}ms | "
                f"total: {(t4-t0)*1000:7.1f}ms"
            )


# ---------------------------------------------------------------------------
# Scene builder (used at init and after each cut re-initialisation)
# ---------------------------------------------------------------------------

def _build_sofa_scene(sofa_comp: SofaSimulationComponent,
                      vertices:   np.ndarray,
                      tetrahedra: np.ndarray,
                      initial_velocities: np.ndarray = None):
    """
    Build and initialise a complete SOFA scene for FEM simulation.

    Called once at startup and again whenever the scene must be re-initialised
    after a cut (if removeTetrahedra is not available in the Python bindings).

    If use_cuda=True, attempts a full CUDA build first.  If any component is
    missing from the installed SOFA build the whole root is discarded and the
    scene is rebuilt entirely on CPU, avoiding template mismatches between
    MechanicalObject and the force field.

    Args:
        sofa_comp:           Component holding simulation parameters.
        vertices:            (N, 3) float32 vertex positions.
        tetrahedra:          (M, 4) int32 tet connectivity.
        initial_velocities:  Optional (N, 3) float32 velocities to restore
                             after init (used when re-initialising after a cut).

    Returns:
        (root, mechanical_object, topology_modifier, cuda_active)
    """
    if sofa_comp.use_cuda:
        try:
            result = _build_sofa_scene_impl(
                sofa_comp, vertices, tetrahedra,
                initial_velocities, use_cuda=True)
            print("[SOFA] Scene built with SofaCUDA (GPU).")
            return result
        except Exception as e:
            print(f"[SOFA] CUDA build failed ({e}) — retrying on CPU.")

    result = _build_sofa_scene_impl(
        sofa_comp, vertices, tetrahedra,
        initial_velocities, use_cuda=False)
    print("[SOFA] Scene built on CPU.")
    return result


def _build_sofa_scene_impl(sofa_comp, vertices, tetrahedra,
                            initial_velocities, use_cuda: bool):
    """
    Internal: build one concrete SOFA scene with either CUDA or CPU components.
    Raises on any addObject failure so the caller can retry with use_cuda=False.
    """
    import Sofa
    import Sofa.Core

    root = Sofa.Core.Node("root")
    root.gravity.value = sofa_comp.gravity
    root.dt.value      = sofa_comp.time_step

    if use_cuda:
        root.addObject('RequiredPlugin', name='SofaCUDA')

    root.addObject('RequiredPlugin', name='Sofa.Component.ODESolver.Backward')
    root.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Iterative')
    root.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Dynamic')
    root.addObject('RequiredPlugin', name='Sofa.Component.StateContainer')
    root.addObject('RequiredPlugin', name='Sofa.Component.Mass')
    root.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Projective')
    root.addObject('RequiredPlugin', name='Sofa.Component.AnimationLoop')
    if not use_cuda:
        root.addObject('RequiredPlugin',
                       name='Sofa.Component.SolidMechanics.FEM.Elastic')

    root.addObject('DefaultAnimationLoop')

    obj = root.addChild('Object')
    obj.addObject('EulerImplicitSolver',
                  rayleighStiffness=0.01, rayleighMass=0.01)
    obj.addObject('CGLinearSolver', iterations=25, tolerance=1e-9, threshold=1e-9)

    obj.addObject('TetrahedronSetTopologyContainer',
                  points=vertices.tolist(),
                  tetrahedra=tetrahedra.tolist())
    modifier = obj.addObject('TetrahedronSetTopologyModifier')

    template = 'CudaVec3f' if use_cuda else 'Vec3d'
    mech = obj.addObject('MechanicalObject',
                         name='dofs', template=template,
                         position=vertices.tolist())

    if use_cuda:
        obj.addObject('CudaTetrahedronFEMForceField',
                      method='large',
                      youngModulus=sofa_comp.young_modulus,
                      poissonRatio=sofa_comp.poisson_ratio)
        obj.addObject('CudaUniformMass', totalMass=sofa_comp.total_mass)
    else:
        obj.addObject('TetrahedralCorotationalFEMForceField',
                      method='large',
                      youngModulus=sofa_comp.young_modulus,
                      poissonRatio=sofa_comp.poisson_ratio)
        obj.addObject('UniformMass', totalMass=sofa_comp.total_mass)

    if sofa_comp.fixed_indices:
        obj.addObject('FixedProjectiveConstraint',
                      indices=sofa_comp.fixed_indices)

    Sofa.Simulation.init(root)

    if initial_velocities is not None:
        mech.velocity.value = initial_velocities.tolist()

    return root, mech, modifier, use_cuda


# ---------------------------------------------------------------------------
# Poke
# ---------------------------------------------------------------------------

def _apply_poke(sofa_comp: SofaSimulationComponent):
    """Apply a downward impulse to the top 5 % of vertices."""
    if sofa_comp.poke_mask is None or sofa_comp.mechanical_object is None:
        return
    vels = np.array(sofa_comp.mechanical_object.velocity.value, dtype=np.float32)
    vels[sofa_comp.poke_mask, 1] -= sofa_comp.poke_speed
    sofa_comp.mechanical_object.velocity.value = vels.tolist()
    n = int(sofa_comp.poke_mask.sum())
    print(f"[Poke] Applied {sofa_comp.poke_speed} m/s downward to {n} vertices")


# ---------------------------------------------------------------------------
# Cut
# ---------------------------------------------------------------------------

def _perform_cut(sofa_comp: SofaSimulationComponent,
                 mesh_comp: StaticMeshComponent):
    """
    Remove tetrahedra above the cutting plane.

    Strategy:
        1. Ask SOFA's TetrahedronSetTopologyModifier to remove the tets.
           If removeTetrahedra is exposed in the Python bindings this is the
           cleanest path: SOFA rebuilds its internal data structures and the
           FEM/mass components see the reduced topology immediately.

        2. If removeTetrahedra raises AttributeError (not yet bound), fall
           back to re-initialising the full SOFA scene with the reduced mesh,
           copying positions and velocities so dynamics are continuous.

    In both cases the rendered surface is updated to reflect the cut.
    """
    positions  = np.array(sofa_comp.mechanical_object.position.value,  dtype=np.float32)
    velocities = np.array(sofa_comp.mechanical_object.velocity.value,   dtype=np.float32)

    origin = np.array(sofa_comp.cut_plane_origin, dtype=np.float32)
    normal = np.array(sofa_comp.cut_plane_normal,  dtype=np.float32)
    normal /= np.linalg.norm(normal)

    # Fetch current topology from SOFA (may have changed from prior cuts).
    try:
        container   = sofa_comp.sofa_root.Object.getObject(
                          'TetrahedronSetTopologyContainer')
        current_tets = np.array(container.tetrahedra.value, dtype=np.int32)
    except Exception:
        current_tets = sofa_comp.tet_mesh.tetrahedra.copy()

    signed_dist = (positions - origin) @ normal          # (N_verts,)
    tet_dists   = signed_dist[current_tets]              # (N_tets, 4)
    any_above   = np.any(tet_dists > 0, axis=1)
    remove_ids  = np.where(any_above)[0].tolist()
    new_tets    = current_tets[~any_above]

    if not remove_ids:
        print("[Cut] No tetrahedra above the cutting plane.")
        return

    print(f"[Cut] Removing {len(remove_ids):,} / {len(current_tets):,} tetrahedra...")

    # --- Attempt 1: SOFA topology modifier ---
    removed_in_sofa = False
    if sofa_comp.topology_modifier is not None:
        try:
            sofa_comp.topology_modifier.removeTetrahedra(remove_ids)
            removed_in_sofa = True
            print("[Cut] Topology updated via SOFA removeTetrahedra.")
        except AttributeError:
            print("[Cut] removeTetrahedra not bound in Python — "
                  "re-initialising SOFA scene.")
        except Exception as e:
            print(f"[Cut] removeTetrahedra failed ({e}) — "
                  "re-initialising SOFA scene.")

    # --- Attempt 2: full scene re-initialisation ---
    if not removed_in_sofa and len(new_tets) > 0:
        root, mech, modifier, cuda_ok = _build_sofa_scene(
            sofa_comp, positions, new_tets,
            initial_velocities=velocities)
        sofa_comp.sofa_root         = root
        sofa_comp.mechanical_object = mech
        sofa_comp.topology_modifier = modifier
        sofa_comp._cuda_active      = cuda_ok
        print("[Cut] SOFA scene re-initialised with reduced topology.")

    if len(new_tets) == 0:
        print("[Cut] All tetrahedra removed — nothing to render.")
        return

    # --- Update rendered surface ---
    new_surface              = _extract_boundary_faces(new_tets, positions)
    sofa_comp.surface_indices = new_surface
    new_normals              = _compute_normals(positions, new_surface)
    print(f"[Cut] Remaining tets: {len(new_tets):,} | "
          f"New surface: {len(new_surface):,} triangles.")

    flat = new_surface.flatten().astype(np.uint32)
    mesh_comp.indices = new_surface   # drives draw_indexed count

    gl.glBindVertexArray(mesh_comp.render_pipeline)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh_comp.index_buffer)
    gl.glBufferData(
        gl.GL_ELEMENT_ARRAY_BUFFER,
        flat.nbytes,
        flat.ctypes.data_as(ctypes.POINTER(gl.GLuint)),
        gl.GL_DYNAMIC_DRAW,
    )
    gl.glBindVertexArray(0)

    _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[1], new_normals)
    print("[Cut] GPU buffers updated.")


# ---------------------------------------------------------------------------
# Shared helpers  (also imported by test files)
# ---------------------------------------------------------------------------

def _extract_boundary_faces(tetrahedra: np.ndarray,
                             vertices:   np.ndarray = None) -> np.ndarray:
    """
    Return the boundary (outer surface) triangles of a tetrahedral mesh.

    A face is on the boundary iff it belongs to exactly one tetrahedron.
    When `vertices` is supplied, face winding is corrected so normals point
    outward (required for correct backface culling).

    Returns np.ndarray of shape (N_faces, 3), dtype uint32.
    """
    face_combos      = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
    opposite         = np.array([3, 2, 1, 0], dtype=np.int32)
    all_faces        = tetrahedra[:, face_combos].reshape(-1, 3)
    all_faces_sorted = np.sort(all_faces, axis=1)

    _, inverse, counts = np.unique(
        all_faces_sorted, axis=0, return_inverse=True, return_counts=True)

    is_boundary    = (counts == 1)[inverse]
    boundary_faces = all_faces[is_boundary].copy()

    if vertices is not None and len(boundary_faces) > 0:
        all_opposite    = tetrahedra[:, opposite].reshape(-1)
        surface_opp     = all_opposite[is_boundary]

        v0  = vertices[boundary_faces[:, 0]]
        v1  = vertices[boundary_faces[:, 1]]
        v2  = vertices[boundary_faces[:, 2]]
        opp = vertices[surface_opp]

        face_normals = np.cross(v1 - v0, v2 - v0)
        to_interior  = opp - v0
        inward       = np.einsum('ij,ij->i', face_normals, to_interior) > 0
        boundary_faces[inward] = boundary_faces[inward][:, [0, 2, 1]]

    return boundary_faces.astype(np.uint32)


def _compute_normals(vertices: np.ndarray,
                     indices:  np.ndarray) -> np.ndarray:
    """Compute per-vertex normals by accumulating area-weighted face normals."""
    normals      = np.zeros_like(vertices, dtype=np.float32)
    v0           = vertices[indices[:, 0]]
    v1           = vertices[indices[:, 1]]
    v2           = vertices[indices[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)

    np.add.at(normals, indices[:, 0], face_normals)
    np.add.at(normals, indices[:, 1], face_normals)
    np.add.at(normals, indices[:, 2], face_normals)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (normals / norms).astype(np.float32)


def _update_vbo(vao: int, vbo: int, data: np.ndarray):
    """Upload new data into an existing VBO (no reallocation)."""
    flat = data.flatten().astype(np.float32)
    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, flat.nbytes, flat)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    gl.glBindVertexArray(0)
