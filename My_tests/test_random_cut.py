"""
Taichi Spring-Mass Simulation — Randomised Blade Cut

Controls:
    Mouse (right-click drag) - rotate camera
    WASD  - move camera
    F     - poke the top of the mesh downward
    B     - start / pause progressive blade cut
    Close window to exit

Usage:
    python My_tests/test_random_cut.py
    python My_tests/test_random_cut.py --mesh dragon_clean
    python My_tests/test_random_cut.py --mesh _bunny_jacobson --seed 12
"""

import argparse
import numpy as np
import glm

from pyGandalf.core.application import Application
from pyGandalf.core.opengl_window import OpenGLWindow

from pyGandalf.systems.link_system import LinkSystem
from pyGandalf.systems.transform_system import TransformSystem
from pyGandalf.systems.camera_system import CameraSystem
from pyGandalf.systems.camera_controller_system import CameraControllerSystem
from pyGandalf.systems.opengl_rendering_system import OpenGLStaticMeshRenderingSystem
from pyGandalf.systems.light_system import LightSystem

from pyGandalf.renderer.opengl_renderer import OpenGLRenderer

from pyGandalf.scene.scene import Scene
from pyGandalf.scene.scene_manager import SceneManager
from pyGandalf.scene.components import *

from pyGandalf.utilities.opengl_material_lib import OpenGLMaterialLib, MaterialData
from pyGandalf.utilities.opengl_texture_lib import OpenGLTextureLib, TextureData
from pyGandalf.utilities.opengl_shader_lib import OpenGLShaderLib
from pyGandalf.utilities.mesh_lib import MeshLib
from pyGandalf.utilities.definitions import SHADERS_PATH, MODELS_PATH
from pyGandalf.utilities.logger import logger

from pyGandalf.thesis_utilities.taichi_simulation_system import (
    TaichiSimulationComponent,
    TaichiSimulationSystem,
    _extract_boundary_faces,
    _compute_normals,
)

# Verified target_faces per mesh (None = use mesh as-is)
MESH_CONFIG = {
    'sphere':                   None,
    'Armadillo_verysimplified': None,
    'dragon_clean':             1500,
    'liver-smooth':             3500,
    'Armadillo_simplified':     None,
    '_bunny_jacobson':          15000,
}


def _random_cut_plane(rng: np.random.Generator, depth_range: float = 0.7):
    normal = rng.standard_normal(3).astype(np.float32)
    normal /= np.linalg.norm(normal)
    offset = rng.uniform(-depth_range, depth_range)
    origin = (normal * offset).astype(np.float32)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(normal, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    blade_dir = ref - float(np.dot(ref, normal)) * normal
    blade_dir /= np.linalg.norm(blade_dir)
    return normal, origin, blade_dir


def main():
    parser = argparse.ArgumentParser(description='Spring-mass cut simulation')
    parser.add_argument('--mesh', default='sphere',
                        choices=list(MESH_CONFIG.keys()),
                        help='Mesh to simulate (default: sphere)')
    parser.add_argument('--seed', type=int, default=5,
                        help='RNG seed for cut plane (default: 5)')
    parser.add_argument('--tet_scale', type=float, default=1.0,
                        help='Interior tet size relative to surface (default: 1.0; try 5-20 for fewer tets)')
    args = parser.parse_args()

    mesh_name    = args.mesh
    target_faces = MESH_CONFIG[mesh_name]

    logger.setLevel(logger.INFO)

    rng = np.random.default_rng(args.seed)
    normal, origin, blade_dir = _random_cut_plane(rng)

    print("=" * 50)
    print(f"Spring-mass simulation -- {mesh_name}")
    print(f"  normal    = [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
    print(f"  origin    = [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")
    print(f"  blade_dir = [{blade_dir[0]:.3f}, {blade_dir[1]:.3f}, {blade_dir[2]:.3f}]")
    print("=" * 50)

    Application().create(OpenGLWindow(f'Spring-Mass -- {mesh_name}', 1280, 720, True), OpenGLRenderer)

    scene = Scene('Random Cut Simulation')

    root   = scene.enroll_entity()
    camera = scene.enroll_entity()
    mesh_e = scene.enroll_entity()
    light  = scene.enroll_entity()

    OpenGLTextureLib().build('white_texture', TextureData(
        image_bytes=0xffffffff.to_bytes(4, byteorder='big'), width=1, height=1))

    OpenGLShaderLib().build('debug_mesh',
        SHADERS_PATH / 'opengl' / 'lit_blinn_phong_debug.vs',
        SHADERS_PATH / 'opengl' / 'lit_blinn_phong_debug.fs')

    OpenGLMaterialLib().build('M_Mesh', MaterialData(
        'debug_mesh', ['white_texture'], glm.vec4(0.3, 0.7, 0.4, 1.0), 1.0))

    print(f"Generating tetrahedral mesh: {mesh_name}.obj ...")
    tet_mesh = MeshLib().build_tetrahedral(
        f'{mesh_name}_tet',
        MODELS_PATH / f'{mesh_name}.obj',
        target_faces=target_faces,
        tet_scale=args.tet_scale,
    )
    print(f"  Vertices:   {len(tet_mesh.vertices):,}")
    print(f"  Tetrahedra: {len(tet_mesh.tetrahedra):,}")

    print("Extracting surface faces...")
    surface_indices = _extract_boundary_faces(tet_mesh.tetrahedra, tet_mesh.vertices)
    initial_normals = _compute_normals(tet_mesh.vertices, surface_indices)
    texcoords       = np.zeros((len(tet_mesh.vertices), 2), dtype=np.float32)
    initial_colors  = np.tile([0.3, 0.7, 0.4], (len(tet_mesh.vertices), 1)).astype(np.float32)
    print(f"  Surface triangles: {len(surface_indices):,}")
    print("=" * 50)

    scene.add_component(root, TransformComponent(glm.vec3(0, 0, 0), glm.vec3(0, 0, 0), glm.vec3(1, 1, 1)))
    scene.add_component(root, InfoComponent('root'))
    scene.add_component(root, LinkComponent(None))

    scene.add_component(mesh_e, InfoComponent(mesh_name))
    scene.add_component(mesh_e, TransformComponent(glm.vec3(0, 0, 0), glm.vec3(0, 0, 0), glm.vec3(1, 1, 1)))
    scene.add_component(mesh_e, LinkComponent(root))
    scene.add_component(mesh_e, StaticMeshComponent(
        f'{mesh_name}_surface',
        attributes=[tet_mesh.vertices.copy(), initial_normals, texcoords, initial_colors],
        indices=surface_indices,
    ))
    scene.add_component(mesh_e, MaterialComponent('M_Mesh'))

    taichi_comp = TaichiSimulationComponent(
        tet_mesh,
        time_step           = 0.005,
        substeps            = 4,
        stiffness           = 200.0,
        damping             = 3.5,
        total_mass          = 100.0,
        gravity             = [0.0, 0.0, 0.0],
        opening_speed       = 20.0,
        poke_speed          = 2.0,
        v_max               = 20.0,
        spring_damping      = 2.0,
        blade_travel_dir    = blade_dir.tolist(),
        blade_speed         = 0.5,
        split_disc_verts    = True,
        opening_ramp_frames = 20,
    )
    taichi_comp.cut_plane_origin = origin.tolist()
    taichi_comp.cut_plane_normal = normal.tolist()
    taichi_comp.hide_wound_faces = True
    scene.add_component(mesh_e, taichi_comp)

    scene.add_component(light, InfoComponent('light'))
    scene.add_component(light, TransformComponent(glm.vec3(0, 5, 0), glm.vec3(0, 0, 0), glm.vec3(1, 1, 1)))
    scene.add_component(light, LinkComponent(root))
    scene.add_component(light, LightComponent(glm.vec3(1.0, 1.0, 1.0), 0.75))

    scene.add_component(camera, InfoComponent('camera'))
    scene.add_component(camera, TransformComponent(glm.vec3(0, 0, 5), glm.vec3(0, 0, 0), glm.vec3(1, 1, 1)))
    scene.add_component(camera, LinkComponent(root))
    scene.add_component(camera, CameraComponent(45, 1.778, 0.1, 1000, 1.2, CameraComponent.Type.PERSPECTIVE))
    scene.add_component(camera, CameraControllerComponent())

    scene.register_system(TransformSystem([TransformComponent]))
    scene.register_system(LinkSystem([LinkComponent, TransformComponent]))
    scene.register_system(CameraSystem([CameraComponent, TransformComponent]))
    scene.register_system(LightSystem([LightComponent, TransformComponent]))
    scene.register_system(TaichiSimulationSystem([TaichiSimulationComponent, StaticMeshComponent]))
    scene.register_system(OpenGLStaticMeshRenderingSystem([StaticMeshComponent, MaterialComponent, TransformComponent]))
    scene.register_system(CameraControllerSystem([CameraControllerComponent, CameraComponent, TransformComponent]))

    SceneManager().add_scene(scene)

    print("Starting simulation.  Press B to cut.  Close window to exit.")
    Application().start()


if __name__ == "__main__":
    main()
