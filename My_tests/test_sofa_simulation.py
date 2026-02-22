"""
SOFA Physics Simulation Test

Loads a tetrahedral sphere mesh, hands it to SOFA for spring-mass simulation,
and renders the deforming surface in pyGandalf each frame.

Controls:
    Mouse (right-click drag) - rotate camera
    WASD  - move camera
    C     - cut the mesh along the horizontal plane at y=0
    Close window to exit
"""

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

from pyGandalf.thesis_utilities.sofa_simulation_system import (
    SofaSimulationComponent,
    SofaSimulationSystem,
    _extract_boundary_faces,
    _compute_normals,
)

import numpy as np
import glm


def main():
    logger.setLevel(logger.INFO)

    Application().create(OpenGLWindow('SOFA FEM Simulation', 1280, 720, True), OpenGLRenderer)

    scene = Scene('SOFA Simulation')

    root = scene.enroll_entity()
    camera = scene.enroll_entity()
    bunny = scene.enroll_entity()
    light = scene.enroll_entity()

    # Resources
    OpenGLTextureLib().build('white_texture', TextureData(
        image_bytes=0xffffffff.to_bytes(4, byteorder='big'), width=1, height=1))

    OpenGLShaderLib().build('default_mesh',
        SHADERS_PATH / 'opengl' / 'lit_blinn_phong.vs',
        SHADERS_PATH / 'opengl' / 'lit_blinn_phong.fs')

    OpenGLMaterialLib().build('M_Bunny', MaterialData(
        'default_mesh', ['white_texture'], glm.vec4(0.3, 0.7, 0.4, 1.0), 1.0))

    # --- Tetrahedral mesh ---
    print("="*50)
    print("Generating tetrahedral mesh...")
    tet_mesh = MeshLib().build_tetrahedral('sphere_tet', MODELS_PATH / 'sphere.obj')
    print(f"  Vertices:   {len(tet_mesh.vertices):,}")
    print(f"  Tetrahedra: {len(tet_mesh.tetrahedra):,}")

    # --- Initial surface geometry for the first frame ---
    # The VBO holds ALL tet vertices. Only boundary faces are drawn.
    # SOFA updates positions in-place, so VBO vertex count never changes.
    print("Extracting surface faces...")
    surface_indices  = _extract_boundary_faces(tet_mesh.tetrahedra, tet_mesh.vertices)
    initial_normals  = _compute_normals(tet_mesh.vertices, surface_indices)
    texcoords        = np.zeros((len(tet_mesh.vertices), 2), dtype=np.float32)
    print(f"  Surface triangles: {len(surface_indices):,}")
    print("="*50)

    # --- Scene graph ---

    scene.add_component(root, TransformComponent(glm.vec3(0, 0, 0), glm.vec3(0, 0, 0), glm.vec3(1, 1, 1)))
    scene.add_component(root, InfoComponent('root'))
    scene.add_component(root, LinkComponent(None))

    scene.add_component(bunny, InfoComponent('sofa_bunny'))
    scene.add_component(bunny, TransformComponent(glm.vec3(0, 0, 0), glm.vec3(0, 10, 0), glm.vec3(1, 1, 1)))
    scene.add_component(bunny, LinkComponent(root))
    # Pass attributes directly — skips MeshLib lookup, load_from_file becomes False.
    scene.add_component(bunny, StaticMeshComponent(
        'sofa_surface',
        attributes=[tet_mesh.vertices.copy(), initial_normals, texcoords],
        indices=surface_indices,
    ))
    scene.add_component(bunny, MaterialComponent('M_Bunny'))
    sofa_comp = SofaSimulationComponent(
        tet_mesh,
        time_step=0.001,
        stiffness=100.0,
        damping=0.5,
        total_mass=50.0,
        gravity=[0, 0, 0],
    )
    sofa_comp.cut_plane_origin = [0.0, 0.0, 0.0]   # cut through centre of sphere
    sofa_comp.cut_plane_normal = [0.0, 1.0, 0.0]   # horizontal cut (Y axis)
    scene.add_component(bunny, sofa_comp)

    scene.add_component(light, InfoComponent('light'))
    scene.add_component(light, TransformComponent(glm.vec3(0, 5, 0), glm.vec3(0, 0, 0), glm.vec3(1, 1, 1)))
    scene.add_component(light, LinkComponent(root))
    scene.add_component(light, LightComponent(glm.vec3(1.0, 1.0, 1.0), 0.75))

    scene.add_component(camera, InfoComponent('camera'))
    scene.add_component(camera, TransformComponent(glm.vec3(0, 0, 5), glm.vec3(0, 0, 0), glm.vec3(1, 1, 1)))
    scene.add_component(camera, LinkComponent(root))
    scene.add_component(camera, CameraComponent(45, 1.778, 0.1, 1000, 1.2, CameraComponent.Type.PERSPECTIVE))
    scene.add_component(camera, CameraControllerComponent())

    # --- Systems ---
    # SofaSimulationSystem must run before OpenGLStaticMeshRenderingSystem so
    # updated positions are already in the VBO when the draw call happens.
    scene.register_system(TransformSystem([TransformComponent]))
    scene.register_system(LinkSystem([LinkComponent, TransformComponent]))
    scene.register_system(CameraSystem([CameraComponent, TransformComponent]))
    scene.register_system(LightSystem([LightComponent, TransformComponent]))
    scene.register_system(SofaSimulationSystem([SofaSimulationComponent, StaticMeshComponent]))
    scene.register_system(OpenGLStaticMeshRenderingSystem([StaticMeshComponent, MaterialComponent, TransformComponent]))
    scene.register_system(CameraControllerSystem([CameraControllerComponent, CameraComponent, TransformComponent]))

    SceneManager().add_scene(scene)

    print("Starting simulation. Close window to exit.")
    Application().start()


if __name__ == "__main__":
    main()
