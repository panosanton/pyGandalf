"""
Test rendering tetrahedral mesh with pyGandalf

This test loads a tetrahedral mesh and renders it using pyGandalf's OpenGL renderer.
For now, it renders the surface of the tetrahedral mesh.
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


def main():
    # Set logger level
    logger.setLevel(logger.INFO)

    # Create application
    Application().create(OpenGLWindow('Tetrahedral Mesh Rendering Test', 1280, 720, True), OpenGLRenderer)

    # Create scene
    scene = Scene('Tetrahedral Mesh Test')

    # Enroll entities
    root = scene.enroll_entity()
    camera = scene.enroll_entity()
    tet_mesh_entity = scene.enroll_entity()
    light = scene.enroll_entity()

    # Build textures
    OpenGLTextureLib().build('white_texture', TextureData(image_bytes=0xffffffff.to_bytes(4, byteorder='big'), width=1, height=1))

    # Build shaders
    OpenGLShaderLib().build('default_mesh', SHADERS_PATH / 'opengl' / 'lit_blinn_phong.vs', SHADERS_PATH / 'opengl' / 'lit_blinn_phong.fs')

    # Build materials
    OpenGLMaterialLib().build('M_TetMesh', MaterialData('default_mesh', ['white_texture'], glm.vec4(0.3, 0.8, 0.5, 1.0), 1.0))

    # Generate tetrahedral mesh
    print("Generating tetrahedral mesh...")
    tet_mesh = MeshLib().build_tetrahedral('bunny_tet', MODELS_PATH / 'bunny.usdc')
    print(f"Generated {len(tet_mesh.tetrahedra):,} tetrahedra")

    # Extract all faces using hybrid rendering (smooth surface + interior structure)
    print("\nExtracting tetrahedral faces with hybrid normals...")
    print("(Smooth surface shading + visible interior structure)")
    tet_faces_mesh = tet_mesh.extract_interior_and_surface_hybrid()

    # Register the mesh with MeshLib so it can be used for rendering
    MeshLib().instance.meshes['tet_faces'] = tet_faces_mesh
    MeshLib().instance.meshes_names['tet_faces'] = 'tet_faces'

    print("\nUsing hybrid rendering:")
    print("  - Surface: smooth normals (natural shading)")
    print("  - Interior: visible tetrahedral structure")
    print("  - No boundary face overlap (clean rendering)")

    # Register components to root
    scene.add_component(root, TransformComponent(glm.vec3(0, 0, 0), glm.vec3(0, 0, 0), glm.vec3(1, 1, 1)))
    scene.add_component(root, InfoComponent('root'))
    scene.add_component(root, LinkComponent(None))

    # Register components to tet mesh entity (rendering all faces)
    scene.add_component(tet_mesh_entity, InfoComponent("tetrahedral_bunny"))
    scene.add_component(tet_mesh_entity, TransformComponent(glm.vec3(0, 0, 0), glm.vec3(0, 10, 0), glm.vec3(1, 1, 1)))
    scene.add_component(tet_mesh_entity, LinkComponent(root))
    scene.add_component(tet_mesh_entity, StaticMeshComponent('tet_faces'))  # Render extracted tetrahedral faces
    scene.add_component(tet_mesh_entity, MaterialComponent('M_TetMesh'))

    # Register components to light
    scene.add_component(light, InfoComponent("light"))
    scene.add_component(light, TransformComponent(glm.vec3(0, 5, 0), glm.vec3(0, 0, 0), glm.vec3(1, 1, 1)))
    scene.add_component(light, LinkComponent(root))
    scene.add_component(light, LightComponent(glm.vec3(1.0, 1.0, 1.0), 0.75))

    # Register components to camera
    scene.add_component(camera, InfoComponent("camera"))
    scene.add_component(camera, TransformComponent(glm.vec3(-0.25, 2, 4), glm.vec3(-15, 0, 0), glm.vec3(1, 1, 1)))
    scene.add_component(camera, LinkComponent(root))
    scene.add_component(camera, CameraComponent(45, 1.778, 0.1, 1000, 1.2, CameraComponent.Type.PERSPECTIVE))
    scene.add_component(camera, CameraControllerComponent())

    # Register systems
    scene.register_system(TransformSystem([TransformComponent]))
    scene.register_system(LinkSystem([LinkComponent, TransformComponent]))
    scene.register_system(CameraSystem([CameraComponent, TransformComponent]))
    scene.register_system(LightSystem([LightComponent, TransformComponent]))
    scene.register_system(OpenGLStaticMeshRenderingSystem([StaticMeshComponent, MaterialComponent, TransformComponent]))
    scene.register_system(CameraControllerSystem([CameraControllerComponent, CameraComponent, TransformComponent]))

    # Add scene to manager
    SceneManager().add_scene(scene)

    # Start application
    print("\nStarting pyGandalf renderer...")
    print("Hybrid tetrahedral rendering with optimal lighting")
    print("Use mouse to rotate and inspect the smooth surface + interior structure")

    Application().start()


if __name__ == "__main__":
    main()
