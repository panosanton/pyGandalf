"""
Tetrahedral Mesh Exploded View Test

This example demonstrates tetrahedral mesh generation and visualization
by creating an "exploded view" where each tetrahedron is moved away from
the center, making the internal structure visible.

This verifies that tetrahedralization is working correctly by showing
all tetrahedra as individual rendered objects.
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
from pyGandalf.thesis_utilities.tet_exploder import explode_tetrahedral_mesh

from pyGandalf.utilities.definitions import SHADERS_PATH, MODELS_PATH
from pyGandalf.utilities.logger import logger


def main():
    # Set logger level
    logger.setLevel(logger.INFO)

    # Create application
    Application().create(OpenGLWindow('Exploded Tetrahedral Mesh Test', 1280, 720, True), OpenGLRenderer)

    # Create scene
    scene = Scene('Exploded Tetrahedral Mesh')

    # Enroll entities
    root = scene.enroll_entity()
    camera = scene.enroll_entity()
    exploded_mesh_entity = scene.enroll_entity()
    light = scene.enroll_entity()

    # Build textures
    OpenGLTextureLib().build('white_texture', TextureData(image_bytes=0xffffffff.to_bytes(4, byteorder='big'), width=1, height=1))

    # Build shaders
    OpenGLShaderLib().build('default_mesh', SHADERS_PATH / 'opengl' / 'lit_blinn_phong.vs', SHADERS_PATH / 'opengl' / 'lit_blinn_phong.fs')

    # Build materials
    OpenGLMaterialLib().build('M_Exploded', MaterialData('default_mesh', ['white_texture'], glm.vec4(0.3, 0.7, 0.9, 1.0), 1.0))

    print("="*60)
    print("TETRAHEDRAL MESH EXPLODED VIEW TEST")
    print("="*60)

    # Generate tetrahedral mesh
    print("\n[1/3] Generating tetrahedral mesh...")
    tet_mesh = MeshLib().build_tetrahedral('bunny_tet', MODELS_PATH / 'bunny.obj')
    print(f"      Generated {len(tet_mesh.tetrahedra):,} tetrahedra from {len(tet_mesh.vertices):,} vertices")

    # Create exploded view
    print("\n[2/3] Creating exploded view...")
    print("      Each tetrahedron will be moved away from center for visualization")
    exploded_mesh = explode_tetrahedral_mesh(tet_mesh, explosion_factor=0.3)
    print(f"      Exploded mesh: {len(exploded_mesh.vertices):,} vertices, {len(exploded_mesh.indices):,} triangles")

    # Register the exploded mesh with MeshLib so it can be used by StaticMeshComponent
    MeshLib().instance.meshes['exploded_bunny'] = exploded_mesh
    MeshLib().instance.meshes_names['exploded_bunny'] = 'exploded_bunny'

    # Register components to root
    scene.add_component(root, TransformComponent(glm.vec3(0, 0, 0), glm.vec3(0, 0, 0), glm.vec3(1, 1, 1)))
    scene.add_component(root, InfoComponent('root'))
    scene.add_component(root, LinkComponent(None))

    # Register components to exploded mesh entity
    scene.add_component(exploded_mesh_entity, InfoComponent("exploded_tetrahedra"))
    scene.add_component(exploded_mesh_entity, TransformComponent(glm.vec3(0, 0, 0), glm.vec3(0, 10, 0), glm.vec3(1, 1, 1)))
    scene.add_component(exploded_mesh_entity, LinkComponent(root))
    scene.add_component(exploded_mesh_entity, StaticMeshComponent('exploded_bunny'))
    scene.add_component(exploded_mesh_entity, MaterialComponent('M_Exploded'))

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
    print("\n[3/3] Starting pyGandalf renderer...")
    print("="*60)
    print("VISUALIZATION:")
    print("  - Each tetrahedron is visible as 4 triangular faces")
    print("  - Gaps between tetrahedra show the exploded structure")
    print("  - Use mouse to rotate and zoom to inspect internal structure")
    print("  - This verifies all tetrahedra are present and correctly generated")
    print("="*60)

    Application().start()


if __name__ == "__main__":
    main()
