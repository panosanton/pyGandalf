"""
Animated Tetrahedral Explosion Test

Demonstrates tetrahedral mesh generation with animated explosion effect.
Tetrahedra continuously move away from and back to the center in a loop.
"""

from pyGandalf.core.application import Application
from pyGandalf.core.opengl_window import OpenGLWindow

from pyGandalf.systems.link_system import LinkSystem
from pyGandalf.systems.transform_system import TransformSystem
from pyGandalf.systems.camera_system import CameraSystem
from pyGandalf.systems.camera_controller_system import CameraControllerSystem
from pyGandalf.systems.opengl_rendering_system import OpenGLStaticMeshRenderingSystem
from pyGandalf.thesis_utilities.animated_explosion_system import AnimatedExplosionSystem, AnimatedExplosionComponent
from pyGandalf.systems.light_system import LightSystem

from pyGandalf.renderer.opengl_renderer import OpenGLRenderer

from pyGandalf.scene.scene import Scene
from pyGandalf.scene.scene_manager import SceneManager
from pyGandalf.scene.components import *

from pyGandalf.utilities.opengl_material_lib import OpenGLMaterialLib, MaterialData
from pyGandalf.utilities.opengl_texture_lib import OpenGLTextureLib, TextureData
from pyGandalf.utilities.opengl_shader_lib import OpenGLShaderLib
from pyGandalf.utilities.mesh_lib import MeshLib
from pyGandalf.thesis_utilities.animated_tet_exploder import AnimatedTetExplosion

from pyGandalf.utilities.definitions import SHADERS_PATH, MODELS_PATH
from pyGandalf.utilities.logger import logger


def main():
    # Set logger level
    logger.setLevel(logger.INFO)

    # Create application
    Application().create(OpenGLWindow('Animated Tetrahedral Explosion', 1280, 720, True), OpenGLRenderer)

    # Create scene
    scene = Scene('Animated Explosion')

    # Enroll entities
    root = scene.enroll_entity()
    camera = scene.enroll_entity()
    animated_mesh_entity = scene.enroll_entity()
    light = scene.enroll_entity()

    # Build textures
    OpenGLTextureLib().build('white_texture', TextureData(image_bytes=0xffffffff.to_bytes(4, byteorder='big'), width=1, height=1))

    # Build shaders
    OpenGLShaderLib().build('default_mesh', SHADERS_PATH / 'opengl' / 'lit_blinn_phong.vs', SHADERS_PATH / 'opengl' / 'lit_blinn_phong.fs')

    # Build materials
    OpenGLMaterialLib().build('M_AnimatedExplosion', MaterialData('default_mesh', ['white_texture'], glm.vec4(0.2, 0.8, 0.9, 1.0), 1.0))

    print("="*60)
    print("ANIMATED TETRAHEDRAL EXPLOSION TEST")
    print("="*60)

    # Generate tetrahedral mesh
    print("\n[1/4] Generating tetrahedral mesh...")
    tet_mesh = MeshLib().build_tetrahedral('bunny_tet', MODELS_PATH / 'bunny.obj')
    print(f"      Generated {len(tet_mesh.tetrahedra):,} tetrahedra")

    # Create animated explosion
    print("\n[2/4] Setting up animation system...")
    animation = AnimatedTetExplosion(
        tet_mesh,
        max_explosion=0.8,  # Maximum explosion distance
        speed=1.5           # Animation speed
    )

    # Create initial mesh
    print("\n[3/4] Creating mesh with all tetrahedral faces...")
    animated_mesh = animation.create_initial_mesh()
    print(f"      Mesh: {len(animated_mesh.vertices):,} vertices, {len(animated_mesh.indices):,} triangles")

    # Register mesh with MeshLib
    MeshLib().instance.meshes['animated_explosion'] = animated_mesh
    MeshLib().instance.meshes_names['animated_explosion'] = 'animated_explosion'

    # Register components to root
    scene.add_component(root, TransformComponent(glm.vec3(0, 0, 0), glm.vec3(0, 0, 0), glm.vec3(1, 1, 1)))
    scene.add_component(root, InfoComponent('root'))
    scene.add_component(root, LinkComponent(None))

    # Register components to animated mesh entity
    scene.add_component(animated_mesh_entity, InfoComponent("animated_tetrahedra"))
    scene.add_component(animated_mesh_entity, TransformComponent(glm.vec3(0, 0, 0), glm.vec3(0, 10, 0), glm.vec3(1, 1, 1)))
    scene.add_component(animated_mesh_entity, LinkComponent(root))
    scene.add_component(animated_mesh_entity, StaticMeshComponent('animated_explosion'))  # Mesh component for rendering
    scene.add_component(animated_mesh_entity, MaterialComponent('M_AnimatedExplosion'))
    scene.add_component(animated_mesh_entity, AnimatedExplosionComponent(animation))  # Animation component

    # Register components to light
    scene.add_component(light, InfoComponent("light"))
    scene.add_component(light, TransformComponent(glm.vec3(0, 5, 0), glm.vec3(0, 0, 0), glm.vec3(1, 1, 1)))
    scene.add_component(light, LinkComponent(root))
    scene.add_component(light, LightComponent(glm.vec3(1.0, 1.0, 1.0), 0.75))

    # Register components to camera
    scene.add_component(camera, InfoComponent("camera"))
    scene.add_component(camera, TransformComponent(glm.vec3(-0.25, 2, 5), glm.vec3(-15, 0, 0), glm.vec3(1, 1, 1)))
    scene.add_component(camera, LinkComponent(root))
    scene.add_component(camera, CameraComponent(45, 1.778, 0.1, 1000, 1.2, CameraComponent.Type.PERSPECTIVE))
    scene.add_component(camera, CameraControllerComponent())

    # Register systems (order matters!)
    scene.register_system(TransformSystem([TransformComponent]))
    scene.register_system(LinkSystem([LinkComponent, TransformComponent]))
    scene.register_system(CameraSystem([CameraComponent, TransformComponent]))
    scene.register_system(LightSystem([LightComponent, TransformComponent]))
    scene.register_system(AnimatedExplosionSystem([AnimatedExplosionComponent, StaticMeshComponent]))  # Updates vertex positions
    scene.register_system(OpenGLStaticMeshRenderingSystem([StaticMeshComponent, MaterialComponent, TransformComponent]))  # Renders meshes
    scene.register_system(CameraControllerSystem([CameraControllerComponent, CameraComponent, TransformComponent]))

    # Add scene to manager
    SceneManager().add_scene(scene)

    # Start application
    print("\n[4/4] Starting animated explosion...")
    print("="*60)
    print("ANIMATION:")
    print("  - Tetrahedra continuously move OUT and back IN")
    print("  - Sine wave animation creates smooth looping effect")
    print("  - Use mouse to rotate camera and view from different angles")
    print("  - Close window to exit")
    print("="*60)

    Application().start()


if __name__ == "__main__":
    main()
