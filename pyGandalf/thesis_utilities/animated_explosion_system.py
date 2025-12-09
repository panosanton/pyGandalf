"""
Animated Explosion System

Updates tetrahedral mesh vertices every frame to create animated explosion effect
"""

import OpenGL.GL as gl
import numpy as np
import time

from pyGandalf.systems.system import System
from pyGandalf.scene.components import Component, StaticMeshComponent
from pyGandalf.thesis_utilities.animated_tet_exploder import AnimatedTetExplosion


class AnimatedExplosionComponent(Component):
    """
    Component that stores animation data for exploding tetrahedra
    """
    def __init__(self, animation: AnimatedTetExplosion):
        super().__init__()
        self.animation = animation


class AnimatedExplosionSystem(System):
    """
    System that updates mesh vertices every frame for explosion animation.
    Works with OpenGLStaticMeshRenderingSystem - this system only updates vertices,
    the rendering system handles actual rendering.
    """

    def __init__(self, component_types):
        super().__init__(component_types)
        # Performance tracking
        self.frame_count = 0
        self.total_buffer_update_time = 0.0

    def on_create_entity(self, entity_id: int, components: tuple[Component]):
        """
        Called when entity is created.
        No setup needed - OpenGLStaticMeshRenderingSystem handles mesh initialization.
        This method exists to satisfy the System base class interface requirement.
        """
        pass

    def on_update_entity(self, ts: float, entity_id: int, components: tuple[Component]):
        """Called every frame to update animation"""
        anim_comp: AnimatedExplosionComponent
        mesh_comp: StaticMeshComponent
        anim_comp, mesh_comp = components

        if anim_comp and mesh_comp:
            # Check if mesh has been initialized by rendering system
            if mesh_comp.render_pipeline is not None and len(mesh_comp.buffers) > 0:
                # Update animation and get new vertex positions
                updated_vertices = anim_comp.animation.update(ts)

                # Update the OpenGL buffer with new vertices
                self._update_vertex_buffer(mesh_comp, updated_vertices)

    def _update_vertex_buffer(self, mesh: StaticMeshComponent, vertices: np.ndarray):
        """Update OpenGL vertex buffer with new positions"""
        start_time = time.perf_counter()

        # Bind VAO
        gl.glBindVertexArray(mesh.render_pipeline)

        # Bind position VBO and update data
        vbo_positions = mesh.buffers[0]  # First VBO is positions
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_positions)

        # Time the data preparation
        prep_start = time.perf_counter()
        vertex_data = vertices.flatten().astype(np.float32)
        prep_time = (time.perf_counter() - prep_start) * 1000

        # Time the GPU upload
        upload_start = time.perf_counter()
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, vertex_data.nbytes, vertex_data)
        gl.glFinish()  # Wait for GPU to complete
        upload_time = (time.perf_counter() - upload_start) * 1000

        # Unbind
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        # Track performance
        total_time = (time.perf_counter() - start_time) * 1000
        self.total_buffer_update_time += total_time
        self.frame_count += 1

        # Print stats every 100 frames
        if self.frame_count % 100 == 0:
            avg_time = self.total_buffer_update_time / self.frame_count
            print(f"[AnimatedExplosionSystem] Avg buffer update: {avg_time:.2f}ms (prep: {prep_time:.2f}ms, upload: {upload_time:.2f}ms, {self.frame_count} frames)")
