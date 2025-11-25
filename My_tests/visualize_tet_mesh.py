"""
Visualize tetrahedral meshes using PyVista
"""

import numpy as np
import pyvista as pv
from pathlib import Path

from pyGandalf.utilities.mesh_lib import MeshLib
from pyGandalf.utilities.definitions import MODELS_PATH


def visualize_tetrahedral_mesh(tet_mesh, show_edges=True, show_surface=True, explode_view=False):
    """
    Visualize a tetrahedral mesh using PyVista

    Args:
        tet_mesh: TetrahedralMeshInstance to visualize
        show_edges: Show wireframe edges
        show_surface: Show outer surface
        explode_view: Separate tetrahedra apart to see interior
    """

    # Create PyVista unstructured grid from tetrahedral mesh
    # Cell type 10 = VTK_TETRA (tetrahedron)
    cell_type = np.full(len(tet_mesh.tetrahedra), 10, dtype=np.uint8)

    # PyVista format: [num_points, p0, p1, p2, p3] for each cell
    cells = np.column_stack((
        np.full(len(tet_mesh.tetrahedra), 4),  # 4 points per tetrahedron
        tet_mesh.tetrahedra
    )).ravel()

    # Create the mesh
    grid = pv.UnstructuredGrid(cells, cell_type, tet_mesh.vertices)

    # Create plotter
    plotter = pv.Plotter()

    if explode_view:
        # Explode view - separates tetrahedra to see interior structure
        # Higher factor = more separation (0.5 = 50% separation)
        exploded = grid.explode(factor=0.8)
        plotter.add_mesh(
            exploded,
            color='lightblue',
            show_edges=show_edges,
            edge_color='black',
            opacity=0.6,
            label='All Tetrahedra (Exploded)'
        )
    elif show_surface:
        # Extract and show outer surface
        surface = grid.extract_surface()
        plotter.add_mesh(
            surface,
            color='lightblue',
            show_edges=show_edges,
            edge_color='black',
            opacity=0.8,
            label='Surface'
        )
    else:
        # Show all tetrahedra (can be slow for large meshes)
        # Completely solid - zoom inside to test if vision is blocked
        plotter.add_mesh(
            grid,
            color='lightblue',
            opacity=1.0,
            show_edges=True,
            edge_color='black',
            label='All Tetrahedra (Solid)'
        )

    # Add info text
    info_text = f"""
    Tetrahedral Mesh: {tet_mesh.name}
    Vertices: {len(tet_mesh.vertices):,}
    Tetrahedra: {len(tet_mesh.tetrahedra):,}
    """
    plotter.add_text(info_text, position='upper_left', font_size=10)

    # Set viewing options
    plotter.add_axes()
    plotter.show_grid()
    plotter.camera_position = 'iso'

    # Show the mesh
    plotter.show()


def visualize_slice(tet_mesh, normal='x'):
    """
    Visualize a slice through the tetrahedral mesh

    Args:
        tet_mesh: TetrahedralMeshInstance to visualize
        normal: Slice plane normal ('x', 'y', or 'z')
    """

    # Create PyVista mesh
    cell_type = np.full(len(tet_mesh.tetrahedra), 10, dtype=np.uint8)
    cells = np.column_stack((
        np.full(len(tet_mesh.tetrahedra), 4),
        tet_mesh.tetrahedra
    )).ravel()
    grid = pv.UnstructuredGrid(cells, cell_type, tet_mesh.vertices)

    # Create plotter
    plotter = pv.Plotter()

    # Extract surface
    surface = grid.extract_surface()
    plotter.add_mesh(surface, opacity=0.3, color='lightblue')

    # Add slice
    slice_mesh = grid.slice(normal=normal)
    plotter.add_mesh(slice_mesh, color='red', show_edges=True, line_width=2)

    # Add info
    plotter.add_text(
        f"Slice through {tet_mesh.name} (normal={normal})",
        position='upper_left',
        font_size=10
    )

    plotter.add_axes()
    plotter.camera_position = 'iso'
    plotter.show()


def compare_surface_vs_tetrahedral(surface_mesh, tet_mesh):
    """
    Show surface mesh and tetrahedral mesh side by side

    Args:
        surface_mesh: Original MeshInstance (surface)
        tet_mesh: TetrahedralMeshInstance
    """

    # Create plotter with two viewports
    plotter = pv.Plotter(shape=(1, 2))

    # Left: Original surface mesh
    plotter.subplot(0, 0)
    surface_pv = pv.PolyData(surface_mesh.vertices,
                             np.column_stack((np.full(len(surface_mesh.indices), 3),
                                            surface_mesh.indices)).ravel())
    plotter.add_mesh(surface_pv, color='lightgreen', show_edges=True, label='Original Surface')
    plotter.add_text('Original Surface Mesh', font_size=12)
    plotter.camera_position = 'iso'

    # Right: Tetrahedral mesh surface
    plotter.subplot(0, 1)
    cell_type = np.full(len(tet_mesh.tetrahedra), 10, dtype=np.uint8)
    cells = np.column_stack((
        np.full(len(tet_mesh.tetrahedra), 4),
        tet_mesh.tetrahedra
    )).ravel()
    grid = pv.UnstructuredGrid(cells, cell_type, tet_mesh.vertices)
    tet_surface = grid.extract_surface()
    plotter.add_mesh(tet_surface, color='lightblue', show_edges=True, label='Tet Surface')
    plotter.add_text('Tetrahedral Mesh Surface', font_size=12)
    plotter.camera_position = 'iso'

    plotter.show()


if __name__ == "__main__":
    print("Loading bunny mesh...")

    # Load surface mesh
    surface_mesh = MeshLib().build("bunny", MODELS_PATH / "bunny.obj")

    print("Generating tetrahedral mesh...")

    # Generate tetrahedral mesh
    tet_mesh = MeshLib().build_tetrahedral("bunny", MODELS_PATH / "bunny.obj")

    print("\nVisualization Options:")
    print("1. Surface view (default)")
    print("2. Slice view")
    print("3. Comparison view (surface vs tetrahedral)")
    print("4. Exploded view (separates tetrahedra to see interior - RECOMMENDED)")

    choice = input("\nChoose visualization (1/2/3/4) [default=1]: ").strip() or "1"

    if choice == "1":
        print("\nShowing tetrahedral mesh surface...")
        visualize_tetrahedral_mesh(tet_mesh, show_edges=True, show_surface=True)
    elif choice == "2":
        normal = input("Slice normal (x/y/z) [default=x]: ").strip() or "x"
        print(f"\nShowing slice through mesh (normal={normal})...")
        visualize_slice(tet_mesh, normal=normal)
    elif choice == "3":
        print("\nComparing surface mesh vs tetrahedral mesh...")
        compare_surface_vs_tetrahedral(surface_mesh, tet_mesh)
    elif choice == "4":
        print(f"\nRendering ALL {len(tet_mesh.tetrahedra):,} tetrahedra in exploded view...")
        print("This separates tetrahedra so you can see interior structure and test performance...")
        visualize_tetrahedral_mesh(tet_mesh, show_edges=True, explode_view=True)
    else:
        print("Invalid choice, showing surface view...")
        visualize_tetrahedral_mesh(tet_mesh, show_edges=True, show_surface=True)
