"""
Simple test script to verify Taichi setup and tetrahedral mesh generation
"""

from pathlib import Path
from pyGandalf.utilities.mesh_lib import MeshLib, TetrahedralMeshInstance
from pyGandalf.thesis_utilities.tet_generator import analyze_tetrahedral_mesh
from pyGandalf.utilities.definitions import MODELS_PATH

def test_taichi_setup():
    """Test that Taichi is working"""
    import taichi as ti

    print("=== Taichi Setup Test ===")
    print(f"Taichi version: {ti.__version__}")
    print(f"Backend: {ti.lang.impl.current_cfg().arch}")
    print("[OK] Taichi is installed and working!\n")


def test_mesh_loading():
    """Test loading a surface mesh"""
    print("=== Testing Surface Mesh Loading ===")

    # Load a simple mesh (bunny)
    mesh_path = MODELS_PATH / "bunny.obj"
    surface_mesh = MeshLib().build("bunny", mesh_path)

    print(f"Loaded mesh: {surface_mesh.name}")
    print(f"Vertices: {len(surface_mesh.vertices)}")
    print(f"Triangles: {len(surface_mesh.indices)}")
    print("[OK] Surface mesh loading works!\n")

    return surface_mesh


def test_tetrahedral_generation(surface_mesh):
    """Test tetrahedral mesh generation (currently just placeholder)"""
    print("=== Testing Tetrahedral Mesh Generation ===")

    # Generate tetrahedral mesh
    tet_mesh = MeshLib().build_tetrahedral("bunny", MODELS_PATH / "bunny.obj")

    print(f"Generated tet mesh: {tet_mesh.name}")
    print(f"Vertices: {len(tet_mesh.vertices)}")
    print(f"Tetrahedra: {len(tet_mesh.tetrahedra)}")
    print("[OK] Tetrahedral generation works (placeholder)!\n")

    return tet_mesh


def test_taichi_analysis(tet_mesh):
    """Test Taichi-accelerated mesh analysis"""
    print("=== Testing Taichi-Accelerated Analysis ===")

    # Analyze the tetrahedral mesh using GPU-accelerated functions
    analyze_tetrahedral_mesh(tet_mesh)

    print("\n[OK] Taichi GPU acceleration works!")


if __name__ == "__main__":
    print("=" * 60)
    print("TETRAHEDRAL MESH GENERATION TEST")
    print("=" * 60)
    print()

    try:
        # Test 1: Taichi setup
        test_taichi_setup()

        # Test 2: Load surface mesh
        surface_mesh = test_mesh_loading()

        # Test 3: Generate tetrahedral mesh
        tet_mesh = test_tetrahedral_generation(surface_mesh)

        # Test 4: Analyze with Taichi
        test_taichi_analysis(tet_mesh)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Implement actual tetrahedralization algorithm in tet_generator.py")
        print("2. Add visualization of tetrahedral meshes")
        print("3. Export to USDA format")

    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
