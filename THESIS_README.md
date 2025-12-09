# Tetrahedral Mesh Generation for pyGandalf

Thesis Project - Tetrahedral Mesh Processing and Physics Simulation Pipeline

---

## Required Modules

Install the following additional packages beyond the base pyGandalf installation:

```bash
pip install taichi
pip install tetgen
```

**Dependencies:**
- **taichi** (1.7.4+) - GPU-accelerated parallel computing framework
- **tetgen** (0.6.7+) - Tetrahedral mesh generation (C++ with Python bindings)
- **pyvista** (0.46+) - 3D visualization (installed automatically with tetgen)
- **numpy** (>=2.0) - Already included in pyGandalf

---

## Modified Files

### `pyGandalf/utilities/mesh_lib.py`

**Added:**
- `TetrahedralMeshInstance` class - Data structure for tetrahedral meshes
  - Stores vertices (Nx3 array) and tetrahedra (Mx4 array of vertex indices)
- `build_tetrahedral()` method - Loads surface mesh and generates tetrahedral mesh

**Fixed:**
- USD basic mesh loading (.usd, .usda, .usdc) - geometry only (single mesh, no materials/animation)
- Automatic normal computation for USD meshes when missing or invalid

### `pyGandalf/utilities/tet_generator.py`

**Added:**
- `generate_tetrahedral_mesh()` function - Uses TetGen library to convert surface mesh to tetrahedral mesh
- `compute_tet_volumes()` Taichi kernel - GPU-accelerated volume calculation
- `compute_tet_quality()` Taichi kernel - GPU-accelerated quality metrics
- `analyze_tetrahedral_mesh()` helper function - Analyzes mesh statistics

---

## New Thesis-Specific Files

### `pyGandalf/thesis_utilities/` (New Directory)

Created a dedicated folder for thesis-specific code to keep it separate from pyGandalf core utilities.

### `pyGandalf/thesis_utilities/tet_exploder.py`

**Added:**
- `explode_tetrahedral_mesh()` - Creates static exploded view where tetrahedra move away from center
- Extracts all 4 triangular faces from each tetrahedron for rendering
- Taichi GPU-accelerated normal computation
  - Handles 711,000 tetrahedra (2.8M triangles)
  - Performance: minutes (Python) → seconds (Taichi GPU)

### `pyGandalf/thesis_utilities/animated_tet_exploder.py`

**Added:**
- `AnimatedTetExplosion` class - Manages animated explosion effect with sine wave motion
- Pre-computes explosion directions for all tetrahedra
- NumPy vectorized operations for per-frame updates
- Taichi GPU-accelerated normal computation

### `pyGandalf/thesis_utilities/animated_explosion_system.py`

**Added:**
- `AnimatedExplosionComponent` - Stores animation data
- `AnimatedExplosionSystem` - ECS system that updates vertex positions in GPU buffers every frame
- Integrates with pyGandalf's rendering system
- Built-in performance profiling

---

## Test Files

### `My_tests/test_taichi.py`
- Tetrahedral mesh generation pipeline test
- GPU-accelerated quality analysis

### `My_tests/test_exploded_tets.py`
- Static exploded tetrahedral visualization
- Verifies interior tetrahedra are visible

### `My_tests/test_animated_explosion.py`
- Interactive animated explosion demo
- Full ECS setup with camera controls
- Verifies tetrahedral structure throughout volume

---

## Project Tasks

### Task 2 - Module 1: Mesh Conversion (2 weeks from 19/11)
Takes OBJ/USD surface mesh files as input and converts them into tetrahedral volumetric meshes suitable for physics simulation.

**Status:** Completed

### Task 3 - Module 2: Spring-Based Slicing Simulator (2 weeks from xx/xx)
Implements spring-mass dynamics on the tetrahedral mesh to simulate cutting forces, deformation, and separation.

**Status:** Not started

### Task 4 - Module 3: Surface Reconstruction (2 weeks from xx/xx)
Extracts the outer surface from the damaged tetrahedral mesh and exports it as a new surface mesh format.

**Status:** Not started

### Task 5 - Module 4: Data Export Pipeline (2 weeks from xx/xx)
Structures and packages simulation parameters (input mesh, cutting plane location/direction, material properties) alongside results (output mesh, damage topology) into standardized formats (JSON metadata with associated mesh files).

**Status:** Not started

---

## Running the Code

```bash
# Generate tetrahedral mesh
python My_tests/test_taichi.py

# Static exploded visualization
python My_tests/test_exploded_tets.py

# Animated explosion (interactive, use mouse to control camera)
python My_tests/test_animated_explosion.py
```

---

## Author

**Thesis Project**
- Author: Antonakakis Panagiotis CSD 5137
- GitHub: https://github.com/panosanton
- Framework: pyGandalf (University of Crete & ICS-FORTH)
- Date: 2025
- Python Version: 3.10+

---

## License

This project extends pyGandalf, which is licensed under Apache 2.0.
