# Tetrahedral Mesh Generation for pyGandalf

Thesis Project - Tetrahedral Mesh Processing and Physics Simulation Pipeline

---

## Required Modules

Install the following additional packages beyond the base pyGandalf installation:

```bash
pip install taichi
pip install tetgen
pip install trimesh
pip install fast-simplification
pip install networkx
pip install pymeshfix
pip install pyvista
pip install PyGLM
```

**Dependencies:**
- **taichi** (1.7.4+) - GPU-accelerated parallel computing framework
- **tetgen** (0.6.7+) - Tetrahedral mesh generation (C++ with Python bindings)
- **trimesh** - Surface mesh loading, repair, and simplification
- **fast-simplification** - Backend for trimesh quadric decimation
- **networkx** - Required by trimesh's `fill_holes()` for topological hole repair on non-watertight meshes
- **pymeshfix** - Self-intersection repair (MeshFix wrapper); required for real-world meshes before TetGen
- **pyvista** (0.46+) - 3D visualization; also required by pymeshfix to read its output (install explicitly)
- **PyGLM** - Python bindings for GLM (OpenGL Mathematics); imported as `glm`
- **numpy** (>=2.0) - Already included in pyGandalf

---

## Modified Files

### `pyGandalf/utilities/mesh_lib.py`

**Added:**
- `TetrahedralMeshInstance` class - Data structure for tetrahedral meshes
  - Stores vertices (Nx3) and tetrahedra (Mx4)
  - `extract_surface()` - Extracts boundary faces with correct outward winding (no vertex duplication)
  - LEGACY methods kept for reference: `extract_all_faces()`, `extract_interior_and_surface_hybrid()`
- `build_tetrahedral(name, path, target_faces=None)` - Loads surface mesh, optionally simplifies, and tetrahedralizes
- GPU-accelerated normal computation kernels (`_accumulate_normals_kernel`, `_normalize_normals_kernel`)

**Fixed:**
- USD mesh loading (.usd, .usda, .usdc)
- Automatic normal recomputation for USD meshes
- Outward winding correction for extracted surface faces (normals were inverted)

### `pyGandalf/thesis_utilities/tet_generator.py`

**Added:**
- `generate_tetrahedral_mesh(surface_mesh, target_faces=None)` - TetGen wrapper
  - Optional `target_faces`: simplifies the input surface mesh before tetrahedralization
  - `_simplify_and_repair()`: tries progressively less aggressive decimation until a watertight mesh is achieved, with fallback to the original mesh
  - `_repair_and_extract()`: runs trimesh repair (fix_winding, fix_normals, fill_holes, merge_vertices), then pymeshfix self-intersection repair, then trimesh process=True rebuild
- Taichi GPU kernels: `compute_tet_volumes()`, `compute_tet_quality()`
- `analyze_tetrahedral_mesh()` - prints mesh statistics

### `pyGandalf/thesis_utilities/taichi_simulation_system.py`

**Added (Task 2 main file):**
- `_SpringMassSimulator` (@ti.data_oriented) - GPU spring-mass physics
  - Explicit Euler integration with substeps for stability
  - Per-spring stiffness array `_sk` (replaces scalar) — allows individual springs to be broken by zeroing stiffness without topology rebuild
  - `rebuild_springs()` - swaps spring network after topology change (cut)
  - `extend_springs()` - appends new springs (used for cutting springs) and returns start index
  - Kernels: `_clear_forces`, `_spring_forces` (atomic adds, uses per-spring `sk` array), `_integrate`
- `TaichiSimulationComponent` - stores params and runtime state
- `TaichiSimulationSystem` - ECS system, runs each frame
- GPU-accelerated normal computation (replaces slow `np.add.at` — 50ms → 4ms on bunny)
- `_cut_topology()` - shared helper: splits tetrahedra at the cut plane, duplicates seam vertices, rewires tets to use original (above) or duplicate (below) vertices
- `_filter_surface_faces()` - partitions raw boundary faces into outer (original surface) and wound (cut face) sets, enforcing correct winding for each half
- `_setup_progressive_cut()` - called once on first B press; splits mesh via `_cut_topology`, adds cutting springs between seam pairs (stiffness = k/2, rest length = 0), builds sorted seam-pair manifest and wound-face manifest, initialises blade cursor just before first seam pair
- `_advance_progressive_blade()` - called every frame while blade is active; advances blade cursor at `blade_speed` m/s, breaks springs behind cursor (zeroes `_sk`), applies opening velocity to separated pairs (batched), reveals wound faces behind cursor by updating the index buffer
- `_perform_cut()` - one-shot cut (C key): removes tets above plane, rebuilds springs, applies wound-opening velocity
- `_apply_poke()` - applies downward velocity impulse to top 5% of vertices
- `_extract_boundary_faces()` - with outward winding correction using opposite-vertex test
- `_build_springs()` - deduplicates 6 edges per tet into unique spring list

**Controls:**
- **F** — poke (downward impulse on top vertices)
- **B** — first press: initialise and start progressive blade cut; subsequent presses: pause/resume blade
- **C** — one-shot cut at the configured plane (disabled once B has been used)
- **P** — pause / resume physics simulation (blade still advances when paused)
- **X** — disc parallelism check (see below)

**Debug coloring + per-face unindexed rendering (active in test_random_cut.py):**
- Requires `lit_blinn_phong_debug.vs/.fs` shaders and a 4th attribute (per-face color, location 3)
- Rendering uses an **unindexed (exploded) layout**: each triangle owns 3 private vertices so face colors never interpolate across boundaries. Full wound-face buffer pre-allocated at cut time; trivial index buffer (`0,1,2, 3,4,5, …`) grows as blade reveals faces.
- `_compute_debug_face_colors(faces, n_orig)` — assigns one color per face (vectorised):
  - **Green** — all vertices are original (index < n_orig): regular outer surface
  - **Blue** — mixed vertices (some original, some new): collar face at cut boundary
  - **Red** — all vertices are new (index >= n_orig): disc (wound surface) face
- Colors stored per-face in `comp._debug_colors` (shape `(N_faces, 3)`); uploaded to GPU as `np.repeat(face_colors, 3, axis=0)` to match the 3×private-vertex layout
- `_check_disc_parallelism(comp, mesh_comp)` — **X key**, runs at any point after cut:
  - Collects all red (disc) faces from `comp._all_render_faces`
  - Computes per-face normals via cross product on current (post-simulation) positions
  - Separates above/below disc halves by sign of `dot(face_normal, cut_normal)`
  - For each half, picks the face nearest to the group centroid as reference
  - Colors **yellow** any disc face with `|dot(face_normal, ref_normal)| < 0.95` (~18° off-plane)
  - Overlays yellow on top of stored base colors; re-uploads only buffer[3]

**Performance notes:**
- `substeps=4`, `time_step=0.005` gives ~30 FPS on bunny with GPU
- Normals: moved from NumPy `np.add.at` (50ms) to Taichi GPU kernels (4ms)
- Stability limit: `sub_dt < dt_crit = 2*sqrt(m_vertex / k_effective)`

### `pyGandalf/resources/shaders/opengl/lit_blinn_phong_debug.vs` / `.fs`

**Added (debug coloring for cut investigation):**
- Vertex shader: same as `lit_blinn_phong.vs` plus `layout(location = 3) in vec3 a_Color` passed through as `v_Color`
- Fragment shader: same as `lit_blinn_phong.fs` but uses `v_Color` per-vertex instead of the `u_Color` material uniform

### `pyGandalf/thesis_utilities/sofa_simulation_system.py`

**Added (SOFA alternative — kept for reference):**
- `SofaSimulationComponent` / `SofaSimulationSystem` using SOFA's corotational FEM
- `TetrahedralCorotationalFEMForceField` (method='large') — physically more accurate than spring-mass
- Optional SofaCUDA support with full CPU fallback
- Cut via `TetrahedronSetTopologyModifier.removeTetrahedra()` with scene re-init fallback
- **Status:** Not actively used — SOFA's Python bindings are for an older version that lacks `CudaTetrahedronFEMForceField`, making GPU acceleration unavailable. CPU-only SOFA is too slow for real-time use.

---

## New Thesis-Specific Files

### `pyGandalf/thesis_utilities/` (New Directory)

### `pyGandalf/thesis_utilities/tet_exploder.py`

- `explode_tetrahedral_mesh(tet_mesh, explosion_factor, use_hybrid_normals)` - exploded view
- `use_hybrid_normals=True`: boundary faces get smooth normals (from geometry accumulation), interior faces get flat normals showing tet structure
- Taichi GPU kernel for exploding all tets in parallel
- Does NOT require `original_surface_mesh` — smooth normals computed from boundary face geometry directly

### `pyGandalf/thesis_utilities/animated_tet_exploder.py`

- `AnimatedTetExplosion` class - animated explosion with sine wave motion

### `pyGandalf/thesis_utilities/animated_explosion_system.py`

- `AnimatedExplosionComponent` / `AnimatedExplosionSystem` - ECS for animated explosion

---

## Test Files

### `My_tests/test_random_cut.py` — **ACTIVE DEVELOPMENT FILE**
- Taichi spring-mass simulation on sphere.obj with randomised progressive blade cut
- Random cut plane angle and depth each run (fixed seed=5 for reproducibility; change seed to explore)
- Controls: F = poke, B = start/pause progressive blade cut, C = one-shot cut (disabled once B used), P = pause/resume physics
- Current params: stiffness=200, damping=3.5, total_mass=100, substeps=4, time_step=0.005, opening_speed=2.0
- Uses `debug_mesh` shader (lit_blinn_phong_debug) + per-vertex debug colors (green/blue/red) for cut debugging

### `My_tests/test_cutting_simulation.py`
- Earlier version with fixed horizontal cut plane — superseded by test_random_cut.py

### `My_tests/test_sofa_simulation.py`
- SOFA FEM simulation (bunny.obj). Uses `SofaSimulationSystem`. Kept for reference.

### `My_tests/test_tet_rendering.py`
- Static rendering of tetrahedral bunny surface using `extract_surface()`

### `My_tests/test_exploded_tets.py`
- Exploded view of bunny tetrahedra with hybrid normals

### `My_tests/test_taichi.py`
- TetGen pipeline + GPU quality analysis

---

## Project Tasks

### Task 1 - Module 1: Mesh Conversion
Takes OBJ/USD surface mesh files as input and converts them into tetrahedral volumetric meshes.

**Status:** Complete
- OBJ and USD loading working
- TetGen tetrahedralization working (bunny.obj confirmed)
- Optional trimesh simplification + repair pipeline before TetGen

### Task 2 - Module 2: Spring-Based Slicing Simulator
Implements spring-mass dynamics on the tetrahedral mesh to simulate cutting forces, deformation, and separation.

**Status:** In progress
- GPU spring-mass simulation working on sphere.obj (~30 FPS)
- Poke deformation (F key) working
- One-shot cut (C key) working — removes tets above plane, updates surface, applies wound-opening velocity
- Progressive blade cut (B key) implemented — virtual node algorithm: duplicates seam vertices, adds cutting springs (k/2, rest=0), breaks them progressively as blade cursor advances; wound surface revealed face-by-face in sync with blade position
- Per-spring stiffness array enables spring breaking without topology rebuild (GPU-friendly)
- Spring damping added (`spring_damp * v_rel` term per spring) — resists oscillation without global velocity overdamping
- Opening velocity ramp — impulse spread over N frames instead of one step, prevents instability at cut time
- Mesh simplification pipeline added to reduce tet count and improve FPS

### Task 3 - Module 3: Surface Reconstruction
Extracts the outer surface from the damaged tetrahedral mesh and exports it as a new surface mesh format.

**Status:** Not started

### Task 4 - Module 4: GNN Surrogate Model
Trains a graph neural network on spring-mass trajectories to predict soft-tissue cutting deformation in real-time, without running the physics simulation.

**Novel contribution:** Existing simulators (SOFA, spring-mass) are too slow for real-time applications requiring sub-millisecond response. The GNN learns the per-frame physics step and runs in pyGandalf as a drop-in replacement for the spring-mass backend.

**Status:** In progress
- Headless simulation runner implemented (`headless_sim.py`) — strips OpenGL/ECS, exposes pure physics
- Dataset generation script implemented (`generate_dataset.py`) — randomised cuts, ~11.5h for 3000 runs
- Data format validated: (T, N, 3) trajectory + post-cut graph per .npz, ~20 MB/file
- GNN model and training script: not yet written

### Task 5 - Module 5: Simulation Method API
Abstract interface (`SimulationMethod`) making the ECS system and headless runner backend-agnostic.

**Status:** In progress
- Abstract base class written (`simulation_method.py`) with full docstrings
- Three planned implementations documented: `SpringMassMethod`, `FEMMethod`, `NeuralMethod`
- `SpringMassMethod` skeleton written (maps existing functions to interface)
- ECS refactor (wiring `TaichiSimulationSystem` to use the interface) not yet done
- `FEMMethod` (corotational FEM) planned for future work
- `NeuralMethod` depends on Task 4 GNN training completing

---

## Known Issues / Decisions

- **SOFA GPU (SofaCUDA):** `CudaTetrahedronFEMForceField` is not available in the installed SOFA version (constrained by pyGandalf's Python version). SOFA is kept as reference but not actively used.
- **Explicit Euler stability:** `sub_dt` must stay below `dt_crit = 2*sqrt(m_vertex / k_eff)`. Current params give ~14x safety margin.
- **Trimesh simplification API:** Installed version uses `target_reduction` (0–1 float) not face count. Code handles both via try/except.
- **Vertex duplication (LEGACY):** `extract_interior_and_surface_hybrid()` duplicated surface vertices for combined interior+surface rendering. Caused non-manifold TetGen input if reused. Now LEGACY — `extract_surface()` is the correct method.

---

## Running the Code

```bash
# Active simulation test (sphere, Taichi spring-mass + progressive cutting)
python My_tests/test_cutting_simulation.py

# Static surface rendering
python My_tests/test_tet_rendering.py

# Exploded view
python My_tests/test_exploded_tets.py
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
