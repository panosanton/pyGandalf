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

## Project Tasks

### Task 1 - Mesh Conversion
**Status: COMPLETE**

Takes OBJ/USD surface mesh files as input and converts them into tetrahedral volumetric meshes.
- OBJ and USD loading working
- TetGen tetrahedralization working (all bundled meshes confirmed)
- Optional trimesh simplification + repair pipeline before TetGen
- `tet_scale` parameter in `build_tetrahedral()` controls interior tet size relative to surface (larger = fewer interior tets)

### Task 2 - Spring-Based Slicing Simulator
**Status: COMPLETE**

Implements spring-mass dynamics on the tetrahedral mesh to simulate cutting forces, deformation, and separation.
- GPU spring-mass simulation working on all bundled meshes
- Poke deformation (F key) working
- One-shot cut (C key) working
- Progressive blade cut (B key) fully implemented with virtual node algorithm
- Per-spring stiffness array enables spring breaking without topology rebuild
- Spring damping, opening velocity ramp, v_max clamping all implemented
- Simulation Method API fully wired (SpringMassMethod backend)

### Task 3 - Surface Reconstruction
**Status: Not started**

Extracts the outer surface from the cut tetrahedral mesh and exports it as a new surface mesh.

### Task 4 - GNN Surrogate Model
**Status: In progress**

Trains a graph neural network on simulation trajectories to predict soft-tissue deformation in real-time.

- Headless simulation runner implemented (`headless_sim.py`) -- strips OpenGL/ECS, exposes pure physics, method-agnostic
- Dataset generation script implemented (`generate_dataset.py`) -- randomised cuts, ~11.5h for 3000 runs
- `inspect_dataset.py` validates a .npz file
- Data format: (T, N, 3) trajectory + post-cut graph per .npz, ~20 MB/file
- `get_graph_data()` added to `SimulationMethod` API for GNN-friendly graph extraction
- GNN model and training script: not yet written

### Task 5 - Simulation Method API
**Status: COMPLETE**

Abstract interface (`SimulationMethod`) making the ECS system and headless runner backend-agnostic.

- `SimulationMethod` ABC with full interface: `initialize`, `step`, `get_positions`, `get_velocities`, `set_velocities`, `setup_cut`, `advance_blade`, `vertex_count`, `n_orig`, `get_graph_data`
- `SpringMassMethod` fully implemented and wired into `TaichiSimulationSystem`
- `FEMMethod` fully implemented (see Task 6)
- `NeuralMethod` planned (depends on Task 4)
- `_cut_topology_physics` extracted as shared helper used by both backends

### Task 6 - FEM Simulator
**Status: In progress**

Corotational FEM with implicit Euler integration and CG solver, running on GPU via Taichi.

- `_FEMSimulator` (@ti.data_oriented) in `fem_simulator.py`
- Corotational neo-Hookean: `P = 2*mu*(F - R)`, R from polar decomposition of F
- Implicit Euler: `(M + dt^2 K) v_new = M v_old + dt f`, solved by matrix-free PCG
- Jacobi preconditioner (per-vertex diagonal of A), significantly reduces CG iterations
- Polar decomposition via `ti.polar_decompose` (replaces slower SVD + sign-correction block)
- Cutting springs applied explicitly on top of implicit FEM (semi-implicit hybrid)
- `sliver_vol_threshold` filters near-degenerate tets near the cut plane to avoid degenerate forces
- Orphan constraint: zero-mass verts (stranded by sliver filtering) follow their nearest master with a fixed positional offset, so they move rigidly with the mesh without teleporting
- `FEMMethod` in `simulation_method.py` wraps `_FEMSimulator` and implements the `SimulationMethod` interface
- No substeps needed -- implicit integrator is unconditionally stable
- `test_fem_cut.py` is the active FEM test

**Pending optimizations (identified, not yet applied):**
- Static loop unrolling in `_matmul` (`ti.static(range(...))`)
- Algebraic multigrid preconditioner for headless/dataset path

---

## Modified Files

### `pyGandalf/utilities/mesh_lib.py`

**Added:**
- `TetrahedralMeshInstance` class - stores vertices (Nx3) and tetrahedra (Mx4)
  - `extract_surface()` - extracts boundary faces with correct outward winding
  - LEGACY methods kept for reference: `extract_all_faces()`, `extract_interior_and_surface_hybrid()`
- `build_tetrahedral(name, path, target_faces=None, tet_scale=1.0)` - loads surface mesh, optionally simplifies, and tetrahedralizes
- GPU-accelerated normal computation kernels (`_accumulate_normals_kernel`, `_normalize_normals_kernel`)

**Fixed:**
- USD mesh loading (.usd, .usda, .usdc)
- Automatic normal recomputation for USD meshes
- Outward winding correction for extracted surface faces

### `pyGandalf/thesis_utilities/simulation_method.py`

**Abstract interface and all backends:**
- `SimulationMethod` ABC -- backend-agnostic interface used by both ECS and headless runner
- `SpringMassMethod` -- wraps `_SpringMassSimulator`; fully wired into `TaichiSimulationSystem`
- `FEMMethod` -- wraps `_FEMSimulator`; corotational FEM with CG implicit solver
  - Reads params: `young_modulus`, `poisson_ratio`, `density`, `cg_iters`, `cg_eps`, `damping`, `v_max`, `gravity`
  - `setup_cut()` calls `_cut_topology_physics`, discards spring-mass result, builds `_FEMSimulator` from split topology
  - Orphan constraint: `{orphan: (master, offset)}` dict; each step sets `pos[orphan] = pos[master] + offset`
- `NeuralMethod` -- placeholder, not yet implemented

### `pyGandalf/thesis_utilities/fem_simulator.py`

**New file -- GPU corotational FEM:**
- `_FEMSimulator` (@ti.data_oriented)
  - `_init_tet_data()` -- computes rest-shape inverse B = Dm^-1, tet volume W, accumulates vertex masses by volume
  - `_get_force()` -- deformation gradient F = Ds * B, polar decomp R, 1st Piola P = 2*mu*(F-R), assembles nodal forces
  - `_matmul()` -- matrix-free A*v product for CG: A = M + dt^2 * K (K assembled from tet stiffness contributions)
  - `_cg()` -- Jacobi-preconditioned CG solver (max `cg_iters` iterations, tolerance `cg_eps`)
  - `_apply_spring_forces()` -- cutting spring forces (semi-implicit; springs break when stiffness zeroed)
  - `_apply_fixed()` -- zero velocity of fixed verts
  - `_integrate()` -- update positions from velocities; clamp speed to v_max
  - `_apply_orphan_constraints()` -- snap zero-mass verts to masters with fixed offset

### `pyGandalf/thesis_utilities/taichi_cut_utils.py`

**Cut-pipeline helpers (extracted from `taichi_simulation_system.py`):**
- `_cut_topology_physics()` -- shared topology helper used by both SpringMassMethod and FEMMethod
  - Pre-snaps near-plane verts (`0 < dist < snap_eps` snapped to `-snap_eps`) to prevent degenerate 1+3 splits
  - Returns `phantom_above_face_keys`: set of face keys from interior crossing tets (used by `_compute_face_categories` for V-key debug)
- `_split_crossed_tets()` -- accepts `surface_tet_indices` parameter; tracks which tets are surface tets for tet-provenance phantom marking
- `_extract_boundary_faces()` -- robust opposite-vertex winding correction; returns faces appearing in exactly one tet
- `_filter_surface_faces()` -- two-stage phantom filter:
  - **PhantomCollar**: drops collar faces with any non-seam original vert outside `orig_surf_verts`
  - **[CollarFilter] reconstruction filter**: for any collar face (containing an inter vert in `[n_orig, n_split)` OR a seam-dup vert `>= n_split`), maps each new vert back to its endpoints (`dup_to_inter` for dups, `ivert_above`/`ivert_below` for inters), then drops the face if the 3-vert reconstruction is not in `orig_surf_set`. Catches non-conforming-split phantoms on both halves.
- `_compute_face_categories()` -- assigns each face to a category (cat1=valid collar, cat2=surface, cat3=phantom, cat4=tet-provenance phantom) for V-key color cycling
- `_compute_normals_post_cut()` -- per-frame normal computation on cut mesh; handles intersection vert lerp and seam dup vert snapping
- `_build_springs()` -- builds spring network from tet edges for spring-mass backend

### `pyGandalf/thesis_utilities/taichi_simulation_system.py`

**Core ECS file:**
- `_setup_progressive_cut()` -- one-time setup for progressive blade cut; appends collar tet face buffer after wound faces for T-key debug overlay
- `_advance_progressive_blade()` -- per-frame: advances blade cursor, breaks cutting springs, reveals wound faces
- `_perform_cut()` -- one-shot cut (C key); reuses `_filter_surface_faces` so the phantom filter applies
- `_realloc_gpu_buffers()` -- reallocates all GPU VBOs and EBO after topology change

**Controls (full list):**

| Key | Action |
|-----|--------|
| F | Poke -- downward impulse on top 5% of vertices |
| B | First press: initialize and start progressive blade cut. Subsequent presses: pause/resume blade |
| C | One-shot cut at the configured plane (disabled once B has been used) |
| P | Pause/resume physics simulation (blade still advances when paused) |
| N | Single-step one frame while paused |
| X | Disc parallelism check: yellow overlay on disc faces not parallel to the cut plane |
| O | Orphan overlay (FEM only): purple overlay on faces touching zero-mass orphan verts |
| V | Cycle face-category debug colors (requires --debug-colors shader) |
| Z | Toggle wireframe mode (backface culling disabled in wireframe) |
| T | Toggle collar tet overlay: shows all 4 faces of every tet containing an intersection vertex, colored yellow |
| WASD | Move camera |
| Right-click drag | Rotate camera |
| Q / E or Space / Shift | Camera up/down |

**Debug coloring (active with `--debug-colors` flag):**
- Requires `lit_blinn_phong_debug.vs/.fs` and 4th vertex attribute (per-face color, location 3)
- Unindexed (exploded) layout: each triangle owns 3 private vertices; trivial index buffer grows as blade reveals faces
- `_compute_debug_face_colors(faces, n_orig)`:
  - **Blue** -- all original verts (< n_orig): regular outer surface
  - **Green** -- mixed: collar face at cut boundary
  - **Red** -- all new verts (>= n_orig): wound/disc face
  - **Yellow** -- collar tet overlay faces (T key)
- V-key color cycling assigns distinct colors per face category for phantom face diagnosis

### `pyGandalf/resources/shaders/opengl/lit_blinn_phong_debug.vs` / `.fs`

**Added:**
- Vertex shader: `layout(location = 3) in vec3 a_Color` passed through as `v_Color`
- Fragment shader: uses `v_Color` per-vertex instead of `u_Color` material uniform

### `pyGandalf/thesis_utilities/tet_generator.py`

**Added:**
- `generate_tetrahedral_mesh(surface_mesh, target_faces=None)` -- TetGen wrapper
  - `_simplify_and_repair()`: progressive decimation fallback until watertight
  - `_repair_and_extract()`: trimesh repair + pymeshfix self-intersection fix
- Taichi GPU kernels: `compute_tet_volumes()`, `compute_tet_quality()`
- `analyze_tetrahedral_mesh()` -- mesh statistics

### `pyGandalf/thesis_utilities/sofa_simulation_system.py`

**Added (kept for reference, not actively used):**
- SOFA corotational FEM via `TetrahedralCorotationalFEMForceField`
- GPU path unavailable in installed SOFA version -- CPU too slow for real-time use

---

## New Thesis-Specific Files

### `pyGandalf/thesis_utilities/tet_exploder.py`
- `explode_tetrahedral_mesh(tet_mesh, explosion_factor, use_hybrid_normals)` -- exploded view with Taichi GPU kernel

### `pyGandalf/thesis_utilities/animated_tet_exploder.py`
- `AnimatedTetExplosion` -- animated explosion with sine wave motion

### `pyGandalf/thesis_utilities/animated_explosion_system.py`
- `AnimatedExplosionComponent` / `AnimatedExplosionSystem` -- ECS for animated explosion

### `pyGandalf/thesis_utilities/headless_sim.py`
- Physics-only runner, no OpenGL/ECS -- used for dataset generation and benchmarking
- Method-agnostic: works with SpringMassMethod or FEMMethod via the SimulationMethod interface

### `My_tests/generate_dataset.py`
- Batch runner: N randomised cuts, each saved as a .npz file with trajectory + graph data

### `My_tests/inspect_dataset.py`
- Validates a single .npz dataset file, prints shape and stats

---

## Test Files

### `My_tests/test_fem_cut.py` -- **ACTIVE FEM TEST**
- Taichi FEM simulation with progressive blade cut
- `--mesh` selects mesh (sphere, bunny, armadillo, dragon, liver)
- `--seed` controls cut plane RNG
- `--cg_iters` sets max CG iterations per step
- `--tet_scale` controls interior tet density (larger = fewer tets)
- `--debug-colors` enables per-face green/blue/red shader
- `hide_wound_faces=False`, `opening_speed=0.0` (no separation force -- FEM handles deformation)

### `My_tests/test_random_cut.py` -- **ACTIVE SPRING-MASS TEST**
- Taichi spring-mass simulation with randomised progressive blade cut
- Same mesh/seed/debug-colors arguments as test_fem_cut.py
- `hide_wound_faces=True`, `opening_speed=2.0`

### `My_tests/test_cutting_simulation.py`
- Earlier fixed-plane cut test, superseded by test_random_cut.py

### `My_tests/test_sofa_simulation.py`
- SOFA FEM simulation, kept for reference

### `My_tests/test_tet_rendering.py`
- Static surface rendering of a tetrahedralized mesh

### `My_tests/test_exploded_tets.py`
- Exploded tet view with hybrid normals

### `My_tests/test_taichi.py`
- TetGen pipeline + GPU quality analysis

---

## Architecture Notes

### Vertex index spaces (after a cut)
- `< n_orig` -- original mesh vertices
- `n_orig .. n_split-1` -- intersection vertices inserted at edge-plane crossings
- `>= n_split` -- seam duplicate vertices (below-half copies)
- `>= n_phys` (if `split_disc_verts=True`) -- rendering-only duplicates for disc/collar shading

### Face classification after cut
- **outer face**: at least 1 original vert (< n_orig); rendered from the start
- **wound face**: all 3 verts are on-plane (intersection or seam); revealed progressively as blade advances
- **phantom collar**: collar face where any original vert is not in `orig_surf_verts` -- interior vert exposed by non-conforming tet split; discarded by PhantomCollar filter

### Non-conforming split problem
When a 2+2 split tet and a 1+3 split tet share an interior face, they triangulate it differently. Sub-triangles that appear in only one split get count=1 in `_extract_boundary_faces` and surface as phantom faces. The PhantomCollar filter (tet-provenance based) discards these.

### Orphan verts
Verts with zero mass after sliver filtering. Sliver tets near the cut plane are excluded from FEM to avoid degenerate forces. Any vert that belongs only to excluded tets gets zero mass. The orphan constraint pins each such vert to its nearest non-zero-mass master with a fixed offset (`pos[orphan] = pos[master] + offset`), so they follow the mesh rigidly.

---

## Known Issues / Decisions

- **SOFA GPU (SofaCUDA):** `CudaTetrahedronFEMForceField` unavailable in the installed SOFA version. SOFA kept as reference only.
- **Explicit Euler stability (spring-mass):** `sub_dt` must stay below `dt_crit = 2*sqrt(m_vertex / k_eff)`. Current params give ~14x safety margin.
- **Trimesh simplification API:** Installed version uses `target_reduction` (0-1 float) not face count. Code handles both via try/except.
- **Phantom inside faces (Bug B):** 554 inward-facing collar faces still present after PhantomCollar filter. Under active investigation on branch `fix-phantom-inside-faces`.

---

## Running the Code

```bash
# FEM simulation (sphere, default seed)
python My_tests/test_fem_cut.py

# FEM simulation (bunny, debug colors)
python My_tests/test_fem_cut.py --mesh _bunny_jacobson --debug-colors

# Spring-mass simulation
python My_tests/test_random_cut.py --mesh sphere

# Dataset generation (headless, 100 runs)
python My_tests/generate_dataset.py --n_runs 100

# Validate a dataset file
python My_tests/inspect_dataset.py path/to/file.npz
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
