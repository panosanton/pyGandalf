# Taichi Example: Computing Tetrahedral Volumes

import numpy as np
import taichi as ti

# Initialize Taichi (choose backend: cpu, cuda, vulkan, metal)
ti.init(arch=ti.gpu)  # or ti.cpu if no GPU

# ============================================
# REGULAR PYTHON/NUMPY
# ============================================
def compute_tet_volumes_numpy(vertices, tetrahedra):
    """Compute volume of each tetrahedron using NumPy"""
    volumes = np.zeros(len(tetrahedra))

    for i, tet in enumerate(tetrahedra):
        # Get 4 vertices of tetrahedron
        v0 = vertices[tet[0]]
        v1 = vertices[tet[1]]
        v2 = vertices[tet[2]]
        v3 = vertices[tet[3]]

        # Compute volume: |det(v1-v0, v2-v0, v3-v0)| / 6
        matrix = np.array([v1 - v0, v2 - v0, v3 - v0])
        volumes[i] = abs(np.linalg.det(matrix)) / 6.0

    return volumes


# ============================================
# TAICHI (GPU ACCELERATED)
# ============================================
@ti.kernel
def compute_tet_volumes_taichi(
    vertices: ti.types.ndarray(),
    tetrahedra: ti.types.ndarray(),
    volumes: ti.types.ndarray()
):
    """Compute volume of each tetrahedron using Taichi (runs on GPU)"""

    # This loop is pararellized on GPU cores using taichi
    for i in range(tetrahedra.shape[0]):
        # Get indices
        i0 = tetrahedra[i, 0]
        i1 = tetrahedra[i, 1]
        i2 = tetrahedra[i, 2]
        i3 = tetrahedra[i, 3]

        # Get vertices
        v0 = ti.Vector([vertices[i0, 0], vertices[i0, 1], vertices[i0, 2]])
        v1 = ti.Vector([vertices[i1, 0], vertices[i1, 1], vertices[i1, 2]])
        v2 = ti.Vector([vertices[i2, 0], vertices[i2, 1], vertices[i2, 2]])
        v3 = ti.Vector([vertices[i3, 0], vertices[i3, 1], vertices[i3, 2]])

        # Compute edges
        e1 = v1 - v0
        e2 = v2 - v0
        e3 = v3 - v0

        # Volume = |e1 · (e2 × e3)| / 6
        cross = ti.Vector([
            e2[1] * e3[2] - e2[2] * e3[1],
            e2[2] * e3[0] - e2[0] * e3[2],
            e2[0] * e3[1] - e2[1] * e3[0]
        ])
        det = e1[0] * cross[0] + e1[1] * cross[1] + e1[2] * cross[2]
        volumes[i] = abs(det) / 6.0


# ============================================
# BENCHMARK
# ============================================
if __name__ == "__main__":
    import time

    # Create test data
    num_vertices = 10000
    num_tets = 50000

    vertices = np.random.rand(num_vertices, 3).astype(np.float32)
    tetrahedra = np.random.randint(0, num_vertices, (num_tets, 4), dtype=np.int32)

    # NumPy version
    start = time.time()
    volumes_numpy = compute_tet_volumes_numpy(vertices, tetrahedra)
    numpy_time = time.time() - start
    print(f"NumPy time: {numpy_time:.4f} seconds")

    # Taichi version
    volumes_taichi = np.zeros(num_tets, dtype=np.float32)

    # First call compiles the kernel (will be slower)
    compute_tet_volumes_taichi(vertices, tetrahedra, volumes_taichi)

    # Second call is the real benchmark
    start = time.time()
    compute_tet_volumes_taichi(vertices, tetrahedra, volumes_taichi)
    taichi_time = time.time() - start
    print(f"Taichi time: {taichi_time:.4f} seconds")

    print(f"\nSpeedup: {numpy_time / taichi_time:.1f}x faster")
    print(f"Results match: {np.allclose(volumes_numpy, volumes_taichi, atol=1e-5)}")
