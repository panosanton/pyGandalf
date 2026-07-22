"""
Export a tetrahedralized mesh as legacy VTK for use in SOFA.

Runs the same MeshLib.build_tetrahedral pipeline as the FEM tests, then
writes an ASCII legacy .vtk (unstructured grid, cell type 10 = tetra) that
SOFA's MeshVTKLoader reads directly.

Usage:
    python export_tet_vtk.py --mesh _bunny_jacobson --out bunny_tets.vtk
"""

import argparse
from pathlib import Path

import numpy as np

from pyGandalf.utilities.mesh_lib import MeshLib
from pyGandalf.utilities.definitions import MODELS_PATH

# Same verified configs as test_fem_cut.py
MESH_CONFIG = {
    'sphere':                   None,
    'Armadillo_verysimplified': None,
    'dragon_clean':             1500,
    'liver-smooth':             3500,
    'Armadillo_simplified':     None,
    '_bunny_jacobson':          15000,
}


def write_legacy_vtk(path: Path, vertices: np.ndarray, tets: np.ndarray) -> None:
    n_v = len(vertices)
    n_t = len(tets)
    with open(path, 'w', newline='\n') as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("tetrahedral mesh exported from pyGandalf pipeline\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {n_v} float\n")
        for p in vertices:
            f.write(f"{p[0]:.7g} {p[1]:.7g} {p[2]:.7g}\n")
        f.write(f"CELLS {n_t} {n_t * 5}\n")
        for t in tets:
            f.write(f"4 {t[0]} {t[1]} {t[2]} {t[3]}\n")
        f.write(f"CELL_TYPES {n_t}\n")
        f.write("10\n" * n_t)


def main():
    parser = argparse.ArgumentParser(description='Export tet mesh to legacy VTK')
    parser.add_argument('--mesh', default='_bunny_jacobson',
                        choices=list(MESH_CONFIG.keys()))
    parser.add_argument('--tet_scale', type=float, default=1.0)
    parser.add_argument('--out', default=None,
                        help='Output path (default: <mesh>_tets.vtk next to this script)')
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path(__file__).parent / f"{args.mesh}_tets.vtk"

    print(f"Tetrahedralizing {args.mesh}.obj ...")
    tet_mesh = MeshLib().build_tetrahedral(
        f'{args.mesh}_tet',
        MODELS_PATH / f'{args.mesh}.obj',
        target_faces=MESH_CONFIG[args.mesh],
        tet_scale=args.tet_scale,
    )
    verts = np.asarray(tet_mesh.vertices, dtype=np.float32)
    tets  = np.asarray(tet_mesh.tetrahedra, dtype=np.int64)
    print(f"  {len(verts):,} vertices, {len(tets):,} tetrahedra")

    write_legacy_vtk(out, verts, tets)
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
