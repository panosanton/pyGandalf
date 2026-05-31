"""
Headless simulation runner for GNN trajectory recording.

No OpenGL, no ECS, no window. Accepts any SimulationMethod backend.

Typical usage:
    method = SpringMassMethod()
    method.initialize(tet_mesh, params)

    sim  = HeadlessSim(method, cut_normal, cut_origin, blade_travel_dir)
    data = sim.run_and_record()
    # data['positions']  (T, N, 3)
    # data['velocities'] (T, N, 3)
    # + whatever keys method.get_graph_data() returns

To use a different backend (e.g. FEMMethod), replace the first two lines.
The rest of the code is unchanged.
"""

import numpy as np

from pyGandalf.thesis_utilities.simulation_method import SimulationMethod


class HeadlessSim:
    """
    Thin driver around any SimulationMethod for trajectory recording.

    The method must be fully initialized before being passed here
    (i.e. method.initialize(tet_mesh, params) already called).

    Recording requires the method to implement get_graph_data().
    If it does not, run_and_record() raises NotImplementedError at
    the point where graph data is requested.
    """

    def __init__(self,
                 method:          SimulationMethod,
                 cut_normal,
                 cut_origin,
                 blade_travel_dir):
        """
        Parameters
        ----------
        method           : initialized SimulationMethod instance
        cut_normal       : (3,) array-like -- cut plane normal
        cut_origin       : (3,) array-like -- point on the cut plane
        blade_travel_dir : (3,) array-like -- direction blade moves in the plane
        """
        cut_normal       = np.array(cut_normal,       dtype=np.float32)
        cut_origin       = np.array(cut_origin,       dtype=np.float32)
        blade_travel_dir = np.array(blade_travel_dir, dtype=np.float32)
        cut_normal       /= np.linalg.norm(cut_normal)
        blade_travel_dir /= np.linalg.norm(blade_travel_dir)

        self._method       = method
        self._cut_normal   = cut_normal
        self._cut_origin   = cut_origin
        self._blade_dir    = blade_travel_dir
        self._blade_travel = 0.0

    def _setup_cut(self) -> bool:
        ok = self._method.setup_cut(self._cut_normal, self._cut_origin, self._blade_dir)
        if ok:
            self._blade_travel = self._method.initial_blade_travel
        return ok

    def run_and_record(self,
                       settle_steps: int = 200,
                       record_every: int = 1) -> dict | None:
        """
        Run cut -> blade advance -> settling and return trajectory data.

        Parameters
        ----------
        settle_steps : physics frames to run after the blade finishes
        record_every : record state every N frames (1 = every frame)

        Returns
        -------
        dict merging per-frame trajectory arrays with the static graph
        data from method.get_graph_data(), or None if the cut misses
        the mesh.

        Raises
        ------
        NotImplementedError if the method does not implement get_graph_data().
        """
        if not self._setup_cut():
            print("[HeadlessSim] Cut missed mesh -- skipping.")
            return None

        method      = self._method
        time_step   = method.time_step
        blade_speed = method.blade_speed

        static = method.get_graph_data()

        positions_list  = []
        velocities_list = []
        blade_prog_list = []

        def _record():
            positions_list.append(method.get_positions().copy())
            velocities_list.append(method.get_velocities().copy())
            blade_prog_list.append(float(self._blade_travel))

        # Phase 1: blade advancing
        frame      = 0
        blade_done = False
        while not blade_done:
            self._blade_travel += blade_speed * time_step
            blade_done          = method.advance_blade(self._blade_travel)
            method.step(time_step)
            if frame % record_every == 0:
                _record()
            frame += 1

        # Phase 2: settling
        for i in range(settle_steps):
            method.step(time_step)
            if i % record_every == 0:
                _record()

        return {
            'cut_normal':     self._cut_normal,
            'cut_origin':     self._cut_origin,
            'n_orig':         np.int32(method.n_orig),
            'n_phys':         np.int32(method.vertex_count),
            'positions':      np.stack(positions_list,  axis=0),
            'velocities':     np.stack(velocities_list, axis=0),
            'blade_progress': np.array(blade_prog_list, dtype=np.float32),
            **static,
        }
