"""
Taichi Spring-Mass Simulation System

GPU-accelerated spring-mass simulator for tetrahedral meshes using explicit
Euler integration.

Controls:
  B - start / pause progressive blade cut
  F - poke the top of the mesh downward
  C - one-shot cut at the plane defined by cut_plane_origin/normal (kept for testing)
  P - pause / resume physics simulation (cut still runs)
  K - clear Taichi kernel_profiler stats (only meaningful with --profile-kernels)
  L - print Taichi kernel_profiler stats snapshot
  Ctrl+LMB - pick nearest rendered face, dump face/tet/vert info,
             highlight face+tet (needs --debug-colors for the color to show)
  Ctrl+RMB - clear pick highlight

Progressive cutting (virtual-node algorithm):
  When B is pressed the mesh topology is split once along the cut plane.
  Intersection vertices on the cut plane are duplicated (one copy per half).
  Cutting springs (stiffness = k/2, rest length = 0) connect each pair of
  duplicated vertices so the mesh looks intact initially.
  Each frame, the blade cursor advances along blade_travel_dir at blade_speed
  m/s.  Cutting springs whose seam vertex lies behind the cursor are broken
  (stiffness → 0) and the wound surface is progressively revealed.

Why per-spring stiffness:
  Breaking a cutting spring only requires setting its stiffness entry to 0 in
  the _sk numpy array.  No topology rebuild is needed per frame.  The Taichi
  spring-force kernel reads _sk via an ndarray parameter that is reuploaded
  to the GPU each substep — cheap for typical spring counts (< 200 k).

Why explicit Euler:
  The mesh starts at rest, so all springs are at their natural length and net
  force is zero.  Explicit Euler is perfectly stable at zero net force.
  dt_crit = 2*sqrt(m_vertex / k_effective).  With default params the safety
  margin is ~2-3x.
"""

import numpy as np
import sys
import time

# Gate the B-press cProfile + ti.profiler dump behind the same CLI flag
# used by mesh_lib.py / taichi_cut_utils.py to enable Taichi's kernel_profiler.
_PROFILE_KERNELS = "--profile-kernels" in sys.argv

# Wall-clock fps / memory reporting for the thesis benchmark numbers. Cheap
# enough to leave in, but it prints once a second, so it is opt-in. Sniffed
# from argv for the same reason as the flag above (ti.init runs before argparse).
_BENCH = "--bench" in sys.argv

import glfw
import taichi as ti
import OpenGL.GL as gl
from pyGandalf.core.input_manager import InputManager
from pyGandalf.systems.system import System
from pyGandalf.scene.components import (Component, StaticMeshComponent,
                                          CameraComponent, TransformComponent)
from pyGandalf.scene.scene_manager import SceneManager
from pyGandalf.core.application import Application

from . import pick_inspector
from . import taichi_cut_utils
from .bench_utils import FrameMeter, mem_report, record_event

from .taichi_cut_utils import (
    _SpringMassSimulator,
    _split_crossed_tets,
    _cut_topology_physics,
    _filter_surface_faces,
    _split_disc_verts_for_rendering,
    _build_springs,
    _print_spring_audit,
    _print_force_audit,
    _extract_boundary_faces,
    _compute_normals,
    _compute_normals_post_cut,
    _finalize_wound_slot_normals,
    _update_vbo,
    _compute_debug_face_colors,
    _compute_face_categories,
    _CAT_COLORS,
    _CAT_NAMES,
)


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

class TaichiSimulationComponent(Component):
    """
    Stores simulation parameters and runtime state for a tetrahedral entity.

    Args:
        tet_mesh:           TetrahedralMeshInstance to simulate.
        time_step:          Integration timestep (s).
        substeps:           Sub-steps per frame for stability.
        gravity:            Gravity vector (default zero).
        stiffness:          Spring stiffness (N/m).
        damping:            Velocity damping coefficient.
        total_mass:         Total object mass (kg) distributed uniformly.
        opening_speed:      Velocity (m/s) given to wound vertices on cut.
        poke_speed:         Velocity (m/s) applied to top vertices on F.
        blade_travel_dir:   Direction the blade moves within the cut plane.
        blade_speed:        Blade advance speed (m/s).
    """

    def __init__(self, tet_mesh,
                 time_step:          float = 0.001,
                 substeps:           int   = 10,
                 gravity:            list  = None,
                 stiffness:          float = 100.0,
                 damping:            float = 1.0,
                 total_mass:         float = 1.0,
                 opening_speed:      float = 1.0,
                 poke_speed:         float = 3.0,
                 v_max:              float = 1.5,
                 spring_damping:     float = 0.0,
                 blade_travel_dir:   list  = None,
                 blade_speed:        float = 0.5,
                 split_disc_verts:   bool  = True,
                 opening_ramp_frames: int  = 20,
                 method_instance             = None):
        super().__init__()
        self.tet_mesh      = tet_mesh
        self.time_step     = time_step
        self.substeps      = substeps
        self.gravity       = gravity if gravity is not None else [0.0, 0.0, 0.0]
        self.stiffness     = stiffness
        self.damping       = damping
        self.total_mass    = total_mass
        self.opening_speed = opening_speed
        self.poke_speed    = poke_speed
        self.v_max         = v_max
        self.spring_damping = spring_damping
        self.opening_ramp_frames = max(1, int(opening_ramp_frames))
        self._method_instance = method_instance  # optional pre-constructed SimulationMethod

        # Populated by TaichiSimulationSystem.on_create_entity
        self.method    = None          # SpringMassMethod — drives physics
        self.simulator:           _SpringMassSimulator = None
        self.surface_indices:     np.ndarray           = None
        self.current_tetrahedra:  np.ndarray           = None
        self.fixed_mask:          np.ndarray           = None
        self.poke_mask:           np.ndarray           = None

        # One-shot cut (C key) — kept for testing
        self.should_cut       = False
        self.cut_plane_origin = [0.0, 0.0, 0.0]
        self.cut_plane_normal = [0.0, 1.0, 0.0]

        # Progressive blade cut (B key)
        self.blade_travel_dir  = blade_travel_dir if blade_travel_dir is not None \
                                  else [1.0, 0.0, 0.0]
        self.blade_speed       = float(blade_speed)
        self.blade_is_active   = False
        self.blade_initialized = False
        self._cut_done         = False  # True once the blade finishes its sweep
        self.blade_travel      = 0.0  # current distance traveled along blade_travel_dir

        # Internal state populated by _setup_progressive_cut
        self._seam_pairs          = []   # [{v_above,v_below,spring_idx,travel_dist,broken}]
        self._outer_faces         = None # np.ndarray (K,3) uint32 — sphere outer surface
        self._wound_faces_by_dist = []   # [(travel_dist:float, face:np.ndarray(3,))]
        self._wound_face_ptr      = 0    # pointer into _wound_faces_by_dist

        # Collar tet debug visualization (C key)
        self._collar_tet_face_offset = 0   # start index in _all_render_faces
        self._n_collar_tet_faces     = 0   # number of collar tet faces appended
        self._show_collar_tets       = False

        # Physics pause (P key) — cut and blade still advance when paused
        self.sim_paused = False

        # Rendering options
        self.split_disc_verts = split_disc_verts  # duplicate rim verts for correct disc/collar normals
        self.use_culling      = False              # GL backface culling; off by default for cut debugging

        # Set after a cut — used for correct normal computation on the cut mesh.
        self._n_orig:            int        = None  # vertex count before splitting
        self._n_split:           int        = None  # n_orig + intersection vertex count
        self._inter_data:        list       = None  # [(new_id, vi_above, vj_below, t), ...]
        self._shared_list:       list       = None  # seam vertex indices (above-half copies)
        self._cut_normal:        np.ndarray = None  # normalized cut plane normal
        self._disc_split_phys_idx: np.ndarray = None  # physics indices of rendering disc duplicates

        # Unindexed (per-face) rendering — active after the first cut.
        # Each triangle gets its own 3 private vertices so face colors never blend.
        # comp.surface_indices keeps original vertex indices for normal computation.
        # The GPU index buffer is trivial ([0,1,2,3,...]) and only covers visible faces.
        self._face_expanded:    bool       = False
        self._all_render_faces: np.ndarray = None  # (N_all_faces, 3) outer+ALL wound, for expansion
        self._n_outer_faces:    int        = 0     # len(outer_faces) — constant after cut

        # Per-face debug colors (N_all_faces, 3) — stored so yellow overlay stays persistent.
        self._debug_colors: np.ndarray = None

        # Set True after a cut; consumed on the next simulated frame to print a force audit.
        self._pending_force_audit: bool = False

        # Opening velocity ramp queue — each entry applies a small velocity increment
        # per frame for opening_ramp_frames frames instead of one large impulse.
        # Format: {'above': np.ndarray, 'below': np.ndarray, 'step_speed': float, 'left': int}
        self._opening_ramp_queue: list = []


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class TaichiSimulationSystem(System):
    """
    Drives a Taichi spring-mass simulation and uploads deformed vertex
    positions to the GPU each frame.

    Must be registered BEFORE OpenGLStaticMeshRenderingSystem.
    """

    def on_create_entity(self, entity, components):
        comp: TaichiSimulationComponent
        mesh_comp: StaticMeshComponent
        comp, mesh_comp = components

        tet = comp.tet_mesh

        y          = tet.vertices[:, 1]
        threshold  = y.min() + (y.max() - y.min()) * 0.05
        fixed_mask = (y < threshold).astype(np.int32)

        comp.fixed_mask         = fixed_mask
        comp.current_tetrahedra = tet.tetrahedra.copy()

        poke_threshold = y.max() - (y.max() - y.min()) * 0.05
        comp.poke_mask = (y >= poke_threshold) & (fixed_mask == 0)
        comp.surface_indices = _extract_boundary_faces(tet.tetrahedra, tet.vertices)

        _sim_params = {
            'stiffness':           comp.stiffness,
            'damping':             comp.damping,
            'spring_damping':      comp.spring_damping,
            'total_mass':          comp.total_mass,
            'gravity':             comp.gravity,
            'v_max':               comp.v_max,
            'time_step':           comp.time_step,
            'substeps':            comp.substeps,
            'opening_speed':       comp.opening_speed,
            'blade_speed':         comp.blade_speed,
            'opening_ramp_frames': comp.opening_ramp_frames,
        }
        if comp._method_instance is not None:
            comp.method = comp._method_instance
        else:
            from pyGandalf.thesis_utilities.simulation_method import SpringMassMethod
            comp.method = SpringMassMethod()
        comp.method.initialize(comp.tet_mesh, _sim_params)
        comp.simulator = comp.method._simulator

        sub_dt = comp.time_step / comp.substeps
        print("[TaichiSimulationSystem] Initialized:")
        print(f"  Vertices:        {len(tet.vertices):,}")
        print(f"  Tetrahedra:      {len(tet.tetrahedra):,}")
        if hasattr(comp.simulator, '_sa'):
            print(f"  Springs:         {len(comp.simulator._sa):,}")
        print(f"  Surface faces:   {len(comp.surface_indices):,}")
        print(f"  Fixed vertices:  {int(fixed_mask.sum())}")
        print(f"  Sub-steps/frame: {comp.substeps}  (sub_dt = {sub_dt:.5f} s)")
        print("  F — poke top of mesh downward")
        print("  B — start / pause progressive blade cut")
        print("  C — one-shot cut (testing only)")
        print("  P — pause / resume physics simulation")
        print("  X — disc parallelism check (colors non-parallel disc faces yellow)")
        print("  V — cycle face-category colors (requires --debug-colors): "
              "white=surface, orange=collar, yellow=reclassified, cyan=wound-above, magenta=wound-dup")
        print("  Z — toggle wireframe mode")

        if _BENCH:
            import atexit
            self._meter = FrameMeter()
            self._meter.enable()
            atexit.register(self._meter.summary)
            print("  M — print host RSS / GPU memory snapshot")
            record_event('mesh',
                         verts=int(len(tet.vertices)),
                         tets=int(len(tet.tetrahedra)),
                         surface_faces=int(len(comp.surface_indices)),
                         fixed_verts=int(fixed_mask.sum()))
            mem_report('after init (uncut)', comp.simulator)

    def on_update_entity(self, ts: float, entity, components):
        comp: TaichiSimulationComponent
        mesh_comp: StaticMeshComponent
        comp, mesh_comp = components

        if comp.simulator is None:
            return
        if mesh_comp.render_pipeline is None or len(mesh_comp.buffers) < 2:
            return

        # --- Wall-clock frame meter (ts is the full frame delta from
        # Application.begin_frame, so it includes draw + buffer swap). Phase
        # labels keep the uncut baseline and the during-cut numbers separate.
        if _BENCH:
            if not comp.blade_initialized:
                _phase = 'idle'
            elif getattr(comp, '_cut_done', False):
                _phase = 'settling'
            elif comp.sim_paused:
                _phase = 'paused'
            else:
                _phase = 'cutting'
            self._meter.tick(ts, _phase)

            m_now = InputManager().get_key_down(glfw.KEY_M)
            if m_now and not getattr(self, '_m_prev', False):
                mem_report(f'on demand ({_phase})', comp.simulator)
            self._m_prev = m_now

        # --- C key: one-shot cut (blocked once progressive cut is initialized) ---
        c_now = InputManager().get_key_down(glfw.KEY_C)
        if c_now and not getattr(self, '_c_prev', False):
            if comp.blade_initialized:
                print("[Cut] Progressive cut already active — C-cut ignored.")
            else:
                comp.should_cut = True
        self._c_prev = c_now

        if comp.should_cut:
            comp.should_cut = False
            _perform_cut(comp, mesh_comp)

        # --- B key: progressive blade cut ---
        b_now = InputManager().get_key_down(glfw.KEY_B)
        if b_now and not getattr(self, '_b_prev', False):
            if not comp.blade_initialized:
                if _PROFILE_KERNELS:
                    # Python + Taichi profile of the one-shot setup, gated behind
                    # --profile-kernels so interactive runs aren't slowed down.
                    import cProfile, pstats, io
                    _prof = cProfile.Profile()
                    _t0 = time.perf_counter()
                    _prof.enable()
                    try:
                        _setup_progressive_cut(comp, mesh_comp)
                    finally:
                        _prof.disable()
                        _dt = time.perf_counter() - _t0
                        _s = io.StringIO()
                        pstats.Stats(_prof, stream=_s).sort_stats('cumulative').print_stats(30)
                        print("=" * 70, flush=True)
                        print(f"[PROFILE] _setup_progressive_cut wall time: {_dt:.3f}s", flush=True)
                        print("[PROFILE] top 30 by cumulative time:", flush=True)
                        print("=" * 70, flush=True)
                        print(_s.getvalue(), flush=True)
                        try:
                            ti.sync()
                            print("=" * 70, flush=True)
                            print("[PROFILE] Taichi kernel profile (cumulative since startup):", flush=True)
                            print("=" * 70, flush=True)
                            ti.profiler.print_kernel_profiler_info()
                            ti.profiler.clear_kernel_profiler_info()
                        except Exception as _e:
                            print(f"[TaichiProfile] unavailable: {_e}", flush=True)
                else:
                    # Unconditional wall time: this is the "one-shot cut setup"
                    # number for the thesis. Measured WITHOUT cProfile, which
                    # inflates it by roughly the interpreter overhead of every
                    # call in the setup path.
                    _t0 = time.perf_counter()
                    _setup_progressive_cut(comp, mesh_comp)
                    _setup_s = time.perf_counter() - _t0
                    print(f"[Bench] cut setup (B press -> ready): "
                          f"{_setup_s:.3f} s", flush=True)
                    record_event('cut_setup', seconds=_setup_s)
                    if _BENCH:
                        mem_report('after cut setup', comp.simulator)
            else:
                comp.blade_is_active = not comp.blade_is_active
                print(f"[Blade] {'Resumed' if comp.blade_is_active else 'Paused'}")
        self._b_prev = b_now

        # --- N key: step one frame while paused ---
        # Polled before the blade-advance and physics-step gates so a single N
        # press advances BOTH the cut cursor and one physics tick.
        n_now = InputManager().get_key_down(glfw.KEY_N)
        if n_now and not getattr(self, '_n_prev', False):
            if comp.sim_paused:
                comp._step_one_frame = True
        self._n_prev = n_now

        t_blade0 = time.perf_counter()
        # Blade advances when physics is running, or when the user requested a
        # single frame step (N) while paused.
        if comp.blade_is_active and (not comp.sim_paused
                                     or getattr(comp, '_step_one_frame', False)):
            _advance_progressive_blade(comp, mesh_comp, ts)
        t_blade1 = time.perf_counter()

        # --- F key: poke ---
        f_now = InputManager().get_key_down(glfw.KEY_F)
        if f_now and not getattr(self, '_f_prev', False):
            _apply_poke(comp)
        self._f_prev = f_now

        # --- P key: pause / resume physics ---
        p_now = InputManager().get_key_down(glfw.KEY_P)
        if p_now and not getattr(self, '_p_prev', False):
            comp.sim_paused = not comp.sim_paused
            print(f"[Sim] Physics {'paused' if comp.sim_paused else 'resumed'}")
        self._p_prev = p_now

        # --- X key: disc parallelism check (yellow non-parallel disc faces) ---
        x_now = InputManager().get_key_down(glfw.KEY_X)
        if x_now and not getattr(self, '_x_prev', False):
            _check_disc_parallelism(comp, mesh_comp)
        self._x_prev = x_now

        # --- O key: orphan vert overlay (purple faces touching orphaned verts, FEM only) ---
        o_now = InputManager().get_key_down(glfw.KEY_O)
        if o_now and not getattr(self, '_o_prev', False):
            _overlay_orphan_colors(comp, mesh_comp)
        self._o_prev = o_now

        # --- V key: cycle face-category colors (requires --debug-colors) ---
        v_now = InputManager().get_key_down(glfw.KEY_V)
        if v_now and not getattr(self, '_v_prev', False):
            _apply_color_cycle(comp, mesh_comp)
        self._v_prev = v_now

        # --- T key: toggle collar tet face overlay ---
        t_now = InputManager().get_key_down(glfw.KEY_T)
        if t_now and not getattr(self, '_t_prev', False):
            if comp._n_collar_tet_faces > 0:
                comp._show_collar_tets = not comp._show_collar_tets
                hide_wound = getattr(comp, 'hide_wound_faces', False)
                wound_vis  = 0 if hide_wound else comp._wound_face_ptr

                # Recolor outer faces that belong to collar tets.
                YELLOW = np.array([1.0, 0.85, 0.0], dtype=np.float32)
                face_colors = comp._debug_colors.copy()
                if comp._show_collar_tets and comp._collar_outer_mask is not None:
                    face_colors[:comp._n_outer_faces][comp._collar_outer_mask] = YELLOW
                # (off: _debug_colors already has original colors, no change needed)

                if len(mesh_comp.buffers) > 3:
                    exp_colors = np.repeat(face_colors, 3, axis=0).astype(np.float32)
                    gl.glBindVertexArray(mesh_comp.render_pipeline)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[3])
                    gl.glBufferData(gl.GL_ARRAY_BUFFER, exp_colors.nbytes, exp_colors,
                                    gl.GL_DYNAMIC_DRAW)
                    gl.glBindVertexArray(0)

                # Update index buffer: outer + collar tet appended faces (skip wound).
                if comp._show_collar_tets:
                    outer_idx  = np.arange(comp._n_outer_faces * 3, dtype=np.uint32)
                    c_start    = comp._collar_tet_face_offset * 3
                    collar_idx = np.arange(c_start, c_start + comp._n_collar_tet_faces * 3,
                                           dtype=np.uint32)
                    flat_idx   = np.concatenate([outer_idx, collar_idx])
                else:
                    flat_idx = np.arange((comp._n_outer_faces + wound_vis) * 3,
                                         dtype=np.uint32)
                gl.glBindVertexArray(mesh_comp.render_pipeline)
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh_comp.index_buffer)
                gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, flat_idx.nbytes, flat_idx,
                                gl.GL_DYNAMIC_DRAW)
                gl.glBindVertexArray(0)
                n_yellow = int(comp._collar_outer_mask.sum()) if comp._collar_outer_mask is not None else 0
                print(f"[CollarTets] {'ON' if comp._show_collar_tets else 'OFF'} — "
                      f"{n_yellow} outer faces + {comp._n_collar_tet_faces} tet faces yellow")
            else:
                print("[CollarTets] No collar tet faces — cut not done yet?")
        self._t_prev = t_now

        # --- Z key: toggle wireframe (surface edges only) ---
        z_now = InputManager().get_key_down(glfw.KEY_Z)
        if z_now and not getattr(self, '_z_prev', False):
            comp._wireframe = not getattr(comp, '_wireframe', False)
            print(f"[Wire] {'ON' if comp._wireframe else 'OFF'}")
        self._z_prev = z_now

        # --- K key: clear Taichi kernel_profiler stats (no-op if profiling off) ---
        k_now = InputManager().get_key_down(glfw.KEY_K)
        if k_now and not getattr(self, '_k_prev', False):
            ti.profiler.clear_kernel_profiler_info()
            print("[Profiler] kernel stats cleared")
        self._k_prev = k_now

        # --- L key: print Taichi kernel_profiler stats (no-op if profiling off) ---
        l_now = InputManager().get_key_down(glfw.KEY_L)
        if l_now and not getattr(self, '_l_prev', False):
            print("[Profiler] kernel stats snapshot:")
            ti.profiler.print_kernel_profiler_info()
        self._l_prev = l_now

        # --- Ctrl + Left click : pick face + parent tet, dump info ---
        # --- Ctrl + Right click: clear highlight ---
        ctrl_held  = InputManager().get_key_down(glfw.KEY_LEFT_CONTROL)
        lmb_now    = InputManager().get_key_down(glfw.MOUSE_BUTTON_1)
        rmb_now    = InputManager().get_key_down(glfw.MOUSE_BUTTON_2)
        lmb_press  = ctrl_held and lmb_now and not getattr(self, '_pick_lmb_prev', False)
        rmb_press  = ctrl_held and rmb_now and not getattr(self, '_pick_rmb_prev', False)
        if lmb_press:
            _run_pick(comp, mesh_comp)
        if rmb_press:
            pick_inspector.clear_highlight(comp, mesh_comp)
        self._pick_lmb_prev = lmb_now
        self._pick_rmb_prev = rmb_now
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK,
                         gl.GL_LINE if getattr(comp, '_wireframe', False) else gl.GL_FILL)
        if comp.use_culling:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

        # --- Simulation sub-steps (via SpringMassMethod — handles ramp internally) ---
        t0 = time.perf_counter()
        step_one = getattr(comp, '_step_one_frame', False)
        if step_one:
            comp._step_one_frame = False
        if not comp.sim_paused or step_one:
            # Time the first few step() calls after a cut to see first-touch JIT cost.
            _n = getattr(comp, '_step_timing_left', 0)
            if _n > 0:
                _t = time.perf_counter()
                comp.method.step(comp.time_step)
                ti.sync()
                _dt = time.perf_counter() - _t
                print(f"[STEP] post-cut step #{6 - _n}: {_dt*1000:.1f} ms", flush=True)
                comp._step_timing_left = _n - 1
            else:
                comp.method.step(comp.time_step)

            # Force audit: spring-mass only (uses _spring_forces kernel).
            if (comp._pending_force_audit and comp._n_orig is not None
                    and hasattr(comp.simulator, '_spring_forces')):
                comp._pending_force_audit = False
                comp.simulator._clear_forces()
                comp.simulator._spring_forces(
                    comp.simulator._sa, comp.simulator._sb,
                    comp.simulator._sr, comp.simulator._sk,
                    float(comp.spring_damping))
                _print_force_audit(
                    comp.simulator._forces.to_numpy(),
                    comp._n_orig,
                    comp.simulator.positions.shape[0])
            elif comp._pending_force_audit:
                comp._pending_force_audit = False
        t1 = time.perf_counter()

        # Route through the method API: FEM allocates Taichi fields at capacity,
        # so the raw `simulator.positions.to_numpy()` returns the full capacity
        # with zero-padded slots in [n_verts, capacity). get_positions() slices
        # to active size for both backends.
        new_positions = comp.method.get_positions()
        t2 = time.perf_counter()

        if comp._disc_split_phys_idx is not None and len(comp._disc_split_phys_idx) > 0:
            render_positions = np.vstack(
                [new_positions, new_positions[comp._disc_split_phys_idx]])
        else:
            render_positions = new_positions

        if comp._n_orig is not None:
            new_normals = _compute_normals_post_cut(
                render_positions, comp.surface_indices,
                comp._n_orig, comp._n_split,
                comp._inter_data, comp._shared_list,
                comp._cut_normal)
        else:
            new_normals = _compute_normals(render_positions, comp.surface_indices)
        t3 = time.perf_counter()

        # In unindexed mode every face owns its 3 vertices — expand before upload.
        if comp._face_expanded and comp._all_render_faces is not None:
            flat = comp._all_render_faces.flatten()
            upload_pos  = render_positions[flat].reshape(-1, 3)
            upload_norm = new_normals[flat].reshape(-1, 3)
            # Per-slot override: wound-face slots snap to +-cut_n regardless of
            # the per-vertex normal (fixes near-plane INTER shading anomalies
            # and sliver-tet winding flips via same-half majority vote).
            if comp._cut_normal is not None and comp._n_outer_faces is not None:
                n_outer = int(comp._n_outer_faces)
                # [outer | wound | collar_tet], stop at wound.
                n_wound_end = (int(comp._collar_tet_face_offset)
                                if comp._n_collar_tet_faces > 0
                                else len(comp._all_render_faces))
                n_wound = n_wound_end - n_outer
                _finalize_wound_slot_normals(
                    upload_norm, upload_pos, comp._all_render_faces,
                    n_wound_faces=n_wound, n_outer_faces=n_outer,
                    n_orig=comp._n_orig, n_split=comp._n_split,
                    n_phys=new_positions.shape[0],
                    disc_split_phys_idx=comp._disc_split_phys_idx,
                    cut_normal=comp._cut_normal)
        else:
            upload_pos  = render_positions
            upload_norm = new_normals

        _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[0], upload_pos)
        _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[1], upload_norm)
        t4 = time.perf_counter()

        if not hasattr(self, '_fc'):
            self._fc = 0
        self._fc += 1
        if self._fc % 10 == 0 and getattr(comp, 'blade_is_active', False):
            ti.sync()  # ensure all GPU work up to here is done, for honest timings
            t_end = time.perf_counter()
            blade_ms = (t_blade1 - t_blade0) * 1000
            print(
                f"[Frame {self._fc:4d}] "
                f"blade {blade_ms:5.1f}ms | "
                f"sim {(t1-t0)*1000:5.1f}ms | "
                f"readback {(t2-t1)*1000:4.1f}ms | "
                f"normals {(t3-t2)*1000:4.1f}ms | "
                f"vbo_upload {(t4-t3)*1000:4.1f}ms | "
                f"gpu_sync {(t_end-t4)*1000:4.1f}ms | "
                f"frame_total {(t_end-t_blade0)*1000:5.1f}ms",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Ctrl+click pick dispatch
# ---------------------------------------------------------------------------

def _run_pick(comp: TaichiSimulationComponent, mesh_comp: StaticMeshComponent):
    """Locate the primary camera + window and hand off to pick_inspector."""
    scene = SceneManager().active_scene
    if scene is None:
        print("[Pick] no active scene")
        return

    camera_comp = None
    camera_xf   = None
    arrays = scene.get_components_array()
    cameras   = arrays.get(CameraComponent,    []) or []
    transforms = arrays.get(TransformComponent, []) or []
    # Match camera to its transform via entity ownership.
    for entity in scene.get_entities():
        refs = scene.get_entity_component_references(entity)
        if CameraComponent in refs and TransformComponent in refs:
            cc = cameras[refs[CameraComponent]]
            if getattr(cc, 'primary', False):
                camera_comp = cc
                camera_xf   = transforms[refs[TransformComponent]]
                break
    if camera_comp is None:
        print("[Pick] no primary camera found")
        return

    win = Application().get_window()
    cursor = InputManager().get_mouse_cursor_pos()
    pick_inspector.pick_and_dump(
        comp, mesh_comp, camera_comp, camera_xf,
        cursor.x, cursor.y, int(win.width), int(win.height))


# ---------------------------------------------------------------------------
# Poke
# ---------------------------------------------------------------------------

def _apply_poke(comp: TaichiSimulationComponent):
    """Apply a downward impulse to the top 5% of vertices."""
    if comp.simulator is None:
        return
    # Active-size positions for thresholding (FEM fields are capacity-allocated).
    pos = comp.method.get_positions()
    n_active = pos.shape[0]
    y = pos[:, 1]
    poke_threshold = y.max() - (y.max() - y.min()) * 0.05
    fixed = comp.simulator._fixed.to_numpy()[:n_active]
    poke_mask = (y >= poke_threshold) & (fixed == 0)
    # Read full-capacity velocities, modify only the active slice via a view,
    # then write the full array back.
    vels = comp.simulator.velocities.to_numpy()
    vels[:n_active][poke_mask, 1] -= comp.poke_speed
    comp.simulator.velocities.from_numpy(vels.astype(np.float32))
    print(f"[Poke] Applied {comp.poke_speed} m/s downward to {int(poke_mask.sum())} verts")


# ---------------------------------------------------------------------------
# One-shot cut  (C key — kept for testing)
# ---------------------------------------------------------------------------

def _cut_topology(comp: TaichiSimulationComponent,
                  origin: np.ndarray,
                  normal: np.ndarray):
    """
    ECS wrapper for _cut_topology_physics.

    Reads arrays from comp, runs the pure computation, and writes the new
    simulator / tetrahedra / fixed_mask back to comp.
    Returns the same 11-tuple callers expect, or None on failure.
    """
    # Active-size views: FEM allocates Taichi fields at capacity, so we must
    # strip the zero-padded tail before handing arrays to topology code.
    n_active = getattr(comp.simulator, 'n_verts', comp.simulator.positions.shape[0])
    _blade_dir_np = np.array(comp.blade_travel_dir, dtype=np.float32)
    _blade_dir_np /= np.linalg.norm(_blade_dir_np)
    result = _cut_topology_physics(
        comp.current_tetrahedra,
        comp.simulator.positions.to_numpy()[:n_active],
        comp.simulator.velocities.to_numpy()[:n_active],
        comp.simulator._masses.to_numpy()[:n_active],
        comp.simulator._fixed.to_numpy()[:n_active],
        comp.stiffness,
        np.array(comp.gravity, dtype=np.float32),
        origin, normal,
        blade_dir=_blade_dir_np,
    )
    if result is None:
        return None

    (new_sim, final_pos, final_vel, final_mass, final_fixed,
     all_tets, n_orig, n_split, shared_list, remap, inter_data, orig_surf_set,
     phantom_above_face_keys, side_label, _all_tets_parent_unused) = result

    comp.simulator          = new_sim
    comp.current_tetrahedra = all_tets
    comp.fixed_mask         = final_fixed
    # Only sync method._simulator when types match — avoids replacing a
    # _FEMSimulator with the _SpringMassSimulator _cut_topology_physics returns.
    if comp.method is not None and type(comp.method._simulator) is type(new_sim):
        comp.method._simulator = new_sim

    return (final_pos, final_vel, final_mass, final_fixed,
            all_tets, n_orig, n_split, shared_list, remap, inter_data, orig_surf_set,
            phantom_above_face_keys, side_label)


def _perform_cut(comp: TaichiSimulationComponent,
                 mesh_comp: StaticMeshComponent):
    """
    One-shot surgical cut along cut_plane_origin / cut_plane_normal  (C key).
    Kept for testing and reference.
    """
    if comp.current_tetrahedra is None or len(comp.current_tetrahedra) == 0:
        print("[Cut] No tetrahedra.")
        return

    origin = np.array(comp.cut_plane_origin, dtype=np.float32)
    normal = np.array(comp.cut_plane_normal,  dtype=np.float32)
    normal /= np.linalg.norm(normal)

    result = _cut_topology(comp, origin, normal)
    if result is None:
        return

    (final_pos, final_vel, final_mass, final_fixed,
     all_tets, n_orig, n_split, shared_list, remap, inter_data,
     orig_surf_set, phantom_above_face_keys, _side_label) = result
    comp._n_orig      = n_orig
    comp._n_split     = n_split
    comp._inter_data  = inter_data
    comp._shared_list = shared_list
    comp._cut_normal  = normal

    # --- Opening velocity (ramped over opening_ramp_frames frames) ---
    if comp.opening_speed > 0.0:
        inter_indices = np.array([d[0] for d in inter_data], dtype=np.int32)
        dup_indices   = np.array([remap[v] for v in inter_indices
                                  if remap[v] >= n_split], dtype=np.int32)
        step_speed = comp.opening_speed / comp.opening_ramp_frames
        comp._opening_ramp_queue.append({
            'above': inter_indices,
            'below': dup_indices,
            'step_speed': step_speed,
            'left': comp.opening_ramp_frames,
        })
        print(f"[Cut] Opening ramp queued: {len(inter_indices)} + {len(dup_indices)} verts "
              f"over {comp.opening_ramp_frames} frames ({step_speed:.4f} m/s/frame)")
    comp._pending_force_audit = True

    # --- Surface ---
    raw_surface = _extract_boundary_faces(all_tets, final_pos)
    mesh_centroid = final_pos[:n_orig].mean(axis=0)
    outer_faces, wound_faces = _filter_surface_faces(
        raw_surface, final_pos, normal, n_orig, n_split, orig_surf_set, mesh_centroid,
        inter_data=inter_data, shared_list=shared_list)

    if comp.split_disc_verts:
        remapped_wound, split_phys = _split_disc_verts_for_rendering(
            outer_faces, wound_faces, len(final_pos), n_orig)
        comp._disc_split_phys_idx = (np.asarray(split_phys, dtype=np.int32)
                                     if len(split_phys) > 0 else None)
        render_pos = (np.vstack([final_pos, final_pos[split_phys]])
                      if len(split_phys) > 0 else final_pos)
        disc_faces = remapped_wound
    else:
        comp._disc_split_phys_idx = None
        render_pos = final_pos
        disc_faces = wound_faces

    all_faces = (np.vstack([outer_faces, np.array(disc_faces, dtype=np.uint32)])
                 if disc_faces else outer_faces)
    comp.surface_indices    = all_faces   # kept as original indices for normal computation
    comp._all_render_faces  = all_faces   # one-shot cut: all faces visible immediately
    comp._n_outer_faces     = len(outer_faces)
    comp._face_expanded     = True

    new_normals       = _compute_normals_post_cut(render_pos, all_faces,
                                                  n_orig, n_split, inter_data, shared_list, normal)
    face_colors       = _compute_debug_face_colors(all_faces, n_orig)
    comp._debug_colors = face_colors.copy()

    # Build expanded (unindexed) arrays — each face owns its 3 private vertices.
    flat              = all_faces.flatten()
    exp_pos           = render_pos[flat].reshape(-1, 3)
    exp_norm          = new_normals[flat].reshape(-1, 3)
    _finalize_wound_slot_normals(
        exp_norm, exp_pos, all_faces,
        n_wound_faces=len(all_faces) - len(outer_faces),
        n_outer_faces=len(outer_faces),
        n_orig=n_orig, n_split=n_split, n_phys=len(final_pos),
        disc_split_phys_idx=comp._disc_split_phys_idx,
        cut_normal=normal)
    exp_tex           = np.zeros((len(all_faces) * 3, 2), dtype=np.float32)
    exp_colors        = np.repeat(face_colors, 3, axis=0)
    trivial_idx       = np.arange(len(all_faces) * 3, dtype=np.uint32).reshape(-1, 3)

    print(f"[Cut] Surface: {len(raw_surface):,} raw → {len(all_faces):,} kept "
          f"({len(exp_pos):,} expanded verts)")

    _realloc_gpu_buffers(mesh_comp, exp_pos, exp_norm, exp_tex, trivial_idx,
                         colors=exp_colors)
    print("[Cut] GPU buffers updated.")


# ---------------------------------------------------------------------------
# Progressive blade cut  (B key)
# ---------------------------------------------------------------------------

def _fast_rebuild_progressive_render(comp: TaichiSimulationComponent,
                                      mesh_comp: StaticMeshComponent,
                                      normal: np.ndarray,
                                      blade_dir: np.ndarray,
                                      origin: np.ndarray):
    """
    Fast render rebuild path for progressive cutting.

    Uses the FEMMethod._precomp_render_cache built by
    _build_render_precompute_cache: a per-face table (over the union of all
    possible sub-tets + all unsplit crossing tets) with wound/outer/drop
    verdicts, categories, and debug colors precomputed from vertex indices.

    Per-frame work:
      - mask rows by current split state
      - packed-key unique+count for boundary detection
      - index precomputed classification arrays
      - winding correction on the outer subset
      - centroid sort for wound reveal
      - normals kernel + GL upload

    Does NOT compute the T-key collar-tet-face debug set (comp._show_collar_tets
    is always False during a live cut and cheap to rebuild lazily if needed).
    """
    _rr_t0 = time.perf_counter()

    method = comp.method
    cache = method._precomp_render_cache
    (_new_sim, final_pos, _final_vel, _final_mass, _final_fixed,
     _all_tets, n_orig, n_split, shared_list, remap, inter_data,
     orig_surf_set, phantom_above_face_keys, _side_label,
     _all_tets_parent) = method._topology_result

    face_verts             = cache['face_verts']
    face_key_idx           = cache['face_key_idx']
    n_unique_keys          = cache['n_unique_keys']
    face_fourth            = cache['face_fourth']
    face_parent            = cache['face_parent']
    face_parent_is_crossing = cache['face_parent_is_crossing']
    face_active_when_split = cache['face_active_when_split']
    face_verdict           = cache['face_verdict']
    split_mask    = method._split_mask
    crossing_mask = method._crossing_mask

    # -- Row activity mask over the full face table --
    # Slivers are physics-disabled (W=0) but stay in the render so their
    # neighboring faces don't get exposed as newly-boundary and produce
    # collar spikes. Their orphan verts are kinematically slaved so the
    # sliver's visible geometry moves rigidly with its master, avoiding
    # visual explosions while staying attached to the rest of the mesh.
    parent_split      = split_mask[face_parent]
    row_active_all    = (~face_parent_is_crossing) | (parent_split == face_active_when_split)

    # -- Boundary detection via bincount on precomputed unique-key indices --
    # A face is boundary iff exactly one active row references its unique key.
    # bincount over int32 indices is a single-pass C-level histogram, faster
    # than np.unique(sort + inverse + count) which was ~65ms/frame on bunny.
    active_key_idx = face_key_idx[row_active_all]
    key_count      = np.bincount(active_key_idx, minlength=n_unique_keys)
    is_boundary_key = (key_count == 1)
    boundary_row_mask = row_active_all & is_boundary_key[face_key_idx]

    outer_row_mask = boundary_row_mask & (face_verdict == 0)
    wound_row_mask = boundary_row_mask & (face_verdict == 1)
    outer_rows = np.where(outer_row_mask)[0]
    wound_rows = np.where(wound_row_mask)[0]

    outer_faces = face_verts[outer_rows].copy()  # (n_outer, 3) uint32
    wound_faces = face_verts[wound_rows]         # (n_wound, 3) uint32

    _rr_t_boundary = time.perf_counter() - _rr_t0
    _rr_t1 = time.perf_counter()

    # -- Winding correction on outer faces --
    if len(outer_faces) > 0:
        fourth_v = face_fourth[outer_rows]
        v0 = final_pos[outer_faces[:, 0]]
        v1 = final_pos[outer_faces[:, 1]]
        v2 = final_pos[outer_faces[:, 2]]
        fp = final_pos[fourth_v]
        face_n    = np.cross(v1 - v0, v2 - v0)
        to_fourth = fp - (v0 + v1 + v2) / 3.0
        inward    = np.einsum('ij,ij->i', face_n, to_fourth) > 0
        if inward.any():
            outer_faces[inward] = outer_faces[inward][:, [0, 2, 1]]

    _rr_t_wind = time.perf_counter() - _rr_t1
    _rr_t2 = time.perf_counter()

    # -- Sort wound faces by blade-travel centroid distance --
    if len(wound_faces) > 0:
        centroids  = final_pos[wound_faces].mean(axis=1)         # (n_w, 3)
        travel_ds  = (centroids - origin) @ blade_dir            # (n_w,)
        order      = np.argsort(travel_ds)
        wound_faces = wound_faces[order]
        sorted_ds  = travel_ds[order]
    else:
        sorted_ds = np.zeros(0, dtype=np.float32)

    # ECS state
    comp._n_orig      = n_orig
    comp._n_split     = n_split
    comp._inter_data  = inter_data
    comp._shared_list = shared_list
    comp._cut_normal  = normal
    comp._seam_pairs  = method._seam_pairs

    # Disc/collar split (vectorized).
    if comp.split_disc_verts:
        remapped_wound, split_phys = _split_disc_verts_for_rendering(
            outer_faces, wound_faces, len(final_pos), n_orig)
        comp._disc_split_phys_idx = (split_phys if len(split_phys) > 0 else None)
        render_pos = (np.vstack([final_pos, final_pos[split_phys]])
                      if len(split_phys) > 0 else final_pos)
        wound_for_render = remapped_wound
    else:
        comp._disc_split_phys_idx = None
        render_pos = final_pos
        wound_for_render = wound_faces

    # Progressive: reveal all wound immediately (parent tet is split before
    # its wound face enters the boundary). _wound_faces_by_dist is consumed
    # only by the D-key disc check -- build it via C-level zip over the
    # sorted arrays (no Python row loop).
    if len(wound_faces) > 0:
        comp._wound_faces_by_dist = list(zip(sorted_ds.tolist(),
                                             wound_for_render))
    else:
        comp._wound_faces_by_dist = []
    comp._wound_face_ptr = len(comp._wound_faces_by_dist)

    wound_arr = (np.asarray(wound_for_render, dtype=np.uint32)
                 if len(wound_faces) > 0 else np.zeros((0, 3), dtype=np.uint32))
    all_render_faces = (np.vstack([outer_faces, wound_arr])
                        if len(wound_arr) > 0 else outer_faces.copy())

    # T-key debug collar-tet-faces: skipped during a live cut. If the user
    # toggles it on with the current topology, the slow rebuild path still
    # handles the initial B-press population.
    comp._collar_tet_face_offset = len(all_render_faces)
    comp._n_collar_tet_faces     = 0
    comp._collar_outer_mask      = np.zeros(len(outer_faces), dtype=bool)
    comp._show_collar_tets       = False

    comp.surface_indices    = outer_faces.copy()
    comp._all_render_faces  = all_render_faces
    comp._outer_faces       = outer_faces
    comp._n_outer_faces     = len(outer_faces)
    comp._face_expanded     = True

    # Debug colors: index the precomputed per-row colors so the shader sees
    # the same blue/green/red material as the pre-progressive path. Wound
    # rows have to be picked up in the sorted order set above.
    face_debug_color_all = cache['face_debug_color']
    outer_colors = face_debug_color_all[outer_rows]
    if len(wound_rows) > 0:
        wound_colors = face_debug_color_all[wound_rows][order]
    else:
        wound_colors = np.zeros((0, 3), dtype=np.float32)
    face_colors = np.vstack([outer_colors, wound_colors])
    comp._debug_colors     = face_colors
    comp._face_categories  = None    # V-key still needs manual precompute
    comp._color_cycle_mode = -1
    n_all_faces = len(all_render_faces)

    _rr_t_class = time.perf_counter() - _rr_t2
    _rr_t3 = time.perf_counter()

    # Normals + expanded buffers.
    new_normals = _compute_normals_post_cut(render_pos, comp.surface_indices,
                                            n_orig, n_split, inter_data, shared_list, normal)

    flat_all   = all_render_faces.flatten()
    exp_pos    = render_pos[flat_all].reshape(-1, 3)
    exp_norm   = new_normals[flat_all].reshape(-1, 3)
    _finalize_wound_slot_normals(
        exp_norm, exp_pos, all_render_faces,
        n_wound_faces=len(wound_faces),
        n_outer_faces=len(outer_faces),
        n_orig=n_orig, n_split=n_split, n_phys=len(final_pos),
        disc_split_phys_idx=comp._disc_split_phys_idx,
        cut_normal=normal)
    exp_tex    = np.zeros((len(all_render_faces) * 3, 2), dtype=np.float32)
    exp_colors = np.repeat(face_colors, 3, axis=0)

    ptr = int(comp._wound_face_ptr)
    n_visible = len(outer_faces) + ptr
    trivial_idx = np.arange(n_visible * 3, dtype=np.uint32).reshape(-1, 3)

    _rr_t_norm = time.perf_counter() - _rr_t3
    _rr_t4 = time.perf_counter()

    _realloc_gpu_buffers(mesh_comp, exp_pos, exp_norm, exp_tex, trivial_idx,
                         colors=exp_colors)
    _rr_t_gl = time.perf_counter() - _rr_t4
    _rr_total = time.perf_counter() - _rr_t0

    print(f"[RenderTiming/fast] total={_rr_total*1000:.1f}ms  "
          f"boundary={_rr_t_boundary*1000:.1f}  "
          f"wind={_rr_t_wind*1000:.1f}  "
          f"class={_rr_t_class*1000:.1f}  "
          f"norm={_rr_t_norm*1000:.1f}  "
          f"gl_upload={_rr_t_gl*1000:.1f}  "
          f"n_active_rows={int(row_active_all.sum()):,}  "
          f"n_faces={n_all_faces:,}",
          flush=True)

    comp._rendered_topology_version = int(method._topology_version)


def _rebuild_progressive_render(comp: TaichiSimulationComponent,
                                 mesh_comp: StaticMeshComponent,
                                 is_initial: bool,
                                 normal: np.ndarray,
                                 blade_dir: np.ndarray,
                                 origin: np.ndarray):
    """
    Rebuild the ECS-side surface + GPU buffers from the current
    comp.method._topology_result.

    Called both from the initial B-press (`is_initial=True`) and from
    _advance_progressive_blade when the method has bumped its
    _topology_version mid-sweep. In the second case blade_travel and
    _wound_face_ptr are preserved by the caller.

    Rebuilds: outer_faces, wound_faces (sorted by blade travel), collar tet
    debug set, per-face debug colors, expanded VBO/index buffers. Records
    comp._topology_version = method._topology_version so the next frame's
    change-detection is accurate.
    """
    # Fast path: use precomputed face table if available (populated by
    # FEMMethod._ensure_full_precompute on the first _maybe_grow_split call).
    # This shortcuts the O(active_tets) boundary + filter + collar rebuild that
    # dominates the frame time on large meshes. Skipped for is_initial=True
    # because setup_cut runs before the first grow, so the cache does not exist
    # yet -- the initial B-press mesh is small (pristine, no INTERs/DUPs).
    if (not is_initial
        and getattr(comp.method, '_precomp_render_cache', None) is not None):
        _fast_rebuild_progressive_render(comp, mesh_comp, normal, blade_dir, origin)
        return

    (_new_sim, final_pos, _final_vel, _final_mass, _final_fixed,
     all_tets, n_orig, n_split, shared_list, remap, inter_data,
     orig_surf_set, phantom_above_face_keys, _side_label,
     _all_tets_parent) = comp.method._topology_result

    comp._n_orig      = n_orig
    comp._n_split     = n_split
    comp._inter_data  = inter_data
    comp._shared_list = shared_list
    comp._cut_normal  = normal

    # Share the seam_pairs list so _advance_progressive_blade can observe broken flags.
    comp._seam_pairs = comp.method._seam_pairs

    _rr_t0 = time.perf_counter()
    # --- Build progressive wound surface ---
    raw_surface = _extract_boundary_faces(all_tets, final_pos)
    _rr_t_boundary = time.perf_counter() - _rr_t0
    _rr_t1 = time.perf_counter()
    mesh_centroid = final_pos[:n_orig].mean(axis=0)
    outer_faces, wound_faces = _filter_surface_faces(
        raw_surface, final_pos, normal, n_orig, n_split, orig_surf_set, mesh_centroid,
        inter_data=inter_data, shared_list=shared_list)
    _rr_t_filter = time.perf_counter() - _rr_t1

    comp._outer_faces = outer_faces

    if comp.split_disc_verts:
        remapped_wound, split_phys = _split_disc_verts_for_rendering(
            outer_faces, wound_faces, len(final_pos), n_orig)
        comp._disc_split_phys_idx = (np.asarray(split_phys, dtype=np.int32)
                                     if len(split_phys) > 0 else None)
        render_pos = (np.vstack([final_pos, final_pos[split_phys]])
                      if len(split_phys) > 0 else final_pos)
        wound_for_render = remapped_wound
    else:
        comp._disc_split_phys_idx = None
        render_pos = final_pos
        wound_for_render = wound_faces

    # Sort wound faces by centroid distance along blade_dir.
    wound_by_dist = []
    for orig_wf, rend_wf in zip(wound_faces, wound_for_render):
        centroid    = final_pos[[int(orig_wf[0]), int(orig_wf[1]), int(orig_wf[2])]].mean(axis=0)
        travel_dist = float(np.dot(centroid - origin, blade_dir))
        wound_by_dist.append((travel_dist, rend_wf))
    wound_by_dist.sort(key=lambda x: x[0])
    comp._wound_faces_by_dist = wound_by_dist
    # Wound reveal:
    #   Legacy (one-shot) cut: all wound faces exist at B-press but are
    #     hidden until the blade cursor's travel_dist passes each centroid.
    #     Initial rebuild -> ptr=0; the reveal loop in _advance_progressive_blade
    #     grows the pointer as blade_travel advances.
    #   Progressive cut: a wound face exists only because its parent tet has
    #     ALREADY been split (mask growth caused the rebuild). Deferring its
    #     visibility by centroid produces see-through gaps behind the blade.
    #     Reveal all wound faces immediately.
    progressive = bool(getattr(comp.method, '_progressive_cut', False))
    if progressive:
        comp._wound_face_ptr = len(wound_by_dist)
    elif is_initial:
        comp._wound_face_ptr = 0
    else:
        travel_now = float(comp.blade_travel)
        comp._wound_face_ptr = sum(1 for td, _ in wound_by_dist if td <= travel_now)

    wound_arr = (np.array([f for _, f in wound_by_dist], dtype=np.uint32)
                 if wound_by_dist else np.zeros((0, 3), dtype=np.uint32))
    all_render_faces = (np.vstack([outer_faces, wound_arr])
                        if len(wound_arr) > 0 else outer_faces.copy())

    # --- Collar tet faces for debug visualization (T key) ---
    seam_arr = np.array(shared_list, dtype=np.int32) if len(shared_list) > 0 else np.array([], dtype=np.int32)
    is_inter = (all_tets >= n_orig) & (all_tets < n_split)
    is_seam  = np.isin(all_tets, seam_arr) if len(seam_arr) > 0 else np.zeros_like(is_inter)
    is_dup   = all_tets >= n_split
    orig_surf_verts_arr = np.array(sorted({v for tri in orig_surf_set
                                           for v in tri}), dtype=np.int32)
    has_cut_plane    = np.any(is_inter | is_seam | is_dup, axis=1)
    has_surf_orig    = np.any(np.isin(all_tets, orig_surf_verts_arr), axis=1)
    collar_tet_mask  = has_cut_plane & has_surf_orig
    collar_tets      = all_tets[collar_tet_mask]
    TET_FACE_TRIPLES = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    if len(collar_tets) > 0:
        collar_tet_faces = np.array(
            [[collar_tets[t, i], collar_tets[t, j], collar_tets[t, k]]
             for t in range(len(collar_tets)) for i, j, k in TET_FACE_TRIPLES],
            dtype=np.uint32)
    else:
        collar_tet_faces = np.zeros((0, 3), dtype=np.uint32)

    collar_face_key_set = {tuple(sorted(f.tolist())) for f in collar_tet_faces}
    collar_outer_mask = np.array(
        [tuple(sorted(f.tolist())) in collar_face_key_set for f in outer_faces],
        dtype=bool)

    collar_tet_face_offset = len(all_render_faces)
    if len(collar_tet_faces) > 0:
        all_render_faces = np.vstack([all_render_faces, collar_tet_faces])

    comp._collar_tet_face_offset = collar_tet_face_offset
    comp._n_collar_tet_faces     = len(collar_tet_faces)
    comp._collar_outer_mask      = collar_outer_mask
    comp._show_collar_tets       = False

    comp.surface_indices    = outer_faces.copy()
    comp._all_render_faces  = all_render_faces
    comp._n_outer_faces     = len(outer_faces)
    comp._face_expanded     = True

    face_colors = _compute_debug_face_colors(all_render_faces, n_orig)
    if len(collar_tet_faces) > 0:
        face_colors[collar_tet_face_offset:] = [1.0, 0.85, 0.0]
    comp._debug_colors = face_colors.copy()

    comp._face_categories  = _compute_face_categories(
        all_render_faces, len(outer_faces), n_orig, n_split,
        inter_data=inter_data, orig_surf_set=orig_surf_set,
        phantom_keys=phantom_above_face_keys)
    comp._color_cycle_mode = -1

    new_normals = _compute_normals_post_cut(render_pos, comp.surface_indices,
                                            n_orig, n_split, inter_data, shared_list, normal)

    flat_all   = all_render_faces.flatten()
    exp_pos    = render_pos[flat_all].reshape(-1, 3)
    exp_norm   = new_normals[flat_all].reshape(-1, 3)
    _finalize_wound_slot_normals(
        exp_norm, exp_pos, all_render_faces,
        n_wound_faces=len(wound_faces),
        n_outer_faces=len(outer_faces),
        n_orig=n_orig, n_split=n_split, n_phys=len(final_pos),
        disc_split_phys_idx=comp._disc_split_phys_idx,
        cut_normal=normal)
    exp_tex    = np.zeros((len(all_render_faces) * 3, 2), dtype=np.float32)
    exp_colors = np.repeat(face_colors, 3, axis=0)

    # Index buffer covers outer + already-revealed wound face slots.
    ptr = int(comp._wound_face_ptr)
    n_visible = len(outer_faces) + ptr
    trivial_idx = np.arange(n_visible * 3, dtype=np.uint32).reshape(-1, 3)

    if is_initial or not getattr(taichi_cut_utils, '_QUIET_CUT_LOGS', False):
        tag = "[Blade]" if is_initial else "[ProgressiveRender]"
        print(f"{tag} Surface: {len(outer_faces):,} outer + "
              f"{len(wound_faces):,} wound (revealed {ptr}) | "
              f"{len(all_render_faces):,} total expanded faces  "
              f"topo_ver={comp.method._topology_version}",
              flush=True)

    _rr_t2 = time.perf_counter()
    _realloc_gpu_buffers(mesh_comp, exp_pos, exp_norm, exp_tex, trivial_idx,
                         colors=exp_colors)
    _rr_t_gl = time.perf_counter() - _rr_t2
    _rr_total = time.perf_counter() - _rr_t0
    _rr_middle = _rr_total - _rr_t_boundary - _rr_t_filter - _rr_t_gl
    print(f"[RenderTiming] total={_rr_total*1000:.1f}ms  "
          f"boundary={_rr_t_boundary*1000:.1f}  "
          f"filter={_rr_t_filter*1000:.1f}  "
          f"collar+normals={_rr_middle*1000:.1f}  "
          f"gl_upload={_rr_t_gl*1000:.1f}  "
          f"n_tets={len(all_tets):,}  n_faces={len(all_render_faces):,}",
          flush=True)

    # Record the version we just rendered so change-detection is accurate.
    comp._rendered_topology_version = int(comp.method._topology_version)


def _setup_progressive_cut(comp: TaichiSimulationComponent,
                             mesh_comp: StaticMeshComponent):
    """
    One-time setup for progressive blade cutting.

    Splits the mesh topology along the cut plane (same tet-splitting logic as
    the one-shot cut), then adds cutting springs between every seam node pair.
    Cutting springs start at stiffness k/2 and rest length 0, so they hold
    the two halves together until the blade cursor passes each pair.

    The wound surface is sorted by blade travel distance so it can be revealed
    face-by-face as the blade advances.
    """
    if comp.current_tetrahedra is None or len(comp.current_tetrahedra) == 0:
        print("[Blade] No tetrahedra.")
        return

    origin = np.array(comp.cut_plane_origin, dtype=np.float32)
    normal = np.array(comp.cut_plane_normal,  dtype=np.float32)
    normal /= np.linalg.norm(normal)

    blade_dir = np.array(comp.blade_travel_dir, dtype=np.float32)
    blade_dir /= np.linalg.norm(blade_dir)

    # Delegate topology split, simulator rebuild, and cutting springs to the method.
    # Each backend (SpringMassMethod, FEMMethod) handles its own physics correctly.
    if not comp.method.setup_cut(normal, origin, blade_dir):
        print("[Blade] Cut plane does not intersect the mesh.")
        return

    # Sync ECS-level aliases — method holds the authoritative new simulator.
    comp.simulator          = comp.method._simulator
    comp.current_tetrahedra = comp.method._current_tets

    _rebuild_progressive_render(comp, mesh_comp, is_initial=True, normal=normal,
                                blade_dir=blade_dir, origin=origin)

    # Blade cursor starts just before the first seam pair (progressive mode
    # with an all-False initial mask has no pairs yet; fall back to the
    # schedule minimum so the blade cursor visibly moves toward the first
    # tet split).
    seam_pairs = comp._seam_pairs
    if seam_pairs:
        comp.blade_travel = seam_pairs[0]['travel_dist'] - 1e-3
    elif (getattr(comp.method, '_progressive_cut', False)
          and comp.method._split_schedule is not None
          and comp.method._crossing_mask is not None
          and comp.method._crossing_mask.any()):
        cross_sched = comp.method._split_schedule[comp.method._crossing_mask]
        comp.blade_travel = float(cross_sched.min()) - 1e-3
    else:
        comp.blade_travel = 0.0

    comp.blade_initialized  = True
    comp.blade_is_active    = True
    comp._step_timing_left  = 5  # time the first 5 step() calls to expose first-touch JIT

    # Warm-up step: trigger first-touch JIT of the FEM per-frame kernels while
    # the user is still inside the setup hitch. Without this the camera freezes
    # for ~1-5s on the first interactive frame after setup_cut returns.
    _t = time.perf_counter()
    comp.method.step(comp.time_step)
    ti.sync()
    print(f"[Blade] JIT warm-up step: {(time.perf_counter() - _t)*1000:.1f} ms", flush=True)

    print(f"[Blade] Setup complete — {len(seam_pairs):,} seam pairs, "
          f"blade starts at travel={comp.blade_travel:.4f}")
    print("[Blade] Press B to pause / resume.")


def _advance_progressive_blade(comp: TaichiSimulationComponent,
                                 mesh_comp: StaticMeshComponent,
                                 dt: float):
    """
    Advance the blade cursor one frame.

    1. Move cursor by blade_speed * dt along blade_travel_dir.
    2. Break all cutting springs whose seam vertex is now behind the cursor.
       Apply opening velocity to each newly separated pair.
    3. Reveal wound surface faces whose centroid is now behind the cursor.
       When new faces are added, reallocate the index buffer on the GPU.
    4. Stop the blade once all springs are broken.
    """
    normal    = np.array(comp.cut_plane_normal, dtype=np.float32)
    normal   /= np.linalg.norm(normal)

    comp.blade_travel += comp.blade_speed * float(dt)

    # --- Break springs behind the cursor ---
    # Check first_break before advance_blade() updates the broken flags.
    newly_to_break = [p for p in comp._seam_pairs
                      if not p['broken'] and p['travel_dist'] <= comp.blade_travel]
    if newly_to_break and not any(p['broken'] for p in comp._seam_pairs):
        comp._pending_force_audit = True
    _adv_t0 = time.perf_counter()
    comp.method.advance_blade(comp.blade_travel)
    _adv_dt = time.perf_counter() - _adv_t0
    # comp._seam_pairs is the same list as comp.method._seam_pairs — broken flags updated in-place.

    # Progressive backends may have grown the topology inside advance_blade
    # (FEMMethod._maybe_grow_split). Detect via the version counter and
    # rebuild the render-side surface + GPU buffers to match.
    method_ver = int(getattr(comp.method, '_topology_version', 0))
    rendered_ver = int(getattr(comp, '_rendered_topology_version', 0))
    _render_dt = 0.0
    if method_ver != rendered_ver:
        origin_np    = np.array(comp.cut_plane_origin, dtype=np.float32)
        blade_dir_np = np.array(comp.blade_travel_dir, dtype=np.float32)
        blade_dir_np /= np.linalg.norm(blade_dir_np)
        # Keep the alias current so downstream picks / debug read the latest tets.
        comp.simulator          = comp.method._simulator
        comp.current_tetrahedra = comp.method._current_tets
        _render_t0 = time.perf_counter()
        taichi_cut_utils._QUIET_CUT_LOGS = True
        try:
            _rebuild_progressive_render(comp, mesh_comp, is_initial=False,
                                        normal=normal, blade_dir=blade_dir_np,
                                        origin=origin_np)
        finally:
            taichi_cut_utils._QUIET_CUT_LOGS = False
        _render_dt = time.perf_counter() - _render_t0
        print(f"[AdvTiming] advance_blade={_adv_dt*1000:.1f}ms  "
              f"render_rebuild={_render_dt*1000:.1f}ms  "
              f"total={( _adv_dt + _render_dt)*1000:.1f}ms",
              flush=True)

    # --- Reveal wound faces behind the cursor ---
    ptr = comp._wound_face_ptr
    wbd = comp._wound_faces_by_dist
    while ptr < len(wbd) and wbd[ptr][0] <= comp.blade_travel:
        ptr += 1

    hide_wound = getattr(comp, 'hide_wound_faces', False)

    faces_added = (not hide_wound) and (ptr > comp._wound_face_ptr)
    if faces_added:
        comp._wound_face_ptr = ptr
        revealed = [f for _, f in wbd[:ptr]]
        comp.surface_indices = (                           # original indices for normals
            np.vstack([comp._outer_faces,
                       np.array(revealed, dtype=np.uint32)])
            if revealed else comp._outer_faces.copy()
        )
        mesh_comp.indices = comp.surface_indices

        # Trivial index buffer grows to cover outer + revealed wound face slots.
        n_vis    = comp._n_outer_faces + ptr
        flat_idx = np.arange(n_vis * 3, dtype=np.uint32)
        gl.glBindVertexArray(mesh_comp.render_pipeline)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh_comp.index_buffer)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, flat_idx.nbytes, flat_idx,
                        gl.GL_DYNAMIC_DRAW)
        gl.glBindVertexArray(0)

    # --- Stop when complete ---
    # Delegates to the method so progressive backends (FEM w/ progressive_cut)
    # can hold "done" open until the blade has traveled past all scheduled
    # tet splits, not just when the currently-existing seam_pairs are broken.
    if comp.method.is_cut_complete(comp.blade_travel):
        # Force-reveal any remaining wound faces (unless hidden for debug).
        if (not hide_wound) and comp._wound_face_ptr < len(wbd):
            comp._wound_face_ptr = len(wbd)
            revealed = [f for _, f in wbd]
            comp.surface_indices = (
                np.vstack([comp._outer_faces,
                           np.array(revealed, dtype=np.uint32)])
                if revealed else comp._outer_faces.copy()
            )
            mesh_comp.indices = comp.surface_indices
            n_vis    = comp._n_outer_faces + len(wbd)
            flat_idx = np.arange(n_vis * 3, dtype=np.uint32)
            gl.glBindVertexArray(mesh_comp.render_pipeline)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh_comp.index_buffer)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, flat_idx.nbytes, flat_idx,
                            gl.GL_DYNAMIC_DRAW)
            gl.glBindVertexArray(0)

        comp.blade_is_active = False
        comp._cut_done       = True
        n_broken = sum(1 for p in comp._seam_pairs if p['broken'])
        print(f"[Blade] Cut complete — {n_broken:,} seam springs broken.")
        record_event('cut_complete',
                     seam_springs_broken=int(n_broken),
                     verts_after_cut=int(comp.method.vertex_count))
        if _BENCH:
            mem_report('after cut complete', comp.simulator)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _check_disc_parallelism(comp: TaichiSimulationComponent,
                             mesh_comp: StaticMeshComponent):
    """
    X key — for each disc half (above and below), pick the face nearest to the
    group's centroid as a reference, then color yellow every disc face whose
    normal is not parallel to that reference (|dot| < 0.95).

    Uses the FULL wound-face set (all seam pairs, even if not yet revealed by
    the blade) so the check works at any point during or after cutting.
    Separates above/below halves by dotting each face's geometric normal with
    the stored cut_normal: above-half faces point in -cut_normal direction,
    below-half faces point in +cut_normal direction.
    """
    if comp._n_orig is None or not comp._wound_faces_by_dist:
        print("[DiscCheck] No cut data available — press B first.")
        return

    n_orig   = comp._n_orig
    cut_n    = comp._cut_normal

    # Current render positions (physics + rendering disc duplicates).
    # Use method API so FEM capacity padding is stripped.
    sim_pos = comp.method.get_positions()
    if comp._disc_split_phys_idx is not None and len(comp._disc_split_phys_idx) > 0:
        render_pos = np.vstack([sim_pos, sim_pos[comp._disc_split_phys_idx]])
    else:
        render_pos = sim_pos

    # Full disc face set (rendering indices, both halves, all faces regardless of blade progress)
    all_disc = np.array([f for _, f in comp._wound_faces_by_dist], dtype=np.uint32)
    if len(all_disc) == 0:
        print("[DiscCheck] No disc faces found.")
        return

    # Compute normalised face normals via cross product
    p0 = render_pos[all_disc[:, 0]]
    p1 = render_pos[all_disc[:, 1]]
    p2 = render_pos[all_disc[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(fn, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-10, 1.0, lengths)
    fn = fn / lengths  # (K, 3)

    # Separate halves: above (fn · cut_n < 0) vs below (fn · cut_n > 0)
    dots_cut   = fn @ cut_n
    above_mask = dots_cut < 0.0
    below_mask = ~above_mask

    PARALLEL_THRESHOLD = 0.95  # |dot| below this ≈ more than ~18° off-plane

    # Overlay yellow on non-parallel faces in the per-face color array.
    all_render_faces = comp._all_render_faces
    if comp._debug_colors is not None and all_render_faces is not None:
        face_colors = comp._debug_colors.copy()   # (N_all_faces, 3)
    else:
        all_faces   = (np.vstack([comp._outer_faces, all_disc])
                       if comp._outer_faces is not None and len(comp._outer_faces) > 0
                       else all_disc)
        face_colors = _compute_debug_face_colors(all_faces, n_orig)

    YELLOW = np.array([0.95, 0.85, 0.1], dtype=np.float32)
    # Map each non-parallel disc face back to its index in _all_render_faces.
    # all_disc comes from _wound_faces_by_dist; wound faces start at _n_outer_faces in all_render_faces.
    n_outer   = comp._n_outer_faces
    disc_face_indices_in_all = np.where(
        np.all(all_render_faces >= n_orig, axis=1)
    )[0]   # indices of disc faces within all_render_faces

    for label, gmask in [("above", above_mask), ("below", below_mask)]:
        gfaces   = all_disc[gmask]
        gnormals = fn[gmask]
        if len(gfaces) < 2:
            print(f"[DiscCheck] {label}-half: only {len(gfaces)} face(s) — skipping.")
            continue
        centroids = render_pos[gfaces].mean(axis=1)
        group_cen = centroids.mean(axis=0)
        ref_idx   = int(np.argmin(np.linalg.norm(centroids - group_cen, axis=1)))
        ref_n     = gnormals[ref_idx]
        abs_dots  = np.abs(gnormals @ ref_n)
        bad       = abs_dots < PARALLEL_THRESHOLD
        n_bad     = int(bad.sum())
        print(f"[DiscCheck] {label}-half: {len(gfaces)} faces, "
              f"ref centroid {centroids[ref_idx].round(3)}, "
              f"{n_bad} non-parallel (|dot|<{PARALLEL_THRESHOLD})")

        # Find global face indices in all_render_faces for the bad faces in this half.
        half_disc_global = disc_face_indices_in_all[gmask]
        for fi in np.where(bad)[0]:
            face_colors[half_disc_global[fi]] = YELLOW

    n_yellow = int(np.all(face_colors == YELLOW, axis=1).sum())

    if len(mesh_comp.buffers) > 3 and all_render_faces is not None:
        exp_colors = np.repeat(face_colors, 3, axis=0).astype(np.float32)
        flat_col   = exp_colors.flatten()
        gl.glBindVertexArray(mesh_comp.render_pipeline)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[3])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_col.nbytes, flat_col, gl.GL_DYNAMIC_DRAW)
        gl.glBindVertexArray(0)

    if n_yellow > 0:
        print(f"[DiscCheck] {n_yellow} non-parallel disc faces colored yellow.")
    else:
        print("[DiscCheck] All disc faces are parallel to their reference — no artifacts detected.")


def _overlay_orphan_colors(comp: TaichiSimulationComponent,
                            mesh_comp: StaticMeshComponent) -> None:
    """
    O key: color purple every surface face that touches an orphaned vert.
    Orphan verts are new-cut verts whose tets were all filtered out (FEM only).
    Use this to confirm whether the visual artifact region matches the orphan region.
    """
    if not hasattr(comp.method, '_debug_orphan_verts') or not comp.method._debug_orphan_verts:
        print("[Orphan] No orphan verts recorded (not an FEM cut, or no cut yet).")
        return

    all_render_faces = comp._all_render_faces
    if all_render_faces is None or comp._debug_colors is None:
        print("[Orphan] No face data yet — cut has not run.")
        return

    ov_arr      = np.array(sorted(comp.method._debug_orphan_verts), dtype=np.int32)
    face_colors = comp._debug_colors.copy()
    PURPLE      = np.array([0.6, 0.0, 0.8], dtype=np.float32)
    hits        = np.isin(all_render_faces, ov_arr).any(axis=1)
    face_colors[hits] = PURPLE
    comp._debug_colors = face_colors.copy()

    n_purple = int(hits.sum())
    print(f"[Orphan] Purple overlay: {n_purple} faces touch an orphan vert "
          f"({len(comp.method._debug_orphan_verts)} orphans total).", flush=True)

    if len(mesh_comp.buffers) > 3:
        exp_colors = np.repeat(face_colors, 3, axis=0).astype(np.float32)
        flat_col   = exp_colors.flatten()
        gl.glBindVertexArray(mesh_comp.render_pipeline)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[3])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_col.nbytes, flat_col, gl.GL_DYNAMIC_DRAW)
        gl.glBindVertexArray(0)


def _realloc_gpu_buffers(mesh_comp: StaticMeshComponent,
                          positions:  np.ndarray,
                          normals:    np.ndarray,
                          texcoords:  np.ndarray,
                          indices:    np.ndarray,
                          colors:     np.ndarray = None):
    """Reallocate all GPU buffers with new data (called after topology changes)."""
    flat_pos  = positions.flatten().astype(np.float32)
    flat_norm = normals.flatten().astype(np.float32)
    flat_tex  = texcoords.flatten().astype(np.float32)
    flat_idx  = indices.flatten().astype(np.uint32)

    gl.glBindVertexArray(mesh_comp.render_pipeline)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[0])
    gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_pos.nbytes,  flat_pos,  gl.GL_DYNAMIC_DRAW)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[1])
    gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_norm.nbytes, flat_norm, gl.GL_DYNAMIC_DRAW)

    if len(mesh_comp.buffers) > 2:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[2])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_tex.nbytes, flat_tex, gl.GL_DYNAMIC_DRAW)

    if colors is not None and len(mesh_comp.buffers) > 3:
        flat_col = colors.flatten().astype(np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh_comp.buffers[3])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, flat_col.nbytes, flat_col, gl.GL_DYNAMIC_DRAW)

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh_comp.index_buffer)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, flat_idx.nbytes, flat_idx,
                    gl.GL_DYNAMIC_DRAW)

    gl.glBindVertexArray(0)
    mesh_comp.indices = indices


# ---------------------------------------------------------------------------
# Face category color cycling  (V key, requires --debug-colors shader)
# ---------------------------------------------------------------------------

def _apply_color_cycle(comp, mesh_comp: 'StaticMeshComponent'):
    """
    Cycle to the next face-category highlight.

    Mode 0 resets to the original debug colors.
    Modes 1-N each highlight ONE category with a bright color while all other
    faces keep their original debug colors (not dimmed).
    """
    if not comp._face_expanded or comp._face_categories is None:
        print("[ColorCycle] No face categories yet — press B first.")
        return
    if len(mesh_comp.buffers) < 4:
        print("[ColorCycle] No color buffer — run with --debug-colors.")
        return

    n_modes = len(_CAT_COLORS) + 1          # 0=reset, 1..N=spotlight one cat
    mode    = (getattr(comp, '_color_cycle_mode', -1) + 1) % n_modes
    comp._color_cycle_mode = mode

    cats     = comp._face_categories         # (N_faces,) int32
    baseline = comp._debug_colors.copy()     # original per-face colors, (N_faces, 3)

    if mode == 0:
        colors = baseline
        label  = "reset to original"
    else:
        c      = mode - 1
        colors = baseline.copy()
        colors[cats == c] = _CAT_COLORS[c]  # override only this category
        label  = _CAT_NAMES[c]

    counts = [int((cats == c).sum()) for c in range(len(_CAT_COLORS))]
    print(f"[ColorCycle] mode {mode} — {label}")
    print(f"  face counts: " +
          ", ".join(f"cat{c}={counts[c]}" for c in range(len(_CAT_COLORS))))

    exp = np.repeat(colors, 3, axis=0).astype(np.float32)
    _update_vbo(mesh_comp.render_pipeline, mesh_comp.buffers[3], exp)
