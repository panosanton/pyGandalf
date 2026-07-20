"""
Taichi corotational FEM simulator.

GPU-accelerated implicit FEM for tetrahedral meshes.

Elasticity model
----------------
Corotational neo-Hookean (mu term only, linearized stiffness for implicit solve):
    F  = Ds @ B          -- deformation gradient
    R  = polar decomp    -- rotation from F = R @ S
    P  = 2*mu*(F - R)    -- first Piola-Kirchhoff stress

Integration
-----------
Implicit Euler via matrix-free preconditioned conjugate gradient (PCG):
    (M - dt^2 * K) v_new = M*v_old + dt*f
This is unconditionally stable regardless of dt or stiffness.

Preconditioner
--------------
Jacobi (diagonal) preconditioning using the exact diagonal of A = M - dt^2*K.
The K diagonal is component-independent by isotropy and precomputed at init.
Reduces CG iteration count by 2-5x compared to unpreconditioned CG.

Cutting springs
---------------
Compatible with the spring-based cutting interface in taichi_simulation_system.py.
The _sa/_sb/_sr/_sk arrays hold cutting springs applied as explicit spring forces
on top of the implicit FEM step.  Zeroing _sk[i] breaks a cutting spring without
any topology rebuild.

Capacity-based allocation
-------------------------
Taichi fields are allocated with shape = capacity (>= initial N/M), not the
active mesh size. Kernels take n_v/n_t i32 args and iterate range(n) so unused
slots are never touched. rebuild_topology() re-uploads new mesh data into the
SAME field objects, keeping SNode IDs stable so Taichi's offline cache hits on
subsequent step() calls -- avoiding the ~4s JIT recompile per cut.
"""
import taichi as ti
import numpy as np


@ti.data_oriented
class _FEMSimulator:
    """
    Implicit corotational FEM on a tetrahedral mesh.

    Interface is intentionally close to _SpringMassSimulator so the ECS system
    can treat both interchangeably via comp.simulator.
    """

    def __init__(self,
                 vertices:        np.ndarray,
                 tetrahedra:      np.ndarray,
                 fixed_mask:      np.ndarray,
                 young_modulus:   float,
                 poisson_ratio:   float,
                 density:         float,
                 gravity:         np.ndarray,
                 damping:         float = 0.0,
                 capacity_verts:  int   = None,
                 capacity_tets:   int   = None,
                 capacity_factor: int   = 4):
        N = len(vertices)
        M = len(tetrahedra)

        if capacity_verts is None:
            capacity_verts = max(N * capacity_factor, N + 1)
        if capacity_tets is None:
            capacity_tets = max(M * capacity_factor * 2, M + 1)

        self._capacity_verts = int(capacity_verts)
        self._capacity_tets  = int(capacity_tets)
        self._n_verts        = int(N)
        self._n_tets         = int(M)

        self._mu      = float(young_modulus / (2.0 * (1.0 + poisson_ratio)))
        self._la      = float(young_modulus * poisson_ratio /
                              ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)))
        self._density = float(density)
        self._gravity = gravity.astype(np.float32)
        self._damping = float(damping)

        cv = self._capacity_verts
        ct = self._capacity_tets

        # Per-vertex state
        self.positions  = ti.Vector.field(3, dtype=ti.f32, shape=cv)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=cv)
        self._forces    = ti.Vector.field(3, dtype=ti.f32, shape=cv)
        self._masses    = ti.field(dtype=ti.f32, shape=cv)
        self._fixed     = ti.field(dtype=ti.i32,  shape=cv)

        # PCG work fields
        self._mul_ans = ti.Vector.field(3, dtype=ti.f32, shape=cv)
        self._b       = ti.Vector.field(3, dtype=ti.f32, shape=cv)
        self._r       = ti.Vector.field(3, dtype=ti.f32, shape=cv)
        self._p       = ti.Vector.field(3, dtype=ti.f32, shape=cv)
        self._z       = ti.Vector.field(3, dtype=ti.f32, shape=cv)
        # [rz, p*Ap, rz_new] -- kept on GPU so the PCG loop has zero Python syncs
        self._cg_scalars = ti.field(dtype=ti.f32, shape=3)

        # Jacobi preconditioner
        # _diag_K[i]: per-vertex diagonal of K (component-independent, precomputed at init)
        # _diag[i]:   diagonal of A = M - dt^2*K, rebuilt each step call
        self._diag_K = ti.field(dtype=ti.f32, shape=cv)
        self._diag   = ti.field(dtype=ti.f32, shape=cv)

        # Per-tet rest-shape data
        self._tets = ti.Vector.field(4, dtype=ti.i32,  shape=ct)
        self._B    = ti.Matrix.field(3, 3, dtype=ti.f32, shape=ct)   # Dm^-1
        self._W    = ti.field(dtype=ti.f32, shape=ct)                 # vol/6

        # Cutting springs -- same interface as _SpringMassSimulator
        self._sa = np.zeros(0, dtype=np.int32)
        self._sb = np.zeros(0, dtype=np.int32)
        self._sr = np.zeros(0, dtype=np.float32)
        self._sk = np.zeros(0, dtype=np.float32)
        self._stiffness = float(young_modulus)  # placeholder for ECS compat

        # Initial upload (padded to capacity)
        self.velocities.fill(0)
        self._forces.fill(0)
        self._masses.fill(0)
        self._upload_padded_vec3(self.positions, vertices)
        self._upload_padded_int(self._fixed,     fixed_mask)
        self._upload_padded_tet(self._tets,      tetrahedra)

        self._init_tet_data(self._n_tets)
        self._compute_diag_K(self._n_verts, self._n_tets)

    # ------------------------------------------------------------------
    # Capacity helpers
    # ------------------------------------------------------------------

    @property
    def n_verts(self) -> int:
        """Active vertex count (<= capacity)."""
        return self._n_verts

    @property
    def n_tets(self) -> int:
        """Active tet count (<= capacity)."""
        return self._n_tets

    @property
    def vertex_count(self) -> int:
        return self._n_verts

    def _upload_padded_vec3(self, field, data: np.ndarray) -> None:
        pad = np.zeros((self._capacity_verts, 3), dtype=np.float32)
        pad[:len(data)] = data.astype(np.float32)
        field.from_numpy(pad)

    def _upload_padded_int(self, field, data: np.ndarray) -> None:
        pad = np.zeros(self._capacity_verts, dtype=np.int32)
        pad[:len(data)] = data.astype(np.int32)
        field.from_numpy(pad)

    def _upload_padded_tet(self, field, data: np.ndarray) -> None:
        pad = np.zeros((self._capacity_tets, 4), dtype=np.int32)
        pad[:len(data)] = data.astype(np.int32)
        field.from_numpy(pad)

    def rebuild_topology(self,
                         vertices:      np.ndarray,
                         tetrahedra:    np.ndarray,
                         fixed_mask:    np.ndarray,
                         velocities:    np.ndarray,
                         rest_positions: np.ndarray = None) -> None:
        """
        Re-upload new mesh state into the existing GPU fields. Avoids JIT
        recompile per cut by keeping SNode IDs stable across calls.

        rest_positions: optional. If provided, temporarily uploaded to the
        positions field before _init_tet_data computes rest-shape inverses B,
        then overwritten with the real `vertices` before returning. This lets
        progressive-cut callers keep the FEM reference config pristine while
        still stepping the sim on the current deformed geometry -- without
        this, every rebuild would set rest shape = current deformed positions
        and the FEM would lose all elastic memory of the pristine mesh.

        Raises if the new mesh exceeds the simulator's allocated capacity.
        """
        N = len(vertices)
        M = len(tetrahedra)
        if N > self._capacity_verts:
            raise RuntimeError(
                f"rebuild_topology: vertex count {N} exceeds capacity {self._capacity_verts}")
        if M > self._capacity_tets:
            raise RuntimeError(
                f"rebuild_topology: tet count {M} exceeds capacity {self._capacity_tets}")

        # Rest-position first (if supplied) so _init_tet_data reads pristine
        # geometry off self.positions. Overwrite with current vertices after.
        if rest_positions is not None:
            self._upload_padded_vec3(self.positions, rest_positions)
        else:
            self._upload_padded_vec3(self.positions, vertices)
        self._upload_padded_vec3(self.velocities, velocities)
        self._upload_padded_int (self._fixed,     fixed_mask)
        self._upload_padded_tet (self._tets,      tetrahedra)

        # _masses is accumulated by _init_tet_data, so it must start at zero.
        self._masses.fill(0)
        self._forces.fill(0)

        # Cutting springs are rebuilt by the caller (extend_springs).
        self._sa = np.zeros(0, dtype=np.int32)
        self._sb = np.zeros(0, dtype=np.int32)
        self._sr = np.zeros(0, dtype=np.float32)
        self._sk = np.zeros(0, dtype=np.float32)

        self._n_verts = int(N)
        self._n_tets  = int(M)

        self._init_tet_data(self._n_tets)
        # Now swap in the true current positions for the step to integrate.
        if rest_positions is not None:
            self._upload_padded_vec3(self.positions, vertices)
        self._compute_diag_K(self._n_verts, self._n_tets)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @ti.kernel
    def _init_tet_data(self, n_t: ti.i32):
        """Compute rest-shape inverses B and volumes W; accumulate vertex masses."""
        for c in range(n_t):
            verts = self._tets[c]
            Dm = ti.Matrix.cols([
                self.positions[verts[i]] - self.positions[verts[3]]
                for i in range(3)
            ])
            self._B[c] = Dm.inverse()
            self._W[c] = ti.abs(Dm.determinant()) / 6.0
            for i in ti.static(range(4)):
                self._masses[verts[i]] += self._W[c] * self._density / 4.0

    @ti.kernel
    def _compute_diag_K(self, n_v: ti.i32, n_t: ti.i32):
        """
        Precompute per-vertex diagonal of the stiffness matrix K (without dt^2 factor).

        Derivation: for tet c, vertex u in {0,1,2}, the diagonal of the
        linearised K contribution is W * 2*mu * (B @ B.T)[u,u].
        For vertex 3 (the anchor), it is W * 2*mu * ||B.T @ [1,1,1]||^2.
        The result is component-independent due to isotropy.
        Called once at init; must be called again if topology changes.
        """
        for i in range(n_v):
            self._diag_K[i] = 0.0
        for c in range(n_t):
            verts = self._tets[c]
            B_c   = self._B[c]
            W_c   = self._W[c]
            scale = W_c * 2.0 * self._mu
            BBt   = B_c @ B_c.transpose()
            for u in ti.static(range(3)):
                self._diag_K[verts[u]] += scale * BBt[u, u]
            # vertex 3: K_diag = scale * 1^T B B^T 1  (sum over i,j of (B@B^T)[i,j])
            one    = ti.Vector([1.0, 1.0, 1.0])
            Bt_one = B_c.transpose() @ one
            self._diag_K[verts[3]] += scale * Bt_one.dot(Bt_one)

    @ti.kernel
    def _build_diag(self, dt: ti.f32, n_v: ti.i32):
        """Build preconditioner diagonal: 1.0 for fixed DOFs, masses+dt^2*K for free DOFs."""
        for i in range(n_v):
            if self._fixed[i] == 1:
                self._diag[i] = 1.0
            else:
                self._diag[i] = self._masses[i] + dt * dt * self._diag_K[i]

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    @ti.kernel
    def _clear_forces(self, n_v: ti.i32):
        for i in range(n_v):
            self._forces[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _get_force(self, gravity: ti.types.ndarray(dtype=ti.f32, ndim=1),
                   n_v: ti.i32, n_t: ti.i32):
        """Corotational elastic forces + gravity."""
        grav = ti.Vector([gravity[0], gravity[1], gravity[2]])
        for c in range(n_t):
            verts = self._tets[c]
            Ds = ti.Matrix.cols([
                self.positions[verts[i]] - self.positions[verts[3]]
                for i in range(3)
            ])
            F = Ds @ self._B[c]

            # Polar decomposition: F = R @ S, R is rotation (faster than SVD)
            R, S = ti.polar_decompose(F)
            P = 2.0 * self._mu * (F - R)
            H = -self._W[c] * P @ self._B[c].transpose()

            for i in ti.static(range(3)):
                f = ti.Vector([H[j, i] for j in range(3)])
                self._forces[verts[i]] += f
                self._forces[verts[3]] -= f

        for i in range(n_v):
            self._forces[i] += grav * self._masses[i]

    @ti.kernel
    def _apply_spring_forces(self,
                              sa: ti.types.ndarray(dtype=ti.i32, ndim=1),
                              sb: ti.types.ndarray(dtype=ti.i32, ndim=1),
                              sr: ti.types.ndarray(dtype=ti.f32, ndim=1),
                              sk: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        """Explicit spring forces -- used for cutting springs."""
        for s in range(sa.shape[0]):
            a = sa[s]
            b = sb[s]
            d = self.positions[b] - self.positions[a]
            length = d.norm()
            if length > 1e-8:
                f = sk[s] * (length - sr[s]) * (d / length)
                self._forces[a] += f
                self._forces[b] -= f

    # ------------------------------------------------------------------
    # Implicit integration -- preconditioned conjugate gradient (PCG)
    # ------------------------------------------------------------------

    @ti.kernel
    def _get_b(self, dt: ti.f32, n_v: ti.i32):
        """RHS of the linear system: b = M*v + dt*f; zero for fixed DOFs (Dirichlet BC)."""
        for i in range(n_v):
            if self._fixed[i] == 1:
                self._b[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                self._b[i] = self._masses[i] * self.velocities[i] + dt * self._forces[i]

    @ti.kernel
    def _matmul(self, ret: ti.template(), vel: ti.template(),
                dt: ti.f32, n_v: ti.i32, n_t: ti.i32):
        """
        Matrix-free product A*vel where A = M - dt^2*K.
        Fixed DOFs use identity rows so CG enforces v=0 there without a post-process.
        """
        for i in range(n_v):
            if self._fixed[i] == 1:
                ret[i] = vel[i]
            else:
                ret[i] = self._masses[i] * vel[i]
        for c in range(n_t):
            verts = self._tets[c]
            W_c = self._W[c]
            B_c = self._B[c]
            for u in ti.static(range(4)):
                for d in ti.static(range(3)):
                    dD = ti.Matrix.zero(ti.f32, 3, 3)
                    if ti.static(u == 3):
                        for j in ti.static(range(3)):
                            dD[d, j] = -1
                    else:
                        dD[d, u] = 1
                    dF = dD @ B_c
                    dP = 2.0 * self._mu * dF
                    dH = -W_c * dP @ B_c.transpose()
                    for i in ti.static(range(3)):
                        for j in ti.static(range(3)):
                            tmp = vel[verts[i]][j] - vel[verts[3]][j]
                            ret[verts[u]][d] += -(dt ** 2) * dH[j, i] * tmp
        # Overwrite fixed DOF rows: tet loop may have added stiffness contributions to them
        for i in range(n_v):
            if self._fixed[i] == 1:
                ret[i] = vel[i]

    @ti.kernel
    def _pcg_init(self, n_v: ti.i32):
        """r = b - A*v0; z = diag^-1 * r; p = z; scalars[0] = r*z"""
        rz = 0.0
        for i in range(n_v):
            r_i = self._b[i] - self._mul_ans[i]
            z_i = r_i / ti.max(self._diag[i], 1e-30)
            self._r[i] = r_i
            self._z[i] = z_i
            self._p[i] = z_i
            rz += r_i.dot(z_i)
        self._cg_scalars[0] = rz

    @ti.kernel
    def _cg_dot_pAp(self, n_v: ti.i32):
        """scalars[1] = p*(A*p) -- stays on GPU"""
        d = 0.0
        for i in range(n_v):
            d += self._p[i].dot(self._mul_ans[i])
        self._cg_scalars[1] = d

    @ti.kernel
    def _pcg_update_step(self, n_v: ti.i32):
        """alpha = rz/pAp; v += alpha*p; r -= alpha*Ap; z = precond(r); scalars[2] = r*z_new"""
        alpha   = self._cg_scalars[0] / ti.max(self._cg_scalars[1], 1e-30)
        rz_new  = 0.0
        for i in range(n_v):
            self.velocities[i] += alpha * self._p[i]
            r_i = self._r[i] - alpha * self._mul_ans[i]
            z_i = r_i / ti.max(self._diag[i], 1e-30)
            self._r[i] = r_i
            self._z[i] = z_i
            rz_new += r_i.dot(z_i)
        self._cg_scalars[2] = rz_new

    @ti.kernel
    def _pcg_update_p(self, n_v: ti.i32):
        """beta = rz_new/rz; p = z + beta*p; advance scalars[0]"""
        beta = self._cg_scalars[2] / ti.max(self._cg_scalars[0], 1e-30)
        for i in range(n_v):
            self._p[i] = self._z[i] + beta * self._p[i]
        self._cg_scalars[0] = self._cg_scalars[2]

    def _cg(self, dt: float, cg_iters: int = 20):
        """Zero-sync PCG: rz, pAp scalars live in _cg_scalars on GPU throughout."""
        nv = self._n_verts
        nt = self._n_tets
        self._get_b(dt, nv)
        self._build_diag(dt, nv)
        self._matmul(self._mul_ans, self.velocities, dt, nv, nt)
        self._pcg_init(nv)
        for _ in range(cg_iters):
            self._matmul(self._mul_ans, self._p, dt, nv, nt)
            self._cg_dot_pAp(nv)
            self._pcg_update_step(nv)
            self._pcg_update_p(nv)

    # ------------------------------------------------------------------
    # Boundary conditions and position update
    # ------------------------------------------------------------------

    @ti.kernel
    def _apply_fixed(self, n_v: ti.i32):
        """Zero velocities of pinned vertices."""
        for i in range(n_v):
            if self._fixed[i] == 1:
                self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _integrate(self, dt: ti.f32, damping: ti.f32, v_max: ti.f32, n_v: ti.i32):
        for i in range(n_v):
            if self._fixed[i] == 0:
                v = self.velocities[i] * (1.0 - damping * dt)
                speed = v.norm()
                if speed > v_max:
                    v *= v_max / speed
                self.velocities[i] = v
                self.positions[i] += dt * v

    # ------------------------------------------------------------------
    # Public step
    # ------------------------------------------------------------------

    def step(self, dt: float, damping: float, v_max: float = 10.0,
             cg_iters: int = 20):
        nv = self._n_verts
        nt = self._n_tets
        self._clear_forces(nv)
        self._get_force(self._gravity, nv, nt)
        if len(self._sa) > 0:
            self._apply_spring_forces(self._sa, self._sb, self._sr, self._sk)
        self._cg(dt, cg_iters)
        self._apply_fixed(nv)
        self._integrate(dt, damping, v_max, nv)

    # ------------------------------------------------------------------
    # Cutting spring interface (mirrors _SpringMassSimulator)
    # ------------------------------------------------------------------

    def extend_springs(self,
                       sa_new: np.ndarray,
                       sb_new: np.ndarray,
                       sr_new: np.ndarray,
                       sk_new: np.ndarray) -> int:
        start = len(self._sa)
        self._sa = np.concatenate([self._sa, sa_new.astype(np.int32)])
        self._sb = np.concatenate([self._sb, sb_new.astype(np.int32)])
        self._sr = np.concatenate([self._sr, sr_new.astype(np.float32)])
        self._sk = np.concatenate([self._sk, sk_new.astype(np.float32)])
        return start

    @ti.kernel
    def _add_vel_scalar(self,
                        indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
                        dx: ti.f32, dy: ti.f32, dz: ti.f32):
        """Add (dx,dy,dz) to velocity[i] for each i in indices -- no CPU sync needed."""
        dv = ti.Vector([dx, dy, dz])
        for k in range(indices.shape[0]):
            j = indices[k]
            self.velocities[j] += dv
