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
                 vertices:      np.ndarray,
                 tetrahedra:    np.ndarray,
                 fixed_mask:    np.ndarray,
                 young_modulus: float,
                 poisson_ratio: float,
                 density:       float,
                 gravity:       np.ndarray,
                 damping:       float = 0.0):
        N = len(vertices)
        M = len(tetrahedra)

        self._mu      = float(young_modulus / (2.0 * (1.0 + poisson_ratio)))
        self._la      = float(young_modulus * poisson_ratio /
                              ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)))
        self._density = float(density)
        self._gravity = gravity.astype(np.float32)
        self._damping = float(damping)

        # Per-vertex state
        self.positions  = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self._forces    = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self._masses    = ti.field(dtype=ti.f32, shape=N)
        self._fixed     = ti.field(dtype=ti.i32,  shape=N)

        # PCG work fields
        self._mul_ans = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self._b       = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self._r       = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self._p       = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self._z       = ti.Vector.field(3, dtype=ti.f32, shape=N)
        # [rz, p*Ap, rz_new] -- kept on GPU so the PCG loop has zero Python syncs
        self._cg_scalars = ti.field(dtype=ti.f32, shape=3)

        # Jacobi preconditioner
        # _diag_K[i]: per-vertex diagonal of K (component-independent, precomputed at init)
        # _diag[i]:   diagonal of A = M - dt^2*K, rebuilt each step call
        self._diag_K = ti.field(dtype=ti.f32, shape=N)
        self._diag   = ti.field(dtype=ti.f32, shape=N)

        # Per-tet rest-shape data
        self._tets = ti.Vector.field(4, dtype=ti.i32,  shape=M)
        self._B    = ti.Matrix.field(3, 3, dtype=ti.f32, shape=M)   # Dm^-1
        self._W    = ti.field(dtype=ti.f32, shape=M)                 # vol/6

        # Cutting springs -- same interface as _SpringMassSimulator
        self._sa = np.zeros(0, dtype=np.int32)
        self._sb = np.zeros(0, dtype=np.int32)
        self._sr = np.zeros(0, dtype=np.float32)
        self._sk = np.zeros(0, dtype=np.float32)
        self._stiffness = float(young_modulus)  # placeholder for ECS compat

        # Upload initial data
        self.positions.from_numpy(vertices.astype(np.float32))
        self.velocities.fill(0)
        self._forces.fill(0)
        self._masses.fill(0)
        self._fixed.from_numpy(fixed_mask.astype(np.int32))
        self._tets.from_numpy(tetrahedra.astype(np.int32))

        self._init_tet_data()
        self._compute_diag_K()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @ti.kernel
    def _init_tet_data(self):
        """Compute rest-shape inverses B and volumes W; accumulate vertex masses."""
        for c in self._tets:
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
    def _compute_diag_K(self):
        """
        Precompute per-vertex diagonal of the stiffness matrix K (without dt^2 factor).

        Derivation: for tet c, vertex u in {0,1,2}, the diagonal of the
        linearised K contribution is W * 2*mu * (B @ B.T)[u,u].
        For vertex 3 (the anchor), it is W * 2*mu * ||B.T @ [1,1,1]||^2.
        The result is component-independent due to isotropy.
        Called once at init; must be called again if topology changes.
        """
        for i in self._diag_K:
            self._diag_K[i] = 0.0
        for c in self._tets:
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
    def _build_diag(self, dt: ti.f32):
        """Build preconditioner diagonal: 1.0 for fixed DOFs, masses+dt^2*K for free DOFs."""
        for i in self._diag:
            if self._fixed[i] == 1:
                self._diag[i] = 1.0
            else:
                self._diag[i] = self._masses[i] + dt * dt * self._diag_K[i]

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    @ti.kernel
    def _clear_forces(self):
        for i in self._forces:
            self._forces[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _get_force(self, gravity: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        """Corotational elastic forces + gravity."""
        grav = ti.Vector([gravity[0], gravity[1], gravity[2]])
        for c in self._tets:
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

        for i in self.positions:
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
    def _get_b(self, dt: ti.f32):
        """RHS of the linear system: b = M*v + dt*f; zero for fixed DOFs (Dirichlet BC)."""
        for i in self._b:
            if self._fixed[i] == 1:
                self._b[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                self._b[i] = self._masses[i] * self.velocities[i] + dt * self._forces[i]

    @ti.kernel
    def _matmul(self, ret: ti.template(), vel: ti.template(), dt: ti.f32):
        """
        Matrix-free product A*vel where A = M - dt^2*K.
        Fixed DOFs use identity rows so CG enforces v=0 there without a post-process.
        """
        for i in ret:
            if self._fixed[i] == 1:
                ret[i] = vel[i]
            else:
                ret[i] = self._masses[i] * vel[i]
        for c in self._tets:
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
        for i in ret:
            if self._fixed[i] == 1:
                ret[i] = vel[i]

    @ti.kernel
    def _pcg_init(self):
        """r = b - A*v0; z = diag^-1 * r; p = z; scalars[0] = r*z"""
        rz = 0.0
        for i in self._r:
            r_i = self._b[i] - self._mul_ans[i]
            z_i = r_i / ti.max(self._diag[i], 1e-30)
            self._r[i] = r_i
            self._z[i] = z_i
            self._p[i] = z_i
            rz += r_i.dot(z_i)
        self._cg_scalars[0] = rz

    @ti.kernel
    def _cg_dot_pAp(self):
        """scalars[1] = p*(A*p) -- stays on GPU"""
        d = 0.0
        for i in self._p:
            d += self._p[i].dot(self._mul_ans[i])
        self._cg_scalars[1] = d

    @ti.kernel
    def _pcg_update_step(self):
        """alpha = rz/pAp; v += alpha*p; r -= alpha*Ap; z = precond(r); scalars[2] = r*z_new"""
        alpha   = self._cg_scalars[0] / ti.max(self._cg_scalars[1], 1e-30)
        rz_new  = 0.0
        for i in self.velocities:
            self.velocities[i] += alpha * self._p[i]
            r_i = self._r[i] - alpha * self._mul_ans[i]
            z_i = r_i / ti.max(self._diag[i], 1e-30)
            self._r[i] = r_i
            self._z[i] = z_i
            rz_new += r_i.dot(z_i)
        self._cg_scalars[2] = rz_new

    @ti.kernel
    def _pcg_update_p(self):
        """beta = rz_new/rz; p = z + beta*p; advance scalars[0]"""
        beta = self._cg_scalars[2] / ti.max(self._cg_scalars[0], 1e-30)
        for i in self._p:
            self._p[i] = self._z[i] + beta * self._p[i]
        self._cg_scalars[0] = self._cg_scalars[2]

    def _cg(self, dt: float, cg_iters: int = 20):
        """Zero-sync PCG: rz, pAp scalars live in _cg_scalars on GPU throughout."""
        self._get_b(dt)
        self._build_diag(dt)
        self._matmul(self._mul_ans, self.velocities, dt)
        self._pcg_init()
        for _ in range(cg_iters):
            self._matmul(self._mul_ans, self._p, dt)
            self._cg_dot_pAp()
            self._pcg_update_step()
            self._pcg_update_p()

    # ------------------------------------------------------------------
    # Boundary conditions and position update
    # ------------------------------------------------------------------

    @ti.kernel
    def _apply_fixed(self):
        """Zero velocities of pinned vertices."""
        for i in self.velocities:
            if self._fixed[i] == 1:
                self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _integrate(self, dt: ti.f32, damping: ti.f32, v_max: ti.f32):
        for i in self.positions:
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
        self._clear_forces()
        self._get_force(self._gravity)
        if len(self._sa) > 0:
            self._apply_spring_forces(self._sa, self._sb, self._sr, self._sk)
        self._cg(dt, cg_iters)
        self._apply_fixed()
        self._integrate(dt, damping, v_max)

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
