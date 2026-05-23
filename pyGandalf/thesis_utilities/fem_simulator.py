"""
Taichi corotational FEM simulator.

GPU-accelerated implicit FEM for tetrahedral meshes.

Elasticity model
----------------
Corotational neo-Hookean (mu term only, linearized stiffness for implicit solve):
    F  = Ds @ B          — deformation gradient
    R  = U @ V.T         — rotation extracted by SVD of F
    P  = 2*mu*(F - R)    — first Piola-Kirchhoff stress

Integration
-----------
Implicit Euler via matrix-free conjugate gradient:
    (M - dt²·K) v_new = M·v_old + dt·f
This is unconditionally stable regardless of dt or stiffness.

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

        # CG work fields
        self._mul_ans   = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self._b         = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self._r         = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self._p         = ti.Vector.field(3, dtype=ti.f32, shape=N)

        # Per-tet rest-shape data
        self._tets      = ti.Vector.field(4, dtype=ti.i32,  shape=M)
        self._B         = ti.Matrix.field(3, 3, dtype=ti.f32, shape=M)   # Dm⁻¹
        self._W         = ti.field(dtype=ti.f32, shape=M)                 # vol/6

        # Cutting springs — same interface as _SpringMassSimulator
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

            # SVD-based rotation extraction
            U, sig, V = ti.svd(F)
            if U.determinant() < 0:
                for i in ti.static(range(3)):
                    U[i, 2] *= -1
                sig[2, 2] = -sig[2, 2]
            if V.determinant() < 0:
                for i in ti.static(range(3)):
                    V[i, 2] *= -1
                sig[2, 2] = -sig[2, 2]

            P = 2.0 * self._mu * (F - U @ V.transpose())
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
        """Explicit spring forces — used for cutting springs."""
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
    # Implicit integration — conjugate gradient
    # ------------------------------------------------------------------

    @ti.kernel
    def _get_b(self, dt: ti.f32):
        """RHS of the linear system: b = M·v + dt·f"""
        for i in self._b:
            self._b[i] = self._masses[i] * self.velocities[i] + dt * self._forces[i]

    @ti.kernel
    def _matmul(self, ret: ti.template(), vel: ti.template(), dt: ti.f32):
        """
        Matrix-free product A·vel where A = M - dt²·K.
        K is the linearized corotational stiffness (2*mu term only).
        """
        for i in ret:
            ret[i] = self._masses[i] * vel[i]
        for c in self._tets:
            verts = self._tets[c]
            W_c = self._W[c]
            B_c = self._B[c]
            for u in range(4):
                for d in range(3):
                    dD = ti.Matrix.zero(ti.f32, 3, 3)
                    if u == 3:
                        for j in range(3):
                            dD[d, j] = -1
                    else:
                        dD[d, u] = 1
                    dF = dD @ B_c
                    dP = 2.0 * self._mu * dF
                    dH = -W_c * dP @ B_c.transpose()
                    for i in range(3):
                        for j in range(3):
                            tmp = vel[verts[i]][j] - vel[verts[3]][j]
                            ret[verts[u]][d] += -(dt ** 2) * dH[j, i] * tmp

    @ti.kernel
    def _vec_add(self, ans: ti.template(), a: ti.template(),
                 k: ti.f32, b: ti.template()):
        for i in ans:
            ans[i] = a[i] + k * b[i]

    @ti.kernel
    def _vec_dot(self, a: ti.template(), b: ti.template()) -> ti.f32:
        ans = 0.0
        for i in a:
            ans += a[i].dot(b[i])
        return ans

    def _cg(self, dt: float, cg_iters: int = 50, cg_eps: float = 1e-6):
        """Conjugate gradient solve for (M - dt²K) v_new = b."""
        self._get_b(dt)
        self._matmul(self._mul_ans, self.velocities, dt)
        self._vec_add(self._r, self._b, -1.0, self._mul_ans)
        self._p.copy_from(self._r)
        r2 = self._vec_dot(self._r, self._r)
        r2_init = r2
        for _ in range(cg_iters):
            self._matmul(self._mul_ans, self._p, dt)
            denom = self._vec_dot(self._p, self._mul_ans)
            if abs(denom) < 1e-30:
                break
            alpha = r2 / denom
            self._vec_add(self.velocities, self.velocities,  alpha, self._p)
            self._vec_add(self._r,         self._r,         -alpha, self._mul_ans)
            r2_new = self._vec_dot(self._r, self._r)
            if r2_new <= r2_init * cg_eps ** 2:
                break
            beta = r2_new / max(r2, 1e-30)
            self._vec_add(self._p, self._r, beta, self._p)
            r2 = r2_new

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
             cg_iters: int = 50, cg_eps: float = 1e-6):
        self._clear_forces()
        self._get_force(self._gravity)
        if len(self._sa) > 0:
            self._apply_spring_forces(self._sa, self._sb, self._sr, self._sk)
        self._cg(dt, cg_iters, cg_eps)
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
