"""
Main LBM simulation class.

Conventions (D2Q9, BGK, lattice units):

- Arrays are indexed as ``f[q, x, y]`` with shape ``(9, nx, ny)``.
- ``flow_speed`` passed to the constructor is interpreted as a *lattice*
  velocity unless ``dx`` and ``dt`` are also supplied (then it is converted
  via ``u_LU = u_phys * dt / dx``). For lattice-unit input keep the default.
- Inlet: prescribed velocity profile on the left wall, populated via Zou/He.
- Outlet: zero-gradient Neumann on the unknown (c_x < 0) populations on the
  right wall.
- Top/bottom (y=0, y=ny-1): free-slip by default. Pass ``wall_mode="periodic"``
  for the canonical Karman-tutorial behaviour or ``"noslip"`` for a channel.
- Obstacle: halfway bounce-back (interior obstacle nodes skip the BGK
  collision; the bounced populations are streamed into the fluid).
"""
from typing import Tuple, Optional, Dict, Any
import warnings
import numpy as np

from .lattice import LatticeConstants
from .validation import ValidationTools


# Maximum safe BGK relaxation parameter. Above this, the kernel is numerically
# unstable; we clamp and warn so the simulation does not silently diverge.
OMEGA_MAX_SAFE = 1.95
OMEGA_MIN_SAFE = 0.05


class LBMSimulation:
    """Lattice Boltzmann (D2Q9, BGK) simulation."""

    def __init__(self, nx: int, ny: int, reynolds: float = 300.0,
                 flow_speed: float = 0.05,
                 omega: Optional[float] = None,
                 dx: float = 1.0, dt: float = 1.0,
                 wall_mode: str = "free_slip"):
        """
        Args:
            nx, ny: grid size.
            reynolds: target Reynolds number based on the obstacle length scale.
                Note that the realised Re is set when the obstacle is created
                (the relaxation parameter is computed from the actual obstacle
                length). Re-computing omega is automatic.
            flow_speed: inlet velocity. Interpreted as a lattice velocity unless
                ``dx != 1`` or ``dt != 1``, in which case it is converted from
                physical units via ``u_LU = u_phys * dt / dx``.
            omega: BGK relaxation parameter. If None it is derived from
                ``reynolds`` and the obstacle length scale at obstacle setup.
            dx, dt: physical spacing/time-step (lattice units by default).
            wall_mode: 'free_slip' (default), 'periodic', or 'noslip'.
        """
        if wall_mode not in ("free_slip", "periodic", "noslip"):
            raise ValueError(f"Unknown wall_mode '{wall_mode}'")

        self.nx = nx
        self.ny = ny
        self.reynolds = reynolds
        self.flow_speed = flow_speed
        self.dx = dx
        self.dt = dt
        self.wall_mode = wall_mode

        self.lattice = LatticeConstants()
        self.validation = ValidationTools()

        self.uLB = LatticeConstants.physical_to_lattice_velocity(flow_speed, dx, dt)

        # omega and L_char are finalised when the obstacle is created so that
        # the relaxation parameter reflects the actual obstacle length scale.
        # We seed both with a coarse domain proxy so attribute access works
        # immediately after construction (the obstacle setup refines them).
        self._user_omega = omega
        self.L_char: float = max(ny / 9.0, 1.0)
        self.omega: float = self._derive_omega(self.L_char)

        self.rho = 1.0
        self.fin: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.vel: Optional[np.ndarray] = None  # target inlet profile (full strength)
        self.pressure: Optional[np.ndarray] = None
        self.obstacle: Optional[np.ndarray] = None

        # Smooth inlet ramp. While ramp_steps > 0, the inlet velocity is
        # multiplied by (1 - exp(-t/tau)) so there is no impulsive start.
        # tau = ramp_steps / 5  (≈99% strength by the end of the window).
        self.ramp_steps: int = 0

        # Sponge / PML layers. Within these strips at the inlet and outlet
        # the distributions are relaxed toward the equilibrium of a target
        # far-field state every step, which absorbs incoming acoustic waves
        # before they can reflect off the Dirichlet inlet / Neumann outlet.
        # Strength ramps quadratically from sponge_strength at the boundary
        # to 0 at the interior edge of the buffer.
        self.sponge_inlet_width: int = 0
        self.sponge_outlet_width: int = 0
        self.sponge_strength: float = 0.0
        self._sponge_profile: Optional[np.ndarray] = None  # cached σ(x)

        self.time_step = 0
        self.is_running = False

        # Most recent integrated force on the obstacle (lattice units).
        self.force: Tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup_cylinder_obstacle(self, cx: float, cy: float, r: float,
                                epsilon: float = 1e-4) -> None:
        """Circular cylinder obstacle with a perturbed uniform inlet."""
        x_idx = np.arange(self.nx)[:, None]
        y_idx = np.arange(self.ny)[None, :]
        self.obstacle = ((x_idx - cx) ** 2 + (y_idx - cy) ** 2) < (r ** 2)
        # Characteristic length for a cylinder is the diameter.
        self._finalise_geometry(L_char=2.0 * r)

        ly = max(self.ny - 1, 1)
        vel = np.fromfunction(
            lambda d, x, y: (1 - d) * self.uLB * (1.0 + epsilon * np.sin(y / ly * 2 * np.pi)),
            (2, self.nx, self.ny),
        )
        self._initialise_fields(vel)

    def setup_rectangle_obstacle(self, cx: float, cy: float,
                                 length: float, width: float,
                                 epsilon: float = 1e-4) -> None:
        """Axis-aligned rectangular obstacle (length along x, width along y)."""
        x_idx = np.arange(self.nx)[:, None]
        y_idx = np.arange(self.ny)[None, :]
        self.obstacle = ((np.abs(x_idx - cx) < length / 2.0) &
                         (np.abs(y_idx - cy) < width / 2.0))
        # Equivalent diameter D_h = 2 * L * W / (L + W) for a rectangle.
        L_char = 2.0 * length * width / (length + width)
        self._finalise_geometry(L_char=L_char)

        ly = max(self.ny - 1, 1)
        vel = np.fromfunction(
            lambda d, x, y: (1 - d) * self.uLB * (1.0 + epsilon * np.sin(y / ly * 2 * np.pi)),
            (2, self.nx, self.ny),
        )
        self._initialise_fields(vel)

    def setup_channel_with_cylinder(self, cx: float, cy: float, r: float,
                                    epsilon: float = 1e-4) -> None:
        """No-slip channel containing a circular cylinder obstacle."""
        x_idx = np.arange(self.nx)[:, None]
        y_idx = np.arange(self.ny)[None, :]
        self.obstacle = ((x_idx - cx) ** 2 + (y_idx - cy) ** 2) < (r ** 2)
        self.wall_mode = "noslip"
        # Use cylinder diameter as the canonical Re length for an internal obstacle.
        self._finalise_geometry(L_char=2.0 * r)

        ly = max(self.ny - 1, 1)
        vel = np.fromfunction(
            lambda d, x, y: (1 - d) * self.uLB * (1.0 + epsilon * np.sin(y / ly * 2 * np.pi)),
            (2, self.nx, self.ny),
        )
        # Don't seed the obstacle interior with momentum.
        vel[:, self.obstacle] = 0.0
        self._initialise_fields(vel)

    def setup_channel(self) -> None:
        """Empty channel — for Poiseuille / channel-flow validation.

        Forces ``wall_mode='noslip'`` and uses the full effective channel
        height (``ny``) as the characteristic length, which is what halfway
        bounce-back produces for the no-slip wall placement.
        """
        self.obstacle = np.zeros((self.nx, self.ny), dtype=bool)
        self.wall_mode = "noslip"
        self._finalise_geometry(L_char=float(self.ny))

        vel = np.zeros((2, self.nx, self.ny))
        vel[0, :, :] = self.uLB
        self._initialise_fields(vel)

    def setup_from_mask(self, mask_file: str, scale: float = 1.0) -> None:
        """Load obstacle geometry from a PNG file. Uniform inlet (no perturbation)."""
        from ..utils.obstacles import load_mask
        mask = load_mask(mask_file, scale).transpose()
        # Resize-on-load may not match (nx, ny) exactly; tolerate by cropping/padding.
        if mask.shape != (self.nx, self.ny):
            cropped = np.zeros((self.nx, self.ny), dtype=bool)
            sx = min(self.nx, mask.shape[0])
            sy = min(self.ny, mask.shape[1])
            cropped[:sx, :sy] = mask[:sx, :sy]
            mask = cropped
        self.obstacle = mask.astype(bool)

        # Equivalent diameter from the actual mask: 4 * area / perimeter.
        area = float(self.obstacle.sum())
        perim = float(self._mask_perimeter(self.obstacle))
        L_char = (4.0 * area / perim) if perim > 0 else max(self.ny / 9.0, 1.0)
        self._finalise_geometry(L_char=L_char)

        vel = np.zeros((2, self.nx, self.ny))
        vel[0, :, :] = self.uLB
        # Don't seed velocity inside the obstacle — avoids a startup pressure pulse.
        vel[:, self.obstacle] = 0.0
        self._initialise_fields(vel)

    def _initialise_fields(self, vel: np.ndarray) -> None:
        """Populate fin = feq(rho=1, vel) and store the immutable inlet."""
        rho_field = self.rho * np.ones((self.nx, self.ny))
        feq = self.lattice.equilibrium_distribution(self.nx, self.ny, rho_field, vel)
        self.fin = feq.copy()
        self.u = vel.copy()
        # Store the inlet profile separately so it never gets overwritten by
        # the macroscopic field on subsequent steps.
        self.vel = vel.copy()
        self.time_step = 0

    def _derive_omega(self, L_char: float) -> float:
        """Compute omega from Re and L_char, clamping to the safe BGK range."""
        if self._user_omega is not None:
            omega = float(self._user_omega)
        else:
            nu_lb = self.uLB * L_char / self.reynolds
            omega = 1.0 / (3.0 * nu_lb + 0.5)

        if not (OMEGA_MIN_SAFE <= omega <= OMEGA_MAX_SAFE):
            warnings.warn(
                f"omega={omega:.4f} outside safe BGK range "
                f"[{OMEGA_MIN_SAFE}, {OMEGA_MAX_SAFE}]; clamping. "
                f"Re={self.reynolds}, uLB={self.uLB:.4f}, L_char={L_char:.2f}. "
                f"Reduce uLB or Re, or use a finer grid.",
                RuntimeWarning,
                stacklevel=2,
            )
            omega = float(np.clip(omega, OMEGA_MIN_SAFE, OMEGA_MAX_SAFE))
        return omega

    def _finalise_geometry(self, L_char: float) -> None:
        """Set L_char and re-derive omega for the actual obstacle."""
        self.L_char = float(L_char)
        self.omega = self._derive_omega(self.L_char)

    @staticmethod
    def _mask_perimeter(mask: np.ndarray) -> int:
        """Count obstacle/fluid links — used as a discrete perimeter measure."""
        m = mask.astype(np.int8)
        diff = 0
        diff += np.sum(m[1:, :] != m[:-1, :])
        diff += np.sum(m[:, 1:] != m[:, :-1])
        return int(diff)

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    def step(self) -> Dict[str, Any]:
        if self.fin is None or self.obstacle is None:
            raise RuntimeError("Simulation not initialized. Call a setup_* method first.")

        self.fin, self.u, _rho, _feq, _fout, self.pressure, self.force = self._compute_fluid_flow()
        self.time_step += 1
        return self._calculate_diagnostics()

    def run(self, max_iterations: int, callback=None) -> None:
        self.is_running = True
        for _ in range(max_iterations):
            if not self.is_running:
                break
            diagnostics = self.step()
            if callback:
                callback(self, diagnostics)

    def stop(self) -> None:
        self.is_running = False

    # ------------------------------------------------------------------
    # Field accessors
    # ------------------------------------------------------------------
    def get_velocity_magnitude(self) -> np.ndarray:
        if self.u is None:
            return np.zeros((self.nx, self.ny))
        return np.sqrt(self.u[0] ** 2 + self.u[1] ** 2)

    def get_pressure_field(self) -> np.ndarray:
        if self.pressure is None:
            return np.zeros((self.nx, self.ny))
        return self.pressure

    def get_pressure_fluctuation(self) -> np.ndarray:
        """Pressure relative to the reference density: c_s^2 * (rho - rho_0)."""
        if self.pressure is None:
            return np.zeros((self.nx, self.ny))
        return self.pressure - self.rho * LatticeConstants.CS2

    def get_drag_lift_coefficients(self) -> Tuple[float, float]:
        """Drag and lift coefficients (Cd, Cl) using the current force estimate.

        Cd = 2 Fx / (rho * U^2 * L_char), Cl = 2 Fy / (rho * U^2 * L_char).
        Returns (0, 0) before the obstacle is set up.
        """
        if self.L_char is None or self.uLB <= 0.0:
            return (0.0, 0.0)
        denom = self.rho * (self.uLB ** 2) * self.L_char
        if denom <= 0.0:
            return (0.0, 0.0)
        return (2.0 * self.force[0] / denom, 2.0 * self.force[1] / denom)

    # ------------------------------------------------------------------
    # Core kernel
    # ------------------------------------------------------------------
    def _build_sponge_profile(self) -> np.ndarray:
        """Return σ(x) of shape (nx,). Quadratic ramp from sponge_strength at
        the boundary cells to 0 at the interior edges of the buffer strips."""
        sigma = np.zeros(self.nx)
        s = float(self.sponge_strength)
        win = int(self.sponge_inlet_width)
        wout = int(self.sponge_outlet_width)
        if s > 0 and win > 0:
            xs = np.arange(win)
            sigma[:win] = s * ((win - xs) / win) ** 2
        if s > 0 and wout > 0:
            xs = np.arange(wout)
            sigma[self.nx - wout:] = s * ((xs + 1) / wout) ** 2
        return sigma

    def _apply_sponge(self, fout: np.ndarray) -> None:
        """Relax fout toward feq_target inside the sponge strips, in place.

        The far-field target is the prescribed inlet plug at unit density.
        Cells outside the sponge are untouched (σ=0).
        """
        if self.sponge_strength <= 0.0:
            return
        if self._sponge_profile is None or self._sponge_profile.shape != (self.nx,):
            self._sponge_profile = self._build_sponge_profile()
        sigma = self._sponge_profile
        if not np.any(sigma > 0):
            return

        # Build target equilibrium once (rho=1, u=self.vel) — this is the
        # undisturbed far-field state we want the sponge to anchor on.
        rho_t = np.ones((self.nx, self.ny))
        feq_t = self.lattice.equilibrium_distribution(self.nx, self.ny,
                                                     rho_t, self.vel)

        # σ broadcast to (1, nx, 1) so it scales each population per x-column.
        s = sigma[None, :, None]
        fout[:] = (1.0 - s) * fout + s * feq_t

    def _apply_y_walls_post_collision(self, fout: np.ndarray) -> None:
        """Apply free-slip / no-slip on y=0 and y=ny-1, between collision and
        streaming. Operates on ``fout`` so the bounced populations stream into
        the fluid one cell on the next streaming step.

        This is the canonical halfway bounce-back placement. Swap is at the
        same y-row (the wall cell), not pulled from the cell above/below — that
        was the bug that violated mass conservation in the no-slip case.
        """
        if self.wall_mode == "periodic":
            return

        if self.wall_mode == "free_slip":
            opp_y = self.lattice.y_mirror
        elif self.wall_mode == "noslip":
            opp_y = self.lattice.noslip
        else:
            raise ValueError(f"Unknown wall_mode '{self.wall_mode}'")

        # At y=0: the populations going INTO the wall are those with c_y < 0
        # (cy_neg). After collision, before streaming, replace the populations
        # going AWAY from the wall (their opp_y partners at the same node) with
        # the values about to head into the wall. Streaming then sends the
        # bounced/mirrored populations back into the fluid at y=1.
        for i in self.lattice.cy_neg:
            fout[opp_y[i], :, 0] = fout[i, :, 0]
        # Symmetric at y=ny-1: populations going INTO the top wall are cy_pos.
        for i in self.lattice.cy_pos:
            fout[opp_y[i], :, -1] = fout[i, :, -1]

    def _compute_fluid_flow(self):
        fin = self.fin
        c = self.lattice.c
        i1, i2, i3 = self.lattice.i1, self.lattice.i2, self.lattice.i3

        # 1) Right wall: zero-gradient outflow on the unknown (c_x < 0) populations only.
        fin[i1, -1, :] = fin[i1, -2, :]

        # 2) Macroscopic fields (with rho-floor to keep u finite if a void forms).
        rho = LatticeConstants.compute_density(fin)
        rho_safe = np.where(rho > 1e-8, rho, 1e-8)
        u = np.dot(c.transpose(), fin.transpose((1, 0, 2))) / rho_safe

        # 3) Left wall: clamp velocity to the prescribed inlet, optionally with
        #    a smooth ramp to avoid an impulsive acoustic shock at t=0.
        if self.ramp_steps > 0 and self.time_step < 5 * self.ramp_steps:
            tau = max(self.ramp_steps, 1)
            ramp = 1.0 - np.exp(-self.time_step / tau)
        else:
            ramp = 1.0
        u[:, 0, :] = self.vel[:, 0, :] * ramp

        # 4) Left wall density from known populations (Zou/He).
        denom = 1.0 - u[0, 0, :]
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        rho[0, :] = (1.0 / denom) * (LatticeConstants.compute_density(fin[i2, 0, :])
                                     + 2.0 * LatticeConstants.compute_density(fin[i1, 0, :]))

        # 5) Equilibrium with the corrected (rho, u) at the inlet.
        feq = self.lattice.equilibrium_distribution(self.nx, self.ny, rho, u)

        # 6) Zou/He inlet for the unknown (c_x > 0) populations:
        #    f_i = f_{opp(i)} + (f_i^eq - f_{opp(i)}^eq).
        opp = self.lattice.noslip
        for i_unknown in i3:
            i_known = opp[i_unknown]
            fin[i_unknown, 0, :] = (fin[i_known, 0, :]
                                    + feq[i_unknown, 0, :] - feq[i_known, 0, :])

        # 7) BGK collision.
        fout = fin - self.omega * (fin - feq)

        # 8) Halfway bounce-back at obstacle nodes. We deliberately read from
        #    pre-collision ``fin`` here: obstacle nodes do not collide; their
        #    populations simply reflect.
        obstacle = self.obstacle
        for i in range(self.lattice.q):
            fout[i, obstacle] = fin[opp[i], obstacle]

        # 9a) Sponge / PML — absorb acoustic perturbations near inlet/outlet
        #     by relaxing post-collision distributions toward the target
        #     equilibrium. No-op when sponge_strength == 0.
        self._apply_sponge(fout)

        # 9b) Top/bottom walls (halfway bounce-back / mirror, on fout).
        self._apply_y_walls_post_collision(fout)

        # 10) Drag/lift via momentum exchange. For each fluid->obstacle link
        #    (i.e., a fluid node whose neighbour in direction c_i is solid) the
        #    momentum transfer is 2 * c_i * f_i^pre-collision.
        force = self._momentum_exchange(fin)

        # 11) Streaming.
        for i in range(self.lattice.q):
            fin[i, :, :] = np.roll(np.roll(fout[i, :, :], c[i, 0], axis=0),
                                   c[i, 1], axis=1)

        # 12) Pressure (lattice equation of state). The interesting visual is
        #     the fluctuation about rho_0 — see ``get_pressure_fluctuation``.
        rho = LatticeConstants.compute_density(fin)
        pressure = rho * LatticeConstants.CS2

        return fin, u, rho, feq, fout, pressure, force

    def _momentum_exchange(self, fin: np.ndarray) -> Tuple[float, float]:
        """Sum 2 c_i f_i over fluid->obstacle links to get the integrated force."""
        c = self.lattice.c
        obstacle = self.obstacle
        fluid = ~obstacle
        Fx = 0.0
        Fy = 0.0
        for i in range(self.lattice.q):
            cx, cy = int(c[i, 0]), int(c[i, 1])
            if cx == 0 and cy == 0:
                continue
            # Neighbour of each cell in direction c_i; uses periodic wrap, which
            # is harmless because boundary fluid cells normally have no obstacle
            # neighbour through the wrap.
            neigh_solid = np.roll(obstacle, shift=(-cx, -cy), axis=(0, 1))
            link_mask = fluid & neigh_solid
            if not link_mask.any():
                continue
            popsum = float(fin[i][link_mask].sum())
            Fx += 2.0 * cx * popsum
            Fy += 2.0 * cy * popsum
        return (Fx, Fy)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _calculate_diagnostics(self) -> Dict[str, Any]:
        if self.u is None:
            return {}

        vel_mag = self.get_velocity_magnitude()
        max_vel = float(np.max(vel_mag))

        stability = self.validation.check_stability_conditions(
            self.u, self.omega, L_char=self.L_char or 1.0,
        )

        Cd, Cl = self.get_drag_lift_coefficients()

        return {
            'time_step': self.time_step,
            'max_velocity': max_vel,
            'max_pressure': float(np.max(self.pressure)) if self.pressure is not None else 0.0,
            'min_pressure': float(np.min(self.pressure)) if self.pressure is not None else 0.0,
            'mach_number': stability.get('max_mach_number', 0.0),
            'is_stable': bool(stability.get('mach_stable', False)
                              and stability.get('omega_stable', False)),
            'reynolds_target': self.reynolds,
            'reynolds_realised': stability.get('estimated_reynolds', 0.0),
            'omega': self.omega,
            'L_char': self.L_char,
            'force_x': self.force[0],
            'force_y': self.force[1],
            'Cd': Cd,
            'Cl': Cl,
        }
