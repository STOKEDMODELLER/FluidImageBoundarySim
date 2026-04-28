"""
Lab scenario 1: plane Poiseuille flow.

Empty channel of (nx, ny) = (300, 41) with no-slip top/bottom walls (halfway
bounce-back), uniform plug inlet U_in on the left, zero-gradient outflow on
the right. After the entrance length the velocity profile must converge to
the analytical parabola

    u(y) = u_max * 4 * (y + 0.5) * (H - (y + 0.5)) / H^2,   H = ny,
    u_max = (3/2) * U_in   (mass conservation: U_avg = U_in).

Tuned for the strict incompressible regime: U_in = 0.02 (Mach = 0.035).
"""
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)
from fluid_sim import LBMSimulation


def parabola(ny: int, u_max: float) -> np.ndarray:
    H = float(ny)
    y = np.arange(ny)
    return u_max * 4.0 * (y + 0.5) * (H - (y + 0.5)) / (H * H)


def report(sim: LBMSimulation, U_in: float, x_probe: int, label: str) -> None:
    u = sim.u[0, x_probe, :].copy()
    rho = (sim.pressure / (1.0 / 3.0)).copy() if sim.pressure is not None else None
    u_pred = parabola(sim.ny, 1.5 * U_in)
    err = u - u_pred
    L2 = np.sqrt((err ** 2).mean()) / (1.5 * U_in) * 100
    Linf = np.abs(err).max() / (1.5 * U_in) * 100
    flux_meas = u.mean()
    print(
        f"[{label}] step={sim.time_step:6d}  "
        f"u_max={u.max():.5f} (pred {1.5*U_in:.5f})  "
        f"<u>={flux_meas:.5f} (pred {U_in:.5f})  "
        f"L2={L2:6.3f}%  Linf={Linf:6.3f}%"
    )
    if rho is not None:
        rho_inlet = float(rho[0, :].mean())
        rho_outlet = float(rho[-1, :].mean())
        rho_at_probe = float(rho[x_probe, :].mean())
        print(
            f"           rho:  inlet {rho_inlet:.5f}  "
            f"x={x_probe} {rho_at_probe:.5f}  outlet {rho_outlet:.5f}  "
            f"(rho_max-min in domain: {rho.max()-rho.min():.4e})"
        )


def main() -> None:
    nx, ny = 300, 41
    U_in = 0.02
    Re = 20.0

    sim = LBMSimulation(nx=nx, ny=ny, reynolds=Re, flow_speed=U_in)
    sim.setup_channel()

    H = float(ny)
    nu_pred = U_in * H / Re
    omega_pred = 1.0 / (3.0 * nu_pred + 0.5)
    print(f"target Re_H={Re},  U_in={U_in},  ny={ny}")
    print(f"predicted nu={nu_pred:.5f},  omega={omega_pred:.5f}")
    print(f"sim    omega={sim.omega:.5f},  L_char={sim.L_char:.2f},  Mach={U_in*np.sqrt(3):.4f}")
    print()

    x_probe = nx - 30  # past entrance length
    report_every = 5000
    n_steps = 50000
    diag = None

    for step in range(1, n_steps + 1):
        diag = sim.step()
        if step % report_every == 0:
            report(sim, U_in, x_probe, f"t={step}")
            sys.stdout.flush()

    print()
    print(f"final Mach={diag['mach_number']:.4f}, stable={diag['is_stable']}")

    # Final detailed profile.
    u = sim.u[0, x_probe, :].copy()
    u_pred = parabola(ny, 1.5 * U_in)
    print()
    print(" y      u_meas       u_pred       error      err/u_max")
    for yi in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
        err = u[yi] - u_pred[yi]
        print(f"{yi:3d}   {u[yi]:.6f}    {u_pred[yi]:.6f}    "
              f"{err:+.6f}    {err/(1.5*U_in)*100:+6.2f}%")


if __name__ == "__main__":
    main()
