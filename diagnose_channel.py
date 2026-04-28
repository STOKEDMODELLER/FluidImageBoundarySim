"""
Localise the Poiseuille mass-flux bug.

Tests A, B, C run at increasing complexity:
  A) wall_mode='periodic'  -> no top/bottom walls. Expect uniform plug u=U_in
     everywhere, rho=1 everywhere. If it deviates, the bug is in inlet/outlet.
  B) wall_mode='free_slip' -> mirror in y. Expect uniform plug too (free-slip
     does not slow anything down for a flow already aligned with walls).
  C) wall_mode='noslip'    -> the failing case from validate_poiseuille.

For each test we report: mass flux at inlet, x=nx/2, x=nx-2 columns,
time-averaged over the last 500 steps after a 5000-step warm-up.
"""
import warnings
import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)
from fluid_sim import LBMSimulation


def cross_section_flux(sim: LBMSimulation, x: int) -> tuple:
    """Returns (rho_mean, u_mean, mass_flux) at column x."""
    rho = sim.pressure[x, :] / (1.0 / 3.0)
    u = sim.u[0, x, :]
    return float(rho.mean()), float(u.mean()), float((rho * u).sum())


def run_test(label: str, wall_mode: str, U_in: float = 0.02,
             nx: int = 200, ny: int = 41, n_warm: int = 5000,
             n_avg: int = 500) -> None:
    sim = LBMSimulation(nx=nx, ny=ny, reynolds=20.0, flow_speed=U_in,
                        wall_mode=wall_mode)
    sim.setup_channel()
    sim.wall_mode = wall_mode  # setup_channel forces noslip; restore caller intent.

    print(f"=== {label}: wall_mode={wall_mode}, U_in={U_in}, nx={nx}, ny={ny} ===")
    print(f"omega={sim.omega:.4f}, Mach={U_in*np.sqrt(3):.4f}")

    for _ in range(n_warm):
        sim.step()

    # Average over the last n_avg steps.
    flux_inlet = []
    flux_mid = []
    flux_outlet = []
    rho_inlet = []
    rho_mid = []
    rho_outlet = []
    u_inlet_max = []
    u_mid_max = []
    u_outlet_max = []
    for _ in range(n_avg):
        sim.step()
        ri, ui, fi = cross_section_flux(sim, 0)
        rm, um, fm = cross_section_flux(sim, nx // 2)
        ro, uo, fo = cross_section_flux(sim, nx - 2)
        flux_inlet.append(fi); flux_mid.append(fm); flux_outlet.append(fo)
        rho_inlet.append(ri); rho_mid.append(rm); rho_outlet.append(ro)
        u_inlet_max.append(sim.u[0, 0, :].max())
        u_mid_max.append(sim.u[0, nx // 2, :].max())
        u_outlet_max.append(sim.u[0, nx - 2, :].max())

    fi, fm, fo = np.mean(flux_inlet), np.mean(flux_mid), np.mean(flux_outlet)
    ri, rm, ro = np.mean(rho_inlet), np.mean(rho_mid), np.mean(rho_outlet)
    print(f"  flux:   inlet={fi:.5f}  mid={fm:.5f}  outlet={fo:.5f}  "
          f"(predicted U_in*ny = {U_in*ny:.5f})")
    print(f"  rho:    inlet={ri:.5f}  mid={rm:.5f}  outlet={ro:.5f}")
    print(f"  u_max:  inlet={np.mean(u_inlet_max):.5f}  "
          f"mid={np.mean(u_mid_max):.5f}  outlet={np.mean(u_outlet_max):.5f}")
    print()


def main() -> None:
    run_test("A", "periodic")
    run_test("B", "free_slip")
    run_test("C", "noslip")


if __name__ == "__main__":
    main()
