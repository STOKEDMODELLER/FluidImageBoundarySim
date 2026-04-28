"""
Lab scenario 2: Karman vortex shedding from a circular cylinder in a
moderately-confined channel.

Setup:
    nx, ny = 600, 80     (8 diameters wide channel)
    cylinder  D = 10  at (cx, cy) = (120, 40)
    U_in     = 0.04
    Re_D     = U_in * D / nu = 100   (well above the shedding threshold)
    walls    = free_slip on top/bottom (free-stream cylinder, not a duct)

Hand prediction (Williamson 1996 fit, valid for 47 <= Re_D <= 180):
    St = 0.2660 - 1.0160 / sqrt(Re_D)
       = 0.2660 - 1.0160 / 10
       = 0.1644
    f  = St * U / D = 0.1644 * 0.04 / 10 = 6.576e-4 lattice^-1
    T  = 1 / f      = 1521 steps per shedding cycle.

Validation:
    - run long enough to capture > 5 shedding cycles (>= 10000 steps after
      the wake develops)
    - record Cl each step
    - peak frequency from FFT == measured Strouhal
    - mean Cd in the shedding regime = O(1.3 - 1.4) for Re=100 cylinders
      (textbook: Cd ~ 1.4 at Re=100 in a free stream).
"""
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)
from fluid_sim import LBMSimulation


def main() -> None:
    nx, ny = 600, 80
    D = 10.0
    cx, cy = 120.0, ny / 2.0
    r = D / 2.0
    U_in = 0.04
    Re = 100.0

    sim = LBMSimulation(nx=nx, ny=ny, reynolds=Re, flow_speed=U_in,
                        wall_mode="free_slip")
    sim.setup_cylinder_obstacle(cx=cx, cy=cy, r=r)

    nu_pred = U_in * D / Re
    omega_pred = 1.0 / (3.0 * nu_pred + 0.5)
    St_pred = 0.2660 - 1.0160 / np.sqrt(Re)
    f_pred = St_pred * U_in / D
    T_pred = 1.0 / f_pred

    print(f"Re_D={Re}, U={U_in}, D={D}, ny={ny}")
    print(f"predicted nu={nu_pred:.5f}, omega={omega_pred:.5f}")
    print(f"sim       omega={sim.omega:.5f}, L_char={sim.L_char:.2f}, "
          f"Mach={U_in*np.sqrt(3):.4f}")
    print(f"predicted St={St_pred:.4f},  f={f_pred:.6e},  T={T_pred:.1f} steps")
    print()

    # warm-up: let the wake develop. Symmetric IC needs a perturbation, which
    # setup_cylinder_obstacle already provides via epsilon.
    n_warmup = 6000
    n_record = 20000
    Cl_history = np.zeros(n_record)
    Cd_history = np.zeros(n_record)

    print("warm-up...")
    for step in range(1, n_warmup + 1):
        d = sim.step()
        if step % 2000 == 0:
            print(f"  warm  t={step}  Mach={d['mach_number']:.4f}  "
                  f"Cd={d['Cd']:+.3f}  Cl={d['Cl']:+.3f}")
            sys.stdout.flush()

    print("recording lift / drag...")
    for k in range(n_record):
        d = sim.step()
        Cl_history[k] = d['Cl']
        Cd_history[k] = d['Cd']
        if (k + 1) % 4000 == 0:
            print(f"  rec   t={sim.time_step}  Mach={d['mach_number']:.4f}  "
                  f"Cd={d['Cd']:+.3f}  Cl={d['Cl']:+.3f}")
            sys.stdout.flush()

    # FFT on Cl to recover Strouhal.
    sig = Cl_history - Cl_history.mean()
    win = np.hanning(len(sig))
    fft = np.fft.rfft(sig * win)
    freqs = np.fft.rfftfreq(len(sig), d=1.0)
    psd = np.abs(fft) ** 2
    # ignore DC bin
    psd[0] = 0.0
    peak = int(np.argmax(psd))
    f_meas = freqs[peak]
    St_meas = f_meas * D / U_in
    T_meas = 1.0 / f_meas if f_meas > 0 else float('inf')

    print()
    print(f"measured peak frequency  f = {f_meas:.6e} (T = {T_meas:.1f} steps)")
    print(f"measured Strouhal        St = {St_meas:.4f}    (predicted {St_pred:.4f})")
    print(f"relative St error        {(St_meas/St_pred - 1.0)*100:+.2f} %")
    print()
    print(f"Cd mean = {Cd_history.mean():+.3f}  std = {Cd_history.std():.3f}")
    print(f"Cl mean = {Cl_history.mean():+.3f}  amplitude (peak-peak) = "
          f"{Cl_history.max() - Cl_history.min():.3f}")


if __name__ == "__main__":
    main()
