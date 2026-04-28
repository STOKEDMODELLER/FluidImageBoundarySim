"""
Channel flow past a circular cylinder.

  Geometry: nx x ny channel with no-slip top/bottom walls.
  Obstacle: circular cylinder of diameter D centred upstream.
  Inlet:    uniform plug velocity U_in.
  Outlet:   zero-gradient on the unknown populations.

  Re_D = U_in * D / nu chosen so that vortex shedding develops (> 47).

Outputs (./output/channel_cyl/):
  frames/*.png             two-row per-frame plot (top: |u|, bottom: p').
  channel_cyl_video.mp4    assembled video.
  channel_cyl_summary.png  final state + Cd/Cl history.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

warnings.simplefilter("ignore", RuntimeWarning)
from fluid_sim import LBMSimulation


OUTDIR = Path("output") / "channel_cyl"
FRAMES_DIR = OUTDIR / "frames"


def make_two_row_frame(path: Path, sim: LBMSimulation,
                       vmag_max: float, p_half: float, step: int) -> None:
    """Top row: |u| heatmap. Bottom row: pressure fluctuation heatmap."""
    vmag = sim.get_velocity_magnitude()
    pflu = sim.get_pressure_fluctuation()
    obstacle = sim.obstacle

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=100,
                             gridspec_kw={"hspace": 0.25})

    im0 = axes[0].imshow(vmag.T, origin="lower", cmap="viridis",
                         vmin=0.0, vmax=vmag_max, aspect="equal")
    axes[0].contour(obstacle.T.astype(float), levels=[0.5],
                    colors="white", linewidths=1.0)
    axes[0].set_title(f"|u|   step={step}")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.012, pad=0.01)

    im1 = axes[1].imshow(pflu.T, origin="lower", cmap="RdBu_r",
                         vmin=-p_half, vmax=p_half, aspect="equal")
    axes[1].contour(obstacle.T.astype(float), levels=[0.5],
                    colors="black", linewidths=1.0)
    axes[1].set_title("pressure fluctuation  (p - p_ref)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.012, pad=0.01)

    fig.savefig(path)
    plt.close(fig)


def assemble_video(frames_dir: Path, out_path: Path, fps: int = 25) -> None:
    files = sorted(frames_dir.glob("*.png"))
    if not files:
        print(f"  no frames in {frames_dir}, skipping video")
        return
    img0 = cv2.imread(str(files[0]))
    h, w = img0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for f in files:
        writer.write(cv2.imread(str(f)))
    writer.release()
    print(f"  wrote {out_path}  ({len(files)} frames @ {fps} fps)")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for f in FRAMES_DIR.glob("*.png"):
        f.unlink()

    nx, ny = 700, 160
    D = 30.0
    cx, cy, r = 220.0, ny / 2.0, D / 2.0   # 60-cell sponge + clearance
    U_in = 0.02      # Mach 0.035 — firmly incompressible
    Re_D = 60.0      # above shedding threshold (~47); omega ~ 1.89

    # Free-stream cylinder (free-slip top/bottom). Acoustic transients leave
    # through the outlet instead of bouncing between two no-slip walls.
    sim = LBMSimulation(nx=nx, ny=ny, reynolds=Re_D, flow_speed=U_in,
                        wall_mode="free_slip")
    sim.setup_cylinder_obstacle(cx=cx, cy=cy, r=r)
    # Smooth ramp the inlet from 0 to U_in over ~ 1500 steps (5*tau).
    sim.ramp_steps = 300
    # Sponge layers absorb acoustic waves before they reflect off the Dirichlet
    # inlet / Neumann outlet. Inlet sponge is wider since the cylinder sits
    # closer to the inlet; outlet sponge can be narrower.
    sim.sponge_inlet_width = 60
    sim.sponge_outlet_width = 40
    sim.sponge_strength = 0.4

    print(f"domain {nx} x {ny}, wall_mode={sim.wall_mode}")
    print(f"cylinder D={D} at ({cx}, {cy}),  Re_D={Re_D}")
    print(f"U_in={U_in},  Mach={U_in*np.sqrt(3):.3f},  omega={sim.omega:.4f}, "
          f"ramp_steps={sim.ramp_steps}")
    print(f"sponge: inlet_width={sim.sponge_inlet_width}, "
          f"outlet_width={sim.sponge_outlet_width}, "
          f"strength={sim.sponge_strength}")

    # Pre-set fixed colour ranges for video stability.
    vmag_max = 1.6 * U_in        # plug + parabolic + acceleration around cyl.
    p_half = 5.0e-3              # symmetric pressure-fluctuation half-range.

    # No warm-up — record everything from t=0 so the wake develops on screen.
    n_warmup = 0
    n_record = 20000
    snapshot_every = 100

    Cl_history, Cd_history = [], []

    print("warm-up...")
    for step in range(1, n_warmup + 1):
        sim.step()
        if step % 1000 == 0:
            print(f"  warm  t={step}")
            sys.stdout.flush()

    print("recording...")
    frame_idx = 0
    for k in range(1, n_record + 1):
        d = sim.step()
        Cl_history.append(d["Cl"])
        Cd_history.append(d["Cd"])
        if k % snapshot_every == 0:
            make_two_row_frame(FRAMES_DIR / f"frame_{frame_idx:04d}.png",
                               sim, vmag_max, p_half, sim.time_step)
            frame_idx += 1
        if k % 1000 == 0:
            print(f"  rec   t={sim.time_step}  Mach={d['mach_number']:.4f}  "
                  f"Cd={d['Cd']:+.3f}  Cl={d['Cl']:+.3f}")
            sys.stdout.flush()
        # Fail-fast on divergence so we don't waste time.
        if not np.isfinite(d['Cd']) or d['mach_number'] > 0.18:
            print(f"  ABORT: numerical instability at t={sim.time_step} "
                  f"(Mach={d['mach_number']:.4f}, Cd={d['Cd']})")
            break

    Cl_history = np.array(Cl_history)
    Cd_history = np.array(Cd_history)

    # Strouhal extraction by FFT of Cl.
    sig = Cl_history - Cl_history.mean()
    win = np.hanning(len(sig))
    fft = np.fft.rfft(sig * win)
    freqs = np.fft.rfftfreq(len(sig), d=1.0)
    psd = np.abs(fft) ** 2
    psd[0] = 0.0
    f_meas = freqs[int(np.argmax(psd))]
    St_meas = f_meas * D / U_in
    St_pred = 0.2660 - 1.0160 / np.sqrt(Re_D)
    print()
    print(f"measured St = {St_meas:.4f}  (Williamson predicts {St_pred:.4f})")
    print(f"Cd mean = {Cd_history.mean():+.3f}  amplitude = "
          f"{Cd_history.max() - Cd_history.min():.3f}")
    print(f"Cl mean = {Cl_history.mean():+.3f}  amplitude = "
          f"{Cl_history.max() - Cl_history.min():.3f}")

    # Final summary panel: two-row final state + Cd/Cl time series.
    vmag = sim.get_velocity_magnitude()
    pflu = sim.get_pressure_fluctuation()

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), dpi=110,
                             gridspec_kw={"height_ratios": [1, 1, 0.8],
                                          "hspace": 0.3})
    im0 = axes[0].imshow(vmag.T, origin="lower", cmap="viridis",
                         vmin=0, vmax=vmag_max, aspect="equal")
    axes[0].contour(sim.obstacle.T.astype(float), levels=[0.5],
                    colors="white", linewidths=1.0)
    axes[0].set_title(f"velocity magnitude   (final, t={sim.time_step})")
    fig.colorbar(im0, ax=axes[0], fraction=0.012, pad=0.01)

    im1 = axes[1].imshow(pflu.T, origin="lower", cmap="RdBu_r",
                         vmin=-p_half, vmax=p_half, aspect="equal")
    axes[1].contour(sim.obstacle.T.astype(float), levels=[0.5],
                    colors="black", linewidths=1.0)
    axes[1].set_title("pressure fluctuation")
    fig.colorbar(im1, ax=axes[1], fraction=0.012, pad=0.01)

    t_axis = np.arange(len(Cl_history)) + n_warmup
    axes[2].plot(t_axis, Cd_history, label="Cd")
    axes[2].plot(t_axis, Cl_history, label="Cl")
    axes[2].set_xlabel("step"); axes[2].grid(True, alpha=0.3)
    axes[2].set_title(f"force coefficients   measured St={St_meas:.4f} "
                      f"(predicted {St_pred:.4f})")
    axes[2].legend(loc="upper right")

    fig.savefig(OUTDIR / "channel_cyl_summary.png")
    plt.close(fig)
    print(f"  wrote {OUTDIR/'channel_cyl_summary.png'}")

    assemble_video(FRAMES_DIR, OUTDIR / "channel_cyl_video.mp4", fps=25)


if __name__ == "__main__":
    main()
