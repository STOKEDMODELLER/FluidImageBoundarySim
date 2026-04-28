"""
Render image / video output for the two lab scenarios.

Outputs (under ./output/):
  poiseuille_summary.png        – velocity field + cross-section profile vs.
                                  analytical parabola.
  poiseuille_frames/*.png       – per-frame velocity heatmaps.
  poiseuille.mp4                – assembled video.
  karman_summary.png            – velocity, vorticity, pressure-fluctuation
                                  panels of the cylinder wake at end of run.
  karman_frames/*.png           – per-frame vorticity heatmaps.
  karman.mp4                    – assembled video.
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cv2

warnings.simplefilter("ignore", RuntimeWarning)
from fluid_sim import LBMSimulation


OUTDIR = Path("output")
OUTDIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def vorticity(u: np.ndarray) -> np.ndarray:
    """Discrete z-component of curl from a (2, nx, ny) velocity field."""
    ux, uy = u[0], u[1]
    duy_dx = np.zeros_like(uy)
    dux_dy = np.zeros_like(ux)
    duy_dx[1:-1, :] = 0.5 * (uy[2:, :] - uy[:-2, :])
    dux_dy[:, 1:-1] = 0.5 * (ux[:, 2:] - ux[:, :-2])
    return duy_dx - dux_dy


def save_heatmap(path: Path, field: np.ndarray, title: str,
                 vmin=None, vmax=None, cmap="viridis",
                 obstacle: np.ndarray = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    im = ax.imshow(field.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal")
    if obstacle is not None:
        ax.contour(obstacle.T.astype(float), levels=[0.5],
                   colors="black", linewidths=0.8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
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


# ----------------------------------------------------------------------
# Lab 1: Poiseuille
# ----------------------------------------------------------------------
def run_poiseuille() -> None:
    print("=" * 60)
    print("Lab 1 — plane Poiseuille channel")
    print("=" * 60)
    nx, ny = 300, 41
    U_in = 0.02
    Re = 20.0

    sim = LBMSimulation(nx=nx, ny=ny, reynolds=Re, flow_speed=U_in)
    sim.setup_channel()

    frames_dir = OUTDIR / "poiseuille_frames"
    frames_dir.mkdir(exist_ok=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()

    n_steps = 30000
    snapshot_every = 200

    frame_idx = 0
    for step in range(1, n_steps + 1):
        sim.step()
        if step % snapshot_every == 0:
            vmag = sim.get_velocity_magnitude()
            save_heatmap(
                frames_dir / f"frame_{frame_idx:04d}.png",
                vmag, f"Poiseuille  t={step}  |u|  (predicted u_max={1.5*U_in:.4f})",
                vmin=0.0, vmax=1.6 * U_in, cmap="viridis",
            )
            frame_idx += 1
            if step % 5000 == 0:
                print(f"  t={step:6d}  max|u|={vmag.max():.5f}  "
                      f"<u> at probe={sim.u[0, nx-30, :].mean():.5f}  "
                      f"(predicted U_in={U_in:.5f})")
                sys.stdout.flush()

    # Final summary: heatmap + cross-section vs analytical parabola.
    H = float(ny)
    y = np.arange(ny)
    u_pred = 1.5 * U_in * 4.0 * (y + 0.5) * (H - (y + 0.5)) / (H * H)
    u_meas = sim.u[0, nx - 30, :]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=100,
                             gridspec_kw={"width_ratios": [3, 1]})
    im = axes[0].imshow(sim.get_velocity_magnitude().T, origin="lower",
                        cmap="viridis", vmin=0, vmax=1.6 * U_in,
                        aspect="equal")
    axes[0].set_title(f"Poiseuille velocity magnitude  (Re_H={Re}, Mach={U_in*np.sqrt(3):.3f})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].axvline(nx - 30, color="white", linestyle="--", linewidth=0.8,
                    label="probe")
    axes[0].legend(loc="upper right")
    fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.02)

    axes[1].plot(u_meas, y, "o-", label="LBM", markersize=4)
    axes[1].plot(u_pred, y, "k--", label="analytical")
    axes[1].set_xlabel("u_x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Profile at probe")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTDIR / "poiseuille_summary.png")
    plt.close(fig)
    print(f"  wrote {OUTDIR/'poiseuille_summary.png'}")

    err = u_meas - u_pred
    print(f"  L2 err / u_max = {np.sqrt((err**2).mean())/(1.5*U_in)*100:.3f} %")
    print(f"  Linf err / u_max = {np.abs(err).max()/(1.5*U_in)*100:.3f} %")

    assemble_video(frames_dir, OUTDIR / "poiseuille.mp4", fps=25)


# ----------------------------------------------------------------------
# Lab 2: Karman cylinder
# ----------------------------------------------------------------------
def run_karman() -> None:
    print("=" * 60)
    print("Lab 2 — Karman vortex shedding from a cylinder")
    print("=" * 60)
    nx, ny = 600, 80
    D = 10.0
    cx, cy, r = 120.0, ny / 2.0, D / 2.0
    U_in = 0.04
    Re = 100.0

    sim = LBMSimulation(nx=nx, ny=ny, reynolds=Re, flow_speed=U_in,
                        wall_mode="free_slip")
    sim.setup_cylinder_obstacle(cx=cx, cy=cy, r=r)

    St_pred = 0.2660 - 1.0160 / np.sqrt(Re)
    f_pred = St_pred * U_in / D
    T_pred = 1.0 / f_pred
    print(f"Re_D={Re}, U={U_in}, D={D}, omega={sim.omega:.4f}, "
          f"Mach={U_in*np.sqrt(3):.3f}")
    print(f"predicted St={St_pred:.4f}, T={T_pred:.1f} steps")

    frames_dir = OUTDIR / "karman_frames"
    frames_dir.mkdir(exist_ok=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()

    n_warmup = 4000
    n_record = 16000
    snapshot_every = 80

    Cl_history = []
    Cd_history = []

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
            vort = vorticity(sim.u)
            vmax = max(np.abs(vort).max(), 1e-6)
            save_heatmap(
                frames_dir / f"frame_{frame_idx:04d}.png",
                vort, f"Karman  t={sim.time_step}  vorticity",
                vmin=-vmax, vmax=vmax, cmap="RdBu_r",
                obstacle=sim.obstacle,
            )
            frame_idx += 1
        if k % 4000 == 0:
            print(f"  rec   t={sim.time_step}  Cd={d['Cd']:+.3f}  Cl={d['Cl']:+.3f}")
            sys.stdout.flush()

    Cl_history = np.array(Cl_history)
    Cd_history = np.array(Cd_history)

    sig = Cl_history - Cl_history.mean()
    win = np.hanning(len(sig))
    fft = np.fft.rfft(sig * win)
    freqs = np.fft.rfftfreq(len(sig), d=1.0)
    psd = np.abs(fft) ** 2
    psd[0] = 0.0
    f_meas = freqs[int(np.argmax(psd))]
    St_meas = f_meas * D / U_in
    T_meas = 1.0 / f_meas if f_meas > 0 else float("inf")

    print()
    print(f"measured peak f={f_meas:.6e}  T={T_meas:.1f} steps")
    print(f"measured St={St_meas:.4f}   (predicted {St_pred:.4f}, "
          f"{(St_meas/St_pred - 1)*100:+.2f} %)")
    print(f"Cd mean={Cd_history.mean():+.3f}  std={Cd_history.std():.3f}")
    print(f"Cl amplitude (peak-peak)={Cl_history.max() - Cl_history.min():.3f}")

    # Final summary panel.
    vmag = sim.get_velocity_magnitude()
    pflu = sim.get_pressure_fluctuation()
    vort = vorticity(sim.u)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), dpi=100,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})
    titles = [
        ("|u|", vmag, "viridis", 0.0, 1.5 * U_in, axes[0]),
        ("vorticity", vort, "RdBu_r",
         -np.abs(vort).max(), np.abs(vort).max(), axes[1]),
        ("pressure fluctuation", pflu, "RdBu_r",
         -np.abs(pflu).max(), np.abs(pflu).max(), axes[2]),
    ]
    for label, arr, cmap, vmin, vmax, ax in titles:
        im = ax.imshow(arr.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="equal")
        ax.contour(sim.obstacle.T.astype(float), levels=[0.5],
                   colors="black", linewidths=0.8)
        ax.set_title(f"{label}  (t={sim.time_step})")
        fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)

    axes[3].plot(np.arange(len(Cl_history)) + n_warmup, Cl_history, label="Cl")
    axes[3].plot(np.arange(len(Cd_history)) + n_warmup, Cd_history, label="Cd")
    axes[3].set_xlabel("step"); axes[3].grid(True, alpha=0.3)
    axes[3].set_title(f"force coefficients  measured St={St_meas:.4f} "
                      f"(predicted {St_pred:.4f})")
    axes[3].legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "karman_summary.png")
    plt.close(fig)
    print(f"  wrote {OUTDIR/'karman_summary.png'}")

    assemble_video(frames_dir, OUTDIR / "karman.mp4", fps=25)


def main() -> None:
    run_poiseuille()
    print()
    run_karman()


if __name__ == "__main__":
    main()
