# FluidImageBoundarySim

A 2D Lattice Boltzmann Method (D2Q9, BGK) fluid simulator with image-based
obstacle boundaries, validated against analytical reference solutions.

This repository was put through a fluid-mechanics audit; the kernel,
boundary conditions, scaling, and diagnostics have been rebuilt to be
mass- and momentum-conserving, and the simulator now carries reproducible
validation scripts that compare its output against textbook results.

---

## Numerics

* **Lattice**: D2Q9, BGK collision, halfway bounce-back at obstacles.
* **Wall modes**: free-slip (default), no-slip (halfway bounce-back applied
  between collision and streaming, on `fout`), or periodic.
* **Inlet**: Zou/He velocity-Dirichlet with proper non-equilibrium
  bounce-back of the unknown populations
  (`f_unknown = f_opp + f_unknown^eq − f_opp^eq`). Optional smooth ramp
  `U(t) = U_in · (1 − e^{−t/τ})` to suppress the impulsive-start acoustic
  shock.
* **Outlet**: zero-gradient Neumann on the unknown (c_x < 0) populations.
* **Sponge / PML layers**: optional inlet/outlet buffer strips that relax
  the post-collision distributions toward the equilibrium of the target
  far-field state, absorbing acoustic waves before they can reflect off the
  Dirichlet inlet or Neumann outlet. Strength ramps quadratically through
  the buffer.
* **ω clamping**: relaxation parameter is held inside `[0.05, 1.95]` with a
  `RuntimeWarning` if a configuration would push it outside the safe BGK
  range. Outside this band BGK becomes numerically unstable.
* **Reynolds length scale**: derived from the *actual* obstacle each
  setup_* call uses — diameter for a cylinder, hydraulic diameter for a
  rectangle, equivalent diameter `4A/P` for a mask.
* **Drag/lift**: integrated force on the obstacle via momentum exchange
  along fluid→solid links; `Cd`, `Cl` exposed in the diagnostics dict.

---

## Project layout

```
FluidImageBoundarySim/
├── fluid_sim/
│   ├── core/
│   │   ├── lattice.py         # D2Q9 constants + index helpers
│   │   ├── simulation.py      # LBMSimulation (kernel, BCs, diagnostics)
│   │   └── validation.py      # stability + Reynolds checks
│   ├── gui/                   # tkinter GUI (controls.py has a pre-existing
│   │                          # one-line-blob syntax issue, isolated from core)
│   └── utils/                 # obstacle tools, config, file utils
├── validate_poiseuille.py     # plane Poiseuille channel — analytical parabola
├── validate_karman.py         # Kármán shedding — Strouhal vs. Williamson 1996
├── diagnose_channel.py        # localises mass-conservation bugs by wall mode
├── run_channel_cylinder.py    # channel + cylinder, two-row velocity/pressure plot + MP4
├── run_visualisation.py       # batch render Poiseuille + Kármán → PNG + MP4
├── model_lib.py               # legacy procedural API (also fixed)
├── model_run.py               # legacy entry point
└── simulation_config.json     # default parameters
```

Outputs land under `output/` (gitignored).

---

## Quick start

```bash
pip install -r requirements.txt
pip install -e .
```

### Validation: plane Poiseuille flow

Empty channel, no-slip walls, uniform plug inlet. Expected developed
profile is parabolic with `u_max = (3/2) · U_in`.

```bash
python validate_poiseuille.py
```

Reports `u_max`, mass flux, and L2/L∞ error against the analytical
parabola. With the corrected wall BC the kernel hits `u_max = 0.030`
versus an analytical `0.030` (≤ 1.5 % across a 5k-step settling check).

### Validation: Kármán vortex shedding

Free-stream cylinder at `Re_D = 100`. Expected Strouhal from Williamson
(1996):

```
St = 0.2660 − 1.0160 / sqrt(Re_D)    (47 ≤ Re_D ≤ 180)
```

```bash
python validate_karman.py
```

Records `Cl(t)` for many shedding cycles, FFTs to recover the dominant
frequency, and prints measured vs. predicted Strouhal.

### Image + video output

Two-row per-frame plot (velocity colormap on top, pressure fluctuation on
bottom), with an MP4 assembled from the frames.

```bash
python run_channel_cylinder.py     # cylinder + sponge + ramp
python run_visualisation.py        # Poiseuille + Kármán batch
```

Outputs:

```
output/channel_cyl/frames/*.png
output/channel_cyl/channel_cyl_video.mp4
output/channel_cyl/channel_cyl_summary.png
output/poiseuille_summary.png
output/poiseuille.mp4
output/karman_summary.png
output/karman.mp4
```

---

## Programmatic use

```python
from fluid_sim import LBMSimulation

sim = LBMSimulation(nx=600, ny=160, reynolds=60.0, flow_speed=0.02,
                    wall_mode="free_slip")
sim.setup_cylinder_obstacle(cx=180, cy=80, r=15)

# Optional: smooth start-up + non-reflecting buffers.
sim.ramp_steps         = 300
sim.sponge_inlet_width = 60
sim.sponge_outlet_width = 40
sim.sponge_strength    = 0.4

for _ in range(20_000):
    d = sim.step()
    if not d["is_stable"]:
        break

print(d["Cd"], d["Cl"], d["reynolds_realised"])
```

Available setup methods:

| Method | Geometry | Walls |
|---|---|---|
| `setup_cylinder_obstacle(cx, cy, r)` | circle | inherits `wall_mode` |
| `setup_rectangle_obstacle(cx, cy, length, width)` | rectangle | inherits `wall_mode` |
| `setup_channel()` | empty | forces no-slip |
| `setup_channel_with_cylinder(cx, cy, r)` | cylinder in channel | forces no-slip |
| `setup_from_mask(path, scale)` | image-defined | inherits `wall_mode` |

Diagnostics dict from `sim.step()`:

| Key | Meaning |
|---|---|
| `time_step`, `max_velocity`, `max_pressure`, `min_pressure` | self-explanatory |
| `mach_number`, `is_stable` | `Mach < 0.1` and `0 < ω < 2` |
| `omega`, `L_char` | derived from the actual obstacle |
| `reynolds_target`, `reynolds_realised` | input vs. measured |
| `Cd`, `Cl`, `force_x`, `force_y` | momentum-exchange force on obstacle |

---

## What was fixed in the audit

Critical (kernel-level):

* Top/bottom wall BCs were mislabelled comments — code was actually
  clobbering the inlet/outlet columns. Replaced with a real free-slip /
  no-slip / periodic switch.
* No-slip implementation was applied at the wrong stage and on `fin`
  pulled from the wrong y-row, leaking ~30 % of mass per step. Now
  applied between collision and streaming, on `fout`, at the correct
  `y` — verified mass-conserving across all three wall modes.
* Zou/He inlet had a self-cancelling correction that reduced to a plain
  equilibrium copy. Now the proper non-equilibrium bounce-back form.
* Inlet velocity drifted because the OO refactor stored the macroscopic
  field in `self.u`; the prescribed inlet now lives in `self.vel` and is
  immutable through the run.
* `setup_cylinder_obstacle` actually generates a circle (was a square).
* Reynolds length scale follows the obstacle (was hard-coded `ny/9`).

Significant:

* `mps_to_lu` validates `dx`, `dt > 0` and is honest about the
  conversion. The default lattice-unit usage is unchanged.
* Pressure plot range is the symmetric fluctuation about the mean
  (fixed `[0.001, 0.006]` window was outside the data range).
* ρ-floor in the macroscopic-velocity divide.
* Drag/lift via momentum exchange.
* `ω` clamped to `[0.05, 1.95]` with a `RuntimeWarning`.
* Smooth inlet ramp and sponge layers added so that visualisation runs
  don't excite the impulsive-start acoustic mode.
* Symmetric morphological `binary_opening` for `smooth_corners`
  (previous structuring-element trick was anisotropic).
* `rotate_obstacle` rebuilt on `scipy.ndimage.rotate` (no holes).

The old `model_lib.py` legacy API has the same kernel-level fixes so that
`model_run.py` produces correct physics.

---

## Requirements

```
numpy >= 1.19
matplotlib >= 3.3
pillow >= 8.0
scipy >= 1.6
opencv-python >= 4.5     # for MP4 assembly
```

Python ≥ 3.8.

---

## License

MIT.
