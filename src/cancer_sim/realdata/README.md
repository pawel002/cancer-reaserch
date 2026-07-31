# Real-patient ground truth (`cancer_sim.realdata`)

Turns the released GliODIL glioma dataset
([Balcerak et al., *Nat. Commun.* 16, 5982, 2025](https://doi.org/10.1038/s41467-025-60366-4))
into the tumour-mass-versus-time trajectories the reduced surrogates consume,
together with the spatial fields the figures are drawn from.

```
real patient (MRI)                     patient.load_geometry
  t1_wm / t1_gm / t1_csf   ─────────▶  anatomy (WM, GM, brain mask)
  segm  (pre-op BraTS)     ─────────▶  tumour core + FLAIR masks
                                       initial cell density u(x,0)
                                       invasiveness  L = sqrt(D/f)
  segm_rec (follow-up)     ─────────▶  recurrence mask (held out)

                                       patient.beam_field
                                  ───▶ conformal PTV per beam configuration

                                       cohort.calibrate_growth   (1 GPU sim)
                                  ───▶ (Dw, f) fixed to a per-patient burden

                                       fk_rt_gpu.simulate        (5 GPU sims)
                                  ───▶ mass(t), U(t), W_eff(t), cross-sections
```

Modules: `patient.py` (geometry, initial condition, beams), `fk_rt_gpu.py` (the
GPU forward model and its NumPy reference), `cohort.py` (screening, growth
calibration, per-patient driver). The older `fk3d.py` / `fk3d_gpu.py` /
`gliodil_io.py` / `to_surrogate.py` are the previous generic-seed bridge, kept
for reference.

## What is measured versus what is assumed

| quantity | source | real? |
| --- | --- | --- |
| brain anatomy, WM/GM anisotropy | patient tissue maps | measured |
| tumour location, size, shape | patient `segm` | measured |
| initial cell density `u(x,0)` | two-threshold reconstruction of `segm` | derived from measurement |
| invasiveness `L = sqrt(D/f)` | the patient's own edema-rim thickness | derived from measurement |
| irradiation field | conformal PTV grown from the real tumour mask | clinical practice, not this patient's plan |
| absolute growth speed `v = 2 sqrt(D f)` | drawn per patient, deterministic | **assumed** — unidentifiable from one timepoint |
| radiosensitivity `gamma`, hypoxia `h` | drawn per patient, deterministic | **assumed** — no dose-response data in the dataset |

Only the last two rows are assumptions, and both are *hidden from every
surrogate*: they are exactly what the assimilation step has to recover. The
released dataset has two static timepoints and no dose-over-time signal, so a
dense response curve cannot be measured; it has to be simulated. What the
benchmark therefore establishes is a comparison of **reduced-model forms on
patient-specific geometry**, not a patient-calibrated clinical prediction.

## Initial cell density

The reconstruction reproduces *both* of the patient's measured contours rather
than fitting a generic blob: `u = 0.50` on the enhancing (T1Gd) surface and
`u = 0.25` on the FLAIR surface, using GliODIL's own density↔label convention
(`synthetic_generator.py:63`). Between the two surfaces the density is
log-linear in the *local* rim coordinate, so a tumour with a thin rim on one
side and a thick rim on the other is honoured. Measured agreement with the
segmentations is exact to the resampling tolerance: median Dice 1.000 at both
iso-levels, worst case 0.980.

## Growth calibration

The Fisher–Kolmogorov equation is exactly invariant under
`(D, f, t) → (cD, cf, t/c)`. One reference simulation per patient therefore
gives the whole one-parameter family, and the scale `c` is chosen so the
*untreated* tumour reaches a per-patient target burden at the horizon. The shape
parameter `L = sqrt(D/f)` stays imaging-derived, so infiltrative and nodular
tumours remain genuinely different; only the clock is normalised.

## Radiotherapy graft

GliODIL infers growth; it models no radiotherapy (grepping its source for
"radiation", "dose" or "fraction" returns nothing). The dose response here is
the spatial analogue of the paper's reduced two-state model, added to GliODIL's
kernel:

```
dA/dt = div(D grad A) + f A (1 - A) - gamma (1 - h A) Z A
dZ/dt = -Z / tau + U(t) beam(x)
```

The `(1 - h A)` factor makes dense tumour regions harder to kill — the
microenvironment channel `M(u, x, t)` of the paper's general model
(Eq. `pde_general`). It is the one deliberate addition, and it is what makes the
reduced 0-D response genuinely under-determined; set `hypoxia = 0` to recover
the plain graft.

The GPU solver is validated against a pure-NumPy float64 reference to machine
precision by `experiments/validate_solver.py`: max relative difference 2.5e-16
in the tumour mass and 3.6e-16 in the damage field with radiotherapy on, and
1.9e-16 with it off. The same script checks the time-rescaling identity the
growth calibration relies on (agreement to 2.6e-05, limited by the output-grid
interpolation, not the solver).

## Beam configurations

`narrow_centered` and `strong_shift` are tuned to deliver almost the same total
dose to the tumour (cohort median coverage 0.42 versus 0.41) with completely
different spatial patterns. A reduced 0-D surrogate sees an identical `U(t)` and
a near-identical effective coverage for the two, so any difference in their
outcome is by construction unresolvable without a closure term.

## Reproduce

```bash
PYTHONPATH=src python experiments/validate_solver.py            # GPU vs NumPy
PYTHONPATH=src python experiments/gen_cohort.py --gpus 4,5,6,7
```

117 of the 152 released patients pass screening (`cohort.screen_cohort`); the
rest are rejected for a degenerate tumour segmentation or an unusable tissue
map. Generating the whole cohort — 6 forward simulations each at 160³ — takes
about 20 GPU-minutes.
