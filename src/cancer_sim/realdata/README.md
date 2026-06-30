# Real-patient bridge: GliODIL → tumour-mass-vs-time surrogate

This package hooks the **real glioma dataset** released with GliODIL
([Balcerak et al., *Nat. Commun.* 16, 5982, 2025](https://doi.org/10.1038/s41467-025-60366-4))
into the ODE / NODE / PI-NODE radiotherapy-response surrogates in this repo.

## Why a bridge is needed (the modality gap)

The two codebases sit at different altitudes, so the real data cannot be fed to
the surrogate as-is:

| | This repo's surrogate | GliODIL | Real GliODIL dataset |
|---|---|---|---|
| Object | 0-D scalar **mass vs time** | 3-D spatial PDE inverse problem | 3-D MRI/PET volumes |
| Input schema | `time, normalized_mass, U_t, W_eff_damage` (~801 pts) | 1 MRI/PET snapshot | seg + WM + GM (+PET) |
| Time | dense trajectory, 2 dose pulses | internal growth only | **2 static timepoints** (pre-op, follow-up) |
| Radiotherapy | explicit dose schedule | **none** (static planning only) | none (just imaging) |

So "running the surrogate on real data" requires converting *static 3-D spatial
snapshots* into a *temporal mass-vs-time RT-response curve*. That conversion is
what this package does.

## The faithful pipeline

```
real patient (MRI/PET)
   │   GliODIL inference  (GPU; run_GliODIL.sh)
   ▼
inferred growth params (Dw, rho, Dw_ratio, focal seed) + WM/GM anatomy
   │   fk3d.simulate    — 3-D Fisher-Kolmogorov forward sim WITH a grafted
   ▼                      radiotherapy damage field, integrated spatially
normalized_mass(t), U_t(t), W_eff(t)
   │   to_surrogate.write_dataset + build_noisy_ensemble
   ▼
datasets/<name>_{full,train,test}.csv  +  noisy_observations/<name>/
   │   run_ensemble.py --cases <name>   (unchanged)
   ▼
ODE / NODE / PI-NODE trained & forecast on a real-anatomy patient
```

**Why the 3-D PDE (not the reduced ODE) makes the ground truth.** The surrogates
exist to bridge the gap between a *spatial* tumour and its *reduced* 0-D model —
that discrepancy is exactly what the PI-NODE residual learns. If we generated the
mass curve with the 2-state ODE itself, the surrogate would trivially fit its own
generator and the experiment would be meaningless. So the bridge keeps a genuine
3-D anisotropic reaction–diffusion PDE (GliODIL's own kernel) as the truth and
reduces it to a mass curve, preserving the paper's premise on real anatomy.

**The radiotherapy graft.** The reduced model's RT physics

```
dy/dt = rho·y(1-y/K) - gamma·z·y - mu·y ,   dz/dt = -z/tau + U(t)
```

is lifted to a spatial analogue: a tumour field `A(x,t)` with a kill term
`-gamma·Z·A`, and a damage field `Z(x,t)` driven by a spatial beam mask,
`dZ/dt = -Z/tau + U(t)·beam(x)`. Spatial integral → `normalized_mass(t)`;
`W_eff(t)` is the tumour-mass-weighted mean damage (0-D reduction of `Z`).

## Honest caveats

- **RT response is synthetic.** The real dataset has no dose-over-time signal,
  so the *growth* (geometry, WM/GM-anisotropic diffusion, proliferation rate) is
  patient-real but the *radiotherapy response* is a physically-motivated graft.
- **Two real timepoints only.** GliODIL infers growth from the pre-op snapshot;
  the follow-up is for recurrence evaluation, not a dense response curve. The
  dense trajectory the surrogate needs is produced by the forward sim, not
  measured.

## Usage

### 0. (one-off) download the dataset — run this yourself

```bash
# ~1.2 GB Git-LFS zip; public (MIT). Either:
.venv/bin/python -m pip install -U "huggingface_hub[cli]"
hf download m1balcerak/GliODIL data_GliODIL_essential.zip \
    --repo-type dataset --local-dir ./real_data
unzip ./real_data/data_GliODIL_essential.zip -d ./real_data/

# …or with plain curl (no auth needed):
curl -L -o data_GliODIL_essential.zip \
  "https://huggingface.co/datasets/m1balcerak/GliODIL/resolve/main/data_GliODIL_essential.zip?download=true"
unzip data_GliODIL_essential.zip -d ./real_data/
```

### 1. CPU demo — verify the whole chain without GPU or download

Fabricates a patient from the WM/GM atlas shipped in `GliODIL/.../precomputed/`:

```bash
.venv/bin/python experiments/build_real_dataset.py --demo --grid 96
.venv/bin/python experiments/run_ensemble.py --cases gliodil_demo \
    --out results/real_data/ensemble
```

### 2. Real dataset layout

The downloaded zip extracts to `data_GliODIL_essential/data_NNN/` (152 patients),
each with **240×240×155** NIfTI volumes:

| file | role |
|---|---|
| `t1_wm.nii.gz`, `t1_gm.nii.gz`, `t1_csf.nii.gz` | tissue probability maps |
| `segm.nii.gz` | pre-op tumour segmentation (BraTS labels) |
| `segm_rec.nii.gz` | recurrence/follow-up segmentation |
| `FET.nii.gz` | FET-PET (subset of patients) |

`load_patient` reads `t1_wm`/`t1_gm` (skips macOS `._` forks), places the focal
seed at the **pre-op tumour centroid** (`tumor_centroid`, from `segm`), and uses
`coeffs.npy` for growth parameters if a GliODIL run produced one.

### 3a. Local run, no GPU (real anatomy, sampled growth)

GliODIL's parameter inference needs a GPU (>18 GB; its TensorFlow stack also does
not support this repo's Python 3.14), so it cannot run here. But the bridge runs
fully on CPU using the **real anatomy + real tumour location**, with growth
parameters either fixed (`--Dw`/`--f`) or drawn per patient from GliODIL's
published ranges (`--sample-params`) to emulate inter-patient heterogeneity:

```bash
# cohort of the first 5 patients, heterogeneous growth, ~2 min build on CPU
.venv/bin/python experiments/build_real_dataset.py \
    --patient-root real_data/data_GliODIL_essential --limit 5 --grid 80 --sample-params
.venv/bin/python experiments/run_ensemble.py \
    --cases gliodil_data_001 gliodil_data_013 gliodil_data_020 gliodil_data_030 gliodil_data_034 \
    --out results/real_data/cohort
```

> Caveat: without GliODIL inference the **growth** is real-anatomy but
> sampled/literature-range, not patient-inferred. With identical params the
> reduced mass curves come out nearly identical across patients (they differ only
> in anatomy+seed); `--sample-params` restores the variability a real cohort
> study needs until the GPU step is available.

### 3b. Full faithful run, with GPU

```bash
# (a) GliODIL inference -> inferred field + coeffs.npy in the patient dir
./GliODIL/run_GliODIL.sh /path/to/data_GliODIL_essential/data_001

# (b) bridge (coeffs.npy now drives patient-specific growth) + (c) surrogates
.venv/bin/python experiments/build_real_dataset.py \
    --patient-dir /path/to/data_GliODIL_essential/data_001 --grid 128
.venv/bin/python experiments/run_ensemble.py --cases gliodil_data_001
```

> `nibabel` (`.venv/bin/python -m pip install nibabel`) is needed to read the
> real `.nii.gz` volumes; the demo and `.npy` paths don't need it.

## Tuning the regime (`FK3DConfig` in `fk3d.py`)

- **Growth aggressiveness:** `f` (proliferation), `Dw` (diffusion), `Dw_ratio`.
  These come from GliODIL per patient; raising them gives larger pre-dose growth
  and steeper regrowth.
- **RT response depth:** `gamma` (radiosensitivity), `dose_amplitude`,
  `dose_duration`, `damage_decay_time` (tau), and the spatial beam
  (`beam_center`, `beam_radius_frac`).
- **Schedule / horizon:** `dose_times`, `T`, `dt_output`. Defaults reproduce the
  surrogate config (T=80, doses at 15 & 45, assimilation window 15–35) so the
  emitted patient is a drop-in case.
- **Speed:** `--grid` downsamples anatomy (96³ ≈ 10 s/patient on CPU,
  192³ ≈ 100 s). `substeps` controls FK stability (auto-checked).
```
