"""Real-patient bridge: GliODIL anatomy/parameters -> tumour-mass-vs-time curves.

This subpackage turns a GliODIL glioma patient (real MRI/PET-derived anatomy and
inferred Fisher-Kolmogorov growth parameters) into the 0-D scalar
``normalized_mass(t)`` radiotherapy-response trajectory that the ODE / NODE /
PI-NODE surrogates in this repository consume.

The bridge is necessary because the two codebases live at different altitudes:

* **This repo's surrogates** ingest a dense *temporal* curve -- tumour mass vs
  time under a radiotherapy dose schedule (schema ``time, normalized_mass, U_t,
  W_eff_damage``), assimilate an early window, and forecast the post-dose
  regrowth.
* **GliODIL** is a 3-D *spatial* PDE-constrained inverse problem: from a single
  MRI/PET snapshot it infers patient-specific Fisher-Kolmogorov parameters
  (water diffusion ``Dw``, proliferation ``f=rho``, WM/GM anisotropy ratio,
  focal seed) on real brain anatomy.  It has **no radiotherapy dynamics**.

The faithful hook-up (chosen design):

1. Run GliODIL on a patient (GPU) -> inferred ``(Dw, rho, Dw_ratio, seed)`` and
   the WM/GM tissue maps (:mod:`gliodil_io`).
2. Forward-simulate the patient's 3-D Fisher-Kolmogorov PDE **with a grafted
   radiotherapy damage field** -- the spatial analogue of the reduced 2-state
   ODE -- and spatially integrate to a mass curve (:mod:`fk3d`).
3. Emit the surrogate dataset CSVs + noisy observation ensemble
   (:mod:`to_surrogate`), after which the existing experiment runners train on
   the patient with ``--cases <patient>`` unchanged.

The 3-D PDE (not the reduced ODE) generates the ground truth on purpose: the
gap between the spatial truth and the 2-state reduction is exactly what the
PI-NODE residual exists to learn, so the paper's premise is preserved -- now
grounded in real patient anatomy instead of a synthetic 2-D field.

**Honest caveat.** The real GliODIL dataset provides only two static spatial
snapshots per patient (pre-op, follow-up) and no dose-over-time signal, so the
radiotherapy *response* is necessarily synthetic (a physically-motivated graft),
while the *growth* (geometry, diffusion anisotropy, proliferation rate) is
patient-real.
"""

from . import fk3d, gliodil_io, to_surrogate  # noqa: F401
