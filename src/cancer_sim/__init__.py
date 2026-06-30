"""Physics-Informed Closure Learning for Reduced Radiotherapy Tumor Dynamics.

A small, modular playground reproducing the surrogate-model experiments from
the paper (latex/main.tex):

    * mechanistic 2-state ODE surrogate          -> cancer_sim.mechanistic
    * data-driven NODE / physics-informed PI-NODE -> cancer_sim.surrogates
    * batched ensemble training                   -> cancer_sim.training
    * data loading + noisy observation ensembles  -> cancer_sim.data
    * metrics / plotting                          -> cancer_sim.metrics, .plotting

Experiment entry points live in ``experiments/``.
"""

from . import config, data, mechanistic, metrics, plotting, surrogates, training  # noqa: F401

__all__ = ["config", "data", "mechanistic", "metrics", "plotting", "surrogates", "training"]
