"""Central configuration for the radiotherapy surrogate experiments.

All settings that the paper refers to live here so the package behaves as a
single, easy-to-manipulate playground.  Values reproduce the paper defaults
(Tables "Assimilation settings" and "Neural surrogate architecture").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets"
NOISY_OBS_DIR = ROOT / "noisy_observations"
RESULTS_DIR = ROOT / "results"

CASES: List[str] = [
    "full_cover",
    "narrow_centered",
    "slight_shift_full_beam",
    "strong_shift_full_beam",
]

# ----------------------------------------------------------------------------
# Assimilation / forecasting window  (paper Table "Assimilation settings")
# ----------------------------------------------------------------------------
TRAIN_SAMPLE_START = 15.0
TRAIN_SAMPLE_END = 35.0
N_TRAIN_POINTS = 20          # N_fit in the paper
N_ENSEMBLE_RUNS = 20         # noisy realisations actually shipped with the repo
NOISE_STD = 0.02             # Gaussian sigma = NOISE_STD * y_i
NOISE_RELATIVE = True
NOISE_SEED = 202601

# Time normalisation used by the neural models. Training (15..35) and
# prediction (15..80) must share the same denominator.
TIME_NORM_DENOM = 80.0


@dataclass
class PDEInit:
    """Mechanistic parameters used to *initialise* the PI-NODE (paper Table).

    These are the ground-truth PDE coefficients, NOT an ODE fit. They remain
    trainable; this object only sets the starting point.
    """

    rho: float = 0.052     # tumour proliferation rate
    beta: float = 1.25     # radiosensitivity coefficient -> gamma init
    K: float = 1.0         # carrying capacity
    mu: float = 1e-4       # background decay
    tau: float = 3.0       # damage relaxation time (damage_decay_time)


PDE_INIT = PDEInit()


@dataclass
class TrainConfig:
    """Hyper-parameters for the neural surrogates (paper architecture table)."""

    hidden: int = 32
    seed: int = 123

    # Learning rates (paper: NODE 1e-3, PI-NODE 2e-4 nominal; repo used the
    # values below, kept as defaults for reproducibility).
    lr_node: float = 2e-3
    lr_pinode: float = 1e-3

    # NODE: single joint phase.
    node_epochs: int = 250

    # PI-NODE alternating schedule: (theta, phi, joint).
    pinode_theta_epochs: int = 0
    pinode_phi_epochs: int = 600
    pinode_joint_epochs: int = 150

    # Physics/residual balance of Eq. (pinode_weighted):
    #   dy = omega * f_RT^(y) + s_r * g_psi
    omega: float = 0.05            # physics weight on mechanistic tumour term
    s_r: float = 0.10              # neural residual scale
    l2_residual: float = 1e-6      # lambda in R(g) = ||g||^2

    # Optional exponential recency weighting of assimilation samples.
    use_time_weights: bool = False
    weight_last_to_first: float = 10.0

    # Gradient clipping (stabilises the stiff RK4 unrolled graph).
    grad_clip: float = 2.0


TRAIN = TrainConfig()


@dataclass
class Ablation:
    name: str
    omega: float
    s_r: float
    label: str = ""


# Physics/residual balance ablation (paper Fig. "uncertainty").
ABLATIONS: List[Ablation] = [
    Ablation("A_current_phys005_res010", 0.05, 0.10, r"$\omega=0.05$, $s_r=0.10$"),
    Ablation("B_weak_phys001_res020", 0.01, 0.20, r"$\omega=0.01$, $s_r=0.20$"),
    Ablation("C_balanced_phys010_res005", 0.10, 0.05, r"$\omega=0.10$, $s_r=0.05$"),
    Ablation("D_strong_phys050_res002", 0.50, 0.02, r"$\omega=0.50$, $s_r=0.02$"),
]
