"""Central configuration: paths, datasets and hyperparameters (Table A.6 of the paper)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

SEED = 42
TARGET = "LoadMargin"
TEST_SIZE = 0.2          # 8,000 train / 2,000 test for N_s = 10,000
N_SYNTH = 10_000         # synthetic samples drawn from each generator

# name -> csv file. "ieee14" is the dataset used in the paper.
DATASETS = {
    "ieee14": DATA / "ieee14" / "Bus14.csv",
    "ieee14_contingency": DATA / "ieee14" / "Bus14Contingency.csv",
    "ieee118_bus17": DATA / "ieee118" / "LPM_Bus17_118.csv",
    "ieee118_bus45": DATA / "ieee118" / "LPM_Bus45_118.csv",
}

# A variable is "informative" when its coefficient of variation (std / |mean|) is at
# least CV_MIN. Voltage magnitudes held at their set-points vary by less than 0.1 % of
# the nominal value (below PMU magnitude accuracy); z-scoring such channels amplifies
# numerical noise, so the *_informative tables leave them out (the target is kept).
CV_MIN = 1e-3

# Features shown in the missing-value figures and in the KDE comparison, per dataset.
# IEEE 14: those of the paper. IEEE 118: a PMU angle and the loads next to the target
# bus, and the load margin (the PMU voltage magnitudes are quasi-constant, see CV_MIN).
_F14 = dict(missing=["V9", "P1"], kde=["V9", "Theta2", "Q4", "LoadMargin"])
FEATURES = {
    "ieee14": _F14,
    "ieee14_contingency": _F14,
    "ieee118_bus17": dict(missing=["Theta18", "Q17"], kde=["Theta18", "P17", "Q17", "LoadMargin"]),
    "ieee118_bus45": dict(missing=["Theta56", "Q45"], kde=["Theta56", "P45", "Q45", "LoadMargin"]),
}

# CTGAN training configuration (paper, Appendix A). Values not listed here are the
# library defaults, which coincide with the remaining rows of the table:
# Adam(lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6), gradient penalty = 10.
CTGAN_PARAMS = dict(
    epochs=1600,
    batch_size=64,
    embedding_dim=128,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    generator_lr=2e-4,
    discriminator_lr=2e-4,
    generator_decay=1e-6,
    discriminator_decay=1e-6,
    discriminator_steps=1,
    pac=1,
)

# The paper does not report TVAE / Gaussian-copula settings; SDV defaults are used.
TVAE_PARAMS = dict(epochs=300, batch_size=500)

MODELS = ("ctgan", "tvae", "copula")
MODEL_LABELS = {"ctgan": "CTGAN", "tvae": "TVAE", "copula": "Gaussian copula"}

# Missing-value experiment (Section 4.1)
MISSING_RATES = (0.05, 0.10, 0.20, 0.30)

# Random-forest reference learner for TRTR / TSTR (Section 4.4)
RF_PARAMS = dict(n_estimators=200, random_state=SEED, n_jobs=-1)


def out_dir(dataset: str) -> Path:
    d = RESULTS / dataset
    for sub in ("synthetic", "tables", "figures", "models"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d
