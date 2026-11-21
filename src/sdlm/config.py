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
