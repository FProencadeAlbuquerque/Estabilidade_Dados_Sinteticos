"""Shared CLI helpers: make `src/` importable and parse the common arguments."""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from sdlm import config  # noqa: E402


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", default="ieee14", choices=list(config.DATASETS))
    p.add_argument("--models", nargs="+", default=list(config.MODELS), choices=list(config.MODELS))
    return p
