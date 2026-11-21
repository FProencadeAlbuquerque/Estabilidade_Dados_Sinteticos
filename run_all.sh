#!/usr/bin/env bash
# Full pipeline for one dataset (default: ieee14, the case study of the paper).
# CTGAN training with the paper configuration takes ~1 h on 2 CPU cores.
set -euo pipefail
DATASET="${1:-ieee14}"
cd "$(dirname "$0")"
python scripts/01_train_generators.py --dataset "$DATASET"
python scripts/02_fidelity.py         --dataset "$DATASET"
python scripts/03_tstr.py             --dataset "$DATASET"
python scripts/04_privacy.py          --dataset "$DATASET"
python scripts/05_missing_values.py   --dataset "$DATASET"
python scripts/06_figures.py          --dataset "$DATASET"
if [ -f "paper/reference_${DATASET}.json" ]; then
  python scripts/07_report.py         --dataset "$DATASET"
fi
