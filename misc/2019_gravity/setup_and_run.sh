#!/usr/bin/env bash
# One-shot setup + run for pbh_fit.py (Strano black-hole emission law vs LCDM).
# By (Anthropic) Claude and Dan Strano
#
# Builds CLASS v3.4.1 with pbh_class.patch, installs its Python wrapper (classy) and Cobaya into a venv,
# downloads Planck 2018 / DESI DR2 / Pantheon+ likelihood data, then runs: LCDM fit, profile over t_evap, report.
# Usage: bash setup_and_run.sh [JOBS]      (JOBS parallel profile points; default 1)
# Expect hours: each minimization is roughly 500-3000 CLASS calls of a few seconds each.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JOBS="${1:-1}"
cd "$HERE"

python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q numpy scipy cython setuptools wheel cobaya

if [ ! -d class_public ]; then
  git clone https://github.com/lesgourg/class_public.git
  (cd class_public && git checkout 814ce59 && git apply "$HERE/pbh_class.patch")
fi
(cd class_public && make -j"$(nproc)" class && make classy)
python3 -c "import classy; print('classy:', classy.__file__)"

python3 pbh_fit.py install --packages packages

CORES="$(nproc)"
export OMP_NUM_THREADS="$(( CORES / JOBS > 0 ? CORES / JOBS : 1 ))"
python3 pbh_fit.py lcdm --packages packages
python3 pbh_fit.py profile --packages packages --jobs "$JOBS"
python3 pbh_fit.py report | tee pbh_out/report.txt
echo "Optional: python3 pbh_fit.py free   (global fit with t_evap free), then report again."
