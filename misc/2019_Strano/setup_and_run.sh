#!/usr/bin/env bash
# One-shot setup + run for pbh_fit.py (Strano black-hole emission law vs LCDM).
# By (Anthropic) Claude and Dan Strano
#
# Builds CLASS v3.4.1 with pbh_class.patch and hi_class v3.4.0.1, installs its Python wrapper (classy) and Cobaya into a venv,
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

# hi_class (scalar-tensor CLASS) for the dilaton / Brans-Dicke channel; installs as 'hiclassy', so it does not
# collide with the patched 'classy' above.
if [ ! -d hi_class_public ]; then
  git clone https://github.com/miguelzuma/hi_class_public.git
  (cd hi_class_public && git checkout ece81a5)
fi
(cd hi_class_public && make -j"$(nproc)" class && pip install -q .)
python3 -c "import hiclassy; print('hiclassy:', hiclassy.__file__)"

python3 pbh_fit.py install --packages packages

CORES="$(nproc)"
export OMP_NUM_THREADS="$(( CORES / JOBS > 0 ? CORES / JOBS : 1 ))"
python3 pbh_fit.py lcdm --packages packages
python3 pbh_fit.py ivacprofile --jobs 2
python3 pbh_fit.py ivac
python3 pbh_fit.py report | tee pbh_out/report.txt
