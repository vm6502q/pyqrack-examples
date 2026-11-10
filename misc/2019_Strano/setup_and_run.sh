#!/usr/bin/env bash
# Created neutrinos + ghost coexistence gas (Strano model) vs Planck 2018.
# By (Anthropic) Claude and Dan Strano.
# Builds CLASS v3.4.1 (commit 814ce59) + created_nu_class.patch, installs Cobaya and the Planck 2018 low-l and
# plik-lite likelihoods, then runs the T_c x m_lightest grid. ~5 min per grid point on 2 cores.
# Usage: bash setup_and_run.sh [extra run_grid.py options, e.g. --Tc 40,50,60,70,80 --m1 0,0.02]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
python3 -m venv .venv && . .venv/bin/activate
pip install -q --upgrade pip && pip install -q numpy scipy cobaya
if [ ! -d class_public ]; then
  git clone https://github.com/lesgourg/class_public.git
  (cd class_public && git checkout 814ce59 && git apply "$HERE/created_nu_class.patch")
fi
(cd class_public && make -j"$(nproc)" class)
cobaya-install planck_2018_lowl.TT planck_2018_lowl.EE planck_2018_highl_plik.TTTEEE_lite_native --packages-path packages
python3 run_grid.py class_public/class packages --out grid_out "$@"
