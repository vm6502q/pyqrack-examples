#!/usr/bin/env bash
# Planck 2018 lensing test of the created-neutrino (Strano) model, on your machine.
# By (Anthropic) Claude and Dan Strano.
#
# Prerequisites: this folder (patch + scripts), git, a C compiler, python3 >= 3.9.
# Re-uses ./class_public and ./.venv from setup_and_run.sh if present.
# Usage: bash run_lensing.sh [quick|refit|all]
#   quick : evaluate LCDM and the model at the existing best fits (cobaya_out/results.json) with lensing added. ~4-5 model
#           evaluations, so tens of minutes.
#   refit : re-minimize LCDM and the model (template iteration) with lensing included in the fit. Hours.
#   all   : both (default).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
MODE="${1:-all}"

[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q numpy scipy cython setuptools wheel cobaya

if [ ! -d class_public ]; then
  git clone https://github.com/lesgourg/class_public.git
  (cd class_public && git checkout 814ce59 && git apply "$HERE/created_nu_class.patch")
fi
# C binary (template ratios, file-pipeline checks) and the Python wrapper in the layout Cobaya looks for
(cd class_public && make -j"$(nproc)" class libclass.a && rm -f python/classy.cpp && python3 setup.py build >/dev/null)
CLASS_DIR="$HERE/class_public"

cobaya-install planck_2018_lowl.TT planck_2018_lowl.EE planck_2018_highl_plik.TTTEEE_lite_native \
    planck_2018_lensing.native --packages-path packages

mkdir -p cobaya_out
if [ ! -f cobaya_out/results.json ] && [ -f cobaya_results.json ]; then cp cobaya_results.json cobaya_out/results.json; fi

if [ "$MODE" = "quick" ] || [ "$MODE" = "all" ]; then
  python3 lensing_test.py --packages packages --class-path "$CLASS_DIR"            | tee cobaya_out/lensing_test_native.txt
  python3 lensing_test.py --packages packages --class-path "$CLASS_DIR" --marged   | tee cobaya_out/lensing_test_marged.txt
fi
if [ "$MODE" = "refit" ] || [ "$MODE" = "all" ]; then
  python3 cobaya_fit.py lcdm  --packages packages --class-path "$CLASS_DIR" --lensing          | tee cobaya_out/refit_lens_lcdm.txt
  python3 cobaya_fit.py model --packages packages --class-path "$CLASS_DIR" --lensing \
      --class-bin "$CLASS_DIR/class" --dof 1 --iters 2                                         | tee cobaya_out/refit_lens_model.txt
fi
echo "Done. Results: cobaya_out/lensing_test_*.json, cobaya_out/results.json (keys *_lens*)."
