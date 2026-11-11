#!/usr/bin/env bash
# Make sure ./class_public carries the created-neutrino patch, then rebuild the binary and the Python wrapper.
# By (Anthropic) Claude and Dan Strano.
#
# Why: setup_and_run.sh / run_lensing.sh skip cloning when class_public already exists. If that tree came from the
# PBH fitting setup (pbh_class.patch), the created-neutrino patch was never applied, and the model run fails.
# The two patches touch different parts of input.c / perturbations.c and apply together cleanly on CLASS 814ce59.
# Usage: bash fix_class_public.sh        (run in the folder holding class_public and created_nu_class.patch)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/class_public"

echo "class_public is at commit $(git rev-parse --short HEAD); modified files:"
git diff --stat | tail -n +1 || true

if grep -q "created_slicing" source/input.c; then
  echo "-> created-neutrino patch already present (latest version, with created_slicing)."
elif grep -q "created_neutrinos" source/input.c; then
  echo "-> an OLDER created-neutrino patch is present; replacing it with the latest."
  had_pbh=0; grep -q "pbh_evaporation_law" source/input.c && had_pbh=1
  git diff > "$HERE/class_public_previous_state.diff"
  echo "   (saved the previous working tree to class_public_previous_state.diff)"
  git checkout -- .                                  # back to clean CLASS 814ce59
  if [ "$had_pbh" = 1 ]; then echo "   re-applying pbh_class.patch"; git apply "$HERE/pbh_class.patch"; fi
  git apply "$HERE/created_nu_class.patch"
else
  echo "-> created-neutrino patch NOT present; applying it now."
  git apply "$HERE/created_nu_class.patch"
fi

make clean >/dev/null
make -j"$(nproc)" class libclass.a
rm -rf python/classy.cpp build/lib.* build/temp.*
python3 setup.py build >/dev/null
echo "Built. Quick check:"
LIB="$(ls -d build/lib.* | tail -1)"
PYTHONPATH="$PWD/$LIB" python3 -c "
import classy
c = classy.Class(); c.set({'output': 'tCl', 'l_max_scalars': 30, 'created_neutrinos': 'no'}); c.compute()
print('patched classy OK:', classy.__file__)"
