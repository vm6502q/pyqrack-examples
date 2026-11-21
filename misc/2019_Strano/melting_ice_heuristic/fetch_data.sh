#!/usr/bin/env bash
# Fetch the public data used by melting_ice_fit.py into ./data (DESI DR2 BAO via Cobaya's bao_data; Pantheon+).
set -euo pipefail
mkdir -p data && cd data
[ -d bao_data ] || git clone --depth 1 https://github.com/CobayaSampler/bao_data.git
if [ ! -d pantheon ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/PantheonPlusSH0ES/DataRelease.git pantheon
  (cd pantheon && git sparse-checkout set "Pantheon+_Data/4_DISTANCES_AND_COVAR")
fi
echo "data ready"
