#!/usr/bin/env bash
set -euo pipefail

DEST_ROOT=${1:?destination root is required}
RECORD=10044268
mkdir -p "$DEST_ROOT"

download_one() {
  local name=$1 size=$2 md5=$3
  local final="$DEST_ROOT/$name" partial="$DEST_ROOT/$name.partial"
  if [[ -f "$final" ]] && [[ $(stat -c %s "$final") == "$size" ]] \
    && [[ $(md5sum "$final" | cut -d' ' -f1) == "$md5" ]]; then
    sha256sum "$final"
    return
  fi
  curl -fL --retry 5 --retry-delay 5 --continue-at - \
    "https://zenodo.org/records/$RECORD/files/$name?download=1" -o "$partial"
  [[ $(stat -c %s "$partial") == "$size" ]]
  [[ $(md5sum "$partial" | cut -d' ' -f1) == "$md5" ]]
  mv "$partial" "$final"
  sha256sum "$final"
}

download_one ReplogleWeissman2022_rpe1.h5ad 1236886900 cc7f1ec50aeb3a3e1b4a6cfa713d80fa
download_one TianKampmann2019_iPSC.h5ad 350848489 8279484264b513fd0419dacdc639ecef
