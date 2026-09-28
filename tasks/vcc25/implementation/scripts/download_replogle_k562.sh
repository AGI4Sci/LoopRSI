#!/usr/bin/env bash
set -euo pipefail

DEST_ROOT=${1:?destination root is required}
NAME=ReplogleWeissman2022_K562_essential.h5ad
ZENODO_RECORD=7416068
URL="https://zenodo.org/records/$ZENODO_RECORD/files/$NAME?download=1"
EXPECTED_SIZE=1546729675
EXPECTED_MD5=d8cba17576d1a8afc0f7d71b79cad0f7
PARTIAL="$DEST_ROOT/$NAME.partial"
FINAL="$DEST_ROOT/$NAME"

mkdir -p "$DEST_ROOT"
if [[ -f "$FINAL" ]] && [[ $(stat -c %s "$FINAL") == "$EXPECTED_SIZE" ]] \
  && [[ $(md5sum "$FINAL" | cut -d' ' -f1) == "$EXPECTED_MD5" ]]; then
  sha256sum "$FINAL"
  exit 0
fi

curl -fL --retry 5 --retry-delay 5 --continue-at - "$URL" -o "$PARTIAL"
[[ $(stat -c %s "$PARTIAL") == "$EXPECTED_SIZE" ]]
[[ $(md5sum "$PARTIAL" | cut -d' ' -f1) == "$EXPECTED_MD5" ]]
mv "$PARTIAL" "$FINAL"
sha256sum "$FINAL"
