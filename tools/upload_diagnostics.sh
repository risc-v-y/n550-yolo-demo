#!/usr/bin/env bash
# Upload one saved diagnostic run using a separate Git work directory.
set -euo pipefail
RUN_DIR="$(realpath -- "${1:?Usage: bash tools/upload_diagnostics.sh RESULT_DIR}")"
UPLOAD_DIR="$(mktemp -d)"
(
  set -e
  test -f "$RUN_DIR/report.json"
  git -C "$UPLOAD_DIR" init -q
  git -C "$UPLOAD_DIR" config user.name "N550 host diagnostics"
  git -C "$UPLOAD_DIR" config user.email "n550-host@users.noreply.github.com"
  git -C "$UPLOAD_DIR" remote add origin git@github.com:risc-v-y/n550-yolo-demo.git
  if git -C "$UPLOAD_DIR" ls-remote --exit-code --heads origin host-diagnostics; then
    git -C "$UPLOAD_DIR" fetch --depth=1 origin host-diagnostics
    git -C "$UPLOAD_DIR" checkout -b host-diagnostics FETCH_HEAD
  else
    test "$?" -eq 2
    git -C "$UPLOAD_DIR" checkout --orphan host-diagnostics
  fi
  cp -R -- "$RUN_DIR" "$UPLOAD_DIR/"
  git -C "$UPLOAD_DIR" add -- "$(basename "$RUN_DIR")"
  git -C "$UPLOAD_DIR" commit -m "Collect N550 board diagnostics $(basename "$RUN_DIR")"
  git -C "$UPLOAD_DIR" push origin HEAD:refs/heads/host-diagnostics
)
