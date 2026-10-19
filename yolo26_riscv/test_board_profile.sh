#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/yolo26_riscv/build/n550-board"
mkdir -p "$out"
cc -std=c11 -O2 -Wall -Wextra -Werror -DBOARD_PROFILE_TEST \
   "$root/yolo26_riscv/board/diagnostics.c" \
   "$root/yolo26_riscv/board/tests/profile_host.c" -o "$out/profile-test"
"$out/profile-test"
