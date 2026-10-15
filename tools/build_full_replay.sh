#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/yolo26_riscv/build/host-verify"
mkdir -p "$out"
cd "$root"
# Default x86_64/SSE2, no AVX requirement; no host-side compiler/runtime install.
gcc -std=c11 -O2 -g -static -march=x86-64 -mtune=generic -ffp-contract=off \
    -fno-fast-math -fexcess-precision=standard -fno-tree-vectorize -fno-strict-aliasing \
    -Wall -Wextra -Werror -I yolo26_riscv -I model \
    tools/full_replay.c yolo26_riscv/scalar_model.c yolo26_riscv/basic.c \
    yolo26_riscv/activation.c model/model_generated.c -lm -o "$out/host-verify"
"$out/host-verify" --selftest
if readelf -l "$out/host-verify" | grep -q INTERP; then echo 'Auditor must be static' >&2; exit 1; fi
sha256sum "$out/host-verify"
