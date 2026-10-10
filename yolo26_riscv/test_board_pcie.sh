#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/yolo26_riscv/build/n550-board"
mkdir -p "$out"
cc -std=c11 -O2 -Wall -Wextra -Werror -fno-pie -no-pie -mcmodel=large \
   -ffunction-sections -fdata-sections -Wl,--gc-sections \
   -Wl,--section-start=.pcie_mailbox=0x81f00000 \
   "$root/yolo26_riscv/board/pcie.c" "$root/yolo26_riscv/board/model_io.c" \
   "$root/yolo26_riscv/board/diagnostics.c" "$root/yolo26_riscv/board/tests/pcie_host.c" \
   -o "$out/pcie-test"
"${HOST_PYTHON:-python3}" -B "$root/yolo26_pc/test_pcie_protocol.py" "$out/pcie-test" -v 2>&1 | tee "$out/pcie-tests.log"
