#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/yolo26_riscv/build/n550-board"
mkdir -p "$out"
cc -std=c11 -O2 -Wall -Wextra -Werror \
    "$root/yolo26_riscv/board/protocol.c" "$root/yolo26_riscv/board/tests/protocol_host.c" \
    -o "$out/protocol-test"
python3 -B "$root/yolo26_pc/test_uart_protocol.py" "$out/protocol-test" -v 2>&1 | tee "$out/protocol-tests.log"
