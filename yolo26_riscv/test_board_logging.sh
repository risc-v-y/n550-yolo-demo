#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/yolo26_riscv/build/n550-board"
mkdir -p "$out"
cc -std=c11 -O2 -Wall -Wextra -Werror -DBOARD_UART_TEST -DBOARD_LOG_TEST -DUART_POLL_LIMIT=4 \
    "$root/yolo26_riscv/board/uart.c" "$root/yolo26_riscv/board/diagnostics.c" \
    "$root/yolo26_riscv/board/tests/logging_host.c" -o "$out/logging-test"
"$out/logging-test"
