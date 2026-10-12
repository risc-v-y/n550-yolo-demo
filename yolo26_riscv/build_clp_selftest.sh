#!/usr/bin/env bash
# Build only. The physical CLP region must be reserved by the platform.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/yolo26_riscv"
out="$src/build/clp-selftest"
tc="${TOOLCHAIN_PREFIX:-$root/toolchain/riscv-elf-toolchain/bin/riscv64-unknown-elf}"
if [[ "${1:-}" == --help ]]; then
    echo 'Build CLP selftest ELF/BIN; does not load or run a board. Override TOOLCHAIN_PREFIX if needed.'
    exit 0
fi
if [[ $# != 0 ]]; then exit 2; fi
mkdir -p "$out"
cd "$src"
flags=(-march=rv64gcv_zicbom_zifencei_xewmatrix1p0 -mabi=lp64d -mcmodel=medany -mno-relax
       -O2 -g -std=c11 -Wall -Wextra -Werror -ffreestanding -fno-builtin -fno-strict-aliasing
       -fno-tree-vectorize -mstringop-strategy=scalar -ffp-contract=off
       -fno-stack-protector -fno-pic -fno-pie -ffunction-sections -fdata-sections
       -fno-asynchronous-unwind-tables -DYOLO_N550_BOARD=1 -DMODEL_PROFILE=0 -I "$src")
"$tc-gcc" "${flags[@]}" -c board/start.S -o "$out/start.o"
"$tc-gcc" "${flags[@]}" -nostdlib -nostartfiles -Wl,--gc-sections \
    -Wl,-Map="$out/clp-selftest.map" -T board/ddr.ld "$out/start.o" \
    board/clp_selftest.c board/uart.c board/diagnostics.c amu.c -o "$out/clp-selftest.elf"
"$tc-objdump" -d "$out/clp-selftest.elf" > "$out/clp-selftest.disasm"
for instruction in vle32.v vse32.v vse16.v mfmacc.s.h; do
    grep -Fq "$instruction" "$out/clp-selftest.disasm"
done
if grep -Eq '\b(vle64\.v|vse64\.v|ebreak|q_call)\b' "$out/clp-selftest.disasm"; then
    echo 'Unexpected implicit RVV copy or semihosting in physical CLP image' >&2; exit 1
fi
"$tc-objcopy" -O binary "$out/clp-selftest.elf" "$out/clp-selftest.bin"
# High DDR is initialized at runtime via RVV, not a padded linker section.
if (( $(wc -c < "$out/clp-selftest.bin") > 1048576 )); then
    echo 'Unexpected CLP image size' >&2; exit 1
fi
"$tc-nm" -n "$out/clp-selftest.elf" > "$out/symbols.txt"
"$tc-size" "$out/clp-selftest.elf"
sha256sum "$out/clp-selftest.elf" "$out/clp-selftest.bin" > "$out/sha256.txt"
echo 'CLP image built. Physical address remapping and completion require board validation.'
