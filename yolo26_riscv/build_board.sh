#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/yolo26_riscv"
out="$src/build/n550-board"
tc="${TOOLCHAIN_PREFIX:-$root/toolchain/riscv-elf-toolchain/bin/riscv64-unknown-elf}"
if [[ "${1:-}" == --help ]]; then
    echo 'Build N550 board ELF (does not run hardware). TOOLCHAIN_PREFIX overrides the ESWIN compiler prefix.'
    exit 0
fi
if [[ $# != 0 ]]; then exit 2; fi
python3 "$root/tools/check_assets.py"
mkdir -p "$out"
cd "$src"
flags=(-march=rv64gcv_zicbom_zifencei_xewmatrix1p0 -mabi=lp64d -mcmodel=medany -mno-relax
       -O2 -g -std=c11 -Wall -Wextra -Werror -ffreestanding -fno-builtin -fno-strict-aliasing
       -fno-tree-vectorize -ffp-contract=off -fno-stack-protector -fno-pic -fno-pie
       -ffunction-sections -fdata-sections -fno-asynchronous-unwind-tables
       -DYOLO_N550_BOARD=1 -DMODEL_PRECISION=1 -DMODEL_PROFILE=0
       -I "$src" -I "$root/model")
"$tc-gcc" "${flags[@]}" -c board/start.S -o "$out/start.o"
cat > "$out/weights.S" <<EOF
.section .rodata.weights,"a",@progbits
.balign 64
.global model_weights,model_weights_end
model_weights:
.incbin "$root/model/constants.bin"
model_weights_end:
.balign 64
EOF
"$tc-gcc" "${flags[@]}" -c "$out/weights.S" -o "$out/weights.o"
sources=(matmul.c conv2d.c activation.c basic.c basic_rvv.c scalar_model.c
         rvv_model_ops.c amu.c accel_model.c board/uart.c board/protocol.c board/main.c board/selftest.c board/sync_selftest.c board/diagnostics.c
         "$root/model/model_generated.c")
"$tc-gcc" "${flags[@]}" -include board/coherent_rvv.h -nostdlib -nostartfiles \
    -Wl,--gc-sections -Wl,-Map="$out/demo.map" -T board/ddr.ld \
    "$out/start.o" "$out/weights.o" "${sources[@]}" -o "$out/demo.elf"
"$tc-objdump" -d "$out/demo.elf" > "$out/demo.disasm"
"$tc-size" "$out/demo.elf" | tee "$out/size.txt"
for instruction in cbo.clean cbo.flush cbo.inval mfmacc.s.h; do
    grep -Fq "$instruction" "$out/demo.disasm"
done
if grep -Eq '\bq_call\b|\bq_open\b|\bebreak\b' "$out/demo.disasm"; then
    echo 'Unexpected semihosting/debug trap in board ELF' >&2; exit 1
fi
"$tc-nm" -n "$out/demo.elf" > "$out/symbols.txt"
sha256sum "$out/demo.elf" "$root/model/constants.bin" "$root/model/model_generated.c" \
    "$root/model/model_generated.h" > "$out/sha256.txt"
"$tc-gcc" --version > "$out/compiler.txt"
"$tc-gcc" "${flags[@]}" -DBOARD_SELFTEST_STANDALONE=1 -nostdlib -nostartfiles \
    -Wl,--gc-sections -T board/ddr.ld "$out/start.o" board/selftest.c board/sync_selftest.c board/uart.c board/diagnostics.c amu.c \
    -o "$out/selftest.elf"
"$tc-objdump" -d "$out/selftest.elf" > "$out/selftest.disasm"
sha256sum "$out/selftest.elf" >> "$out/sha256.txt"
"$tc-gcc" "${flags[@]}" -DYOLO_PCIE=1 -include board/coherent_rvv.h -nostdlib -nostartfiles \
    -Wl,--gc-sections -Wl,-Map="$out/demo-pcie.map" -T board/ddr.ld \
    "$out/start.o" "$out/weights.o" "${sources[@]}" board/pcie.c -o "$out/demo-pcie.elf"
"$tc-objdump" -d "$out/demo-pcie.elf" > "$out/demo-pcie.disasm"
"$tc-nm" -n "$out/demo-pcie.elf" > "$out/pcie-symbols.txt"
"$tc-size" "$out/demo-pcie.elf" > "$out/pcie-size.txt"
"$tc-objcopy" -O binary "$out/demo-pcie.elf" "$out/demo-pcie.bin"
sha256sum "$out/demo-pcie.elf" "$out/demo-pcie.bin" >> "$out/sha256.txt"
printf 'Board firmware compiled; physical UART/PCIe/cache/AMU verification is still required.\n'
