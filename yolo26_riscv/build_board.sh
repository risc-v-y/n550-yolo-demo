#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/yolo26_riscv"
out="$src/build/n550-board"
tc="${TOOLCHAIN_PREFIX:-$root/toolchain/riscv-elf-toolchain/bin/riscv64-unknown-elf}"
image=demo-pcie
extra=()
probe_sources=()
case "${1:-}" in
    --tile-probe) image=tile-probe; extra=(-DBOARD_TILE_PROBE=1); probe_sources=(board/tile_probe.c); shift ;;
    --tile-probe-qemu)
        image=tile-probe-qemu
        extra=(-DBOARD_TILE_PROBE=1 -DBOARD_CLP_SIM=1 -DBOARD_UART_TEST=1 -DCLP_MODEL_QEMU=1
               -Wl,--defsym=board_clp_sim=1 -Wl,--wrap=board_pcie_loop)
        probe_sources=(board/tile_probe.c board/tests/tile_probe_qemu.c board/tests/clp_qemu.c); shift ;;
esac
if [[ "${1:-}" == --help ]]; then
    echo 'Build N550 ELF; optional --tile-probe or --tile-probe-qemu. Does not run hardware.'
    exit 0
fi
if [[ $# != 0 ]]; then exit 2; fi
python3 "$root/tools/check_assets.py"
mkdir -p "$out"
cd "$src"
# Tree vectorization flags do not disable the backend's RVV aggregate copies.
# Keep compiler-generated copies scalar; explicit RVV asm/intrinsics remain.
flags=(-march=rv64gcv_zicbom_zifencei_xewmatrix1p0 -mabi=lp64d -mcmodel=medany -mno-relax
       -O2 -g -std=c11 -Wall -Wextra -Werror -ffreestanding -fno-builtin -fno-strict-aliasing
       -fno-tree-vectorize -mstringop-strategy=scalar -ffp-contract=off -fno-stack-protector -fno-pic -fno-pie
       -ffunction-sections -fdata-sections -fno-asynchronous-unwind-tables
       -DYOLO_N550_BOARD=1 -DMODEL_PRECISION=1 -DMODEL_PROFILE=0
       -I "$src" -I "$root/model" "${extra[@]}")
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
         rvv_model_ops.c amu.c accel_model.c board/uart.c board/model_io.c board/main.c board/clp_selftest.c board/diagnostics.c
         "$root/model/model_generated.c")
"$tc-gcc" "${flags[@]}" -DCLP_PROBE_EMBEDDED=1 -include board/coherent_rvv.h -nostdlib -nostartfiles \
    -Wl,--gc-sections -Wl,-Map="$out/$image.map" -T board/ddr.ld \
    "$out/start.o" "$out/weights.o" "${sources[@]}" "${probe_sources[@]}" board/pcie.c -o "$out/$image.elf"
"$tc-objdump" -d "$out/$image.elf" > "$out/$image.disasm"
"$tc-size" "$out/$image.elf" | tee "$out/$image-size.txt"
for instruction in cbo.clean cbo.flush cbo.inval mfmacc.s.h; do
    grep -Fq "$instruction" "$out/$image.disasm"
done
if [[ "$image" != tile-probe-qemu ]] && grep -Eq '\bq_call\b|\bq_open\b|\bebreak\b' "$out/$image.disasm"; then
    echo 'Unexpected semihosting/debug trap in board ELF' >&2; exit 1
fi
symbols="$image-symbols.txt"
if [[ "$image" == demo-pcie ]]; then symbols=pcie-symbols.txt; fi
"$tc-nm" -n "$out/$image.elf" > "$out/$symbols"
"$tc-objcopy" -O binary "$out/$image.elf" "$out/$image.bin"
sha256sum "$out/$image.elf" "$out/$image.bin" "$root/model/constants.bin" "$root/model/model_generated.c" \
    "$root/model/model_generated.h" > "$out/sha256.txt"
"$tc-gcc" --version > "$out/compiler.txt"
# The model contains the CLP startup test. Do not regenerate the historical
# cached-RVV selftest; independent CLP builds use build_clp_selftest.sh.
if (( $(wc -c < "$out/$image.bin") > 12582912 )); then
    echo 'Unexpected model BIN size; check CLP NOLOAD layout' >&2; exit 1
fi
printf 'CLP model firmware compiled; physical board inference still requires validation.\n'
