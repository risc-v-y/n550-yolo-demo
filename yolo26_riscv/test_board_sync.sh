#!/usr/bin/env bash
# Optional developer regression. QEMU does not model N550 noncoherent DCache.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/yolo26_riscv"
out="$src/build/board-sync-test"
tc="${TOOLCHAIN_PREFIX:-$root/toolchain/riscv-elf-toolchain/bin/riscv64-unknown-elf}"
qemu="${QEMU_SYSTEM_RISCV64:-$root/toolchain/qemu_linux/bin/qemu-system-riscv64}"
mkdir -p "$out"
cat > "$out/harness.c" <<'EOF'
#include "board/diagnostics.h"
#include "amu.h"
int board_selftest(void);
static uintptr_t call(uintptr_t op,uintptr_t argument) {
    register uintptr_t a0 __asm__("a0")=op,a1 __asm__("a1")=argument;
    __asm__ volatile(".balign 16\n.option push\n.option norvc\n"
                     "slli zero,zero,31\nebreak\nsrai zero,zero,7\n.option pop"
                     : "+r"(a0),"+r"(a1) : : "memory");
    return a0;
}
static void print(const char *s) { (void)call(4,(uintptr_t)s); }
__attribute__((noreturn)) void platform_exit(uintptr_t code) {
    uintptr_t args[]={0x20026,code}; (void)call(0x20,(uintptr_t)args);
    for(;;) __asm__ volatile("wfi");
}
void platform_trap(uintptr_t cause,uintptr_t pc,uintptr_t value,const uintptr_t *regs) {
    board_diag_trap(cause,pc,value,regs); print("FAIL trap\n"); platform_exit(2);
}
int main(void) {
    board_diag_init(); board_diag_stage(DIAG_AMU_INIT);
    if(amu_init()) { print("FAIL AMU init\n"); return 1; }
    board_diag_stage(DIAG_SELFTEST);
    int result=board_selftest();
#if TEST_FAULT == 1
    int ok=result==-103 && board_diag.frame==13 && board_diag.tensor==3 && board_diag.element==16;
#elif TEST_FAULT == 2
    int ok=result==-102 && board_diag.frame==7 && board_diag.tensor==2 && board_diag.element==0;
#else
    int ok=result==0 && board_diag.error==0;
#endif
    print(ok ? "PASS board sync regression\n" : "FAIL board sync regression\n");
    return !ok;
}
EOF
# Inject faults only into temporary copies, never the board source.
python3 - "$src/board/sync_selftest.c" "$out" <<'PY'
import pathlib,sys
source=pathlib.Path(sys.argv[1]).read_text()
out=pathlib.Path(sys.argv[2])
changes=[('(float)(round+1)', '(float)(round==13 ? 0 : round+1)'),
         ('board_diag.amu_flags=flags;', 'if(round==7) matrix_out[0]=0;\n        board_diag.amu_flags=flags;')]
for i,(before,after) in enumerate(changes,1):
    assert source.count(before)==1
    (out/f'fault{i}.c').write_text(source.replace(before,after))
PY
flags=(-march=rv64gcv_zicbom_zifencei_xewmatrix1p0 -mabi=lp64d -mcmodel=medany -mno-relax
       -O2 -g -std=c11 -Wall -Wextra -Werror -ffreestanding -fno-builtin -fno-tree-vectorize
       -ffp-contract=off -fno-stack-protector -ffunction-sections -fdata-sections
       -DYOLO_N550_BOARD=1 -DMODEL_PROFILE=0 -I "$src" -I "$src/board")
for fault in 0 1 2; do
    sync="$src/board/sync_selftest.c"
    if [[ $fault != 0 ]]; then sync="$out/fault$fault.c"; fi
    "$tc-gcc" "${flags[@]}" -DTEST_FAULT="$fault" -nostdlib -nostartfiles -Wl,--gc-sections \
        -T "$src/board/ddr.ld" "$src/board/start.S" "$src/board/selftest.c" "$sync" \
        "$src/board/diagnostics.c" "$src/board/uart.c" "$src/amu.c" "$out/harness.c" -o "$out/test$fault.elf"
    for vlen in 128 256 512; do
        echo "fault=$fault vlen=$vlen"
        timeout 60s "$qemu" -machine virt \
            -cpu "rv64,v=true,vlen=$vlen,zicbom=true,cbom_blocksize=64,matrix=on,xtlen=65536,xtrlen=512,xelen=32" \
            -m 64M -smp 1 -nographic -monitor none -serial none -bios none \
            -kernel "$out/test$fault.elf" -semihosting-config enable=on,target=native -no-reboot
    done
done
echo 'Arithmetic, RVV tails, reuse and failure diagnostics checked; physical coherency remains unverified.'
