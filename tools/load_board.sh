#!/usr/bin/env bash
# S2C deployment only. Frame I/O is handled separately by the PCIe backend.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
reset_dir=''
pbcopy='pbcopy'
firmware='demo-pcie.bin'
usage() {
    echo 'Usage: bash tools/load_board.sh --reset-dir /path/to/s2c [--pbcopy /path/to/pbcopy] [--selftest]'
    echo 'reset1.sh holds reset; reset.sh releases reset. Upload failure keeps reset asserted.'
}
while (( $# )); do
    case "$1" in
        --reset-dir|--pbcopy)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            if [[ "$1" == --reset-dir ]]; then reset_dir="$2"; else pbcopy="$2"; fi
            shift 2 ;;
        --selftest) firmware='selftest.bin'; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ -z "$reset_dir" ]]; then usage >&2; exit 2; fi
reset_dir="$(cd -- "$reset_dir" && pwd)"
for script in reset1.sh reset.sh; do
    if [[ ! -r "$reset_dir/$script" ]]; then echo "Missing reset script: $reset_dir/$script" >&2; exit 1; fi
done
pbcopy="$(command -v -- "$pbcopy")"
# Reset scripts change the working directory, so retain an absolute tool path.
if [[ "$pbcopy" != /* ]]; then pbcopy="$(cd -- "$(dirname -- "$pbcopy")" && pwd)/$(basename -- "$pbcopy")"; fi
bin="$root/firmware/$firmware"
if [[ ! -s "$bin" || "$bin" == *:* ]]; then echo "Invalid firmware path: $bin" >&2; exit 1; fi
# Check the selected image before changing hardware state.
entry="$(awk -v path="firmware/$firmware" '$2==path {print}' "$root/firmware/SHA256SUMS")"
if [[ -z "$entry" || "$entry" == *$'\n'* ]]; then echo 'Missing/duplicate firmware checksum' >&2; exit 1; fi
(cd -- "$root" && printf '%s\n' "$entry" | sha256sum -c -)
mkdir -p "$root/yolo26_pc/outputs"
out="$(mktemp -d "$root/yolo26_pc/outputs/deploy-XXXXXXXX")"
log="$out/load.log"
# Cooperates with PBTools; deployment must not overwrite an active inference.
exec 9>"${TMPDIR:-/tmp}/yolo26-pcie.lock"
if ! flock -n 9; then echo 'Another host process owns the YOLO PCIe link.' >&2; exit 1; fi
held=0
trap 'echo "Deployment interrupted; reset may be held (held=$held). Log: $log" >&2; exit 130' INT TERM
run_step() {
    local name="$1" rc
    shift
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$name" | tee -a "$log"
    # Run in a subshell so a failing pipeline cannot bypass our failure handling.
    if (set -o pipefail; "$@" 2>&1 | tee "$out/step.log" | tee -a "$log"); then
        if grep -Eiq '(^|[^[:alnum:]_])(failed|failure|fail!)([^[:alnum:]_]|$)' "$out/step.log"; then
            echo "$name reported failure despite exit 0. Log: $log" >&2
            return 1
        fi
        return 0
    else
        rc=$?
        echo "$name failed (exit $rc). Log: $log" >&2
        return "$rc"
    fi
}
reset_script() { (cd -- "$reset_dir" && bash "./$1"); }
printf 'Firmware: %s\nCPU address: 0x80000000; PCIe offset: 0x0\n' "$bin" | tee -a "$log"
if ! run_step 'HOLD RESET: reset1.sh' reset_script reset1.sh; then
    echo 'Reset assertion failed; firmware was not uploaded or released.' >&2; exit 1
fi
held=1
if ! run_step 'LOAD DDR: pbcopy' "$pbcopy" -d "$bin:0x0"; then
    echo 'Upload failed; reset.sh was NOT called. Keep S2C in reset and inspect the log.' >&2; exit 1
fi
if ! run_step 'RELEASE RESET: reset.sh' reset_script reset.sh; then
    echo 'Reset release failed; board state is uncertain. Inspect the log and platform.' >&2; exit 1
fi
held=0
echo "Deployment completed. Log: $log" | tee -a "$log"
echo 'Check UART startup/selftest text, then use the PCIe client to verify READY and inference.'
