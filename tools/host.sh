#!/usr/bin/env bash
# Launch the Linux host menu from any working directory.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$root/tools/host_control.py" "$@"
