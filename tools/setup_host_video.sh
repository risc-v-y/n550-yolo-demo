#!/usr/bin/env bash
# Install pinned video wheels locally, then test host processing without board I/O.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${1:-/usr/bin/python3}"
wheel_dir="$repo_dir/host_deps/py38-linux-x86_64"
target_dir="$repo_dir/yolo26_pc/.venv-host-video/lib"

"$python_bin" -c 'import platform,sys
if sys.version_info[:2] != (3,8) or sys.platform != "linux" or platform.machine() != "x86_64":
    raise SystemExit("Offline video wheels require Linux x86_64 and Python 3.8; got " + sys.version)
print("Video Python:", sys.executable, sys.version.split()[0])'

(
    cd -- "$wheel_dir"
    sha256sum -c SHA256SUMS
)

# Use pip's supported sysconfig switch in this process only. Python 3.8 hosts
# can have incompatible system distutils files; neither they nor site-packages
# need to be imported or modified for installing these prebuilt wheels.
# --isolated/--no-index keep user configuration and network sources out.
PYTHONPATH="$wheel_dir/pip-24.1.2-py3-none-any.whl" "$python_bin" -c '
import runpy, sys, sysconfig
sysconfig._PIP_USE_SYSCONFIG = True
sys.argv[0] = "pip"
runpy.run_module("pip", run_name="__main__")
' \
    --isolated --disable-pip-version-check --no-cache-dir install \
    --no-index --no-deps --only-binary=:all: --upgrade --target "$target_dir" \
    "$wheel_dir/numpy-1.24.4-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "$wheel_dir/opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

# Use exactly the environment which the video entry point will load.
"$python_bin" "$repo_dir/tools/host_video.py" --check --no-display
printf '%s\n' 'VIDEO SETUP COMPLETED: offline dependencies and host processing passed.' \
    'Next: bash tools/host.sh video (graphical session required).'
