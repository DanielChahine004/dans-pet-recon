#!/usr/bin/env bash
# One-time machine setup: OMEGA's fully-supported GPU path on WSL2.
#
# Why: omegatomo's pip wheels ship prebuilt libs linked against CUDA 13 +
# ArrayFire, neither of which this machine has (driver tops out at CUDA 12.9).
# OMEGA's supported answer is compileOMEGA(): the wheel ships the C++ sources
# and rebuilds its libs with plain g++ against *your* ArrayFire + CUDA.
# This script installs those two system-level dependencies once; after that,
# every new Python environment needs only the two commands at the bottom.
#
# Needs sudo. Idempotent — safe to re-run.
set -euo pipefail

# ---- 1. CUDA dev headers + NVRTC (runtime kernel compiler) ------------------
# Minimal set (~200 MB, not the full 3 GB toolkit): gives /usr/local/cuda with
# cuda.h, nvrtc.h and libnvrtc.so, which compileOMEGA's search paths expect.
# libcuda.so itself comes from the Windows driver via /usr/lib/wsl/lib.
if [ ! -e /usr/local/cuda/targets/x86_64-linux/include/crt/host_defines.h ]; then
    wget -qO /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb
    sudo apt-get update
    sudo apt-get install -y cuda-cudart-dev-12-9 cuda-nvrtc-dev-12-9 cuda-crt-12-9
fi

# ---- 2. ArrayFire 3.10 (built on CUDA 12.8 — matches this driver) -----------
if [ ! -e /opt/arrayfire/lib64/libafcuda.so.3 ]; then
    if [ ! -d /opt/ArrayFire-3.10.0-Linux ]; then
        wget -O /tmp/ArrayFire.sh https://arrayfire.gateway.scarf.sh/linux/3.10.0/ArrayFire.sh
        sudo sh /tmp/ArrayFire.sh --include-subdir --prefix=/opt  # -> /opt/ArrayFire-<ver>-Linux
        rm /tmp/ArrayFire.sh
    fi
    # stable, version-independent path that compileOMEGA autodetects
    sudo ln -sfn /opt/ArrayFire-3.10.0-Linux /opt/arrayfire
fi

# ---- 3. Register ArrayFire libs system-wide (no per-env LD_LIBRARY_PATH) ----
echo /opt/arrayfire/lib64 | sudo tee /etc/ld.so.conf.d/arrayfire.conf >/dev/null
sudo ldconfig

# ---- 4. Sanity check ---------------------------------------------------------
python - <<'PY'
import ctypes
ctypes.CDLL("libafcuda.so.3")
print("ArrayFire CUDA backend loads OK")
PY

cat <<'EOF'

Machine setup done. In EVERY new Python environment that needs OMEGA, run:

    pip install omegatomo arrayfire
    python -c "from omegatomo.util.compile import compileOMEGA; compileOMEGA()"

The second command is OMEGA's supported post-install hook: it rebuilds the
env's own omegatomo/libs/*.so against /opt/arrayfire and /usr/local/cuda
(seconds; the "OpenCL build failed" line is expected on WSL2 — NVIDIA has no
OpenCL here; the CUDA and CPU backends are what matter).
EOF
