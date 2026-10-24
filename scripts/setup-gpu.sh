#!/usr/bin/env bash
# Install the GPU profile that matches the cards in this machine.
# Hopper: uv only. Consumer, Ada, and Ampere: uv, then
# compile SageAttention v2.2.0 (the latest tag) into the same environment.
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"

install_sage() {
    local arch="$1"
    local python src cuda_home
    # SageAttention 2.2.0 ships an SM120 kernel and no SM121 kernel.
    # SM121 (GB10) is in the 12.x family, so the build targets 12.0.
    if [ "$arch" = "12.1" ]; then
        arch="12.0"
    fi
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        python="$VIRTUAL_ENV/bin/python"
    else
        python=".venv/bin/python"
    fi
    if "$python" -c "import sageattention" >/dev/null 2>&1; then
        echo "SageAttention is already installed"
        return 0
    fi
    cuda_home="${CUDA_HOME:-/usr/local/cuda}"
    if [ ! -x "$cuda_home/bin/nvcc" ]; then
        echo "nvcc not found at $cuda_home. Set CUDA_HOME to the CUDA toolkit." >&2
        exit 1
    fi
    src="$(mktemp -d)"
    git clone --depth 1 --branch v2.2.0 https://github.com/thu-ml/SageAttention.git "$src/SageAttention"
    TORCH_CUDA_ARCH_LIST="$arch" CUDA_HOME="$cuda_home" \
        uv pip install --python "$python" --no-build-isolation --no-deps "$src/SageAttention"
    rm -rf "$src"
}

if ! command -v nvidia-smi >/dev/null; then
    echo "nvidia-smi is not on PATH" >&2
    exit 1
fi

mapfile -t caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/ //g' | sort -u)
if [ "${#caps[@]}" -ne 1 ]; then
    echo "GPUs report more than one compute capability: ${caps[*]}" >&2
    exit 1
fi
cap="${caps[0]}"

case "$cap" in
    9.0) extra=hopper ;;
    8.0 | 8.6 | 8.9 | 12.0 | 12.1) extra=consumer ;;
    *)
        echo "No dependency profile for compute capability $cap" >&2
        exit 1
        ;;
esac

echo "compute capability $cap -> $extra"

if [ "$extra" = consumer ]; then
    uv sync --extra consumer --inexact "$@"
    install_sage "$cap"
else
    uv sync --extra "$extra" "$@"
fi

# Show which device preset this machine should use (VRAM *and* system RAM).
echo
uv run --no-sync kandy doctor || true
