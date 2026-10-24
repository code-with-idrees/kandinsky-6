<#
.SYNOPSIS
    Install the Kandinsky 6 runtime that matches the GPU in this Windows machine.

.DESCRIPTION
    The Linux entry point is `scripts/setup-gpu.sh` through `just setup`. This is
    its Windows counterpart: it reads the compute capability from nvidia-smi,
    maps it to the same `hopper` / `consumer` extras, and runs `uv sync`.

    Hopper (9.0) gets PyTorch + FlashAttention 3 wheels. Ampere, Ada, and
    consumer Blackwell (8.0, 8.6, 8.9, 12.0, 12.1) get PyTorch only.

    SageAttention is optional on Windows because its CUDA extension needs a
    local CUDA toolkit and MSVC toolchain. Without it the preset's
    `attention: auto` falls back to PyTorch SDPA, which dispatches to the fused
    FlashAttention kernel inside the wheel. Pass -Sage to compile it anyway.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/setup-gpu.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/setup-gpu.ps1 -Sage -Force
#>
[CmdletBinding()]
param(
    # Compile SageAttention into the environment (needs nvcc + MSVC).
    [switch]$Sage,
    # Reinstall even when the environment already imports torch.
    [switch]$Force,
    # Python for the venv. The project supports 3.13 and 3.14.
    [string]$PythonVersion = "3.13"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv is not on PATH. Install it with: powershell -c ""irm https://astral.sh/uv/install.ps1 | iex"""
    }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw "nvidia-smi is not on PATH. Install an NVIDIA driver first."
    }

    $caps = @(& nvidia-smi --query-gpu=compute_cap --format=csv,noheader |
        ForEach-Object { $_.Trim() } | Sort-Object -Unique)
    if ($caps.Count -ne 1) {
        throw "GPUs report more than one compute capability: $($caps -join ', ')"
    }
    $cap = $caps[0]
    if ($cap -notmatch '^\d+\.\d+$') {
        throw "nvidia-smi returned an unexpected compute capability: '$cap'"
    }

    switch ($cap) {
        "9.0" { $extra = "hopper" }
        { $_ -in @("8.0", "8.6", "8.9", "12.0", "12.1") } { $extra = "consumer" }
        default {
            throw "No dependency profile for compute capability $cap. Supported: 8.0, 8.6, 8.9, 9.0, 12.0, 12.1."
        }
    }
    Write-Host "compute capability $cap -> $extra"

    if (-not $Force -and (Test-Path ".venv\Scripts\python.exe")) {
        $torchOk = & ".venv\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available())" 2>$null
        if ($LASTEXITCODE -eq 0 -and $torchOk -eq "True") {
            Write-Host "torch already imports with CUDA in .venv; reusing it (-Force to reinstall)"
        }
        else {
            $Force = $true
        }
    }

    if ($Force -or -not (Test-Path ".venv\Scripts\python.exe")) {
        & uv sync --python $PythonVersion --extra $extra --inexact
        if ($LASTEXITCODE -ne 0) { throw "uv sync failed with exit code $LASTEXITCODE" }
    }

    $python = ".venv\Scripts\python.exe"

    if ($Sage) {
        $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
        $cl = Get-Command cl -ErrorAction SilentlyContinue
        if (-not $nvcc -or -not $cl) {
            throw "SageAttention needs nvcc and cl on PATH. Install the CUDA toolkit and 'Desktop development with C++', then reopen the shell."
        }
        $already = & $python -c "import sageattention" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "SageAttention is already installed"
        }
        else {
            # SageAttention 2.2.0 ships an SM120 kernel and no SM121 kernel; GB10
            # is in the 12.x family, so the build targets 12.0 like the shell script.
            $arch = if ($cap -eq "12.1") { "12.0" } else { $cap }
            $src = Join-Path $env:TEMP "SageAttention"
            if (Test-Path $src) { Remove-Item -Recurse -Force $src }
            & git clone --depth 1 --branch v2.2.0 https://github.com/thu-ml/SageAttention.git $src
            if ($LASTEXITCODE -ne 0) { throw "git clone of SageAttention failed" }
            $env:TORCH_CUDA_ARCH_LIST = $arch
            $env:DISTUTILS_USE_SDK = "1"
            $env:CUDA_HOME = Split-Path -Parent (Split-Path -Parent $nvcc.Source)
            & uv pip install --python $python --no-build-isolation --no-deps $src
            if ($LASTEXITCODE -ne 0) { throw "SageAttention build failed" }
            Remove-Item -Recurse -Force $src
        }
    }
    else {
        Write-Host "Skipping SageAttention (optional on Windows). attention:auto uses PyTorch SDPA."
    }

    & $python -c @"
import torch
name = torch.cuda.get_device_name(0)
cc = ".".join(map(str, torch.cuda.get_device_capability(0)))
free, total = torch.cuda.mem_get_info()
print(f"torch {torch.__version__} cuda {torch.version.cuda}")
print(f"{name} sm_{cc.replace('.', '')} {total / 1024**3:.1f} GiB")
"@
    if ($LASTEXITCODE -ne 0) { throw "torch cannot see the GPU" }

    & $python -m kandinsky doctor
    Write-Host ""
    Write-Host 'Next: uv run kandy generate "a cat on a mat"'
}
finally {
    Pop-Location
}
