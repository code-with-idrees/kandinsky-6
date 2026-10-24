"""Host discovery and device-preset advice.

The YAML presets in ``kandinsky/configs/devices`` encode VRAM, not system RAM.
That is enough on an 80 GB card, where every component is resident at once, but
not on a 24 GB consumer card: ``offload: block`` keeps the whole checkpoint on
the host, so a 29B Pro checkpoint needs ~75 GiB of RAM even though it runs in
24 GB of VRAM. This module answers the two questions a preset cannot:

* which preset matches the card in this machine, and
* does the checkpoint that preset names actually fit in VRAM *and* RAM.

The estimates are deliberately simple. Weight bytes come from
``weights.CHECKPOINTS`` (safetensors sizes, which are already the runtime bf16
or fp16 footprint), plus a fixed activation headroom and the residency rule of
each offload strategy. They are accurate enough to stop an 80 GB Pro download
on a 32 GB workstation, and cheap enough to run before every generation.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..pipeline.config import PipelineConfig, load_config
from .home import weights_dir
from .weights import CHECKPOINTS, Checkpoint

logger = logging.getLogger("kandinsky")

GIB = 1 << 30
MIB = 1 << 20

# A desktop compositor keeps a slice of VRAM, and Windows/WDDM counts it against
# every process. Budget below the reported total instead of pretending it is free.
DISPLAY_RESERVE_GIB = 1.0
# Denoising activations, the audio branch, and allocator slack.
ACTIVATION_HEADROOM_GIB = 2.0
# 4-bit NF4 with double quant keeps roughly 28% of a bf16 text encoder (Qwen2.5-VL 8B).
NF4_RATIO = 0.28
# Host RAM above this fraction is where Windows starts to page and Linux starts
# to reclaim file-backed weights, which turns a load into a swap storm.
RAM_SAFE_FRACTION = 0.85
# Weights above this fraction of RAM cannot be staged at all.
RAM_IMPOSSIBLE_FRACTION = 1.0

PRESET_DIR = Path(__file__).resolve().parents[1] / "configs" / "devices"

_STRATEGY_RANK = {"none": 3, "module": 2, "block": 1}


@dataclass(frozen=True)
class HostInfo:
    """What this machine can hold, as reported by the driver/OS."""

    gpu: str | None
    vram_gib: float
    compute_capability: str | None
    ram_gib: float
    disk_free_gib: float
    attention: tuple[str, ...]
    python: str
    torch_version: str
    cuda: str | None
    platform: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["attention"] = list(self.attention)
        return payload


@dataclass(frozen=True)
class MemoryEstimate:
    """Approximate residency of one pipeline configuration on one host."""

    checkpoint: str
    strategy: str
    quantized_qwen: bool
    weights_gib: float
    cpu_gib: float
    vram_gib: float
    fits_vram: bool
    fits_ram: bool

    @property
    def fits(self) -> bool:
        return self.fits_vram and self.fits_ram

    def problems(self) -> list[str]:
        problems = []
        if not self.fits_vram:
            problems.append(
                f"{self.vram_gib:.1f} GiB of VRAM needed; "
                f"the card has less after the display reserve"
            )
        if not self.fits_ram:
            problems.append(
                f"{self.cpu_gib:.1f} GiB of host RAM needed for offload; "
                f"this machine cannot stage it"
            )
        return problems


@dataclass(frozen=True)
class Recommendation:
    """The preset that best matches a host, with the estimate behind it."""

    preset: Path
    cfg: PipelineConfig
    estimate: MemoryEstimate
    score: int
    reasons: tuple[str, ...]


def system_ram_gib() -> float:
    """Total physical memory, without adding ``psutil`` to the dependency set."""
    if sys.platform == "win32":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / GIB
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB
    except (AttributeError, OSError, ValueError):
        return 0.0


def _disk_free_gib(path: Path) -> float:
    current = path
    while True:
        try:
            return shutil.disk_usage(current).free / GIB
        except OSError:
            parent = current.parent
            if parent == current:
                return 0.0
            current = parent


def attention_backends() -> tuple[str, ...]:
    """Attention engines importable in this environment, best first."""
    engines = ["sdpa"]
    try:
        from .kernels.flash_attention_2 import flash_attention_2
        from .kernels.flash_attention_3 import flash_attention_3
        from .kernels.sage import sage
    except Exception:  # pragma: no cover - broken installs report sdpa only
        return ("sdpa",)
    if sage is not None:
        engines.append("sage")
    if flash_attention_2 is not None:
        engines.append("flash_attention_2")
    if flash_attention_3 is not None:
        engines.append("flash_attention_3")
    return tuple(engines)


def detect_host(device: str | torch.device = "cuda") -> HostInfo:
    """Read GPU, VRAM, compute capability, RAM, and free disk for this host."""
    target = torch.device(device)
    gpu = None
    vram_gib = 0.0
    capability = None
    if target.type == "cuda" and torch.cuda.is_available():
        index = target.index if target.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        gpu = props.name
        vram_gib = props.total_memory / GIB
        capability = f"{props.major}.{props.minor}"
    return HostInfo(
        gpu=gpu,
        vram_gib=vram_gib,
        compute_capability=capability,
        ram_gib=system_ram_gib(),
        disk_free_gib=_disk_free_gib(weights_dir()),
        attention=attention_backends(),
        python=platform.python_version(),
        torch_version=torch.__version__,
        cuda=torch.version.cuda,
        platform=f"{platform.system()} {platform.machine()}",
    )


def _checkpoint(name: str) -> Checkpoint:
    return CHECKPOINTS[name]


def text_encoder_gib(spec: Checkpoint, *, quantized: bool) -> float:
    bf16 = spec.text_bytes / GIB
    return bf16 * NF4_RATIO if quantized else bf16


def estimate_memory(cfg: PipelineConfig, host: HostInfo, strategy: str | None = None) -> MemoryEstimate:
    """Estimate VRAM, host RAM, and fit for ``cfg`` on ``host``.

    ``offload: none`` keeps every component on the card. ``module`` and ``block``
    keep the whole checkpoint on the host and stage one module group (or two DiT
    blocks) at a time, so their VRAM floor is the largest single group and their
    RAM floor is the full checkpoint. ``strategy`` overrides the YAML value, for
    callers that only override it on the command line.
    """
    spec = _checkpoint(cfg.checkpoint)
    strategy = strategy if strategy is not None else cfg.offload.strategy
    quantized = cfg.text_embedder.quantized_qwen

    dit = spec.dit_bytes / GIB
    text_bf16 = spec.text_bytes / GIB
    text = text_encoder_gib(spec, quantized=quantized)
    aux = spec.aux_bytes / GIB
    weights = dit + text_bf16 + aux

    if strategy == "none":
        cpu = 0.0
        vram = dit + text + aux + ACTIVATION_HEADROOM_GIB
    elif strategy == "module":
        # NF4 Qwen is device-bound (bitsandbytes) and never returns to the host,
        # so a quantized text encoder is GPU-resident at every stage.
        cpu = dit + aux + (0.0 if quantized else text_bf16)
        vram = max(dit, text, aux) + ACTIVATION_HEADROOM_GIB
    else:
        # Block offload stages DiT blocks only; the text encoder is either
        # host-resident until its stage (bf16) or pinned on the card (NF4).
        cpu = dit + aux + (0.0 if quantized else text_bf16)
        blocks = max(1, cfg.dit.num_visual_blocks)
        resident_text = text if quantized else 0.0
        vram = 2.0 * (dit / blocks) * 1.15 + resident_text + aux * 0.25 + ACTIVATION_HEADROOM_GIB

    usable_vram = max(0.0, host.vram_gib - DISPLAY_RESERVE_GIB)
    return MemoryEstimate(
        checkpoint=cfg.checkpoint,
        strategy=strategy,
        quantized_qwen=quantized,
        weights_gib=weights,
        cpu_gib=cpu,
        vram_gib=vram,
        fits_vram=vram <= usable_vram,
        fits_ram=cpu <= host.ram_gib * RAM_SAFE_FRACTION,
    )


def _capability_allows(cfg: PipelineConfig, host: HostInfo) -> bool:
    """FlashAttention 3 only builds for Hopper; other engines run anywhere on CUDA."""
    if cfg.attention.engine != "flash_attention_3":
        return True
    return host.compute_capability == "9.0"


def _quality_score(cfg: PipelineConfig) -> int:
    """Prefer a bigger checkpoint, then the least invasive residency, then bf16 text."""
    pro = 2 if cfg.checkpoint.startswith("pro") else 1
    strategy = _STRATEGY_RANK.get(cfg.offload.strategy, 0)
    # NF4 only buys VRAM when the text encoder is resident for the whole run.
    precision = 1 if cfg.text_embedder.quantized_qwen and cfg.offload.strategy == "none" else 2
    return pro * 100 + strategy * 10 + precision


def _preset_matches_gpu(path: Path, gpu: str | None) -> bool:
    """True when every token of the preset stem appears in the GPU name."""
    if not gpu:
        return False
    tokens = {token for token in re.split(r"[^a-z0-9]+", gpu.lower()) if token}
    return all(part in tokens for part in path.stem.split("-"))


def preset_paths(preset_dir: Path = PRESET_DIR) -> list[Path]:
    return sorted(preset_dir.glob("*.yaml"))


def recommend(host: HostInfo | None = None) -> Recommendation:
    """Pick the highest-quality preset that fits this host.

    Presets are ranked by checkpoint size, then residency, then text precision;
    an exact GPU-name match breaks ties (``rtx-6000`` presets otherwise tie on
    every H100 because they share a strategy). Presets that need a kernel this
    card cannot build are skipped.

    Raises ``LookupError`` when no shipped preset fits, which means the machine
    needs a smaller checkpoint than the catalog offers or more RAM for offload.
    """
    host = host if host is not None else detect_host()
    considered: list[tuple[int, bool, float, Recommendation]] = []
    for path in preset_paths():
        cfg = load_config(path)
        if not _capability_allows(cfg, host):
            continue
        estimate = estimate_memory(cfg, host)
        if not estimate.fits:
            continue
        score = _quality_score(cfg)
        considered.append((
            score,
            _preset_matches_gpu(path, host.gpu),
            -estimate.vram_gib,
            Recommendation(
                preset=path,
                cfg=cfg,
                estimate=estimate,
                score=score,
                reasons=_reasons(host, cfg, estimate),
            ),
        ))
    if not considered:
        raise LookupError(
            "no shipped device preset fits this host: "
            f"{host.vram_gib:.0f} GiB VRAM, {host.ram_gib:.0f} GiB RAM"
        )
    considered.sort(key=lambda item: item[:3], reverse=True)
    return considered[0][3]


def _reasons(host: HostInfo, cfg: PipelineConfig, estimate: MemoryEstimate) -> tuple[str, ...]:
    reasons = [
        f"{estimate.vram_gib:.1f} GiB of {max(0.0, host.vram_gib - DISPLAY_RESERVE_GIB):.1f} GiB usable VRAM",
    ]
    if cfg.offload.strategy == "none":
        reasons.append("every component is resident, so no per-stage transfers")
    else:
        reasons.append(
            f"offload:{cfg.offload.strategy} stages {estimate.cpu_gib:.1f} GiB from host RAM"
        )
    if cfg.text_embedder.quantized_qwen:
        reasons.append("NF4 text encoder keeps the 8B Qwen branch near 4 GiB instead of 15.5 GiB")
    return tuple(reasons)


def check_config(path: str | Path, host: HostInfo | None = None) -> tuple[PipelineConfig, MemoryEstimate]:
    """Load one preset or full YAML and estimate it against this host."""
    host = host if host is not None else detect_host()
    cfg = load_config(path)
    return cfg, estimate_memory(cfg, host)


def guard(
    cfg: PipelineConfig,
    host: HostInfo | None = None,
    strategy: str | None = None,
) -> MemoryEstimate:
    """Log (and for impossible RAM, raise) when ``cfg`` cannot fit this host.

    Called from the factory before any weight is read, so a mistyped preset
    fails in seconds instead of after an 80 GB download.
    """
    host = host if host is not None else detect_host()
    estimate = estimate_memory(cfg, host, strategy)
    if estimate.fits:
        return estimate
    if not estimate.fits_ram and estimate.cpu_gib > host.ram_gib * RAM_IMPOSSIBLE_FRACTION:
        raise MemoryError(
            f"checkpoint {cfg.checkpoint!r} needs {estimate.cpu_gib:.1f} GiB of host RAM "
            f"with offload:{estimate.strategy}, but this machine has "
            f"{host.ram_gib:.1f} GiB. Use a lite checkpoint or a machine with more RAM."
        )
    logger.warning(
        "config may not fit this host: %s. Run `kandy doctor` for the preset it recommends.",
        "; ".join(estimate.problems()),
    )
    return estimate


def format_report(
    host: HostInfo,
    recommendation: Recommendation | None,
    estimate: MemoryEstimate | None = None,
    *,
    config_path: str | Path | None = None,
) -> str:
    """Plain-text doctor output. One fact per line, then the recommendation."""
    lines = [
        f"gpu            {host.gpu or 'no CUDA device'}",
        f"vram           {host.vram_gib:.1f} GiB"
        + (f" (compute capability {host.compute_capability})" if host.compute_capability else ""),
        f"system ram     {host.ram_gib:.1f} GiB",
        f"free disk      {host.disk_free_gib:.1f} GiB at {weights_dir()}",
        f"attention      {', '.join(host.attention)}",
        f"torch          {host.torch_version} (CUDA {host.cuda or 'n/a'})",
        f"platform       {host.platform}, Python {host.python}",
    ]
    if config_path is not None and estimate is not None:
        verdict = "fits" if estimate.fits else "does not fit"
        lines.append("")
        lines.append(f"config         {config_path} ({verdict})")
        lines.append(
            f"estimate       {estimate.vram_gib:.1f} GiB VRAM, "
            f"{estimate.cpu_gib:.1f} GiB host RAM, "
            f"{estimate.weights_gib:.1f} GiB weights"
        )
        for problem in estimate.problems():
            lines.append(f"               - {problem}")
    if recommendation is not None:
        lines.append("")
        lines.append(f"recommended    {recommendation.preset}")
        for reason in recommendation.reasons:
            lines.append(f"               - {reason}")
        lines.append("")
        lines.append(f'  kandy generate "a cat on a mat" --config "{recommendation.preset}"')
    return "\n".join(lines)


__all__ = (
    "HostInfo",
    "MemoryEstimate",
    "Recommendation",
    "attention_backends",
    "check_config",
    "detect_host",
    "estimate_memory",
    "format_report",
    "guard",
    "preset_paths",
    "recommend",
    "system_ram_gib",
    "text_encoder_gib",
)
