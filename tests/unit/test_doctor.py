from __future__ import annotations

import pytest

from kandinsky.pipeline.config import load_config
from kandinsky.runtime import doctor
from kandinsky.runtime.doctor import (
    PRESET_DIR,
    HostInfo,
    estimate_memory,
    recommend,
    system_ram_gib,
)


def host(
    vram_gib: float,
    ram_gib: float,
    capability: str = "8.6",
    gpu: str = "NVIDIA GeForce RTX 3090",
) -> HostInfo:
    """A host with nothing but the numbers doctor reads."""
    return HostInfo(
        gpu=gpu,
        vram_gib=vram_gib,
        compute_capability=capability,
        ram_gib=ram_gib,
        disk_free_gib=500.0,
        attention=("sdpa",),
        python="3.13.0",
        torch_version="2.14.1",
        cuda="13.0",
        platform="Windows AMD64",
    )


def test_physical_ram_is_reported() -> None:
    assert system_ram_gib() > 1.0


def test_3090_workstation_picks_lite_without_offload() -> None:
    recommendation = recommend(host(24.0, 32.0))

    assert recommendation.preset.name == "rtx-3090.yaml"
    assert recommendation.cfg.checkpoint == "lite-distill"
    assert recommendation.cfg.offload.strategy == "none"
    assert recommendation.estimate.vram_gib < 24.0


def test_pro_on_a_32gb_workstation_is_reported_unstageable() -> None:
    cfg = load_config(PRESET_DIR / "rtx-4090.yaml")

    estimate = estimate_memory(cfg, host(24.0, 32.0))

    assert not estimate.fits
    assert not estimate.fits_ram
    assert estimate.cpu_gib > 60.0
    assert any("host RAM" in problem for problem in estimate.problems())


def test_4090_with_enough_ram_stages_pro_blocks() -> None:
    recommendation = recommend(
        host(24.0, 128.0, capability="8.9", gpu="NVIDIA GeForce RTX 4090")
    )

    assert recommendation.cfg.checkpoint == "pro-distill"
    assert recommendation.cfg.offload.strategy == "block"
    assert recommendation.preset.name == "rtx-4090.yaml"


def test_h100_prefers_its_own_preset_over_a100() -> None:
    recommendation = recommend(
        host(80.0, 512.0, capability="9.0", gpu="NVIDIA H100 80GB HBM3")
    )

    assert recommendation.preset.name == "h100.yaml"
    assert recommendation.cfg.attention.engine == "flash_attention_3"


def test_hopper_only_presets_are_skipped_on_ampere() -> None:
    recommendation = recommend(
        host(80.0, 512.0, capability="8.6", gpu="NVIDIA A100-SXM4-80GB")
    )

    assert recommendation.cfg.attention.engine != "flash_attention_3"


def test_5090_prefers_its_own_preset() -> None:
    recommendation = recommend(
        host(32.0, 128.0, capability="12.0", gpu="NVIDIA GeForce RTX 5090")
    )

    assert recommendation.preset.name == "rtx-5090.yaml"


def test_recommendation_explains_vram_and_strategy() -> None:
    recommendation = recommend(host(24.0, 32.0))

    text = " ".join(recommendation.reasons)

    assert "VRAM" in text
    assert "resident" in text


def test_guard_refuses_a_checkpoint_that_cannot_be_staged() -> None:
    cfg = load_config(PRESET_DIR / "rtx-4090.yaml")

    with pytest.raises(MemoryError, match="host RAM"):
        doctor.guard(cfg, host(24.0, 32.0))


def test_guard_accepts_the_3090_preset() -> None:
    cfg = load_config(PRESET_DIR / "rtx-3090.yaml")

    estimate = doctor.guard(cfg, host(24.0, 32.0))

    assert estimate.fits


def test_preset_gpu_name_match_needs_every_token() -> None:
    assert doctor._preset_matches_gpu(PRESET_DIR / "rtx-3090.yaml", "NVIDIA GeForce RTX 3090")
    assert not doctor._preset_matches_gpu(PRESET_DIR / "rtx-3090.yaml", "NVIDIA GeForce RTX 4090")
    assert not doctor._preset_matches_gpu(PRESET_DIR / "rtx-3090.yaml", None)
