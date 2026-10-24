from __future__ import annotations

from pathlib import Path

import pytest

from kandinsky.pipeline.config import load_config
from kandinsky.runtime.doctor import preset_paths
from kandinsky.runtime.weights import CHECKPOINTS

PRESETS = preset_paths()
STRATEGIES = {"none", "module", "block"}
ENGINES = {"flash_attention_3", "flash_attention_2", "sage", "sdpa", "auto"}


def test_presets_are_shipped() -> None:
    assert PRESETS


@pytest.mark.parametrize("path", PRESETS, ids=lambda path: path.stem)
def test_device_preset_loads(path: Path) -> None:
    cfg = load_config(path)

    assert cfg.checkpoint in CHECKPOINTS
    assert cfg.offload.strategy in STRATEGIES
    assert cfg.attention.engine in ENGINES
    assert cfg.paths.dit and cfg.paths.qwen


def test_rtx_3090_preset_targets_the_24gb_ampere_card() -> None:
    cfg = load_config(Path(__file__).resolve().parents[2] / "kandinsky" / "configs" / "devices" / "rtx-3090.yaml")

    assert cfg.checkpoint == "lite-distill"
    assert cfg.offload.strategy == "none"
    assert cfg.text_embedder.quantized_qwen
    assert cfg.generation.width == 864
    assert cfg.generation.height == 480
    assert cfg.generation.num_steps == 10
