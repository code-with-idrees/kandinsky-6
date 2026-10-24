"""Pick one installed attention kernel for a DiT attention slot."""

from __future__ import annotations

import logging

from .flash_attention_2 import flash_attention_2
from .flash_attention_3 import flash_attention_3
from .sage import sage
from .sdpa import sdpa

logger = logging.getLogger("kandinsky")

# Historic name used by export code that forces the SDPA implementation.
_sdpa = sdpa

_INSTALLED = {
    "flash_attention_3": flash_attention_3,
    "flash_attention_2": flash_attention_2,
    "sage": sage,
}


def resolve_attention_engine(requested: str) -> str:
    """Resolve a config or CLI name to an engine that can be constructed.

    A requested kernel whose Python package is missing falls back to ``auto``,
    which prefers the fastest installed backend and ends at SDPA. That is what
    makes the same preset usable on Windows (no SageAttention/FA wheels) as on
    the Linux machine it was tuned for.
    """
    if requested in _INSTALLED and _INSTALLED[requested] is None:
        logger.warning("%s requested but not installed; falling back to auto", requested)
        return "auto"
    return requested


class SelfAttentionEngine:
    """Selects one attention kernel at construction time."""

    _ENGINES = ("flash_attention_3", "flash_attention_2", "sage", "sdpa", "auto")

    def __init__(self, engine: str = "auto"):
        assert engine in self._ENGINES, f"Unknown attention engine: {engine!r}"
        engine = resolve_attention_engine(engine)
        if engine == "flash_attention_3":
            if flash_attention_3 is None:
                raise RuntimeError("flash_attention_3 requested but not installed.")
            self._fn = flash_attention_3
        elif engine == "flash_attention_2":
            if flash_attention_2 is None:
                raise RuntimeError("flash_attention_2 requested but not installed.")
            self._fn = flash_attention_2
        elif engine == "sage":
            if sage is None:
                raise RuntimeError("sageattention requested but not installed.")
            self._fn = sage
        elif engine == "sdpa":
            self._fn = sdpa
        else:
            self._fn = sdpa
            if sage is not None:
                self._fn = sage
            if flash_attention_2 is not None:
                self._fn = flash_attention_2
            if flash_attention_3 is not None:
                self._fn = flash_attention_3

    def get_attention(self):
        return self._fn
