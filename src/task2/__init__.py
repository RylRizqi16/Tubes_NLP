"""Task 2 — Named Entity Recognition (NER) & LDA Topic Modeling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.task2.predictor import T2Predictor, predict, predict_topic

__all__ = ["T2Predictor", "predict", "predict_topic"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from src.task2 import predictor

        return getattr(predictor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
