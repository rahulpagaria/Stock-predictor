from __future__ import annotations

import math


def validate_threshold(threshold: float) -> None:
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Signal threshold must be a finite, non-negative value.")


def classify_signal(predicted_return: float, threshold: float = 0.01) -> str:
    validate_threshold(threshold)
    if not math.isfinite(predicted_return):
        raise ValueError("Predicted return must be finite.")
    if predicted_return > threshold:
        return "BUY"
    if predicted_return < -threshold:
        return "SELL"
    return "FLAT"
