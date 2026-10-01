"""Single-trial cosine and pooled rho; zero-norm alignment is undefined."""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


def _array(value: Any) -> NDArray[np.float64]:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu", dtype=torch.float64).numpy()
    return np.asarray(value, dtype=np.float64)


def _alignment(first: Any, second: Any, *, flatten: bool) -> float | None:
    a, b = _array(first), _array(second)
    if flatten:
        a, b = a.ravel(), b.ravel()
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Finite matched vectors required")
    if a.size == 0:
        return None
    scale_a, scale_b = float(np.max(np.abs(a))), float(np.max(np.abs(b)))
    if scale_a == 0 or scale_b == 0:
        return None
    # Independent global scaling leaves pooled rho unchanged, avoids over/underflow.
    a, b = a / scale_a, b / scale_b
    denominator = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
    return float(np.clip(np.sum(a * b) / denominator, -1, 1))


def credit_alignment(update: Any, negative_gradient: Any) -> dict[str, bool | float | None]:
    """Cosine of an actual update with a reference negative gradient."""
    cosine = _alignment(update, negative_gradient, flatten=True)
    return {"defined": cosine is not None, "cosine": cosine}


def aggregate_alignment(updates: Any, negative_gradients: Any) -> dict[str, bool | float | None]:
    """sum dot / sqrt(sum squared norms), not an average of trial cosines."""
    rho = _alignment(updates, negative_gradients, flatten=False)
    return {"defined": rho is not None, "rho": rho}
