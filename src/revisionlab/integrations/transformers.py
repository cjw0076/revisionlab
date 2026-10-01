"""Issue one delayed scalar forecast from a Transformers regression output."""

import math
from numbers import Integral

import numpy as np
import torch
from numpy.typing import NDArray

try:
    from transformers.modeling_outputs import SequenceClassifierOutput
except ModuleNotFoundError as error:
    if error.name != "transformers":
        raise
    raise ModuleNotFoundError(
        "Transformers integration requires the optional extra: "
        'python -m pip install -e ".[transformers]"',
        name="transformers",
    ) from error

from ..protocols import ForecastTicket
from ..replay import DelayedReplay


def issue_regression(
    runner: DelayedReplay,
    output: SequenceClassifierOutput,
    *,
    issue_hour: int,
    read_key: NDArray[np.float64] | torch.Tensor,
    sample_index: int,
) -> ForecastTicket:
    """Issue the selected ``[batch, 1]`` regression logit using an explicit key.

    The scalar is detached without casting its dtype. ``loss`` and labels are
    ignored; targets enter only through ``runner.release`` after maturity.
    Key validation, snapshots, clipping, and ownership remain runner contracts.
    Calls must be serialized, just like direct ``DelayedReplay`` calls.
    """
    if not isinstance(output, SequenceClassifierOutput):
        raise TypeError("Expected a Transformers SequenceClassifierOutput")
    logits = output.logits
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Regression logits must be a torch.Tensor")
    if not logits.is_floating_point():
        raise ValueError("Regression logits must have a floating dtype")
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] != 1:
        raise ValueError("Expected nonempty regression logits with shape [batch, 1]")
    if isinstance(sample_index, bool) or not isinstance(sample_index, Integral):
        raise ValueError("sample_index must be an integer row")
    if not 0 <= sample_index < logits.shape[0]:
        raise ValueError("sample_index is outside the regression batch")
    baseline = float(logits[int(sample_index), 0].detach().cpu().item())
    if not math.isfinite(baseline):
        raise ValueError("Selected regression logit must be finite")
    return runner.issue(issue_hour, baseline, read_key)
