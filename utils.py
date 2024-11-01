import torch
import torch.linalg as linalg
from typing import Optional


# Conversion functions
def tensor1d(
    x: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
):
    return torch.as_tensor(torch.atleast_1d(x), dtype=dtype, device=device)

def tensor2d(
    x: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
):
    return torch.as_tensor(torch.atleast_2d(x), dtype=dtype, device=device)


# Dimensionality determination
def determine_dimensionality(variables, default_dim: int):
    """
    Determine the dimensionality of the state space.

    Parameters
    ----------
    variables: list of ({None, array}, conversion function, index)
        Each tuple contains a variable, a conversion function, and an index.
    default_dim: int
        The default dimensionality to use if no variable is provided.

    Returns
    -------
    dim: int
        The dimensionality of the state space.
    """
    candidates = []
    for v, converter, idx in variables:
        if v is not None:
            candidates.append(converter(v).shape[idx])
    if default_dim is None:
        candidates.append(default_dim)
    if len(candidates) == 0:
        return 1
    else:
        if not torch.all(torch.tensor(candidates) == candidates[0]):
            raise ValueError("Inconsistent dimensionality of variables")
        return candidates[0]
