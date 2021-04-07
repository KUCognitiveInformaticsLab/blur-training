import itertools
from typing import Dict, Optional, List

import torch

from .lowpass_filter import GaussianBlurAll


def apply_bandpass_filter(
    images: torch.Tensor, sigma1: float, sigma2: float
) -> torch.Tensor:
    """Apply band-pass filter to input images
    Args:
        images (torch.Tensor): (N, C, H, W)
        sigma1 (float): sigma1
        sigma2 (float): sigma2

    Returns (torch.Tensor): band-passed images (N, C, H, W)
    """
    low1 = GaussianBlurAll(images, sigma=sigma1)
    if sigma2 == None:
        return low1
    else:
        low2 = GaussianBlurAll(images, sigma=sigma2)
        return low1 - low2


def make_bandpass_filters(
    num_filters: int = 6,
) -> Dict[int, Optional[List[int]]]:
    filters = {}
    filters[0] = [0, 1]
    for i in range(1, num_filters):
        if i == (num_filters - 1):
            filters[i] = [2 ** (i - 1), None]  # last band-pass is low-pass filter
        else:
            filters[i] = [2 ** (i - 1), 2 ** i]

    return filters


def make_filter_combinations(filters: dict) -> list:
    """Makes all combinations from filters
    Args:
        filters (dict): Dict[int, List[int, int]]

    Returns (list): all combinations of filter id
    """
    filter_comb = []
    for i in range(1, len(filters)):
        filter_comb += list(itertools.combinations(filters.keys(), i))

    return filter_comb
