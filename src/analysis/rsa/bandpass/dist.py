import os
import pathlib
import sys
from typing import Tuple

import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.analysis.rsa.bandpass.activations import compute_activations_with_bandpass


def compute_dist(
    RSA,
    data_loader: iter,
    filters: dict,
    metric: str = "correlation",  # ("correlation")
    device: torch.device = torch.device("cuda:0"),
) -> Tuple[list, list, list, list]:
    """Computes embedded activations of all band-pass images by t-SNE.
    Args:

    Returns:
        dist_within, dist_btw
    """
    num_filters = len(filters)

    all_activations = {}  # {L: (N, F+1, activations)}
    for layer in RSA.layers:
        all_activations[layer] = []  # (N, F+1, activations)

    labels = []

    # compute RSM for each image (with some filters applied)
    for image_id, (image, label) in tqdm(
        enumerate(data_loader), desc="Feed test images", leave=False
    ):
        """Note that data_loader SHOULD return a single image for each loop.
        image (torch.Tensor): torch.Size([1, C, H, W])
        label (torch.Tensor): e.g. tensor([0])
        """
        labels += [f"l{label.item():02d}_f{i}" for i in range(num_filters + 1)]

        activations = compute_activations_with_bandpass(
            RSA=RSA,
            image=image,
            filters=filters,
            device=device,
            add_noise=False,
        )  # Dict: {L: (F+1, C, H, W)}

        for layer in RSA.layers:
            all_activations[layer] += [
                activations[layer].reshape(num_filters + 1, -1)
            ]  # (F+1, activations)

    dist_s_within = []
    dist_s_btw = []
    dist_b_within = []
    dist_b_btw = []

    for layer in tqdm(RSA.layers, desc="layers", leave=False):
        layer_activations = np.array(all_activations[layer])  # (N, 1+F, D)

        layer_activations_s = layer_activations[
            :, 0, :
        ]  # (N, D): sharp (original) images
        dist_within, dist_btw = compute_dist_within_btw(
            activations=layer_activations_s, metric=metric
        )
        dist_s_within += [dist_within]
        dist_s_btw += [dist_btw]

        layer_activations_b = layer_activations[:, 1, :]  # (N, D): blurred images
        dist_within, dist_btw = compute_dist_within_btw(
            activations=layer_activations_b, metric=metric
        )
        dist_b_within += [dist_within]
        dist_b_btw += [dist_btw]

    return dist_s_within, dist_s_btw, dist_b_within, dist_b_btw


def compute_dist_within_btw(activations, metric="correlation"):
    """
    @param metric:
    @param activations: (N, D)
    @return: dist_within, dist_btw
    """
    rsm_s = 1 - squareform(pdist(activations, metric=metric))  # 1 - (1 - corr.) = corr.

    # within classes
    results = []
    for i in range(16):
        results += [
            np.triu(rsm_s[i * 100 : i * 100 + 100, i * 100 : i * 100 + 100], k=1).sum()
        ]
    dist_within = sum(results) / (16 * ((100 * 99) / 2))

    # btw classes
    results = []
    for i in range(16):
        for j in range(i + 1, 16):
            results += [rsm_s[i * 100 : i * 100 + 100, j * 100 : j * 100 + 100].sum()]

    dist_btw = sum(results) / (120 * 100 * 100)

    return dist_within, dist_btw
