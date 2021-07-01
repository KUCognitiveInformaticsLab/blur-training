import os
import pathlib
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import squareform, pdist
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.analysis.rsa.bandpass.activations import compute_activations_with_bandpass


def compute_corr2dist(
    RSA,
    data_loader: iter,
    filters: dict,
    metric: str = "correlation",  # ("correlation")
    device: torch.device = torch.device("cuda:0"),
) -> Tuple[list, list, list, list]:
    """Computes embedded activations of all band-pass images by t-SNE.
    Args:

    Returns:
        dist_same, dist_diff
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

    dist_s_same = []
    dist_s_diff = []
    dist_b_same = []
    dist_b_diff = []
    dist_sb_idt = []
    dist_sb_same = []
    dist_sb_diff = []

    for layer in tqdm(RSA.layers, desc="layers", leave=False):
        layer_activations = np.array(
            all_activations[layer]
        )  # (N, 1+F, D)  1+F = 2(Shape and Blur)
        # layer_activations = layer_activations.reshape(1600 * 2, -1)  # (N * 2, D)

        # corr.
        rsm = np.corrcoef(
            layer_activations[:, 0, :], layer_activations[:, 1, :]
        )  # (3200, 3200)

        rsm_s = rsm[0:1600, 0:1600]  # S vs. S
        rsm_b = rsm[1600 : 1600 * 2, 1600 : 1600 * 2]  # B vs. B
        rsm_sb = rsm[0:1600, 1600 : 1600 * 2]  # S vs. B

        dist_same, dist_diff = compute_dist_same_diff(rsm=rsm_s)
        dist_s_same += [dist_same]
        dist_s_diff += [dist_diff]

        dist_same, dist_diff = compute_dist_same_diff(rsm=rsm_b)
        dist_b_same += [dist_same]
        dist_b_diff += [dist_diff]

        dist_idt, dist_same, dist_diff = compute_dist_idt_same_diff(rsm=rsm_sb)
        dist_sb_idt += [dist_idt]
        dist_sb_same += [dist_same]
        dist_sb_diff += [dist_diff]

    # all_results = [dist_s_same] + [dist_s_diff] + \
    #               [dist_b_same] + [dist_b_diff] + \
    #               [dist_sb_idt] + [dist_sb_same] + [dist_sb_diff]
    all_results = [
        dist_s_same,
        dist_s_diff,
        dist_b_same,
        dist_b_diff,
        dist_sb_idt,
        dist_sb_same,
        dist_sb_diff,
    ]
    index = [
        "sharp_same",
        "sharp_different",
        "blur_same",
        "blur_different",
        "sharp-blur_identical",
        "sharp-blur_same",
        "sharp-blur_different",
    ]

    df_dist = pd.DataFrame(
        all_results,
        index=index,
        columns=RSA.layers,
    )

    # save
    # df_dist.to_csv(result_path)

    # return dist_s_same, dist_s_diff, dist_b_same, dist_b_diff
    return df_dist


def compute_rsm2dist(
    RSA,
    data_loader: iter,
    filters: dict,
    metric: str = "correlation",  # ("correlation")
    device: torch.device = torch.device("cuda:0"),
) -> Tuple[list, list, list, list]:
    """Computes embedded activations of all band-pass images by t-SNE.
    Args:

    Returns:
        dist_same, dist_diff
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

    dist_s_same = []
    dist_s_diff = []
    dist_b_same = []
    dist_b_diff = []
    dist_sb_idt = []
    dist_sb_same = []
    dist_sb_diff = []

    for layer in tqdm(RSA.layers, desc="layers", leave=False):
        layer_activations = np.array(
            all_activations[layer]
        )  # (N, 1+F, D)  1+F = 2(Shape and Blur)
        layer_activations = layer_activations.reshape(1600 * 2, -1)  # (N * 2, D)

        # 1 - (1 - corr.) = corr.
        rsm = 1 - squareform(pdist(layer_activations, metric=metric))  # (3200, 3200)

        rsm_s = rsm[0:1600, 0:1600]  # S vs. S
        rsm_b = rsm[1600 : 1600 * 2, 1600 : 1600 * 2]  # B vs. B
        rsm_sb = rsm[0:1600, 1600 : 1600 * 2]  # S vs. B

        dist_same, dist_diff = compute_dist_same_diff(rsm=rsm_s)
        dist_s_same += [dist_same]
        dist_s_diff += [dist_diff]

        dist_same, dist_diff = compute_dist_same_diff(rsm=rsm_b)
        dist_b_same += [dist_same]
        dist_b_diff += [dist_diff]

        dist_idt, dist_same, dist_diff = compute_dist_idt_same_diff(rsm=rsm_sb)
        dist_sb_idt += [dist_idt]
        dist_sb_same += [dist_same]
        dist_sb_diff += [dist_diff]

    # all_results = [dist_s_same] + [dist_s_diff] + \
    #               [dist_b_same] + [dist_b_diff] + \
    #               [dist_sb_idt] + [dist_sb_same] + [dist_sb_diff]
    all_results = [
        dist_s_same,
        dist_s_diff,
        dist_b_same,
        dist_b_diff,
        dist_sb_idt,
        dist_sb_same,
        dist_sb_diff,
    ]
    index = [
        "sharp_same",
        "sharp_different",
        "blur_same",
        "blur_different",
        "sharp-blur_identical",
        "sharp-blur_same",
        "sharp-blur_different",
    ]

    df_dist = pd.DataFrame(
        all_results,
        index=index,
        columns=RSA.layers,
    )

    # save
    # df_dist.to_csv(result_path)

    # return dist_s_same, dist_s_diff, dist_b_same, dist_b_diff
    return df_dist


def compute_dist_same_diff(rsm):
    """
    @param metric:
    @param activations: (N, D)

        test dataset: 16-class-ImageNet
            num_classes: 16
            num_images_each_class: 100

    @return: dist_same, dist_diff
    """
    # same classes
    dists = []
    for i in range(16):
        dists += [
            np.triu(rsm[i * 100 : i * 100 + 100, i * 100 : i * 100 + 100], k=1).sum()
        ]  # Diagonal values (identical images) are not included.
    dist_same = sum(dists) / (16 * ((100 * 99) / 2))

    # diff classes
    dists = []
    for i in range(16):
        for j in range(i + 1, 16):
            dists += [rsm[i * 100 : i * 100 + 100, j * 100 : j * 100 + 100].sum()]

    dist_diff = sum(dists) / (120 * 100 ** 2)

    return dist_same, dist_diff


def compute_dist_idt_same_diff(rsm):
    """
    @param rsm: (1600, 1600) (e.g. Sharp vs. Blur)
    @return: dist_idt, dist_same, dist_diff
    """
    # identical
    dist_idt = np.diag(rsm).mean()

    # same classes
    dists = []
    for i in range(16):
        dists += [rsm[i * 100 : i * 100 + 100, i * 100 : i * 100 + 100].sum()]
    dist_same = (sum(dists) - np.diag(rsm).sum()) / (16 * 100 ** 2 - 1600)

    # diff classes
    dists = []
    for i in range(16):
        for j in range(16):
            if i != j:
                dists += [rsm[i * 100 : i * 100 + 100, j * 100 : j * 100 + 100].sum()]
    dist_diff = sum(dists) / (240 * 100 ** 2)

    return dist_idt, dist_same, dist_diff


def plot_dist(
    dist: pd.DataFrame,
    stimuli,
    layers,
    title,
    plot_path,
):
    blur_sigma = 4

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(
        1,
        1,
        1,
        # xlabel="layers",
        ylabel=f"Correlation",
        ylim=(-0.5, 1),
    )

    if stimuli == "s-b":
        ax.plot(
            layers,
            dist.loc["sharp-blur_identical"].values,
            label=f"S-B (σ={blur_sigma}), identical images",
            ls="-",
        )
        ax.plot(
            layers,
            dist.loc["sharp-blur_same"].values,
            label=f"S-B (σ={blur_sigma}), same classes",
            ls="--",
        )
        ax.plot(
            layers,
            dist.loc["sharp-blur_different"].values,
            label=f"S-B (σ={blur_sigma}), different classes",
            ls=":",
        )
    elif stimuli == "separate":  # S, B separately plotted
        ax.plot(layers, dist.loc["sharp_same"].values, label="S, same classes", ls="-")
        ax.plot(
            layers,
            dist.loc["sharp_different"].values,
            label="S, different classes",
            ls="-",
        )
        ax.plot(
            layers,
            dist.loc["blur_same"].values,
            label=f"B (σ={blur_sigma}), same classes",
            ls="--",
        )
        ax.plot(
            layers,
            dist.loc["blur_different"].values,
            label=f"B (σ={blur_sigma}), different classes",
            ls="--",
        )

    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.legend()
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.grid(ls=":")

    if title:
        ax.set_title(title)

    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def load_dist(file_path):
    return pd.read_csv(file_path, index_col=0)
