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
    excluded_labels=[],
    device: torch.device = torch.device("cuda:0"),
    # metric="correlation"
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

    dist_s_same_seen = []
    dist_s_diff_seen = []
    dist_b_same_seen = []
    dist_b_diff_seen = []
    dist_sb_idt_seen = []
    dist_sb_same_seen = []
    dist_sb_diff_seen = []

    # if excluded_labels
    dist_s_same_unseen = []
    dist_s_diff_seen_unseen = []
    dist_s_diff_unseen_unseen = []

    dist_b_same_unseen = []
    dist_b_diff_seen_unseen = []
    dist_b_diff_unseen_unseen = []

    dist_sb_idt_unseen = []
    dist_sb_same_unseen = []
    dist_sb_diff_seen_unseen = []
    dist_sb_diff_unseen_unseen = []

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

        if excluded_labels:
            (
                dist_same_seen,
                dist_same_unseen,
                dist_diff_seen,
                dist_diff_seen_unseen,
                dist_diff_unseen_unseen,
            ) = compute_dist_same_diff_ex_labels(
                rsm=rsm_s, excluded_labels=excluded_labels
            )
            dist_s_same_seen += [dist_same_seen]
            dist_s_same_unseen += [dist_same_unseen]
            dist_s_diff_seen += [dist_diff_seen]
            dist_s_diff_seen_unseen += [dist_diff_seen_unseen]
            dist_s_diff_unseen_unseen += [dist_diff_unseen_unseen]
        else:
            dist_same, dist_diff = compute_dist_same_diff(rsm=rsm_s)
            dist_s_same_seen += [dist_same]
            dist_s_diff_seen += [dist_diff]

        if excluded_labels:
            (
                dist_same_seen,
                dist_same_unseen,
                dist_diff_seen,
                dist_diff_seen_unseen,
                dist_diff_unseen_unseen,
            ) = compute_dist_same_diff_ex_labels(
                rsm=rsm_b, excluded_labels=excluded_labels
            )
            dist_b_same_seen += [dist_same_seen]
            dist_b_same_unseen += [dist_same_unseen]
            dist_b_diff_seen += [dist_diff_seen]
            dist_b_diff_seen_unseen += [dist_diff_seen_unseen]
            dist_b_diff_unseen_unseen += [dist_diff_unseen_unseen]
        else:
            dist_same, dist_diff = compute_dist_same_diff(rsm=rsm_b)
            dist_b_same_seen += [dist_same]
            dist_b_diff_seen += [dist_diff]

        if excluded_labels:
            (
                dist_idt_seen,
                dist_idt_unseen,
                dist_same_seen,
                dist_same_unseen,
                dist_diff_seen,
                dist_diff_seen_unseen,
                dist_diff_unseen_unseen,
            ) = compute_dist_idt_same_diff_ex_labels(
                rsm=rsm_sb, excluded_labels=excluded_labels
            )
            dist_sb_idt_seen += [dist_idt_seen]
            dist_sb_idt_unseen += [dist_idt_unseen]
            dist_sb_same_seen += [dist_same_seen]
            dist_sb_same_unseen += [dist_same_unseen]
            dist_sb_diff_seen += [dist_diff_seen]
            dist_sb_diff_seen_unseen += [dist_diff_seen_unseen]
            dist_sb_diff_unseen_unseen += [dist_diff_unseen_unseen]
        else:
            dist_idt, dist_same, dist_diff = compute_dist_idt_same_diff(rsm=rsm_sb)
            dist_sb_idt_seen += [dist_idt]
            dist_sb_same_seen += [dist_same]
            dist_sb_diff_seen += [dist_diff]

    all_results = [
        dist_s_same_seen,
        dist_s_diff_seen,
        dist_b_same_seen,
        dist_b_diff_seen,
        dist_sb_idt_seen,
        dist_sb_same_seen,
        dist_sb_diff_seen,
    ]
    index = [
        "sharp_same_seen",
        "sharp_different_seen",
        "blur_same_seen",
        "blur_different_seen",
        "sharp-blur_identical_seen",
        "sharp-blur_same_seen",
        "sharp-blur_different_seen",
    ]
    if excluded_labels:
        all_results += [
            dist_s_same_unseen,
            dist_s_diff_seen_unseen,
            dist_s_diff_unseen_unseen,
            dist_b_same_unseen,
            dist_b_diff_seen_unseen,
            dist_b_diff_unseen_unseen,
            dist_sb_idt_unseen,
            dist_sb_same_unseen,
            dist_sb_diff_seen_unseen,
            dist_sb_diff_unseen_unseen,
        ]
        index += [
            "sharp_same_unseen",
            "sharp_different_seen-unseen",
            "sharp_different_unseen-unseen",
            "blur_same_unseen",
            "blur_different_seen-unseen",
            "blur_different_unseen-unseen",
            "sharp-blur_identical_unseen",
            "sharp-blur_same_unseen",
            "sharp-blur_different_seen-unseen",
            "sharp-blur_different_unseen-unseen",
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


def compute_corr2dist_h_l(
    RSA,
    data_loader: iter,
    filters: dict,
    excluded_labels=[],
    device: torch.device = torch.device("cuda:0"),
    # metric="correlation"
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

    dist_h_same_seen = []
    dist_h_diff_seen = []
    dist_l_same_seen = []
    dist_l_diff_seen = []
    dist_lb_idt_seen = []
    dist_lb_same_seen = []
    dist_lb_diff_seen = []

    # if excluded_labels
    dist_h_same_unseen = []
    dist_h_diff_seen_unseen = []
    dist_h_diff_unseen_unseen = []

    dist_l_same_unseen = []
    dist_l_diff_seen_unseen = []
    dist_l_diff_unseen_unseen = []

    dist_lb_idt_unseen = []
    dist_lb_same_unseen = []
    dist_lb_diff_seen_unseen = []
    dist_lb_diff_unseen_unseen = []

    for layer in tqdm(RSA.layers, desc="layers", leave=False):
        layer_activations = np.array(
            all_activations[layer]
        )  # (N, 1+F, D)  1+F = 3(Sharp, high, low)

        # corr.
        rsm = np.corrcoef(
            layer_activations[:, 1, :], layer_activations[:, 2, :]
        )  # (3200, 3200)

        rsm_h = rsm[0:1600, 0:1600]  # H vs. H
        rsm_b = rsm[1600 : 1600 * 2, 1600 : 1600 * 2]  # L vs. L
        rsm_hb = rsm[0:1600, 1600 : 1600 * 2]  # H vs. L

        if excluded_labels:
            (
                dist_same_seen,
                dist_same_unseen,
                dist_diff_seen,
                dist_diff_seen_unseen,
                dist_diff_unseen_unseen,
            ) = compute_dist_same_diff_ex_labels(
                rsm=rsm_h, excluded_labels=excluded_labels
            )
            dist_h_same_seen += [dist_same_seen]
            dist_h_same_unseen += [dist_same_unseen]
            dist_h_diff_seen += [dist_diff_seen]
            dist_h_diff_seen_unseen += [dist_diff_seen_unseen]
            dist_h_diff_unseen_unseen += [dist_diff_unseen_unseen]
        else:
            dist_same, dist_diff = compute_dist_same_diff(rsm=rsm_h)
            dist_h_same_seen += [dist_same]
            dist_h_diff_seen += [dist_diff]

        if excluded_labels:
            (
                dist_same_seen,
                dist_same_unseen,
                dist_diff_seen,
                dist_diff_seen_unseen,
                dist_diff_unseen_unseen,
            ) = compute_dist_same_diff_ex_labels(
                rsm=rsm_b, excluded_labels=excluded_labels
            )
            dist_l_same_seen += [dist_same_seen]
            dist_l_same_unseen += [dist_same_unseen]
            dist_l_diff_seen += [dist_diff_seen]
            dist_l_diff_seen_unseen += [dist_diff_seen_unseen]
            dist_l_diff_unseen_unseen += [dist_diff_unseen_unseen]
        else:
            dist_same, dist_diff = compute_dist_same_diff(rsm=rsm_b)
            dist_l_same_seen += [dist_same]
            dist_l_diff_seen += [dist_diff]

        if excluded_labels:
            (
                dist_idt_seen,
                dist_idt_unseen,
                dist_same_seen,
                dist_same_unseen,
                dist_diff_seen,
                dist_diff_seen_unseen,
                dist_diff_unseen_unseen,
            ) = compute_dist_idt_same_diff_ex_labels(
                rsm=rsm_hb, excluded_labels=excluded_labels
            )
            dist_lb_idt_seen += [dist_idt_seen]
            dist_lb_idt_unseen += [dist_idt_unseen]
            dist_lb_same_seen += [dist_same_seen]
            dist_lb_same_unseen += [dist_same_unseen]
            dist_lb_diff_seen += [dist_diff_seen]
            dist_lb_diff_seen_unseen += [dist_diff_seen_unseen]
            dist_lb_diff_unseen_unseen += [dist_diff_unseen_unseen]
        else:
            dist_idt, dist_same, dist_diff = compute_dist_idt_same_diff(rsm=rsm_hb)
            dist_lb_idt_seen += [dist_idt]
            dist_lb_same_seen += [dist_same]
            dist_lb_diff_seen += [dist_diff]

    all_results = [
        dist_h_same_seen,
        dist_h_diff_seen,
        dist_l_same_seen,
        dist_l_diff_seen,
        dist_lb_idt_seen,
        dist_lb_same_seen,
        dist_lb_diff_seen,
    ]
    index = [
        "high_same_seen",
        "high_different_seen",
        "low_same_seen",
        "low_different_seen",
        "high-low_identical_seen",
        "high-low_same_seen",
        "high-low_different_seen",
    ]
    if excluded_labels:
        all_results += [
            dist_h_same_unseen,
            dist_h_diff_seen_unseen,
            dist_h_diff_unseen_unseen,
            dist_l_same_unseen,
            dist_l_diff_seen_unseen,
            dist_l_diff_unseen_unseen,
            dist_lb_idt_unseen,
            dist_lb_same_unseen,
            dist_lb_diff_seen_unseen,
            dist_lb_diff_unseen_unseen,
        ]
        index += [
            "high_same_unseen",
            "high_different_seen-unseen",
            "high_different_unseen-unseen",
            "low_same_unseen",
            "low_different_seen-unseen",
            "low_different_unseen-unseen",
            "high-low_identical_unseen",
            "high-low_same_unseen",
            "high-low_different_seen-unseen",
            "high-low_different_unseen-unseen",
        ]

    df_dist = pd.DataFrame(
        all_results,
        index=index,
        columns=RSA.layers,
    )

    # save
    # df_dist.to_csv(result_path)

    # return dist_h_same, dist_h_diff, dist_l_same, dist_l_diff
    return df_dist


def compute_corr2dist_s_h(
    RSA,
    data_loader: iter,
    filters: dict,
    device: torch.device = torch.device("cuda:0"),
    # metric="correlation"
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

    dist_s_same_seen = []
    dist_s_diff_seen = []
    dist_h_same_seen = []
    dist_h_diff_seen = []
    dist_sh_idt_seen = []
    dist_sh_same_seen = []
    dist_sh_diff_seen = []

    for layer in tqdm(RSA.layers, desc="layers", leave=False):
        layer_activations = np.array(
            all_activations[layer]
        )  # (N, 1+F, D)  1+F = 2(Shape and High)
        # layer_activations = layer_activations.reshape(1600 * 2, -1)  # (N * 2, D)

        # corr.
        rsm = np.corrcoef(
            layer_activations[:, 0, :], layer_activations[:, 1, :]
        )  # (3200, 3200)

        rsm_s = rsm[0:1600, 0:1600]  # S vs. S
        rsm_h = rsm[1600 : 1600 * 2, 1600 : 1600 * 2]  # H vs. H
        rsm_sh = rsm[0:1600, 1600 : 1600 * 2]  # S vs. H

        dist_same, dist_diff = compute_dist_same_diff(rsm=rsm_s)
        dist_s_same_seen += [dist_same]
        dist_s_diff_seen += [dist_diff]

        dist_same, dist_diff = compute_dist_same_diff(rsm=rsm_h)
        dist_h_same_seen += [dist_same]
        dist_h_diff_seen += [dist_diff]

        dist_idt, dist_same, dist_diff = compute_dist_idt_same_diff(rsm=rsm_sh)
        dist_sh_idt_seen += [dist_idt]
        dist_sh_same_seen += [dist_same]
        dist_sh_diff_seen += [dist_diff]

    all_results = [
        dist_s_same_seen,
        dist_s_diff_seen,
        dist_h_same_seen,
        dist_h_diff_seen,
        dist_sh_idt_seen,
        dist_sh_same_seen,
        dist_sh_diff_seen,
    ]
    index = [
        "sharp_same_seen",
        "sharp_different_seen",
        "high_same_seen",
        "high_different_seen",
        "sharp-high_identical_seen",
        "sharp-high_same_seen",
        "sharp-high_different_seen",
    ]

    df_dist = pd.DataFrame(
        all_results,
        index=index,
        columns=RSA.layers,
    )

    # save
    # df_dist.to_csv(result_path)

    # return dist_s_same, dist_s_diff, dist_h_same, dist_h_diff
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
        "sharp_same_seen",
        "sharp_different_seen",
        "blur_same_seen",
        "blur_different_seen",
        "sharp-blur_identical_seen",
        "sharp-blur_same_seen",
        "sharp-blur_different_seen",
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


def compute_dist_same_diff_ex_labels(
    rsm,
    excluded_labels=[],
):
    """
    @param metric:
    @param activations: (N, D)

        test dataset: 16-class-ImageNet
            num_classes: 16
            num_images_each_class: 100

    @return: dist_same, dist_diff
    """
    # same classes
    dists_seen = []
    dists_unseen = []
    count_seen = 0
    count_unseen = 0
    for i in range(16):
        # Diagonal values (identical images) are not included.
        d = np.triu(rsm[i * 100 : i * 100 + 100, i * 100 : i * 100 + 100], k=1).sum()

        if i in excluded_labels:
            dists_unseen += [d]
            count_unseen += 1
        else:
            dists_seen += [d]
            count_seen += 1

    dist_same_seen = sum(dists_seen) / (count_seen * ((100 * 99) / 2))
    try:
        dist_same_unseen = sum(dists_unseen) / (count_unseen * ((100 * 99) / 2))
    except ZeroDivisionError:
        dist_same_unseen = 0

    # diff classes
    dists_seen = []
    dists_seen_unseen = []
    dists_unseen_unseen = []
    dists_unseen = []
    count_seen = 0
    count_seen_unseen = 0
    count_unseen_unseen = 0
    for i in range(16):
        for j in range(i + 1, 16):
            d = rsm[i * 100 : i * 100 + 100, j * 100 : j * 100 + 100].sum()

            if j in excluded_labels:
                if i in excluded_labels:
                    dists_unseen_unseen += [d]
                    count_unseen_unseen += 1
                else:
                    dists_seen_unseen += [d]
                    count_seen_unseen += 1
            else:
                dists_seen += [d]
                count_seen += 1

    dist_diff_seen = sum(dists_seen) / (count_seen * 100 ** 2)
    try:
        dist_diff_seen_unseen = sum(dists_seen_unseen) / (count_seen_unseen * 100 ** 2)
    except ZeroDivisionError:
        dist_diff_seen_unseen = 0
    try:
        dist_diff_unseen_unseen = sum(dists_unseen_unseen) / (
            count_unseen_unseen * 100 ** 2
        )
    except ZeroDivisionError:
        dist_diff_unseen_unseen = 0

    # dist_diff_unseen = sum(dists) / ((120 - (16-len(excluded_labels)) * (15-len(excluded_labels)) / 2) * 100 ** 2)

    return (
        dist_same_seen,
        dist_same_unseen,
        dist_diff_seen,
        dist_diff_seen_unseen,
        dist_diff_unseen_unseen,
    )


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


def compute_dist_idt_same_diff_ex_labels(
    rsm,
    excluded_labels=[],
):
    """
    @param rsm: (1600, 1600) (e.g. Sharp vs. Blur)
    @return: dist_idt, dist_same, dist_diff
    """
    # identical
    dists_idt_seen = []
    dists_idt_unseen = []
    num_idt_seen = 0
    num_idt_unseen = 0
    for i in range(16):
        d = np.diag(rsm[i * 100 : i * 100 + 100, i * 100 : i * 100 + 100]).sum()

        if i in excluded_labels:
            dists_idt_unseen += [d]
            num_idt_unseen += 100
        else:
            dists_idt_seen += [d]
            num_idt_seen += 100

    dist_idt_seen = sum(dists_idt_seen) / num_idt_seen
    dist_idt_unseen = sum(dists_idt_unseen) / num_idt_unseen

    # same classes
    dists_same_seen = []
    dists_same_unseen = []
    num_same_seen = 0
    num_same_unseen = 0
    for i in range(16):
        d = (
            rsm[i * 100 : i * 100 + 100, i * 100 : i * 100 + 100].sum()
            - np.diag(rsm[i * 100 : i * 100 + 100, i * 100 : i * 100 + 100]).sum()
        )

        if i in excluded_labels:
            dists_same_unseen += [d]
            num_same_unseen += 100 ** 2 - 100
        else:
            dists_same_seen += [d]
            num_same_seen += 100 ** 2 - 100

    dist_same_seen = sum(dists_same_seen) / num_same_seen
    dist_same_unseen = sum(dists_same_unseen) / num_same_unseen

    # diff classes
    dists_seen = []
    dists_seen_unseen = []
    dists_unseen_unseen = []
    dists_unseen = []
    num_diff_seen = 0
    num_diff_seen_unseen = 0
    num_diff_unseen_unseen = 0
    for i in range(16):
        for j in range(i, 16):
            if i != j:
                d = rsm[i * 100 : i * 100 + 100, j * 100 : j * 100 + 100].sum()

                if j in excluded_labels:
                    if i in excluded_labels:
                        dists_unseen_unseen += [d]
                        num_diff_unseen_unseen += 100 ** 2
                    else:
                        dists_seen_unseen += [d]
                        num_diff_seen_unseen += 100 ** 2
                else:
                    dists_seen += [d]
                    num_diff_seen += 100 ** 2

    dist_diff_seen = sum(dists_seen) / num_diff_seen
    try:
        dist_diff_seen_unseen = sum(dists_seen_unseen) / num_diff_seen_unseen
    except ZeroDivisionError:
        dist_diff_seen_unseen = 0
    try:
        dist_diff_unseen_unseen = sum(dists_unseen_unseen) / num_diff_unseen_unseen
    except ZeroDivisionError:
        dist_diff_unseen_unseen = 0

    return (
        dist_idt_seen,
        dist_idt_unseen,
        dist_same_seen,
        dist_same_unseen,
        dist_diff_seen,
        dist_diff_seen_unseen,
        dist_diff_unseen_unseen,
    )


def plot_dist(
    dist: pd.DataFrame,
    stimuli,
    compare,
    layers,
    title,
    plot_path,
    excluded_labels=[],
    full=False,
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

    if compare == "cross":
        if excluded_labels:
            ax.plot(
                layers,
                dist.loc["sharp-blur_identical_seen"].values,
                label=f"S-B (σ={blur_sigma}), identical images, seen class",
                ls="-",
            )
            ax.plot(
                layers,
                dist.loc["sharp-blur_identical_unseen"].values,
                label=f"S-B (σ={blur_sigma}), identical images, unseen class",
                ls="-",
            )
            ax.plot(
                layers,
                dist.loc["sharp-blur_same_seen"].values,
                label=f"S-B (σ={blur_sigma}), same classes, seen class",
                ls="--",
            )
            ax.plot(
                layers,
                dist.loc["sharp-blur_same_unseen"].values,
                label=f"S-B (σ={blur_sigma}), same classes, unseen class",
                ls="--",
            )

            if full:
                ax.plot(
                    layers,
                    dist.loc["sharp-blur_different_seen"].values,
                    label=f"S-B (σ={blur_sigma}), different classes, seen vs seen class",
                    ls=":",
                )
                ax.plot(
                    layers,
                    dist.loc["sharp-blur_different_seen-unseen"].values,
                    label=f"S-B (σ={blur_sigma}), different classes, seen vs unseen class",
                    ls=":",
                )
                if dist.loc["sharp-blur_different_unseen-unseen"].values.sum() != 0:
                    ax.plot(
                        layers,
                        dist.loc["sharp-blur_different_unseen-unseen"].values,
                        label=f"S-B (σ={blur_sigma}), different classes, unseen vs unseen class",
                        ls=":",
                    )
        elif stimuli == "s-b":
            ax.plot(
                layers,
                dist.loc["sharp-blur_identical_seen"].values,
                label=f"S-B (σ={blur_sigma}), identical images",
                ls="-",
            )
            ax.plot(
                layers,
                dist.loc["sharp-blur_same_seen"].values,
                label=f"S-B (σ={blur_sigma}), same classes",
                ls="--",
            )
            ax.plot(
                layers,
                dist.loc["sharp-blur_different_seen"].values,
                label=f"S-B (σ={blur_sigma}), different classes",
                ls=":",
            )
        elif stimuli == "s-h":
            ax.plot(
                layers,
                dist.loc["sharp-high_identical_seen"].values,
                label=f"S-H(1-2), identical images",
                ls="-",
            )
            ax.plot(
                layers,
                dist.loc["sharp-high_same_seen"].values,
                label=f"S-H(1-2), same classes",
                ls="--",
            )
            ax.plot(
                layers,
                dist.loc["sharp-high_different_seen"].values,
                label=f"S-H(1-2), different classes",
                ls=":",
            )
        elif stimuli == "h-l":
            ax.plot(
                layers,
                dist.loc["high-low_identical_seen"].values,
                label=f"High(1-2)-Low(4), identical images",
                ls="-",
            )
            ax.plot(
                layers,
                dist.loc["high-low_same_seen"].values,
                label=f"High(1-2)-Low(4), same classes",
                ls="--",
            )
            ax.plot(
                layers,
                dist.loc["high-low_different_seen"].values,
                label=f"High(1-2)-Low(4), different classes",
                ls=":",
            )

    elif compare == "separate":  # S, B separately plotted
        if excluded_labels:
            ax.plot(
                layers,
                dist.loc["sharp_same_seen"].values,
                label="S, same classes, seen class",
                ls="-",
            )
            ax.plot(
                layers,
                dist.loc["sharp_same_unseen"].values,
                label="S, same classes, unseen class",
                ls="-",
            )

            if full:
                ax.plot(
                    layers,
                    dist.loc["sharp_different_seen"].values,
                    label="S, different classes, seen vs seen class",
                    ls="-",
                )
                ax.plot(
                    layers,
                    dist.loc["sharp_different_seen-unseen"].values,
                    label="S, different classes, seen vs unseen class",
                    ls="-",
                )
                if dist.loc["blur_different_unseen-unseen"].values.sum() != 0:
                    ax.plot(
                        layers,
                        dist.loc["sharp_different_unseen-unseen"].values,
                        label="S, different classes, unseen vs unseen class",
                        ls="-",
                    )

            ax.plot(
                layers,
                dist.loc["blur_same_seen"].values,
                label=f"B (σ={blur_sigma}), same classes, seen class",
                ls="--",
            )
            ax.plot(
                layers,
                dist.loc["blur_same_unseen"].values,
                label=f"B (σ={blur_sigma}), same classes, unseen class",
                ls="--",
            )

            if full:
                ax.plot(
                    layers,
                    dist.loc["blur_different_seen"].values,
                    label=f"B (σ={blur_sigma}), different classes, seen vs seen class",
                    ls="--",
                )
                ax.plot(
                    layers,
                    dist.loc["blur_different_seen-unseen"].values,
                    label=f"B (σ={blur_sigma}), different classes, seen vs unseen class",
                    ls="--",
                )
                if dist.loc["blur_different_unseen-unseen"].values.sum() != 0:
                    ax.plot(
                        layers,
                        dist.loc["blur_different_unseen-unseen"].values,
                        label=f"B (σ={blur_sigma}), different classes, unseen vs unseen class",
                        ls="--",
                    )
        elif stimuli == "s-b":
            ax.plot(
                layers,
                dist.loc["sharp_same_seen"].values,
                label="S, same classes",
                ls="-",
            )
            ax.plot(
                layers,
                dist.loc["sharp_different_seen"].values,
                label="S, different classes",
                ls="-",
            )
            ax.plot(
                layers,
                dist.loc["blur_same_seen"].values,
                label=f"B (σ={blur_sigma}), same classes",
                ls="--",
            )
            ax.plot(
                layers,
                dist.loc["blur_different_seen"].values,
                label=f"B (σ={blur_sigma}), different classes",
                ls="--",
            )
        elif stimuli == "s-h":
            ax.plot(
                layers,
                dist.loc["sharp_same_seen"].values,
                label="S, same classes",
                ls="-",
            )
            ax.plot(
                layers,
                dist.loc["sharp_different_seen"].values,
                label="S, different classes",
                ls="-",
            )
            ax.plot(
                layers,
                dist.loc["high_same_seen"].values,
                label=f"H(1-2), same classes",
                ls="--",
            )
            ax.plot(
                layers,
                dist.loc["high_different_seen"].values,
                label=f"H(1-2), different classes",
                ls="--",
            )
        elif stimuli == "h-l":
            ax.plot(
                layers,
                dist.loc["high_same_seen"].values,
                label="High(1-2), same classes",
                ls="-",
            )
            ax.plot(
                layers,
                dist.loc["high_different_seen"].values,
                label="High(1-2), different classes",
                ls="-",
            )
            ax.plot(
                layers,
                dist.loc["low_same_seen"].values,
                label=f"Low(4), same classes",
                ls="--",
            )
            ax.plot(
                layers,
                dist.loc["low_different_seen"].values,
                label=f"Low(4), different classes",
                ls="--",
            )

    ax.set_xticklabels(layers, rotation=45, ha="right")
    if excluded_labels:
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)  # upper right
        ax.legend(
            bbox_to_anchor=(0, -0.25), loc="upper left", borderaxespad=0, fontsize=10
        )  # down left
    else:
        ax.legend()
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.grid(ls=":")

    if title:
        ax.set_title(title)

    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def load_dist(file_path):
    return pd.read_csv(file_path, index_col=0)
