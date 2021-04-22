import os
import pathlib
import sys

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.analysis.rsa.rsa import alexnet_layers
from src.analysis.rsa.bandpass.activations import compute_activations_with_bandpass


def compute_bandpass_RSMs(
    RSA,
    data_loader: iter,
    filters: dict,
    add_noise: bool = False,
    mean: float = 0.0,
    var: float = 0.1,
    metrics: str = "correlation",  # ("correlation", "1-covariance", "negative-covariance")
    device: torch.device = torch.device("cuda:0"),
) -> dict:
    """Computes RSM for each image and return mean RSMs.
    Args:
        in_dir: path to input directory
        num_filters: number of band-pass filter
        num_images: number of images

    Returns:
        Mean RSMs (Dict)
    """
    rsms = {}
    for layer in RSA.layers:
        rsms[layer] = []

    # compute RSM for each image (with some filters applied)
    for image_id, (image, label) in tqdm(
        enumerate(data_loader), desc="test images", leave=False
    ):
        """Note that data_loader SHOULD return a single image for each loop.
        image (torch.Tensor): torch.Size([1, C, H, W])
        label (torch.Tensor): e.g. tensor([0])
        """
        activations = compute_activations_with_bandpass(
            RSA=RSA,
            image=image,
            filters=filters,
            device=device,
            add_noise=add_noise,
            mean=mean,
            var=var,
        )

        # add parameter settings of this analysis
        # activations["label_id"] = label.item()
        # activations["num_filters"] = len(filters)

        # save (This file size is very big with iterations!)
        # file_name = f"image{image_id:04d}_f{len(filters):02d}.pkl"
        # file_path = os.path.join(out_dir, file_name)
        # save_activations(activations=activations, file_path=file_path)

        for layer in RSA.layers:
            # reshape activations for computing rsm
            activation = activations[layer].reshape(len(filters) + 1, -1)

            rsms[layer] += [compute_RSM(activation=activation, metrics=metrics)]

    mean_rsms = {}
    mean_rsms["num_filters"] = len(filters)
    # mean_rsms["num_images"] = len(data_loader)
    # mean_rsms["target_id"] = target_id
    for layer in RSA.layers:
        all_rsms = np.array(rsms[layer])

        mean_rsms[layer] = all_rsms.mean(0)

    return mean_rsms

def compute_noise_RSMs(
    RSA,
    data_loader: iter,
    mean: float = 0.0,
    var: float = 0.1,
    metrics: str = "correlation",  # ("correlation", "1-covariance", "negative-covariance")
    device: torch.device = torch.device("cuda:0"),
) -> dict:
    """TODO: Computes RSM with independent noises and return mean RSMs.
    Args:

    Returns:
        Mean RSMs (Dict)
    """
    rsms = {}
    for layer in RSA.layers:
        rsms[layer] = []

    # compute RSM with independent noises
    for image_id, (image, label) in tqdm(
        enumerate(data_loader), desc="test images", leave=False
    ):
        """Note that data_loader SHOULD return a single image for each loop.
        image (torch.Tensor): torch.Size([1, C, H, W])
        label (torch.Tensor): e.g. tensor([0])
        """

        # compute activations with noise
        # TODO: compute_activations_with_noise()
        """
        activations = compute_activations_with_bandpass(
            RSA=RSA,
            image=image,
            device=device,
            mean=mean,
            var=var,
        )
        """

        # add parameter settings of this analysis
        # activations["label_id"] = label.item()
        # activations["num_filters"] = len(filters)

        # save (This file size is very big with iterations!)
        # file_name = f"image{image_id:04d}_f{len(filters):02d}.pkl"
        # file_path = os.path.join(out_dir, file_name)
        # save_activations(activations=activations, file_path=file_path)

        # compute RSM
        """
        for layer in RSA.layers:
            # reshape activations for computing rsm
            activation = activations[layer].reshape(len(filters) + 1, -1)

            rsms[layer] += [compute_RSM(activation=activation, metrics=metrics)]
        """


    # compute mean RSM
    mean_rsms = {}
    """
    mean_rsms["num_filters"] = len(filters)
    # mean_rsms["num_images"] = len(data_loader)
    # mean_rsms["target_id"] = target_id
    for layer in RSA.layers:
        all_rsms = np.array(rsms[layer])

        mean_rsms[layer] = all_rsms.mean(0)
    """

    return mean_rsms


def compute_RSM(activation, metrics):
    """Computes RSM.
    Args:
        activation: (N, dim_features)
            N: number of stimuli
            dim_features: dimension of features (e.g. height x width of a feature map)
        metrics: ("correlation", "covariance")
    Returns: RSM (N, N)
    """
    if metrics == "correlation":
        rsm = 1 - squareform(
            pdist(activation, metric=metrics)
        )  # 1 - (1 - corr.) = corr.
    elif metrics == "covariance":
        rsm = squareform(
            pdist(
                activation,
                lambda u, v: np.average((u - np.average(u)) * (v - np.average(v))),
            )  # cov.
        )  # TODO: compute diagonal coefficients (They are zeros after passed to "squareform()")

    return rsm


def plot_bandpass_RSMs(
    rsms,
    layers=alexnet_layers,
    num_filters=6,
    vmin=0,
    vmax=2,
    title="",
    out_file="rsms.png",
    show_plot=False,
):
    """Plot several layers RSMs in one figure.
    Args:
        model_name: name of the model to examine.
    """
    fig = plt.figure(dpi=300)

    # color bar
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])

    for i, layer in enumerate(layers):
        ax = fig.add_subplot(2, 4, i + 1)
        # sns.set(font_scale=0.5)  # adjust the font size of labels
        ax.set_title(layer)

        sns.heatmap(
            rsms[layer],
            ax=ax,
            square=True,
            vmin=vmin,
            vmax=vmax,
            xticklabels=["0", "0-1", "1-2", "2-4", "4-8", "8-16", "16-"],
            yticklabels=["0", "0-1", "1-2", "2-4", "4-8", "8-16", "16-"],
            cmap="coolwarm_r",
            cbar=False,
            # --- show values ---
            annot=True,
            fmt="1.2f",
            annot_kws={"size": 3},
            # ---  ---
            # cbar_ax=cbar_ax,  # show color bar
        )

        ax.hlines(
            [i for i in range(2, num_filters + 1)],
            *ax.get_xlim(),
            linewidth=0.1,
            colors="gray",
        )
        ax.vlines(
            [i for i in range(2, num_filters + 1)],
            *ax.get_ylim(),
            linewidth=0.1,
            colors="gray",
        )
        ax.hlines(
            1,  # diff. line for separating raw images and bandpass images
            *ax.get_xlim(),
            linewidth=1,
            colors="gray",
        )
        ax.vlines(
            1,  # diff. line for separating raw images and bandpass images
            *ax.get_xlim(),
            linewidth=1,
            colors="gray",
        )

    # show color bar
    # cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
    # sns.heatmap(
    #     rsms[layer],
    #     cbar=True,
    #     cbar_ax=cbar_ax,
    #     vmin=vmin,
    #     vmax=vmax,
    #     cmap="coolwarm_r",
    #     xticklabels=False,
    #     yticklabels=False,
    # )

    # sns.set(font_scale=0.5)  # adjust the font size of title
    if title:
        fig.suptitle(title)
    # fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.tight_layout()
    plt.savefig(out_file)
    if show_plot:
        plt.show()
    plt.close()
