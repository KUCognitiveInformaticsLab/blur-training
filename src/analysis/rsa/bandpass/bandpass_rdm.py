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
from src.analysis.rsa.utils import load_activations
from src.analysis.rsa.bandpass.activations import compute_activations_with_bandpass


def compute_bandpass_RDMs(
    RSA,
    data_loader: iter,
    filters: dict,
    add_noise: bool = True,
    mean: float = 0.0,
    var: float = 0.1,
    metrics: str = "correlation",  # ("correlation", "1-covariance", "negative-covariance")
    device: torch.device = torch.device("cuda:0"),
) -> dict:
    """Computes RDM for each image and return mean RDMs.
    Args:
        in_dir: path to input directory
        num_filters: number of band-pass filter
        num_images: number of images

    Returns:
        Mean RDMs (Dict)
    """
    mean_rdms = {}
    mean_rdms["num_filters"] = len(filters)
    # mean_rdms["num_images"] = len(data_loader)
    # mean_rdms["target_id"] = target_id

    for layer in tqdm(RSA.layers, desc="layers", leave=False):
        rdms = []
        # compute RDM for each image (with some filters applied)
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
            activations["label_id"] = label.item()
            activations["num_filters"] = len(filters)

            # save (This file size is very big with iterations!)
            # file_name = f"image{image_id:04d}_f{len(filters):02d}.pkl"
            # file_path = os.path.join(out_dir, file_name)
            # save_activations(activations=activations, file_path=file_path)

            # reshape activations for computing rdm
            activation = activations[layer].reshape(len(filters) + 1, -1)

            rdm = compute_RDM(activation=activation, metrics=metrics)

            rdms.append(rdm)

        rdms = np.array(rdms)

        mean_rdms[layer] = rdms.mean(0)

    return mean_rdms


def compute_bandpass_RDMs_2(
    RSA,
    data_loader: iter,
    filters: dict,
    add_noise: bool = True,
    mean: float = 0.0,
    var: float = 0.1,
    metrics: str = "correlation",  # ("correlation", "1-covariance", "negative-covariance")
    device: torch.device = torch.device("cuda:0"),
) -> dict:
    """Computes RDM for each image and return mean RDMs.
    Args:
        in_dir: path to input directory
        num_filters: number of band-pass filter
        num_images: number of images

    Returns:
        Mean RDMs (Dict)
    """
    rdms = {}
    for layer in RSA.layers:
        rdms[layer] = []

    # compute RDM for each image (with some filters applied)
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
            # reshape activations for computing rdm
            activation = activations[layer].reshape(len(filters) + 1, -1)

            rdms[layer] += [compute_RDM(activation=activation, metrics=metrics)]

    mean_rdms = {}
    mean_rdms["num_filters"] = len(filters)
    # mean_rdms["num_images"] = len(data_loader)
    # mean_rdms["target_id"] = target_id
    for layer in RSA.layers:
        all_rdms = np.array(rdms[layer])

        mean_rdms[layer] = all_rdms.mean(0)

    return mean_rdms


def compute_RDM(activation, metrics):
    """Computes RDM.
    Args:
        activation: (N, dim_features)
            N: number of stimuli
            dim_features: dimension of features (e.g. height x width of a feature map)
        metrics: ("correlation", "negative-covariance")
    Returns: RDM (N, N)
    """
    if metrics == "correlation":
        rdm = squareform(pdist(activation, metric=metrics))  # 1 - corr.
    elif metrics == "negative-covariance":
        rdm = squareform(
            pdist(
                activation,
                lambda u, v: -np.average((u - np.average(u)) * (v - np.average(v))),
            )  # - cov.
        )  # TODO: compute diagonal coefficients (They are zeros after passed to "squareform()")
    elif metrics == "1-covariance":
        rdm = squareform(
            pdist(
                activation,
                lambda u, v: 1 - np.average((u - np.average(u)) * (v - np.average(v))),
            )  # 1 - cov.
        )

    return rdm


def load_compute_mean_rdms(
    in_dir: str,
    num_filters: int = 6,
    num_images: int = 1600,
    metrics: str = "correlation",  # or "covariance"
) -> dict:
    """Computes RDM for each image and return mean RDMs.
    Args:
        in_dir: path to input directory
        num_filters: number of band-pass filter
        num_images: number of images
    Returns: Mean RDMs (Dict)
    """
    mean_rdms = {}
    mean_rdms["num_filters"] = num_filters
    mean_rdms["num_images"] = num_images
    # mean_rdms["target_id"] = target_id

    for layer in alexnet_layers:
        rdms = []
        # compute RDM for each image (with some filters applied)
        for image_id in range(num_images):
            file_name = f"image{image_id:04d}_f{num_filters:02d}.pkl"
            file_path = os.path.join(in_dir, file_name)
            activations = load_activations(file_path=file_path)
            activation = activations[layer].reshape(num_filters + 1, -1)
            if metrics == "correlation":
                rdm = squareform(pdist(activation, metric=metrics))  # 1 - corr.
            elif metrics == "negative-covariance":
                rdm = squareform(
                    pdist(
                        activation,
                        lambda u, v: -np.average(
                            (u - np.average(u)) * (v - np.average(v))
                        ),
                    )  # - cov.
                )
            elif metrics == "1-covariance":
                rdm = squareform(
                    pdist(
                        activation,
                        lambda u, v: 1
                        - np.average((u - np.average(u)) * (v - np.average(v))),
                    )  # 1 - cov.
                )
            rdms.append(rdm)

        rdms = np.array(rdms)

        mean_rdms[layer] = rdms.mean(0)

    return mean_rdms


def plot_bandpass_rdms(
    rdms,
    layers=alexnet_layers,
    num_filters=6,
    vmin=0,
    vmax=2,
    title="",
    out_file="rdms.png",
    show_plot=False,
):
    """Plot several layers RDMs in one figure.
    Args:
        model_name: name of the model to examine.
    """
    fig = plt.figure(dpi=300)

    for i, layer in enumerate(layers):
        ax = fig.add_subplot(2, 4, i + 1)
        # sns.set(font_scale=0.5)  # adjust the font size of labels
        ax.set_title(layer)

        sns.heatmap(
            rdms[layer],
            ax=ax,
            square=True,
            vmin=vmin,
            vmax=vmax,
            xticklabels=["0", "0-1", "1-2", "2-4", "4-8", "8-16", "16-"],
            yticklabels=["0", "0-1", "1-2", "2-4", "4-8", "8-16", "16-"],
            cmap="coolwarm",
            cbar=False,
            # --- show values ---
            # annot=True,
            # fmt="1.2f",
            # annot_kws={'size': 3}
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
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])
    # sns.heatmap(rdms[layer], cbar=True, cbar_ax=cbar_ax,
    #             vmin=-1, vmax=1, cmap='coolwarm',
    #             xticklabels=False, yticklabels=False)

    # sns.set(font_scale=0.5)  # adjust the font size of title
    if title:
        fig.suptitle(title)
    # fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.tight_layout()
    plt.savefig(out_file)
    if show_plot:
        plt.show()
    plt.close()
