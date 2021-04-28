import os
import pathlib
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.analysis.rsa.bandpass.activations import compute_activations_with_bandpass


def compute_bandpass_tSNE(
    RSA,
    num_images: int,
    data_loader: iter,
    filters: dict,
    num_dim: int,
    device: torch.device = torch.device("cuda:0"),
) -> dict:
    """Computes RSM for each image and return mean RSMs.
    Args:
        num_filters: number of band-pass filter
        num_images: number of images
        num_dim: dim of embedding

    Returns:
        embedded_activations: (L, N, F+1, D)
    """
    num_layers = len(RSA.layers)
    num_filters = len(filters)

    embedded_activations = np.zeros((num_layers, num_images, num_filters + 1, num_dim))

    tsne = TSNE(n_components=num_dim, random_state=0, perplexity=30, n_iter=1000)

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
            add_noise=False,
        )

        for layer_id, layer in enumerate(RSA.layers):
            X = activations[layer].reshape(num_filters + 1, -1)  # (7, -1)
            embedded_activations[layer_id, image_id] = tsne.fit_transform(X)  # (7, 2)

    return embedded_activations


def plot_tSNE(
    embedded_activations: np.ndarray, layers: iter, model_name: str, out_dir: str
):
    """
    Args:
        embedded_activations: (L, N, F+1, D)
    """
    num_images, num_layers, num_filters, num_dim = embedded_activations.shape
    colors = ["k", "r", "g", "b", "c", "m", "y"]

    for layer_id, layer in tqdm(enumerate(layers), desc="plotting layers", leave=False):
        fig = plt.figure(dpi=150)

        for image_id in tqdm(
            range(num_images - 1500), desc="plotting images", leave=False
        ):
            for filter_id in range(num_filters + 1):
                plt.scatter(
                    x=embedded_activations[layer_id, image_id, filter_id, 0],
                    y=embedded_activations[layer_id, image_id, filter_id, 1],
                    label=f"f{filter_id}",
                    color=colors[filter_id],
                    alpha=0.5,
                )
            if image_id == 0:
                fig.legend(
                    bbox_to_anchor=(0.91, 0.88),
                    loc="upper left",
                    borderaxespad=0,
                    fontsize=8,
                )

        plt.title(layer, fontsize=10)
        # fig.tight_layout()
        filename = f"{model_name}_{layer}.png"
        out_file = os.path.join(out_dir, filename)
        fig.savefig(out_file)
