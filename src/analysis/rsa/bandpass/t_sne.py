import os
import pathlib
import sys

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.analysis.rsa.rsa import alexnet_layers
from src.analysis.rsa.bandpass.activations import compute_activations_with_bandpass

def compute_bandpass_tSNE(
    RSA,
    num_images: int,
    data_loader: iter,
    filters: dict,
    add_noise: bool = False,
    mean: float = 0.0,
    var: float = 0.1,
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
    num_layers = len(RSA.layers)
    num_filters = len(filters)
    num_dim = 2

    embedded_activations = np.zeros((num_images, num_layers, num_filters + 1, num_dim))

    tsne = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=1000)

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

        for layer_id, layer in tqdm(enumerate(RSA.layers), desc="t-SNE layers", leave=False):
            X = activations[layer].reshape(num_filters + 1, -1)  # (7, -1)
            # X_embedded += [tsne.fit_transform(X)]  # (7, 2)
            embedded_activations[image_id, layer_id] = tsne.fit_transform(X)  # (7, 2)

    return embedded_activations


def plot_tSNE(embedded_activations, layers):
    num_images, num_layers, num_filters, num_dim = embedded_activations.shape
    colors = ["r", "g", "b", "c", "m", "y", "k", "orange", "pink"]

    for layer_id, layer in tqdm(enumerate(layers), desc="plotting layers", leave=False):
        plt.figure(figsize=(30, 30))
        for image_id in range(num_images):
            for filter_id in range(num_filters + 1):
                plt.scatter(
                    x=embedded_activations[image_id, layer_id, filter_id, 0],
                    y=embedded_activations[image_id, layer_id, filter_id, 1],
                    label=filter_id,
                    color=colors[filter_id]
                )
            if image_id == 0:
                plt.legend(fontsize=30)
