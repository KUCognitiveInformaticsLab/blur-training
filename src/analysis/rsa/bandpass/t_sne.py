import os
import pathlib
import pickle
import sys
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.analysis.rsa.bandpass.activations import compute_activations_with_bandpass


def compute_tSNE(
    RSA,
    num_images: int,
    data_loader: iter,
    filters: dict,
    num_dim: int,
    random_state: int = 0,
    perplexity: int = 30,
    n_iter: int = 1000,
    device: torch.device = torch.device("cuda:0"),
) -> np.ndarray:
    """Computes RSM for each image and return mean RSMs.
    Args:
        num_images: number of images
        num_dim: dim of embedding

    Returns:
        embedded_activations: (L, N, F+1, D)
    """
    num_layers = len(RSA.layers)
    num_filters = len(filters)

    all_activations = np.zeros((num_layers, num_images, num_filters + 1, num_dim))

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
        )  # Dict: {L: (F+1, C, H, W)}

    tsne = TSNE(
        n_components=num_dim,
        random_state=random_state,
        perplexity=perplexity,
        n_iter=n_iter,
    )

    for layer_id, layer in enumerate(RSA.layers):
        X = activations[layer].reshape(num_filters + 1, -1)  # (F+1, -1)
        embedded_activations[layer_id, image_id] = tsne.fit_transform(X)  # (F+1, D)

    return embedded_activations


def compute_bandpass_tSNE(
    RSA,
    num_images: int,
    data_loader: iter,
    filters: dict,
    num_dim: int,
    random_state: int = 0,
    perplexity: int = 30,
    n_iter: int = 1000,
    device: torch.device = torch.device("cuda:0"),
) -> Tuple[np.ndarray, list]:
    """Computes RSM for each image and return mean RSMs.
    Args:
        num_images: number of images
        num_dim: dim of embedding

    Returns:
        embedded_activations (np.ndarray): (F+1, L, N, D)
        labels (list): (N)
    """
    num_layers = len(RSA.layers)
    num_filters = len(filters)

    all_activations = {}
    for layer in RSA.layers:
        all_activations[layer] = []

    labels = []

    # compute RSM for each image (with some filters applied)
    for image_id, (image, label) in tqdm(
        enumerate(data_loader), desc="Feed test images", leave=False
    ):
        """Note that data_loader SHOULD return a single image for each loop.
        image (torch.Tensor): torch.Size([1, C, H, W])
        label (torch.Tensor): e.g. tensor([0])
        """
        labels += [label.item()]

        activations = compute_activations_with_bandpass(
            RSA=RSA,
            image=image,
            filters=filters,
            device=device,
            add_noise=False,
        )  # Dict: {L: (F+1, C, H, W)}

        for layer in RSA.layers:
            all_activations[layer] += [activations[layer].reshape(num_filters + 1, -1)]

    tsne = TSNE(
        n_components=num_dim,
        random_state=random_state,
        perplexity=perplexity,
        n_iter=n_iter,
    )

    embedded_activations = np.zeros((num_filters + 1, num_layers, num_images, num_dim))

    for filter_id in tqdm(range(num_filters + 1), desc="Computing t-SNE (filters)", leave=False):
        for layer_id, layer in tqdm(enumerate(RSA.layers), desc="Computing t-SNE (layers)", leave=False):
            X = np.array(all_activations[layer])[:, filter_id]  # (N, activations)
            embedded_activations[filter_id, layer_id] = tsne.fit_transform(X)  # (N, D)

    return embedded_activations, labels


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

        for image_id in tqdm(range(num_images), desc="plotting images", leave=False):
            for filter_id in range(num_filters + 1):
                target = embedded_activations[layer_id, image_id, filter_id]
                if num_dim == 2:
                    plt.scatter(
                        x=target[0],
                        y=target[1],
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
        filename = f"{model_name}_{layer}_{num_dim}d.png"
        out_file = os.path.join(out_dir, filename)
        fig.savefig(out_file)


def save_embedded_activations(embedded_activations: dict, labels: list, file_path: str):
    object = {"embedded_activations": embedded_activations, "labels": labels}
    with open(file_path, "wb") as f:
        pickle.dump(object, f)


def load_embedded_activations(file_path: str):
    with open(file_path, "rb") as f:
        object = pickle.load(f)
        return object["embedded_activations"], object["labels"]
