import os
import pathlib
import pickle
import sys
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.analysis.rsa.bandpass.activations import compute_activations_with_bandpass
from src.model.model_names import rename_model_name


def compute_tSNE_each_bandpass(
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
    """Computes embedded activations of each band-pass images by t-SNE.
    Args:
        num_images: number of images
        num_dim: dim of embedding

    Returns:
        embedded_activations (np.ndarray): (F+1, L, N, D)
        labels (list): (N)
    """
    num_layers = len(RSA.layers)
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
        labels += [label.item()]

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

    tsne = TSNE(
        n_components=num_dim,
        random_state=random_state,
        perplexity=perplexity,
        n_iter=n_iter,
    )

    embedded_activations = np.zeros(
        (num_filters + 1, num_layers, num_images, num_dim)
    )  # (F+1, L, N, D)

    for filter_id in tqdm(
        range(num_filters + 1), desc="Computing t-SNE (filters)", leave=False
    ):
        for layer_id, layer in tqdm(
            enumerate(RSA.layers), desc="Computing t-SNE (layers)", leave=False
        ):
            # all_activations: dict = {L: (N, F+1, activations)}
            X = np.array(all_activations[layer])[:, filter_id]  # (N, activations)
            embedded_activations[filter_id, layer_id] = tsne.fit_transform(X)  # (N, D)

    return embedded_activations, labels


def compute_tSNE_all_bandpass(
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
    """Computes embedded activations of all band-pass images by t-SNE.
    Args:
        num_images: number of images
        num_dim: dim of embedding

    Returns:
        embedded_activations (np.ndarray): (L, N * (F+1), D)
        labels (list): (N * (F+1))
    """
    num_layers = len(RSA.layers)
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

    tsne = TSNE(
        n_components=num_dim,
        random_state=random_state,
        perplexity=perplexity,
        n_iter=n_iter,
    )

    embedded_activations = np.zeros(
        (num_layers, num_images * (num_filters + 1), num_dim)
    )  # (L, N * (F+1), D)

    for layer_id, layer in tqdm(
        enumerate(RSA.layers), desc="Computing t-SNE (layers)", leave=False
    ):
        # all_activations: dict = {L: (N, F+1, activations)}
        X = np.array(all_activations[layer]).reshape(
            num_images * (num_filters + 1), -1
        )  # (N * (F+1), activations)
        embedded_activations[layer_id] = tsne.fit_transform(X)  # (N * (F+1), D)

    return embedded_activations, labels


def save_embedded_activations(embedded_activations: dict, labels: list, file_path: str):
    object = {"embedded_activations": embedded_activations, "labels": labels}
    with open(file_path, "wb") as f:
        pickle.dump(object, f)


def load_embedded_activations(file_path: str):
    with open(file_path, "rb") as f:
        object = pickle.load(f)
        return object["embedded_activations"], object["labels"]


def plot_tSNE_each_bandpass(
    embedded_activations: np.ndarray,
    labels: list,
    num_filters,
    layers,
    num_dim,
    plots_dir,
    analysis,
    perplexity,
    n_iter,
    num_classes,
    model_name,
    title=True,
):
    for filter_id in tqdm(
        range(num_filters + 1), desc="platting (each filters)", leave=False
    ):
        for layer_id, layer in tqdm(
            enumerate(layers), "plotting (each layer)", leave=False
        ):
            target = embedded_activations[filter_id, layer_id]

            if num_dim == 2:
                fig = plt.figure(dpi=150)

                plt.scatter(
                    x=target[:, 0],
                    y=target[:, 1],
                    c=labels,
                    cmap="jet",
                    alpha=0.5,
                )

                plt.colorbar()

                if title:
                    plt.title(
                        f"{analysis}, f={filter_id}, p={perplexity}, i={n_iter}, {num_classes}-class, {rename_model_name(model_name)}, {layer}",
                        fontsize=8,
                    )

            elif num_dim == 3:
                # fig = plt.figure(dpi=150).gca(projection="3d")
                fig = plt.figure(dpi=150)
                ax = Axes3D(fig)

                sc = ax.scatter(
                    xs=target[:, 0],
                    ys=target[:, 1],
                    zs=target[:, 2],
                    c=labels,
                    cmap="jet",
                    alpha=0.5,
                )

                fig.colorbar(sc, shrink=0.75)

                if title:
                    ax.set_title(
                        title=title,
                        fontsize=10,
                    )

            # fig.tight_layout()
            plot_file = f"{analysis}_{num_dim}d_f{filter_id}_p{perplexity}_i{n_iter}_{num_classes}-class_{model_name}_{layer}.png"
            plot_path = os.path.join(plots_dir, plot_file)
            plt.savefig(plot_path)
            plt.close()


def plot_tSNE_all_bandpass(
    embedded_activations: np.ndarray,
    labels: list,
    layers,
    num_dim,
    plots_dir,
    analysis,
    perplexity,
    n_iter,
    num_classes,
    model_name,
    title=True,
):
    """
    embedded_activations (np.ndarray): (L, N * (F+1), D)
    """
    for layer_id, layer in tqdm(
        enumerate(layers), "plotting (each layer)", leave=False
    ):
        target = embedded_activations[layer_id]

        if num_dim == 2:
            fig = plt.figure(dpi=150)

            plt.scatter(
                x=target[:, 0],
                y=target[:, 1],
                c=labels,
                cmap="jet",
                alpha=0.5,
            )

            plt.colorbar()

            if title:
                plt.title(
                    f"{analysis}, p={perplexity}, i={n_iter}, {num_classes}-class, {rename_model_name(model_name)}, {layer}",
                    fontsize=8,
                )

        elif num_dim == 3:
            # fig = plt.figure(dpi=150).gca(projection="3d")
            fig = plt.figure(dpi=150)
            ax = Axes3D(fig)

            sc = ax.scatter(
                xs=target[:, 0],
                ys=target[:, 1],
                zs=target[:, 2],
                c=labels,
                cmap="jet",
                alpha=0.5,
            )

            fig.colorbar(sc, shrink=0.75)

            if title:
                ax.set_title(
                    title=title,
                    fontsize=10,
                )

        # fig.tight_layout()
        plot_file = f"{analysis}_{num_dim}d_p{perplexity}_i{n_iter}_{num_classes}-class_{model_name}_{layer}.png"
        plot_path = os.path.join(plots_dir, plot_file)
        plt.savefig(plot_path)
        plt.close()


markers = {
    0: "s",
    1: "P",
}
# colors = {
#     0: "blue",
#     1: "orange",
# }
colors = {
    0: "magenta",
    1: "red",
    2: "coral",
    3: "orange",
    4: "gold",
    5: "yellow",
    6: "cyan",
    7: "blue",
    8: "skyblue",
    9: "lime",
    10: "green",
    11: "yellowgreen",
    12: "purple",
    13: "brown",
    14: "gray",
    15: "k",
}


def plot_tSNE_s_b_each_layer(
    embedded_activations: np.ndarray,
    labels: list,
    layers,
    num_dim,
    plots_dir,
    analysis,
    perplexity,
    n_iter,
    num_classes,
    model_name,
    title=True,
):
    """
    embedded_activations (np.ndarray): (L, N * (1+F), D)
    labels (list): (N * (1+F))
    """

    for layer_id, layer in tqdm(
        enumerate(layers), "plotting (each layer)", leave=False
    ):
        # idx = np.random.permutation(len(labels))[:100]  # random-100
        idx = [
            i + j for i in range(0, 3200, 200) for j in range(20)
        ]  # 10 images (S-B pair) for each class
        # idx = [i for i in range(0, 3200, 200)]  # for showing a legend

        target = embedded_activations[layer_id][idx]
        target_labels = np.array(labels)[idx]

        fig = plt.figure(dpi=150)
        for i in range(len(target)):
            if num_dim == 2:
                # get (x, y)
                x, y = target[i]

                # get label and filter_id
                # l (int): label (0 - 15)
                # f (int): filter id (0: sharp, 1: blur)
                l, f = map(
                    int, target_labels[i].replace("l", "").replace("f", "").split("_")
                )

                plt.scatter(
                    x=x,
                    y=y,
                    marker=markers[
                        f
                    ],  # Change the marker according to "sharpe" or "blur"
                    c=colors[l],  # Change the color according to the colour
                    alpha=0.5,
                    label=f"{l:02d} " + ("S" if f == 0 else "B"),
                )
                # plt.annotate(l, xy=(x, y))

        if title:
            plt.title(
                f"{analysis}, p={perplexity}, i={n_iter}, {num_classes}-class, {rename_model_name(model_name)}, {layer}",
                fontsize=8,
            )

        """Only S and B
        target = embedded_activations[layer_id][idx]
        target_labels = list(np.array([int(l[-1]) for l in labels])[idx])

        if num_dim == 2:
            fig = plt.figure(dpi=150)

            plt.scatter(
                x=target[:, 0],
                y=target[:, 1],
                c=target_labels,
                cmap="jet",
                alpha=0.5,
            )

            plt.colorbar()
        
            if title:
                plt.title(
                    f"{analysis}, p={perplexity}, i={n_iter}, {num_classes}-class, {rename_model_name(model_name)}, {layer}",
                    fontsize=8,
                )
        """

        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        # fig.tight_layout()
        plot_file = f"{analysis}_{num_dim}d_p{perplexity}_i{n_iter}_{num_classes}-class_{model_name}_{layer}.png"
        plot_path = os.path.join(plots_dir, plot_file)
        plt.savefig(plot_path)
        plt.close()


def plot_tSNE_s_b(
    embedded_activations: np.ndarray,
    labels: list,
    layers,
    num_dim,
    plots_dir,
    analysis,
    perplexity,
    n_iter,
    num_classes,
    model_name,
    title=True,
):
    """
    embedded_activations (np.ndarray): (L, N * (1+F), D)
    labels (list): (N * (1+F))
    """
    fig = plt.figure(dpi=300, figsize=(24, 12))

    for layer_id, layer in tqdm(
        enumerate(layers), "plotting (each layer)", leave=False
    ):
        # idx = np.random.permutation(len(labels))[:100]  # random-100
        idx = [
            i + j for i in range(0, 3200, 200) for j in range(20)
        ]  # 10 images (S-B pair) for each class
        # idx = [i for i in range(0, 3200, 200)]  # for showing a legend

        target = embedded_activations[layer_id][idx]
        target_labels = np.array(labels)[idx]

        ax = fig.add_subplot(2, 4, layer_id + 1)  # plot one of 8 layers
        ax.set_title(layer, fontsize=12)

        for i in range(len(target)):
            if num_dim == 2:
                # get (x, y)
                x, y = target[i]

                # get label and filter_id
                # l (int): label (0 - 15)
                # f (int): filter id (0: sharp, 1: blur)
                l, f = map(
                    int, target_labels[i].replace("l", "").replace("f", "").split("_")
                )

                ax.scatter(
                    x=x,
                    y=y,
                    marker=markers[
                        f
                    ],  # Change the marker according to "sharpe" or "blur"
                    c=colors[l],  # Change the color according to the colour
                    alpha=0.5,
                    label=f"{l:02d} " + ("S" if f == 0 else "B"),
                )
                # plt.annotate(l, xy=(x, y))

    if title:
        plt.suptitle(
            f"{analysis}, p={perplexity}, i={n_iter}, {num_classes}-class, {rename_model_name(model_name)}",
            fontsize=24,
            fontweight="bold",
        )

    """Only S and B
    target = embedded_activations[layer_id][idx]
    target_labels = list(np.array([int(l[-1]) for l in labels])[idx])

    if num_dim == 2:
        fig = plt.figure(dpi=150)x

        plt.scatter(
            x=target[:, 0],
            y=target[:, 1],
            c=target_labels,
            cmap="jet",
            alpha=0.5,
        )

        plt.colorbar()

        if title:
            plt.title(
                f"{analysis}, p={perplexity}, i={n_iter}, {num_classes}-class, {rename_model_name(model_name)}, {layer}",
                fontsize=8,
            )
    """

    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # fig.tight_layout()
    plot_file = f"{analysis}_{num_dim}d_p{perplexity}_i{n_iter}_{num_classes}-class_{model_name}.png"
    plot_path = os.path.join(plots_dir, plot_file)
    plt.savefig(plot_path)
    plt.close()
