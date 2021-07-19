import argparse
import os
import pathlib
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.rsa import alexnet_layers, vone_alexnet_layers
from src.model.model_names import rename_model_name

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="alexnet",
)
parser.add_argument(
    "--num_classes",
    default=1000,
    type=int,
)
parser.add_argument(
    "--epoch",
    default=60,
    type=int,
)
parser.add_argument("--model_names", nargs="+", type=str)
parser.add_argument(
    "--num_filters",
    default=6,
    type=int,
)
parser.add_argument(
    "-d",
    "--num_dim",
    default=2,
    type=int,
)
parser.add_argument(
    "--perplexity",
    default=30,
    type=int,
)
parser.add_argument(
    "--n_iter",
    default=1000,
    type=int,
)


if __name__ == "__main__":
    # ===== args =====
    args = parser.parse_args()

    arch = args.arch
    num_classes = args.num_classes
    epoch = args.epoch
    n_iter = args.n_iter

    in16_test_path = "/Users/sou/lab1-mnt/data1/imagenet16/test/"

    analysis = f"bandpass_activations_tSNE"
    num_dim = args.num_dim
    perplexity = args.perplexity

    num_images = 100
    filter_id = 0  # raw
    num_labels = 16

    # I/O settings
    results_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/results/{analysis}/{num_classes}-class/"
    # results_dir = f"./results/{analysis}/{num_classes}-class/"
    plots_dir = f"./plots/{analysis}/{num_classes}-class/"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # models to compare
    model_names = [
        "alexnet_normal",
        # "alexnet_all_s04",
        # "alexnet_mix_s04",
        # sin_names[arch],
        # "vone_alexnet",
        # "untrained_alexnet",
    ]

    colors = [
        "k",
        "r",
        "g",
        "b",
        "c",
        "m",
        "y",
        "orange",
        "pink",
        "brown",
        "purple",
        "navy",
        "lime",
        "crimson",
        "gold",
        "gray",
    ]

    print("===== arguments =====")
    print("analysis:", analysis)
    print("num_classes:", num_classes)
    print("num_dim:", num_dim)
    print("perplexity:", perplexity)
    print("n_iter:", n_iter)

    print("===== I/O =====")
    print("IN, results_dir:", results_dir)
    print("OUT, plots_dir:", plots_dir)
    print()

    print("===== models to analyze =====")
    print(model_names)
    print()

    # ===== main =====
    print("===== main =====")

    # random seed settings
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    for model_name in tqdm(model_names, desc="models"):
        if "vone" in model_name:
            layers = vone_alexnet_layers
        else:
            layers = alexnet_layers

        result_file = f"{analysis}_embedded_activations_{num_dim}d_p{perplexity}_i{n_iter}_{num_classes}-class_{model_name}.npy"
        result_path = os.path.join(results_dir, result_file)

        # load t-SNE
        embedded_activations = np.load(result_path)  # (L, N, F+1)

        # plot t-SNE
        for layer_id, layer in tqdm(enumerate(layers), "plotting (each layer)"):
            if num_dim == 2:
                fig = plt.figure(dpi=300)
            elif num_dim == 3:
                fig = plt.figure(dpi=300).gca(projection="3d")

            for image_id in range(num_images):
                for label_id in range(num_labels):
                    target = embedded_activations[
                        layer_id, label_id * 100 + image_id, filter_id
                    ]
                    if num_dim == 2:
                        plt.scatter(
                            x=target[0],
                            y=target[1],
                            label=f"l{label_id}",
                            color=colors[label_id],
                            alpha=0.5,
                        )
                    else:
                        fig.scatter(
                            xs=target[0],
                            ys=target[1],
                            zs=target[2],
                            label=f"l{label_id}",
                            color=colors[label_id],
                            alpha=0.5,
                        )

                if image_id == 0 and num_dim == 2:
                    fig.legend(
                        bbox_to_anchor=(0.91, 0.88),
                        loc="upper left",
                        borderaxespad=0,
                        fontsize=8,
                    )
                elif image_id == 0 and num_dim == 3:
                    fig.legend(
                        bbox_to_anchor=(0.01, 0.92),
                        loc="upper left",
                        borderaxespad=0,
                        fontsize=6,
                    )

            plt.title(
                f"{analysis}, p={perplexity}, i={n_iter}, {num_classes}, {rename_model_name(arch=arch, model_name=model_name)}, {layer}",
                fontsize=8,
            )
            # fig.tight_layout()
            plot_file = f"{analysis}_{num_dim}d_p{perplexity}_i{n_iter}_f{filter_id}_nl{num_labels}_ni{num_images}_{num_classes}-class_{model_name}_{layer}.png"
            plot_path = os.path.join(plots_dir, plot_file)
            plt.savefig(plot_path)
