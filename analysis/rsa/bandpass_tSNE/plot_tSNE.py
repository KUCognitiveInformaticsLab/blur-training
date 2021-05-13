import argparse
import os
import pathlib
import sys
from distutils.util import strtobool

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.bandpass.t_sne import (
    load_embedded_activations,
    plot_tSNE,
)
from src.analysis.rsa.rsa import alexnet_layers, vone_alexnet_layers
from src.image_process.bandpass_filter import make_bandpass_filters
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
    default=3,
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
parser.add_argument(
    "--compute",
    type=strtobool,
    default=1,
)


if __name__ == "__main__":
    # ===== args =====
    args = parser.parse_args()

    arch = args.arch
    num_classes = args.num_classes
    epoch = args.epoch

    imagenet_path = "/mnt/data1/ImageNet/ILSVRC2012/"
    in16_test_path = "/mnt/data1/imagenet16/test/"

    analysis = f"t-SNE_bandpass_activations"
    compute = args.compute
    num_filters = args.num_filters
    num_dim = args.num_dim
    perplexity = args.perplexity
    n_iter = args.n_iter

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

    print("===== arguments =====")
    print("analysis:", analysis)
    print("compute:", compute)
    print("num_classes:", num_classes)
    print("num_filters:", num_filters)
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
    print()

    # random seed settings
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # make filters
    filters = make_bandpass_filters(num_filters=num_filters)

    for model_name in tqdm(model_names, desc="models"):
        if "vone" in model_name:
            layers = vone_alexnet_layers
        else:
            layers = alexnet_layers

        result_file = f"{analysis}_embedded_activations_{num_dim}d_p{perplexity}_i{n_iter}_{num_classes}-class_{model_name}.npy"
        result_path = os.path.join(results_dir, result_file)

        embed, labels = load_embedded_activations(
            file_path=result_path
        )  # (F+1, L, N, D), (N)

        plot_tSNE(
            embedded_activations=embed,
            labels=labels,
            num_filters=num_filters,
            num_dim=num_dim,
            plots_dir=plots_dir,
            analysis=analysis,
            perplexity=perplexity,
            n_iter=n_iter,
            num_classes=num_classes,
            model_name=model_name,
            title=True,
        )
