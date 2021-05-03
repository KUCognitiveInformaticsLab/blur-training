import argparse
import os
import pathlib
import sys

import numpy as np
import torch
import vonenet
from matplotlib import pyplot as plt
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.bandpass.t_sne import (
    compute_bandpass_tSNE,
)
from src.analysis.rsa.rsa import AlexNetRSA, VOneNetAlexNetRSA
from src.dataset.imagenet16 import load_imagenet16, make_local_in16_test_loader
from src.image_process.bandpass_filter import make_bandpass_filters
from src.model.utils import load_model
from src.model.model_names import rename_model_name
from src.model.load_sin_pretrained_models import load_sin_model


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
    "--num_dim",
    default=2,
    type=int,
)
parser.add_argument(
    "--perplexity",
    default=30,
    type=int,
)


if __name__ == "__main__":
    # ===== args =====
    args = parser.parse_args()

    arch = args.arch
    num_classes = args.num_classes
    epoch = args.epoch

    imagenet_path = "/mnt/data1/ImageNet/ILSVRC2012/"
    in16_test_path = "/mnt/data1/imagenet16/test/"

    analysis = f"bandpass_activations_tSNE"
    num_filters = args.num_filters
    num_dim = args.num_dim
    perplexity = args.perplexity

    # I/O settings
    models_dir = "/mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
        16 if num_classes == 16 else ""  # else is (num_classes == 1000)
    )
    results_dir = f"./results/{analysis}/{num_classes}-class/"
    plots_dir = f"./plots/{analysis}/{num_classes}-class/"

    assert os.path.exists(models_dir), f"{models_dir} does not exist."
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
    print("num_classes:", num_classes)
    print("num_filters:", num_filters)
    print("num_dim:", num_dim)
    print("perplexity:", perplexity)

    print("===== I/O =====")
    print("OUT, results_dir:", results_dir)
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make Dataloader
    # ** batch_size must be 1 **
    # _, test_loader = load_imagenet16(imagenet_path=imagenet_path, batch_size=1)
    test_loader = make_local_in16_test_loader(
        data_path=in16_test_path, batch_size=1, shuffle=False
    )

    # make filters
    filters = make_bandpass_filters(num_filters=num_filters)

    for model_name in tqdm(model_names, desc="models"):
        print()

        if num_classes == 1000 and "SIN" in model_name:
            # Stylized-ImageNet
            model = load_sin_model(model_name).to(device)
            model.features = model.features.module
            RSA = AlexNetRSA(model)
        elif num_classes == 1000 and "vone" in model_name:
            model = vonenet.get_model(model_arch=arch, pretrained=True).to(device)
            RSA = VOneNetAlexNetRSA(model)
        elif "untrained" in model_name:
            model_path = ""  # load untrained model
            model = load_model(
                arch=arch, num_classes=num_classes, model_path=model_path
            ).to(device)
            RSA = AlexNetRSA(model)
        else:
            model_path = os.path.join(
                models_dir, model_name, f"epoch_{epoch:02d}.pth.tar"
            )
            model = load_model(
                arch=arch, num_classes=num_classes, model_path=model_path
            ).to(device)
            RSA = AlexNetRSA(model)

        result_file = f"{analysis}_embedded_activations_{num_dim}d_p{perplexity}_{num_classes}-class_{model_name}.npy"
        result_path = os.path.join(results_dir, result_file)

        # load t-SNE
        embedded_activations = np.load(result_path)

        colors = ["k", "r", "g", "b", "c", "m", "y"]

        # plot t-SNE
        for layer_id, layer in tqdm(enumerate(RSA.layers), "plotting (each layer)"):
            for image_id in range(test_loader.num_images):
                for filter_id in range(num_filters + 1):
                    fig = plt.figure(dpi=150)
                    target = embedded_activations[layer_id, image_id, filter_id]
                    if num_dim == 2:
                        plt.scatter(
                            x=target[0],
                            y=target[1],
                            label=f"f{filter_id}",
                            color=colors[filter_id],
                            alpha=0.5,
                        )
                    else:
                        fig = plt.figure(dpi=150).gca(projection="3d")
                        fig.scatter(
                            xs=target[0],
                            ys=target[1],
                            zs=target[2],
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

            # fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
            plt.title(
                f"{analysis}, p={perplexity}, {num_classes}, {rename_model_name(model_name)}, {layer}",
                fontsize=10,
            )
            # fig.tight_layout()
            plot_file = f"{analysis}_{num_dim}d_p{perplexity}_{num_classes}-class_{model_name}_{layer}.png"
            plot_path = os.path.join(plots_dir, plot_file)
            fig.savefig(plot_path)
