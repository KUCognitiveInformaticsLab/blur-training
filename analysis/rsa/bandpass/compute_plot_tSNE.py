import argparse
import os
import pathlib
import sys
from distutils.util import strtobool

import numpy as np
import torch
import vonenet
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.bandpass.t_sne import (
    compute_bandpass_tSNE,
    save_embedded_activations,
    load_embedded_activations,
    plot_tSNE,
)
from src.analysis.rsa.rsa import AlexNetRSA, VOneNetAlexNetRSA
from src.dataset.imagenet16 import make_local_in16_test_loader
from src.image_process.bandpass_filter import make_bandpass_filters
from src.model.utils import load_model
from src.model.load_sin_pretrained_models import load_sin_model, sin_names


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
        "alexnet_all_s04",
        "alexnet_mix_s04",
        sin_names[arch],
        "vone_alexnet",
        "untrained_alexnet",
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
    print("IN, models_dir:", models_dir)
    print("OUT, results_dir:", results_dir)
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
        # ===== compute RSM =====
        # make RSA instance
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

        # compute bandpass tSNE
        if compute:
            print(f"{model_name} computing...")
            embed, labels = compute_bandpass_tSNE(
                RSA=RSA,
                num_images=test_loader.num_images,
                data_loader=test_loader,
                filters=filters,
                num_dim=num_dim,
                perplexity=perplexity,
                n_iter=n_iter,
                device=device,
            )  # (F+1, L, N, D), (N)

        result_file = f"{analysis}_embedded_activations_{num_dim}d_p{perplexity}_i{n_iter}_{num_classes}-class_{model_name}.npy"
        result_path = os.path.join(results_dir, result_file)

        if compute:
            # save t-SNE embedded activations
            save_embedded_activations(
                embedded_activations=embed, labels=labels, file_path=result_path
            )

        # === plot t-SNE ===
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
