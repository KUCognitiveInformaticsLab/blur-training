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
    compute_tSNE_each_bandpass,
)
from src.analysis.rsa.rsa import AlexNetRSA, VOneNetAlexNetRSAParallel
from src.dataset.imagenet16 import load_imagenet16
from src.image_process.bandpass_filter import make_bandpass_filters
from src.model.utils import load_model
from src.model.load_sin_pretrained_models import load_sin_model, sin_names

if __name__ == "__main__":
    # ===== args =====
    arch = "alexnet"
    num_classes = 1000
    epoch = 60

    imagenet_path = "/mnt/data1/ImageNet/ILSVRC2012/"

    analysis = f"bandpass_activations_tSNE"
    num_filters = 6
    num_dim = int(sys.argv[1])

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
    print("num_classes:", num_classes)
    print("num_filters:", num_filters)
    print("num_dim:", num_dim)

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

    # random seed settings
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make Dataloader
    # ** batch_size must be 1 **
    _, test_loader = load_imagenet16(imagenet_path=imagenet_path, batch_size=1)

    # make filters
    filters = make_bandpass_filters(num_filters=num_filters)

    for model_name in tqdm(model_names, desc="models"):
        # ===== compute RSM =====
        print()
        print(f"{model_name}: computing RSM...")
        # make RSA instance

        if num_classes == 1000 and "SIN" in model_name:
            # Stylized-ImageNet
            model = load_sin_model(model_name).to(device)
            model.features = model.features.module
            RSA = AlexNetRSA(model)
        elif num_classes == 1000 and "vone" in model_name:
            model = vonenet.get_model(model_arch=arch, pretrained=True).to(device)
            RSA = VOneNetAlexNetRSAParallel(model)
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
        embedded_activations = compute_tSNE_each_bandpass(
            RSA=RSA,
            num_images=test_loader.num_images,
            data_loader=test_loader,
            filters=filters,
            num_dim=num_dim,
            device=device,
        )

        # save t-SNE embedded activations
        result_file = f"{num_classes}-class_{model_name}_{analysis}_embedded_activations_{num_dim}d.npy"
        result_path = os.path.join(results_dir, result_file)
        np.save(result_path, embedded_activations)
