import os
import pathlib
import pickle
import sys

import numpy as np
import torch

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.utils.model import load_model
from src.analysis.rsa.rsa import AlexNetRSA
from src.dataset.imagenet16 import (
    load_imagenet16,
    num_channels,
    height,
    width,
)
from src.image_process.bandpass_filter import (
    make_bandpass_filters,
    apply_bandpass_filter,
)


def main(
    arch: str = "alexnet",
    num_classes: int = 16,
    model_names: list = ["alexnet_normal"],
    epoch: int = 60,
    models_dir: str = "/mnt/work/blur-training/imagenet16/logs/models/",  # model directory
    out_dir: str = "./results/alexnet_bandpass/activations",
    dataset_path="/mnt/data1/ImageNet/ILSVRC2012/",
    # all_filter_combinations: bool = False,
    num_filters: int = 6,  # number of band-pass filters
    seed: int = 42,
):
    """Computes band-pass test images."""
    # I/O settings
    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    os.makedirs(out_dir, exist_ok=True)

    # data settings
    # num_data = 1600

    # random seed settings
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make Dataloader
    _, test_loader = load_imagenet16(imagenet_path=dataset_path, batch_size=1)

    # make filters
    filters = make_bandpass_filters(num_filters=num_filters)

    for model_name in model_names:
        analyze(
            models_dir=models_dir,
            arch=arch,
            num_classes=num_classes,
            model_name=model_name,
            epoch=epoch,
            device=device,
            data_loader=test_loader,
            filters=filters,
            out_dir=out_dir,
        )


def analyze(
    models_dir: str,
    arch: str,
    num_classes: int,
    model_name: str,
    epoch: int,
    device: torch.device,
    data_loader: iter,
    filters: dict,
    out_dir: str,
):
    model_path = os.path.join(models_dir, model_name, f"epoch_{epoch:02d}.pth.tar")
    model = load_model(arch=arch, num_classes=num_classes, model_path=model_path).to(
        device
    )

    out_model_dir = os.path.join(out_dir, f"{model_name}_e{epoch:02d}")
    os.makedirs(out_model_dir, exist_ok=True)

    RSA = AlexNetRSA(model)

    for image_id, (image, label) in enumerate(data_loader):
        """Note that data_loader returns single image for each loop
        image (torch.Tensor): torch.Size([1, 3, 375, 500])
        label (torch.Tensor): e.g. tensor([0])
        """
        # make bandpass images
        test_images = torch.zeros([len(filters) + 1, 1, num_channels, height, width])
        test_images[0] = image  # add raw images
        for i, (s1, s2) in enumerate(filters.values(), 1):
            test_images[i] = apply_bandpass_filter(images=image, sigma1=s1, sigma2=s2)

        # change the order of num_images and num_filters(+1)
        test_images = test_images.transpose(1, 0)  # (1, F+1, C, H, W)

        activations = RSA.compute_activations(test_images[0].to(device))
        # print(activations["conv-relu-1"].shape)  # torch.Size([F+1, 64, 55, 55])

        # add parameter settings of this analysis
        activations["label_id"] = label.item()
        activations["num_filters"] = len(filters)

        # save
        file_name = f"image{image_id:04d}_f{len(filters):02d}.pkl"
        file_path = os.path.join(out_model_dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(activations, f)


if __name__ == "__main__":
    # arguments
    arch = "alexnet"
    num_classes = 16
    epoch = 60
    models_dir = "/mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
        16 if num_classes == 16 else ""  # else is (num_classes == 1000)
    )
    out_dir = f"/home/sou/work/blur-training/analysis/rsa/bandpass/results/activations/{num_classes}-class-{arch}"
    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # all_filter_combinations = False
    # if all_filter_combinations:
    #     out_dir = f"./results/{arch}_bandpass_all_filter_comb/activations"
    # else:
    #     out_dir = f"./results/{arch}_bandpass/activations"

    # models to compare
    modes = [
        "normal",
        # "all",
        # "mix",
        # "random-mix",
        # "single-step",
        # "fixed-single-step",
        # "reversed-single-step",
        # "multi-steps",
    ]

    # sigmas to compare
    sigmas_mix = [s for s in range(1, 6)] + [10]
    sigmas_random_mix = ["00-05", "00-10"]

    # make model name list
    model_names = []
    for mode in modes:
        if mode in ("normal", "multi-steps"):
            model_names += [f"{arch}_{mode}"]
        elif mode == "random-mix":
            for min_max in sigmas_random_mix:
                model_names += [f"{arch}_{mode}_s{min_max}"]
        elif mode == "mix":
            for sigma in sigmas_mix:
                model_names += [f"{arch}_{mode}_s{sigma:02d}"]
        else:
            for s in range(4):
                model_names += [f"{arch}_{mode}_s{s + 1:02d}"]

    main(
        arch=arch,
        num_classes=num_classes,
        model_names=model_names,
        models_dir=models_dir,  # model directory
        out_dir=out_dir,
        dataset_path="/mnt/data1/ImageNet/ILSVRC2012/",
        # all_filter_combinations=all_filter_combinations,
        num_filters=6,  # number of band-pass filters
        seed=42,
    )
