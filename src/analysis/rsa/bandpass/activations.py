import os
import pathlib
import sys

import numpy as np
import torch

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.model.utils import load_model
from src.analysis.rsa.rsa import AlexNetRSA
from src.analysis.rsa.utils import save_activations
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
from src.image_process.noise import gaussian_noise


def main(
    arch: str = "alexnet",
    num_classes: int = 16,
    model_names: list = ["alexnet_normal"],
    epoch: int = 60,
    models_dir: str = "/mnt/work/blur-training/imagenet16/logs/models/",  # model directory
    results_dir: str = "./results/alexnet_bandpass/activations",
    dataset_path="/mnt/data1/ImageNet/ILSVRC2012/",
    # all_filter_combinations: bool = False,
    num_filters: int = 6,  # number of band-pass filters
    seed: int = 42,
):
    """Computes band-pass test images."""
    # I/O settings
    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)

    # data settings
    # num_data = 1600

    # random seed settings
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make Dataloader
    # ** batch_size must be 1 **
    _, test_loader = load_imagenet16(imagenet_path=dataset_path, batch_size=1)

    # make filters
    filters = make_bandpass_filters(num_filters=num_filters)

    for model_name in model_names:
        model_path = os.path.join(models_dir, model_name, f"epoch_{epoch:02d}.pth.tar")
        model = load_model(
            arch=arch, num_classes=num_classes, model_path=model_path
        ).to(device)

        out_dir = os.path.join(results_dir, f"{model_name}_e{epoch:02d}")
        os.makedirs(out_dir, exist_ok=True)

        compute_save(
            model,
            device=device,
            data_loader=test_loader,
            filters=filters,
            out_dir=out_dir,
        )


def compute_save(
    model,
    device: torch.device,
    data_loader: iter,
    filters: dict,
    out_dir: str,
):
    RSA = AlexNetRSA(model)

    for image_id, (image, label) in enumerate(data_loader):
        """Note that data_loader returns single image for each loop
        image (torch.Tensor): torch.Size([1, 3, 375, 500])
        label (torch.Tensor): e.g. tensor([0])
        """
        activations = compute_activations_with_bandpass(
            RSA=RSA, image=image, label=label, filters=filters, device=device
        )

        # save (This file size is very big with iterations!)
        file_name = f"image{image_id:04d}_f{len(filters):02d}.pkl"
        file_path = os.path.join(out_dir, file_name)
        save_activations(activations=activations, file_path=file_path)


def compute_activations_with_bandpass(
    RSA,
    image: torch.Tensor,
    filters: dict,
    add_noise: bool = False,
    mean: float = 0.0,
    var: float = 0.1,
    device: torch.device = torch.device("cuda:0"),
):
    """Computes activations of a single image with band-pass filters applied.
    Args:
        image (torch.Tensor): torch.Size([1, C, H, W])
        label (torch.Tensor): e.g. tensor([0])

    Returns:
        activations (Dict)
    """
    test_images = torch.zeros([len(filters) + 1, 1, num_channels, height, width])

    test_images[0] = image  # add raw images

    # if add_noise:  # for smoothing high-freq. activations
    #     image = gaussian_noise(images=image, mean=mean, var=var)

    for i, (s1, s2) in enumerate(filters.values(), 1):
        test_images[i] = apply_bandpass_filter(images=image, sigma1=s1, sigma2=s2)

        if add_noise:  # for smoothing high-freq. activations
            test_images[i] = gaussian_noise(images=image, mean=mean, var=var)

    # change the order of num_images and num_filters(+1)
    test_images = test_images.transpose(1, 0)  # (F+1, 1, C, H, W) -> (1, F+1, C, H, W)

    activations = RSA.compute_activations(test_images[0].to(device))

    return activations
