import os
import pathlib
import pickle
import sys

import numpy as np
import torch
import torchvision

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.utils.model import load_model
from src.utils.image import imsave
from src.analysis.rsa.rsa import AlexNetRSA
from src.image_process.bandpass_images import (
    make_bandpass_images,
    make_bandpass_images_all_comb,
)
from src.dataset.imagenet16 import label_map


def analyze(
    models_dir: str,
    arch: str,
    model_name: str,
    epoch: int,
    device: torch.device,
    test_images: torch.Tensor,
    target_id: int,
    num_filters: int,
    num_images: int,
    out_dir: str,
):
    model_path = os.path.join(models_dir, model_name, f"epoch_{epoch:02d}.pth.tar")
    model = load_model(arch=arch, model_path=model_path).to(device)

    RSA = AlexNetRSA(model)

    for n in range(num_images):
        activations = RSA.compute_activations(test_images[n])
        # print(activations["conv-relu-1"].shape)  # torch.Size([F+1, 64, 55, 55])

        # add parameter settings of this analysis
        activations["target_id"] = target_id
        activations["num_filters"] = num_filters

        # save
        file_name = f"{model_name}_e{epoch:02d}_l{target_id:02d}_f{num_filters:02d}_n{n:03d}.pkl"
        file_path = os.path.join(out_dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(activations, f)


def main(
    arch: str = "alexnet",
    model_names: list = ["alexnet_normal"],
    epoch: int = 60,
    models_dir: str = "/mnt/work/blur-training/imagenet16/logs/models/",  # model directory
    out_dir: str = "./results/alexnet_bandpass/activations",
    dataset_path="/mnt/data1/ImageNet/ILSVRC2012/",
    all_filter_combinations: bool = False,
    test_images_dir: str = "./test-images",  # directory for test images overview file
    save_test_images: bool = False,
    target_id: int = 1,  # bear
    num_filters: int = 6,  # number of band-pass filters
    num_images: int = 10,  # number of images for each class.
    seed: int = 42,
):
    """Computes band-pass test images."""
    # I/O settings
    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if save_test_images and not os.path.exists(test_images_dir):
        os.makedirs(test_images_dir)

    # data settings
    # num_data = 1600

    # random seed settings
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load and make test images
    if all_filter_combinations:
        test_images = make_bandpass_images_all_comb(
            dataset_path=dataset_path,
            target_id=target_id,
            num_filters=num_filters,
            num_images=num_images,
        ).to(device)
    else:
        test_images = make_bandpass_images(
            dataset_path=dataset_path,
            target_id=target_id,
            num_filters=num_filters,
            num_images=num_images,
        ).to(
            device
        )  # (N, F+1, C, H, W)

    # save test images (if needed)
    if save_test_images:
        image_name = f"bandpass_{label_map[target_id]}_n{num_images}.png"
        imsave(
            torchvision.utils.make_grid(
                test_images.reshape(-1, *test_images.shape[2:]).cpu(),
                nrow=test_images.shape[1],
            ),
            filename=os.path.join(test_images_dir, image_name),
            unnormalize=True,
        )

    for model_name in model_names:
        analyze(
            models_dir=models_dir,
            arch=arch,
            model_name=model_name,
            epoch=epoch,
            device=device,
            test_images=test_images,
            target_id=target_id,
            num_filters=num_filters,
            num_images=num_images,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    arch = "alexnet"
    mode = "normal"
    model_names = [f"{arch}_{mode}"]
    out_dir = f"./results/{arch}/activations"

    all_filter_combinations = False
    # if all_filter_combinations:
    #     out_dir = f"./results/{arch}_bandpass_all_filter_comb/activations"
    # else:
    #     out_dir = f"./results/{arch}_bandpass/activations"

    main(
        arch=arch,
        model_names=model_names,
        epoch=60,
        models_dir="/mnt/work/blur-training/imagenet16/logs/models/",  # model directory
        out_dir=out_dir,
        dataset_path="/mnt/data1/ImageNet/ILSVRC2012/",
        all_filter_combinations=all_filter_combinations,
        test_images_dir="./test-images",  # directory for test images overview file
        save_test_images=False,
        target_id=1,  # bear
        num_filters=6,  # number of band-pass filters
        num_images=10,  # number of images for each class.
        seed=42,
    )
