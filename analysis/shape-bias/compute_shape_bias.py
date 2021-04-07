import os
import pathlib
import sys

import torch

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

from src.analysis.shape_bias.shape_bias import compute_shape_bias
from src.model.load_sin_pretrained_models import load_sin_model
from src.model.utils import load_model

from vonenet import get_model


if __name__ == "__main__":
    arch = "alexnet"
    num_classes = 1000
    epoch = 60
    models_dir = "/mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
        16 if num_classes == 16 else ""  # else is (num_classes == 1000)
    )
    results_dir = f"./results/{num_classes}-class-{arch}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # models to compare
    model_names = [
        f"{arch}_normal",
        f"{arch}_multi-steps",
    ]
    modes = [
        f"{arch}_all",
        f"{arch}_mix",
        f"{arch}_random-mix",
        f"{arch}_single-step",
        # f"{arch}_fixed-single-step",
        # f"{arch}_reversed-single-step",
    ]

    # sigmas to compare
    sigmas_mix = [s for s in range(1, 6)] + [10]
    sigmas_random_mix = ["00-05", "00-10"]

    # add sigma to compare to the model names
    for mode in modes:
        if mode == f"{arch}_random-mix":
            for min_max in sigmas_random_mix:
                model_names += [f"{mode}_s{min_max}"]
        elif mode == f"{arch}_mix":
            for sigma in sigmas_mix:
                model_names += [f"{mode}_s{sigma:02d}"]
        else:
            for sigma in range(1, 5):
                model_names += [f"{mode}_s{sigma:02d}"]

    # VOneNet
    model_names += ["{}_vonenet".format(arch)]

    # Stylized-ImageNet
    sin_names = {
        "alexnet": "alexnet_trained_on_SIN",
        "vgg16": "vgg16_trained_on_SIN",
        "resnet50": "resnet50_trained_on_SIN",
    }
    model_names += sin_names[arch]

    for model_name in model_names:
        print(model_name)
        # load model
        if "vonenet" in model_name:
            model = get_model(model_arch=arch, pretrained=True).to(device)
            all_file = os.path.join(
                results_dir, "all_decisions_{}.csv".format(model_name)
            )
            correct_file = os.path.join(
                results_dir, "correct_decisions_{}.csv".format(model_name)
            )
        elif "SIN" in model_name:
            model = load_sin_model(model_name).to(device)
            all_file = os.path.join(
                results_dir, "all_decisions_{}.csv".format(model_name)
            )
            correct_file = os.path.join(
                results_dir, "correct_decisions_{}.csv".format(model_name)
            )
        else:
            model_path = os.path.join(
                models_dir, model_name, "epoch_{}.pth.tar".format(epoch)
            )
            model = load_model(model_path).to(device)
            all_file = os.path.join(
                results_dir, "all_decisions_{}_e{}.csv".format(model_name, epoch)
            )
            correct_file = os.path.join(
                results_dir, "correct_decisions_{}_e{}.csv".format(model_name, epoch)
            )

        # compute
        df_all_decisions, df_correct_decisions = compute_shape_bias(model=model)

        # save
        df_all_decisions.to_csv(all_file)
        df_correct_decisions.to_csv(correct_file)
