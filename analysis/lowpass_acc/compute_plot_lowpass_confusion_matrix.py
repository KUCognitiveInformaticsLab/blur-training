#!/usr/bin/env python
# coding: utf-8

# !pip install robustness==1.1  # (or 1.1.post2)

import os
import pathlib
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import vonenet
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

from src.dataset.imagenet16 import load_imagenet16
from src.dataset.imagenet import load_imagenet
from src.model.utils import load_model
from src.model.load_sin_pretrained_models import load_sin_model, sin_names
from src.analysis.lowpass_acc.lowpass_acc import (
    compute_confusion_matrix,
    plot_confusion_matrix,
)


if __name__ == "__main__":
    # ===== args =====
    arch = str(sys.argv[1])  # e.g.: ("alexnet", "vone_alexnet")
    num_classes = int(sys.argv[2])  # number of last output of the models
    test_dataset = str(sys.argv[3])  # test_dataset to use
    compare = str(sys.argv[4])  # models to compare e.g.: ("vss", "all_blur-training", "mix_no-blur", "mix_no-sharp")

    epoch = 60
    batch_size = 64
    stimuli = "lowpass"
    analysis = f"{stimuli}_confusion_matrix_{test_dataset}"
    max_sigma = 20

    pretrained = False  # True if you want to use pretrained vone_alexnet.

    machine = "server"  # ("server", "local")

    imagenet_path = (
        "/mnt/data1/ImageNet/ILSVRC2012/"
        if machine == "server"
        else ("/Users/sou/lab2-mnt/data1/ImageNet/ILSVRC2012/")
    )

    # I/O
    models_dir = (
        "/mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
            16 if num_classes == 16 else 1000  # else is (num_classes == 1000)
        )
        if machine == "server"
        else (
            "/Users/sou/lab2-mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
                16 if num_classes == 16 else 1000  # else means (num_classes == 1000)
            )
        )
    )
    results_dir = f"./results/{analysis}/{num_classes}-class/"
    plots_dir = f"./plots/{analysis}/{num_classes}-class/"

    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # models to compare
    from src.model.model_names import get_model_names

    model_names = get_model_names(arch=arch, compare=compare)

    print("===== arguments =====")
    print("num_classes:", num_classes)
    print("batch_size:", batch_size)
    print("test_dataset:", test_dataset)
    print("max_sigma:", max_sigma)
    print()

    print("===== I/O =====")
    print("IN, models_dir:", models_dir)
    print("OUT, results_dir:", results_dir)
    print()

    print("===== models to analyze =====")
    print(model_names)
    print()

    # ===== main =====
    print("===== main =====")

    cudnn.benchmark = True  # for fast running

    # random seed settings
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    # loading data
    if test_dataset == "imagenet16":
        _, test_loader = load_imagenet16(
            imagenet_path=imagenet_path, batch_size=batch_size
        )
    elif test_dataset == "imagenet1000":
        _, _, test_loader = load_imagenet(
            imagenet_path=imagenet_path,
            batch_size=batch_size,
            distributed=False,
            workers=4,
        )

    for model_name in tqdm(model_names, desc="models", leave=False):
        print()
        print(f"{model_name}: computing lowpass confusion matrix...")
        # load model
        if "SIN" in model_name:
            if test_dataset == "imagenet16":
                continue
            # Stylized-ImageNet
            model = load_sin_model(model_name).to(device)
            model.num_classes = num_classes
        elif "vone" in model_name and pretrained:
            model = vonenet.get_model(model_arch=arch, pretrained=True).to(device)
            model.num_classes = num_classes
        elif "untrained" in model_name:
            model_path = ""  # load untrained model
            model = load_model(
                arch=arch, num_classes=num_classes, model_path=model_path
            ).to(device)
            model.num_classes = num_classes
        else:
            model_path = os.path.join(
                models_dir, model_name, "epoch_{}.pth.tar".format(epoch)
            )
            model = load_model(
                arch=arch,
                num_classes=num_classes,
                model_path=model_path,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            ).to(device)
            model.num_classes = num_classes

        # set path to output
        out_path = os.path.join(
            results_dir, f"{analysis}_{num_classes}-class_{model_name}_acc1.csv"
        )  # acc1&acc5 will be saved (when ImageNet is test dataset)

        for s in tqdm(range(max_sigma + 1), desc="lowpass filters", leave=False):
            conf_matrix = compute_confusion_matrix(
                model=model,
                test_loader=test_loader,
                sigma=s,
                device=device,
            )

            # save confusion matrix
            result_name = f"{num_classes}-class_{model_name}_{analysis}_s{s:02d}.npy"
            result_path = os.path.join(results_dir, result_name)
            np.save(result_path, conf_matrix)

            # load confusion matrix
            # conf_matrix = np.load(result_path)

            # normalize confusion matrix. (divided by # of each class)
            norm_conf_matrix = conf_matrix / (conf_matrix.sum() / num_classes)

            # plot confusion matrix
            title = (
                f"{test_dataset}, {stimuli} s{s:02d}, {num_classes}-class, {model_name}"
            )
            plot_name = f"{num_classes}-class_{model_name}_{analysis}_s{s:02d}.png"
            plot_path = os.path.join(plots_dir, plot_name)
            plot_confusion_matrix(
                confusion_matrix=norm_conf_matrix,
                vmin=0,
                vmax=1,
                title=title,
                out_path=plot_path,
            )

    print(f"{analysis}: All done!!")
