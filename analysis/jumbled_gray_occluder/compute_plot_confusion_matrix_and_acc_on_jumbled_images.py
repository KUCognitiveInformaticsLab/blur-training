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
from src.model.load_sin_pretrained_models import load_sin_model
from src.analysis.jumbled_gray_occluder.jumbled_gray_occluder import (
    compute_confusion_matrix,
)
from src.analysis.classification.confusion_matrix import plot_confusion_matrix
from src.analysis.classification.acc import save_acc1


if __name__ == "__main__":
    # ===== args =====
    num_classes = int(sys.argv[1])  # number of last output of the models
    test_dataset = str(sys.argv[2])  # test_dataset to use
    stimuli = str(
        sys.argv[3]
    )  # ("jumbled", "gray_occluder", "jumbled_with_gray_occluder")
    div_v = int(sys.argv[4])  # (4, 8, 16, 32)
    div_h = div_v
    compare = str(sys.argv[5])  # models to compare

    arch = "alexnet"
    epoch = 60
    batch_size = 64

    analysis = f"{stimuli}_{test_dataset}"

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

    model_names = [
        f"{arch}_normal",
    ]

    print("===== arguments =====")
    print("num_classes:", num_classes)
    print("batch_size:", batch_size)
    print("test_dataset:", test_dataset)
    print(f"stimuli: {stimuli} {div_v}x{div_h}", stimuli)
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
        print(f"{model_name}: computing ...")
        # load model
        if "SIN" in model_name:
            if test_dataset == "imagenet16":
                continue
            # Stylized-ImageNet
            model = load_sin_model(model_name).to(device)
            model.num_classes = num_classes
        elif "vone" in model_name:
            if test_dataset == "imagenet16":
                continue
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

        conf_matrix, acc1 = compute_confusion_matrix(
            model=model,
            test_loader=test_loader,
            stimuli=stimuli,
            div_v=div_v,
            div_h=div_h,
            device=device,
        )

        # save acc
        acc1_name = (
            f"{num_classes}-class_{model_name}_{analysis}_{div_v}x{div_h}_acc1.csv"
        )
        acc1_path = os.path.join(results_dir, acc1_name)
        save_acc1(acc1=acc1, save_file=acc1_path)

        # save confusion matrix
        conf_name = f"{num_classes}-class_{model_name}_{analysis}_{div_v}x{div_h}.npy"
        conf_path = os.path.join(results_dir, conf_name)
        np.save(conf_path, conf_matrix)

        # load confusion matrix
        # conf_matrix = np.load(conf_path)

        # normalize confusion matrix. (divided by # of each class)
        norm_conf_matrix = conf_matrix / (conf_matrix.sum() / num_classes)

        # plot confusion matrix
        title = f"{test_dataset}, {stimuli} {div_v}x{div_h}, {num_classes}-class, {model_name}"
        plot_name = f"{num_classes}-class_{model_name}_{analysis}_{div_v}x{div_h}.png"
        plot_path = os.path.join(plots_dir, plot_name)
        plot_confusion_matrix(
            confusion_matrix=norm_conf_matrix,
            vmin=0,
            vmax=1,
            title=title,
            out_path=plot_path,
        )

    print(f"{analysis}: All done!!")
