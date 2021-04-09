import os
import pathlib
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

from src.dataset.imagenet16 import load_imagenet16
from src.dataset.imagenet import load_imagenet
from src.model.utils import load_model
from src.image_process.bandpass_filter import make_bandpass_filters
from src.analysis.bandpass_acc.bandpass_acc import test_performance


if __name__ == "__main__":
    # ===== args =====
    arch = "alexnet"
    num_classes = 16  # number of last output of the models
    epoch = 60
    batch_size = 64

    imagenet_path = "/Users/sou/lab1-mnt/data1/ImageNet/ILSVRC2012/"

    test_dataset = "imagenet16"  # test_dataset to use

    num_filters = 6  # the number of bandpass filters

    print("===== arguments =====")
    print("num_classes:", num_classes)
    print("num_filters:", num_filters)
    print("batch_size:", batch_size)
    print("test_dataset:", test_dataset)
    print()

    models_dir = "/Users/sou/lab1-mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
        16 if num_classes == 16 else ""  # else is (num_classes == 1000)
    )
    results_dir = f"/Users/sou/work/blur-training/analysis/bandpass_acc/results/{num_classes}-class/{arch}/"
    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("===== I/O =====")
    print("IN, models_dir:", models_dir)
    print("OUT, results_dir:", results_dir)
    print()

    # models to compare
    modes = [
        "normal",
        "all",
        "mix",
        "random-mix",
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
    elif test_dataset == "imagenet":
        _, _, test_loader = load_imagenet(
            imagenet_path=imagenet_path,
            batch_size=batch_size,
            distributed=False,
            workers=4,
        )

    # make bandpass bandpass_filters
    bandpass_filters = make_bandpass_filters(num_filters=num_filters)

    for model_name in model_names:
        # load model
        model_path = os.path.join(
            models_dir, model_name, "epoch_{}.pth.tar".format(epoch)
        )
        model = load_model(
            arch=arch,
            num_classes=num_classes,
            model_path=model_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        ).to(device)

        # set path to output
        out_file = os.path.join(
            results_dir, f"{num_classes}-class_{model_name}_e{epoch}_acc1.csv"
        )

        test_performance(
            model=model,
            test_loader=test_loader,
            bandpass_filters=bandpass_filters,
            device=device,
            out_file=out_file,
        )
