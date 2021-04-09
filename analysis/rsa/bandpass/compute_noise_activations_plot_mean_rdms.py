import os
import pathlib
import sys

import numpy as np

import torch

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.bandpass.mean_rdms import (
    compute_mean_rdms_with_bandpass,
    plot_bandpass_rdms,
)
from src.analysis.rsa.rsa import AlexNetRSA
from src.analysis.rsa.utils import save_rdms
from src.dataset.imagenet16 import load_imagenet16
from src.image_process.bandpass_filter import make_bandpass_filters
from src.model.utils import load_model


if __name__ == "__main__":
    # ===== args =====
    arch = "alexnet"
    num_classes = 16
    epoch = 60

    imagenet_path = "/mnt/data1/ImageNet/ILSVRC2012/"

    num_filters = 6
    add_noise = True
    mean, var = 0, 0.1  # gaussian noise parameters
    metrics = "correlation"  # "1-covariance", "negative-covariance"

    print("===== arguments =====")
    print("num_classes:", num_classes)
    print("num_filters:", num_filters)
    print("add_noise:", add_noise)
    print("mean:", mean)
    print("var:", var)
    print("metrics:", metrics)
    print()

    # I/O settings
    models_dir = "/mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
        16 if num_classes == 16 else ""  # else is (num_classes == 1000)
    )
    results_dir = f"./results/mean_rdms_noise_{metrics}/{num_classes}-class-{arch}/"
    plots_dir = f"./plots/mean_rdms_noise_{metrics}/{num_classes}-class-{arch}/"

    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("===== I/O =====")
    print("IN, models_dir:", models_dir)
    print("OUT, results_dir:", results_dir)
    print("OUT, plots_dir:", plots_dir)
    print()

    # models to compare
    model_names = [
        f"{arch}_normal",
        # f"{arch}_multi-steps",
    ]
    modes = [
        # f"{arch}_all",
        # f"{arch}_mix",
        # f"{arch}_random-mix",
        # f"{arch}_single-step",
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

    print("===== models to analyze =====")
    print(model_names)
    print()

    # ===== main =====
    print("===== main =====")

    seed = 42
    # random seed settings
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

    for model_name in model_names:
        model_path = os.path.join(models_dir, model_name, f"epoch_{epoch:02d}.pth.tar")
        model = load_model(
            arch=arch, num_classes=num_classes, model_path=model_path
        ).to(device)

        # ===== compute RDM =====
        print(model_name, "computing RDM ...")
        # make RSA instance
        RSA = AlexNetRSA(model)

        # compute mean RDMs
        mean_rdms = compute_mean_rdms_with_bandpass(
            RSA=RSA,
            data_loader=test_loader,
            filters=filters,
            add_noise=True,
            mean=mean,
            var=var,
            metrics=metrics,
            device=device,
        )

        # save mean RDMs
        print("saving RDM...")
        result_file = f"{model_name}_e{epoch:02d}.pkl"
        result_path = os.path.join(results_dir, result_file)
        save_rdms(mean_rdms=mean_rdms, file_path=result_path)

        print(model_name, "computing RDM is DONE!")

        # ===== plot RDM =====
        print("plotting ...")
        # get analysis parameters.
        # num_images = mean_rdms["num_images"]
        num_filters = mean_rdms["num_filters"]

        # (optional) set title
        title = f"RDM ({metrics}), {num_classes}-class, {model_name}, epoch={epoch}"

        # set filename
        filename = f"mean_rdms_{metrics}_{num_classes}-class_{model_name}_e{epoch}_f{num_filters}.png"
        out_file = os.path.join(plots_dir, filename)

        # colour value range of the plots
        vmin = 0 if metrics == "correlation" else -5
        vmax = 2 if metrics == "correlation" else 5

        # plot_rdms(rdms=diff_rdms, out_file=out_file, plot_show=True)
        plot_bandpass_rdms(
            rdms=mean_rdms,
            num_filters=num_filters,
            vmin=vmin,
            vmax=vmax,
            title=title,
            out_file=out_file,
            show_plot=False,
        )

        print(model_name, "plotting is DONE!", end="\n\n")
