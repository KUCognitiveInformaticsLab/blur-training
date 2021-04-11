import os
import pathlib
import sys

import numpy as np
import torch
import vonenet
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.bandpass.mean_rdms import (
    compute_mean_rdms_with_bandpass,
    plot_bandpass_rdms,
)
from src.analysis.rsa.rsa import VOneNetAlexNetRSA, AlexNetRSA
from src.analysis.rsa.utils import save_rdms
from src.dataset.imagenet16 import load_imagenet16
from src.model.utils import load_model
from src.model.load_sin_pretrained_models import load_sin_model, sin_names
from src.image_process.bandpass_filter import make_bandpass_filters

if __name__ == "__main__":
    # ===== args =====
    arch = "alexnet"
    num_classes = 1000
    epoch = 60

    num_filters = 6
    add_noise = False
    metrics = "correlation"  # "1-covariance", "negative-covariance"
    analysis = f"bandpass_rdm_{metrics}"

    imagenet_path = "/mnt/data1/ImageNet/ILSVRC2012/"

    # I/O settings
    models_dir = "/mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
        16 if num_classes == 16 else ""  # else is (num_classes == 1000)
    )
    results_dir = f"./results/{analysis}/{num_classes}-class-{arch}/"
    plots_dir = f"./plots/{analysis}/{num_classes}-class-{arch}/"

    # models to compare
    model_names = [
        # f"{arch}_normal",
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

    # models to compare
    model_names += [
        # "vone_alexnet",
    ]
    model_names += [sin_names[arch]]

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

    # assert os.path.exists(models_dir), f"{models_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("===== arguments =====")
    # print("num_classes:", num_classes)
    print("num_filters:", num_filters)
    print("add_noise:", add_noise)
    print("metrics:", metrics)
    print()

    print("===== I/O =====")
    # print("IN, models_dir:", models_dir)
    print("OUT, results_dir:", results_dir)
    print("OUT, plots_dir:", plots_dir)
    print()

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

    for model_name in tqdm(model_names, desc="models"):
        if "SIN" in model_name:
            # Stylized-ImageNet
            load_sin_model(model_name).to(device)
            RSA = AlexNetRSA(model)
        elif "vone" in model_name:
            model = vonenet.get_model(model_arch=arch, pretrained=True).to(device)
            RSA = VOneNetAlexNetRSA(model)
        else:
            model_path = os.path.join(
                models_dir, model_name, f"epoch_{epoch:02d}.pth.tar"
            )
            model = load_model(
                arch=arch, num_classes=num_classes, model_path=model_path
            ).to(device)

        # ===== compute RDM =====
        print()
        print(f"{model_name}: computing RDM...")

        # compute mean RDMs
        mean_rdms = compute_mean_rdms_with_bandpass(
            RSA=RSA,
            data_loader=test_loader,
            filters=filters,
            add_noise=add_noise,
            metrics=metrics,
            device=device,
        )

        # save mean RDMs
        # print("saving RDM...")
        result_file = f"{analysis}_{model_name}.pkl"
        result_path = os.path.join(results_dir, result_file)
        save_rdms(mean_rdms=mean_rdms, file_path=result_path)

        # ===== plot RDM =====
        # print(f"{model_name}: plotting RDM...")
        # get analysis parameters.
        # num_images = mean_rdms["num_images"]
        num_filters = mean_rdms["num_filters"]

        # (optional) set title
        title = f"RDM ({metrics}) , {model_name}"

        # set filename
        filename = f"{analysis}_{model_name}.png"
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

    print("All done!!")
