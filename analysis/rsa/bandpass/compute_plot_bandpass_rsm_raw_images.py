import os
import pathlib
import sys

import numpy as np
import torch
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.bandpass.bandpass_rsm import (
    compute_raw_RSM,
    plot_bandpass_RSM_raw_images,
)
from src.analysis.rsa.utils import save_rsms
from src.dataset.imagenet16 import load_imagenet16
from src.image_process.bandpass_filter import make_bandpass_filters

if __name__ == "__main__":
    # ===== args =====
    arch = "alexnet"
    num_classes = 1000
    epoch = 60

    imagenet_path = "/mnt/data1/ImageNet/ILSVRC2012/"

    num_filters = 6
    add_noise = False
    args = sys.argv
    metrics = "correlation"  # "1-covariance", "negative-covariance"
    analysis = f"bandpass_rsm_{metrics}"

    # I/O settings
    results_dir = f"./results/{analysis}/"
    plots_dir = f"./plots/{analysis}/"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("===== arguments =====")
    print("num_classes:", num_classes)
    print("num_filters:", num_filters)
    print("add_noise:", add_noise)
    # print("mean:", mean)
    # print("var:", var)
    print("metrics:", metrics)
    print()

    print("===== I/O =====")
    print("OUT, results_dir:", results_dir)
    print("OUT, plots_dir:", plots_dir)
    print()

    # models to compare
    model_names = [
        "raw_images",
    ]

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
        # compute mean RSMs
        mean_rsm = compute_raw_RSM(
            data_loader=test_loader,
            filters=filters,
            add_noise=add_noise,
            metrics=metrics,
            device=device,
        )

        # save mean RSMs
        # print("saving RSM...")
        result_file = f"{analysis}_{model_name}.pkl"
        result_path = os.path.join(results_dir, result_file)
        save_rsms(mean_rsms=mean_rsm, file_path=result_path)

        # # ===== plot RSM =====
        # print(f"{model_name}: plotting RSM...")
        # get analysis parameters.
        # num_images = mean_rsms["num_images"]
        num_filters = mean_rsm["num_filters"]

        # (optional) set title
        plot_title = f"{analysis}, {model_name}"

        # set plot filename
        plot_file = f"{analysis}_{model_name}.png"
        plot_path = os.path.join(plots_dir, plot_file)

        # colour value range of the plots
        vmin = -1
        vmax = 1

        # plot_rsms(rsms=diff_rsms, out_file=out_file, plot_show=True)
        plot_bandpass_RSM_raw_images(
            rsm=mean_rsm["raw"],
            num_filters=num_filters,
            vmin=vmin,
            vmax=vmax,
            title=plot_title,
            out_file=plot_path,
            show_plot=False,
        )

    print("All done!!")
