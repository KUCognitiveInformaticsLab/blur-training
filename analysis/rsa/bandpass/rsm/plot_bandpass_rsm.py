import os
import pathlib
import sys

import numpy as np
import torch
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../../")

from src.analysis.rsa.bandpass.bandpass_rsm import (
    plot_bandpass_RSMs,
    plot_bandpass_RSMs_flatt,
)
from src.analysis.rsa.utils import load_rsms
from src.model.load_sin_pretrained_models import sin_names
from src.model.model_names import rename_model_name
from src.analysis.rsa.rsa import alexnet_layers, vone_alexnet_layers

if __name__ == "__main__":
    # ===== args =====
    arch = "alexnet"
    args = sys.argv
    num_classes = int(args[1])
    epoch = 60

    imagenet_path = "/mnt/data1/ImageNet/ILSVRC2012/"

    num_filters = 6
    add_noise = False
    metrics = "correlation"  # "1-covariance", "negative-covariance"
    analysis = f"bandpass_rsm_{metrics}"

    # I/O settings
    in_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/rsa/bandpass/rsm/results/{analysis}/{num_classes}-class/"
    # in_dir = f"./results/{analysis}/{num_classes}-class/"
    plots_dir = f"./plots/{analysis}/{num_classes}-class/"

    os.makedirs(in_dir, exist_ok=True)
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
    print("OUT, in_dir:", in_dir)
    print("OUT, plots_dir:", plots_dir)
    print()

    # models to compare
    model_names = [
        "untrained_alexnet",
        "alexnet_normal",
        "alexnet_all_s04",
        "alexnet_mix_s04",
        f"{arch}_multi-steps",
        sin_names[arch],
        "vone_alexnet",
    ]

    # model_names = get_model_names(arch=arch)

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

    for model_name in tqdm(model_names, desc="models"):
        if "vone" in model_name:
            layers = vone_alexnet_layers
        else:
            layers = alexnet_layers

        # load mean RSMs
        # print("saving RSM...")
        result_file = f"{analysis}_{num_classes}-class_{model_name}.pkl"
        result_path = os.path.join(in_dir, result_file)
        mean_rsms = load_rsms(file_path=result_path)

        # ===== plot RSM =====
        print(f"{model_name}: plotting RSM...")
        # get analysis parameters.
        # num_images = mean_rsms["num_images"]
        num_filters = mean_rsms["num_filters"]

        # (optional) set title
        plot_title = f"{num_classes}-class, {rename_model_name(model_name)}"

        # set plot filename
        plot_file = f"{analysis}_flatt_{num_classes}-class_{model_name}.png"
        plot_path = os.path.join(plots_dir, plot_file)

        # colour value range of the plots
        vmin = -1
        vmax = 1

        # plot_rsms(rsms=diff_rsms, out_file=out_file, plot_show=True)
        plot_bandpass_RSMs_flatt(
            rsms=mean_rsms,
            layers=layers,
            num_filters=num_filters,
            vmin=vmin,
            vmax=vmax,
            title=plot_title,
            out_file=plot_path,
            show_plot=False,
        )

    print(f"{analysis}: Plotting done!!")
