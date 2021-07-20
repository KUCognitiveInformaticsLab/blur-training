#!/usr/bin/env python
# coding: utf-8

# !pip install robustness==1.1  # (or 1.1.post2)

import os
import pathlib
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

from src.analysis.lowpass_acc.lowpass_acc import plot_confusion_matrix
from src.model.model_names import rename_model_name

if __name__ == "__main__":
    # ===== args =====
    arch = "alexnet"
    num_classes = int(sys.argv[1])  # number of last output of the models
    test_dataset = str(sys.argv[2])  # test_dataset to use
    compare = str(sys.argv[3])  # models to compare. e.g. "vss", "mix_no-blur"

    epoch = 60
    batch_size = 64
    stimuli = "lowpass"
    analysis = f"{stimuli}_confusion_matrix_{test_dataset}"
    max_sigma = 10  # 20

    machine = "local"  # ("server", "local")

    # I/O
    results_dir = (
        f"./results/{analysis}/{num_classes}-class/"
        if machine == "server"
        else f"/Users/sou/lab2-work/blur-training-dev/analysis/lowpass_acc/results/{analysis}/{num_classes}-class/"
    )

    plots_dir = f"./plots/{analysis}/{num_classes}-class/"
    plots_server_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/lowpass_acc/plots/{analysis}/{num_classes}-class/"

    # assert os.path.exists(models_dir), f"{models_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # models to compare
    from src.model.model_names import get_model_names

    model_names = get_model_names(arch=arch, models=compare)

    print("===== arguments =====")
    print("num_classes:", num_classes)
    print("batch_size:", batch_size)
    print("test_dataset:", test_dataset)
    print("max_sigma:", max_sigma)
    print()

    print("===== I/O =====")
    print("IN, results_dir:", results_dir)
    print("OUT, plots_dir:", plots_dir)
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

    for model_name in tqdm(model_names, desc="models", leave=False):
        for s in tqdm(range(max_sigma + 1), desc="lowpass filters", leave=False):
            # load confusion matrix
            result_name = f"{num_classes}-class_{model_name}_{analysis}_s{s:02d}.npy"
            result_path = os.path.join(results_dir, result_name)
            conf_matrix = np.load(result_path)

            # compute acc
            acc = np.diag(conf_matrix).sum() / conf_matrix.sum()

            # normalize confusion matrix. (divided by # of each class)
            norm_conf_matrix = conf_matrix / (conf_matrix.sum() / num_classes)

            # plot confusion matrix
            # title = f"{test_dataset}, {stimuli} s{s:02d}, {num_classes}-class, {model_name}, acc={acc:.2f}"
            title = f"{num_classes}-class, {rename_model_name(model_name)}, {stimuli} σ={s}"
            plot_name = f"{num_classes}-class_{model_name}_{analysis}_s{s:02d}.png"
            plot_path = os.path.join(plots_dir, plot_name)
            plot_confusion_matrix(
                confusion_matrix=norm_conf_matrix,
                vmin=0,
                vmax=1,
                title=title,
                out_path=plot_path,
            )
            if machine == "local":
                # save to a server
                plot_path = os.path.join(plots_server_dir, plot_name)
                plot_confusion_matrix(
                    confusion_matrix=norm_conf_matrix,
                    vmin=0,
                    vmax=1,
                    title=title,
                    out_path=plot_path,
                )

    print(f"{analysis}: All done!!")
