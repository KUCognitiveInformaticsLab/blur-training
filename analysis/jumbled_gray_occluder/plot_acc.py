import os
import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import pandas as pd

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

from src.model.model_names import rename_model_name
from src.model.plot import colors, lines
from src.analysis.classification.acc import load_acc1


if __name__ == "__main__":
    num_classes = int(sys.argv[1])  # number of last output of the models
    test_dataset = str(sys.argv[2])  # test_dataset to use
    stimuli = str(
        sys.argv[3]
    )  # ("jumbled", "gray_occluder", "jumbled_with_gray_occluder")
    scales = [1, 4, 8, 16, 32]  # 1 == original

    metrics = "acc1"

    arch = "alexnet"
    epoch = 60
    batch_size = 64

    analysis = f"{stimuli}_{test_dataset}"

    machine = "server"  # ("server", "local")

    # directories and model settings
    in_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/jumbled_gray_occluder/results/{analysis}/{num_classes}-class/"
    # out_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/lowpass_acc/plots/{analysis}/{num_classes}-class/"
    lowpass_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/lowpass_acc/results/lowpass_acc_{test_dataset}/{num_classes}-class/"  # for acc on original
    out_dir = f"./plots/{num_classes}-class/"
    out_server_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/jumbled_gray_occluder/plots/{analysis}_vs_humans/{num_classes}-class/"

    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    os.makedirs(out_dir, exist_ok=True)

    # models to plot
    model_names = [
        f"{arch}_normal",
        f"{arch}_all_s04",
        f"{arch}_mix_s04",
        f"{arch}_multi-steps",
        # f"{arch}_mix_s10",
        # f"{arch}_random-mix_s00-05",
        # f"{arch}_random-mix_s00-10",
        # f"{arch}_trained_on_SIN",
        # f"vone_{arch}",
    ]

    # model_names = [
    #     f"{arch}_normal",
    #     # f"{arch}_all_s01",
    #     # f"{arch}_all_s02",
    #     # f"{arch}_all_s03",
    #     # f"{arch}_all_s04",
    #     # f"{arch}_mix_s01",
    #     # f"{arch}_mix_s02",
    #     # f"{arch}_mix_s03",
    #     f"{arch}_mix_s04",
    #     # f"{arch}_mix_s01_no-blur-1label",
    #     # f"{arch}_mix_s01_no-blur-8label",
    #     # f"{arch}_mix_s04_no-blur-1label",
    #     # f"{arch}_mix_s04_no-blur-8label",
    #     f"{arch}_mix_s04_no-sharp-1label",
    #     f"{arch}_mix_s04_no-sharp-8label",
    #     # f"{arch}_mix_p-blur_s01",
    #     # f"{arch}_mix_p-blur_s04",
    #     # f"{arch}_mix_p-blur_s01_no-blur-1label",
    #     # f"{arch}_mix_p-blur_s01_no-blur-8label",
    #     # f"{arch}_mix_p-blur_s04_no-blur-1label",
    #     # f"{arch}_mix_p-blur_s04_no-blur-8label",
    #     # f"{arch}_multi-steps",
    # ]

    model_names = [
        f"{arch}_normal",
        f"{arch}_all_s04",
        f"{arch}_mix_s04",
        f"{arch}_mix_s04",
        f"{arch}_multi-steps",
        f"vone_{arch}_normal",
        f"vone_{arch}_all_s04",
        f"vone_{arch}_mix_s04",
        f"vone_{arch}_random-mix_s00-04",
        f"vone_{arch}_multi-steps",
    ]

    x = [f"{div_v}x{div_v}" for div_v in scales]
    x[0] = "original"

    # read accuracy results
    acc = {}
    for model_name in model_names:
        acc_scales = []
        for div_v in scales:
            div_h = div_v
            if div_v == 1:  # original images
                file_path = os.path.join(
                    lowpass_dir,
                    f"lowpass_acc_{test_dataset}_{num_classes}-class_{model_name}_{metrics}.csv",
                )
            else:
                file_path = os.path.join(
                    in_dir,
                    f"{num_classes}-class_{model_name}_{analysis}_{div_v}x{div_h}_{metrics}.csv",
                )
            if "SIN" in model_name:
                file_path = file_path.replace("16-class", "1000-class")
                file_path = file_path.replace("imagenet16", "imagenet1000")

            if div_v == 1:  # original images
                acc_scales += [pd.read_csv(file_path, index_col=0).values[0][0] / 100]
            else:
                acc_scales += [load_acc1(file_path=file_path)]

        acc[model_name] = acc_scales

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(
        1,
        1,
        1,
        title=(
            f"Top-{metrics[-1]} acc. on ImageNet, {stimuli}"
            if test_dataset == "imagenet1000"
            else f"Top-{metrics[-1]} acc. on 16-class-ImageNet, {stimuli}"
        ),
        xlabel="",
        ylabel=f"Top-{metrics[-1]} accuracy",
        ylim=(0, 1),
    )
    for model_name in model_names:
        ax.plot(x[0], acc[model_name][0], marker="o", color=colors[model_name])
        ax.plot(
            list(reversed(x[1:])),
            list(reversed(acc[model_name][1:])),
            label=rename_model_name(model_name),
            marker="o",
            ls=lines[model_name],
            # ls=":" if model_name == f"{arch}_normal" else "-",
            color=colors[model_name],
        )

    # plot chance level performance
    plt.hlines(
        [1 / 16],
        xmin=0,
        xmax=len(x) - 1,
        colors="k",
        linestyles="dashed",
        label="Chance performance",
    )

    ax.legend()
    # import numpy as np
    # ax.set_xticks(np.arange(scales))
    plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(10))
    ax.xaxis.set_major_locator(tick.MultipleLocator(1))
    ax.grid(which="major")
    ax.grid(which="minor")

    # set plot file name.
    plot_file = f"{analysis}_{metrics}_{num_classes}-class_{model_names}.png"

    # fig.show()
    fig.savefig(os.path.join(out_dir, plot_file))
