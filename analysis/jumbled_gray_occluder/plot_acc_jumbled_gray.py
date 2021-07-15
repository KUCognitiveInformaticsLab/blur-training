import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

from src.model.model_names import rename_model_name
from src.model.plot import colors
from src.analysis.classification.acc import load_acc1


if __name__ == "__main__":
    num_classes = int(sys.argv[1])  # number of last output of the models
    test_dataset = str(sys.argv[2])  # test_dataset to use
    stimuli = [
        "original",
        "jumbled",
        "gray_occluder",
    ]
    xticks = [
        "Original",
        "Jumbled",
        "Gray Occluder",
    ]
    scales = [4, 8, 16, 32]  # 1 == original

    metrics = "acc1"

    arch = "alexnet"
    epoch = 60
    batch_size = 64

    analysis = f"{stimuli}_{test_dataset}"

    machine = "server"  # ("server", "local")

    # directories and model settings
    # out_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/lowpass_acc/plots/{analysis}/{num_classes}-class/"
    lowpass_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/lowpass_acc/results/lowpass_acc_{test_dataset}/{num_classes}-class/"  # for acc on original
    out_dir = f"./plots/{num_classes}-class/"
    out_server_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/jumbled_gray_occluder/plots/{analysis}_vs_humans/{num_classes}-class/"

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

    model_names = [
        f"{arch}_normal",
        # f"vone_{arch}_normal",
        f"{arch}_all_s04",
        # f"vone_{arch}_all_s04",
        f"{arch}_mix_s04",
        # f"vone_{arch}_mix_s04",
        # f"{arch}_random-mix_s00-04",
        # f"vone_{arch}_random-mix_s00-04",
        f"{arch}_multi-steps",
        # f"vone_{arch}_multi-steps",
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

    # set plot file name.
    plot_file = f"{analysis}_{metrics}_{num_classes}-class_{model_names}.png"
    # plot_file = f"{analysis}_{metrics}_{num_classes}-class_vone_alexnet.png"

    x = stimuli

    # read accuracy results
    acc = {}
    original_acc = {}
    for model_name in model_names:
        results = []
        for stimulus in stimuli:
            if stimulus == "original":
                file_path = os.path.join(
                    lowpass_dir,
                    f"lowpass_acc_{test_dataset}_{num_classes}-class_{model_name}_{metrics}.csv",
                )
                results += [pd.read_csv(file_path, index_col=0).values[0][0] / 100]
            else:
                acc_all = 0
                for div_v in scales:
                    div_h = div_v
                    in_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/jumbled_gray_occluder/results/{stimulus}_{test_dataset}/{num_classes}-class/"
                    file_path = os.path.join(
                        in_dir,
                        f"{num_classes}-class_{model_name}_{stimulus}_{test_dataset}_{div_v}x{div_h}_{metrics}.csv",
                    )
                    # if "vone" in model_name or "SIN" in model_name:
                    #     file_path = file_path.replace("16-class", "1000-class")
                    #     file_path = file_path.replace("imagenet16", "imagenet1000")

                    acc_all += load_acc1(file_path=file_path)

                results += [acc_all / len(scales)]  # take mean

        acc[model_name] = results

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(
        1,
        1,
        1,
        title=(
            f"Top-{metrics[-1]} acc. on ImageNet"
            if test_dataset == "imagenet1000"
            else f"Top-{metrics[-1]} acc. on 16-class-ImageNet"
        ),
        # xlabel="",
        ylabel=f"Top-{metrics[-1]} accuracy",
        ylim=(0, 1),
    )

    # set width of bars
    barWidth = 0.15
    # barWidth = 0.4  # for B+S-Net comparison

    # Set position of bar on X axis
    r1 = np.arange(len(acc[model_names[0]]))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    # Ref: https://www.python-graph-gallery.com/11-grouped-barplot
    for model_name in model_names:
        plt.bar(
            r1,
            acc[model_name],
            color=colors[model_name],
            width=barWidth,
            edgecolor="w",
            hatch="////" if "vone_" in model_name else None,
            label=rename_model_name(model_name=model_name, arch=arch),
        )
        r1 = [x + barWidth for x in r1]

    # Add xticks on the middle of the group bars
    plt.xticks(
        [r + barWidth for r in range(len(acc[model_names[0]]))],
        # [r + 0.2 for r in range(len(acc[model_names[0]]))],  # for B+S-Net comparison
        xticks,
        rotation=45,
        ha="right",
    )

    # plot chance level performance
    plt.hlines(
        [1 / 16],
        xmin=-0.1,
        # xmin=-0.3,  # for B+S-Net comparison
        xmax=len(x) - 0.45,
        # xmax=len(x) - 0.3,  # for B+S-Net comparison
        colors="k",
        linestyles="dashed",
        label="Chance performance",
    )

    ax.legend()
    # plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(10))
    # ax.xaxis.set_major_locator(tick.MultipleLocator(1))
    # ax.grid(which="major", linestyle="dotted")
    # ax.grid(which="minor", linestyle="dotted")
    ax.yaxis.grid(ls="dotted")

    # fig.show()
    fig.savefig(os.path.join(out_dir, plot_file), bbox_inches="tight")
