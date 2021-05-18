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


if __name__ == "__main__":
    arch = "alexnet"
    epoch = 60
    num_classes = int(sys.argv[1])  # number of last output of the models
    test_dataset = str(sys.argv[2])
    metrics = str(sys.argv[3])  # ("acc1", "acc5")
    max_sigma = 10
    analysis = f"lowpass_acc_{test_dataset}"

    # directories and model settings
    in_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/lowpass_acc/results/{analysis}/{num_classes}-class/"
    # out_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/lowpass_acc/plots/{analysis}/{num_classes}-class/"
    out_dir = f"./plots/{analysis}/{num_classes}-class/"

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
        f"{arch}_mix_s01",
        "mix_no-blur-1label",
        "mix_no-blur-8label",
    ]

    # set plot file name.
    plot_file = f"{analysis}_{metrics}_max-s{max_sigma}_{num_classes}-class_{model_names}.png"

    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    x = [s for s in range(max_sigma + 1)]
    x[0] = "0(sharp)"

    # read band-pass accuracy results
    acc = {}
    for model_name in model_names:
        # if "SIN" in model_name or "vone" in model_name:
        #     # Stylized-ImageNet
        #     file_path = os.path.join(in_dir, f"{analysis}_{num_classes}-class_{model_name}_{metrics}.csv")
        # else:
        #     file_path = os.path.join(
        #         in_dir, f"{analysis}_{num_classes}-class_{model_name}_e{epoch}_{metrics}.csv"
        #     )
        file_path = os.path.join(
            in_dir, f"{analysis}_{num_classes}-class_{model_name}_{metrics}.csv"
        )
        if "vone" in model_name or "SIN" in model_name:
            file_path = file_path.replace("16-class", "1000-class")
            file_path = file_path.replace("imagenet16", "imagenet")

        acc[model_name] = pd.read_csv(file_path, index_col=0).values[0]

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(
        1,
        1,
        1,
        title=(
            f"Top-{metrics[-1]} acc. on ImageNet with low-pass filters "
            if test_dataset == "imagenet"
            else f"Top-{metrics[-1]} acc. on 16-class-ImageNet with low-pass filters"
        ),
        xlabel="Low-pass filters (σ of GaussianBlur)",
        ylabel=f"Top-{metrics[-1]} accuracy",
        ylim=(0, 1),
    )
    for model_name in model_names:
        # ax.plot(x[0], acc[model_name][0], marker="o", color=colors[model_name])
        ax.plot(
            x,
            acc[model_name][: max_sigma + 1],
            label=rename_model_name(model_name),
            marker="o",
            ls=lines[model_name],
            color=colors[model_name],
        )

    ax.legend()
    # ax.set_xticks(np.arange(0, max_sigma+1, 5))
    plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(10))
    # ax.xaxis.set_major_locator(tick.MultipleLocator(1))
    ax.grid(which="major")
    ax.grid(which="minor")
    # plt.xlim()
    plt.ylim(0, 100)

    # fig.show()
    fig.savefig(os.path.join(out_dir, plot_file))
