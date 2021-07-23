import os
import pathlib
import sys

import matplotlib.pyplot as plt

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

from src.analysis.bandpass_acc.utils import load_result
from src.model.model_names import rename_model_name
from src.model.plot import colors, lines, get_marker


if __name__ == "__main__":
    arch = str(sys.argv[1])  # "resnet50", "vgg16", "alexnet"
    num_classes = int(sys.argv[2])  # number of last output of the models
    epoch = 60
    test_dataset = str(sys.argv[3])  # "imagenet16", "imagenet1000"
    metrics = str(sys.argv[4])  # "acc1"

    analysis = f"bandpass_acc_{test_dataset}"

    # directories and model settings
    in_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/bandpass_acc/results/{num_classes}-class/"
    # out_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/bandpass_acc/plots/{num_classes}-class/"
    out_dir = f"./plots/{num_classes}-class/"

    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # models to plot
    model_names = [
        f"{arch}_normal",
        f"{arch}_all_s04",
        f"{arch}_mix_s04",
        f"{arch}_random-mix_s00-04",
        f"{arch}_multi-steps",
        # f"{arch}_mix_s10",
        # f"{arch}_random-mix_s00-10",
        # f"{arch}_trained_on_SIN",
        # f"vone_{arch}",
    ]
    model_names = [
        f"{arch}_normal",
        f"{arch}_all_s04",
        f"{arch}_mix_s04",
        # f"{arch}_random-mix_s00-04",
        f"{arch}_multi-steps",
        # f"vone_{arch}_normal",
        # f"vone_{arch}_all_s04",
        # f"vone_{arch}_random-mix_s00-04",
        # f"vone_{arch}_mix_s04",
        # f"vone_{arch}_multi-steps",
        # f"{arch}_trained_on_SIN",
        "humans",
    ]

    # set plot file name.
    plot_file = f"{analysis}_{num_classes}-class_{arch}_{model_names}.png"
    # plot_file = f"{analysis}_{num_classes}-class_normal_{arch}_humans.png"

    x = ["{}-{}".format(2 ** i, 2 ** (i + 1)) for i in range(4)] + ["16-"]
    x.insert(0, "0-1")
    x.insert(0, "0(original)")

    # read band-pass accuracy results
    acc1 = {}
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
        if "SIN" in model_name:
            file_path = file_path.replace("16-class", "1000-class")

        if num_classes == 1000:
            file_path = file_path.replace("imagenet16_", "")

        if model_name == "humans":
            continue

        acc1[model_name] = load_result(file_path=file_path).values[0] / 100

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(
        1,
        1,
        1,
        title=(
            f"Top-{metrics[-1]} acc. on ImageNet with band-pass filters "
            if test_dataset == "imagenet1000"
            else f"Top-{metrics[-1]} acc. on 16-class-ImageNet with band-pass filters"
        ),
        xlabel="Band-pass filters",
        ylabel=f"Top-{metrics[-1]} accuracy",
        ylim=(0, 1),
    )
    for model_name in model_names:
        if model_name == "humans":
            # plot humans data
            ax.plot(x[0], [0.8932], marker="x", color=colors[model_name])
            ax.plot(
                ["1-2", "4-8", "16-"],
                [0.7885, 0.6391, 0.2336],
                label=rename_model_name(arch=arch, model_name=model_name),
                marker="x",
                ls=lines[model_name],
                # ls=":" if model_name == f"{arch}_normal" else "-",
                color=colors[model_name],
            )
        else:
            ax.plot(
                x[0], acc1[model_name][0],
                marker=get_marker(model_name=model_name, num_classes=num_classes),
                color=colors[model_name]
            )
            ax.plot(
                x[1:],
                acc1[model_name][1:],
                label=rename_model_name(arch=arch, model_name=model_name),
                marker=get_marker(model_name=model_name, num_classes=num_classes),
                ls=lines[model_name],
                color=colors[model_name],
            )

    # plot chance level performance
    if test_dataset == "imagenet16":
        chance = [1 / 16]
    elif test_dataset == "imagenet1000":
        if metrics == "acc1":
            chance = [1 / 1000]
        elif metrics == "acc5":
            chance = [5 / 1000]
    plt.hlines(
        chance,
        xmin=0,
        xmax=len(x) - 1,
        colors="k",
        linestyles="dashed",
        label="Chance performance",
    )

    # ax.legend()
    ax.legend(fontsize=8)
    # ax.set_xticks(np.arange(0, max_sigma+1, 5))
    # plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(10))
    # ax.xaxis.set_major_locator(tick.MultipleLocator(1))
    ax.grid(which="major")
    ax.grid(which="minor")
    # plt.xlim()
    plt.ylim(0, 1)

    # fig.show()
    fig.savefig(os.path.join(out_dir, plot_file))
