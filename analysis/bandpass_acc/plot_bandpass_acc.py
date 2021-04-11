import os
import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as tick

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

from src.analysis.bandpass_acc.utils import load_result


if __name__ == "__main__":
    arch = "alexnet"
    epoch = 60
    num_classes = 16  # number of last output of the models

    # directories and model settings
    in_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/bandpass_acc/results/{num_classes}-class-{arch}/"
    out_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/bandpass_acc/plots/{num_classes}-class-{arch}/"

    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # models to plot
    model_names = [
        f"{num_classes}-class-{arch}_normal",
        f"{num_classes}-class-{arch}_all_s04",
        f"{num_classes}-class-{arch}_mix_s04",
        # f"{num_classes}-class-{arch}_mix_s10",
        # f"{num_classes}-class-{arch}_random-mix_s00-05"
        # f"{num_classes}-class-{arch}_random-mix_s00-10"
    ]

    # set plot file name.
    plot_file = (
        f"bandpass-acc1_{num_classes}-class-{arch}_e{epoch}_normal_all-s04_mix-s04.png"
    )

    colors = {
        f"{num_classes}-class-{arch}_normal": "#1f77b4",
        f"{num_classes}-class-{arch}_all_s04": "darkorange",
        f"{num_classes}-class-{arch}_mix_s04": "limegreen",
        f"{num_classes}-class-{arch}_mix_s10": "hotpink",
        f"{num_classes}-class-{arch}_random-mix_s00-05": "green",
        f"{num_classes}-class_{arch}_random-mix_s00-10": "mediumvioletred",
    }

    lines = {
        f"{num_classes}-class-{arch}_normal": ":",
        f"{num_classes}-class-{arch}_all_s04": "-",
        f"{num_classes}-class-{arch}_mix_s04": "-",
        f"{num_classes}-class-{arch}_mix_s10": "-",
        f"{num_classes}-class-{arch}_random-mix_s00-05": "-",
        f"{num_classes}-class-{arch}_random-mix_s00-10": "-",
    }

    x = ["{}-{}".format(2 ** i, 2 ** (i + 1)) for i in range(4)] + ["16-"]
    x.insert(0, "0-1")
    x.insert(0, "0(raw)")

    # read band-pass accuracy results
    value = "acc1"
    acc1 = {}
    for model_name in model_names:
        file_path = os.path.join(in_dir, f"{model_name}_e{epoch}_{value}.csv")
        acc1[model_name] = load_result(file_path=file_path).values[0]

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(
        1,
        1,
        1,
        title=f"top-1 acc. on band-pass 16-class-imagenet",
        xlabel="Test images",
        ylabel="Top-1 accuracy",
        ylim=(0, 1),
    )
    for model_name in model_names:
        ax.plot(x[0], acc1[model_name][0], marker="o", color=colors[model_name])
        ax.plot(
            x[1:],
            acc1[model_name][1:],
            label=model_name,
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
    fig.show()
    fig.savefig(os.path.join(out_dir, plot_file))
