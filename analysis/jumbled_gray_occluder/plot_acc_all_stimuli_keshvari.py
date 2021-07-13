import os
import pathlib
import sys

import matplotlib.pyplot as plt

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

if __name__ == "__main__":
    scales = [4, 8, 16, 32]  # 1 == original

    arch = "alexnet"
    epoch = 60
    batch_size = 64

    machine = "server"  # ("server", "local")

    # directories and model settings
    # out_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/lowpass_acc/plots/{analysis}/{num_classes}-class/"
    out_dir = f"./plots/"
    # out_server_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/jumbled_gray_occluder/plots/{analysis}_vs_humans/{num_classes}-class/"

    os.makedirs(out_dir, exist_ok=True)

    x = ["Original", "Jumbled", "Gray Occluder", "Jumbled with Gray Occluder"]  # "Chance performance"
    acc_humans = [0.9691, 0.4531, 0.8377, 0.5353]
    acc_vgg = [0.9640, 0.6042, 0.3690, 0.3891]
    colors = ["#3372B7", "#C75B2E", "#E3B245", "#73378A"]

    # === humans ===
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(
        1,
        1,
        1,
        # xlabel="",
        ylabel=f"Accuracy",
        ylim=(0, 1),
    )

    # plot acc
    plt.bar(
        x,
        acc_humans,
        color=colors,
        width=0.5,
        # edgecolor="w",
    )

    # plot chance level performance
    plt.hlines(
        0.1266,
        xmin=-0.5,
        xmax=len(x)-0.5,
        colors="k",
        linestyles="dashed",
        label="Chance performance",
    )

    # Add xticks
    plt.xticks(
        x,
        rotation=45,
        ha="right",
    )

    ax.legend()
    # plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(10))
    # ax.xaxis.set_major_locator(tick.MultipleLocator(1))
    ax.grid(which="major", linestyle="dotted")
    ax.grid(which="minor", linestyle="dotted")

    ax.set_title("Humans (Keshvari et al, 2021)", weight="bold")
    # set plot file name.
    plot_file = f"keshvari2021_results_humans.png"

    # fig.show()
    fig.savefig(os.path.join(out_dir, plot_file), bbox_inches="tight")

    # === vgg ===
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(
        1,
        1,
        1,
        # xlabel="",
        ylabel=f"Accuracy",
        ylim=(0, 1),
    )

    # plot acc
    plt.bar(
        x,
        acc_vgg,
        color=colors,
        width=0.5,
        # edgecolor="w",
    )

    # plot chance level performance
    plt.hlines(
        0.1266,
        xmin=-0.5,
        xmax=len(x) - 0.5,
        colors="k",
        linestyles="dashed",
        label="Chance performance",
    )

    # Add xticks
    plt.xticks(
        x,
        rotation=45,
        ha="right",
    )

    ax.legend()
    # plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(10))
    # ax.xaxis.set_major_locator(tick.MultipleLocator(1))
    ax.grid(which="major", linestyle="dotted")
    ax.grid(which="minor", linestyle="dotted")

    ax.set_title("Pretrained VGG (Keshvari et al, 2021)", weight="bold")
    # set plot file name.
    plot_file = f"keshvari2021_results_vgg.png"

    # fig.show()
    fig.savefig(os.path.join(out_dir, plot_file), bbox_inches="tight")

