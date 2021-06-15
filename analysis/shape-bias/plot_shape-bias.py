import os
import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

from src.analysis.shape_bias.shape_bias import get_shape_bias, get_cue_conf_acc
from src.model.load_sin_pretrained_models import sin_names
from src.model.plot import colors
from src.model.model_names import rename_model_name


if __name__ == "__main__":
    arch = "alexnet"
    num_classes = 16
    epoch = 60

    in_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/shape-bias/results/{num_classes}-class/"
    # in_dir = f'./results/{num_classes}-class'
    out_dir = f"./plots/{num_classes}-class"
    assert in_dir
    os.makedirs(out_dir, exist_ok=True)

    # models to plot
    model_names = [
        f"{arch}_normal",
        f"vone_{arch}_normal",
        f"{arch}_all_s04",
        f"vone_{arch}_all_s04",
        f"{arch}_mix_s04",
        f"vone_{arch}_mix_s04",
        f"{arch}_multi-steps",
        f"vone_{arch}_multi-steps",
        # f"vone_{arch}_normal",
        # f"{arch}_vonenet",
        # "resnet50-1x_simclr",
        # "resnet50-2x_simclr",
        # "resnet50-4x_simclr",
        sin_names[arch.replace("vone_", "")],
    ]
    model_names = [
        f"{arch}_normal",
        f"{arch}_all_s04",
        f"{arch}_mix_s04",
        f"{arch}_random-mix_s00-04",
        f"{arch}_multi-steps",
        # f"vone_{arch}_normal",
        # f"{arch}_vonenet",
        # "resnet50-1x_simclr",
        # "resnet50-2x_simclr",
        # "resnet50-4x_simclr",
        sin_names[arch.replace("vone_", "")],
    ]
    model_names = [
        f"{arch}_normal",
        f"vone_{arch}_normal",
        f"{arch}_all_s04",
        f"vone_{arch}_all_s04",
        f"{arch}_mix_s04",
        f"vone_{arch}_mix_s04",
        f"{arch}_random-mix_s00-04",
        f"vone_{arch}_random-mix_s00-04",
        f"{arch}_multi-steps",
        f"vone_{arch}_multi-steps",
        # f"vone_{arch}_normal",
        # f"{arch}_vonenet",
        # "resnet50-1x_simclr",
        # "resnet50-2x_simclr",
        # "resnet50-4x_simclr",
        sin_names[arch.replace("vone_", "")],
    ]

    filename = f"shape-bias_{model_names}.png"

    shape_bias = {}
    cue_conf_acc = {}

    # load and get results
    for model_name in model_names:
        if (
            "SIN" in model_name
            or model_name == "alexnet_vonenet"
            or "simclr" in model_name
        ):
            file_path = os.path.join(
                in_dir.replace("16", "1000"), f"correct_decisions_{model_name}.csv"
            )
        else:
            file_path = os.path.join(
                in_dir, f"correct_decisions_{model_name}_e{epoch}.csv"
            )
        shape_bias[model_name] = get_shape_bias(file_path=file_path)
        cue_conf_acc[model_name] = get_cue_conf_acc(file_path=file_path)

    # sigma to compare
    fig = plt.figure(dpi=150, figsize=(4, 5))
    ax = fig.add_subplot(
        1,
        1,
        1,
        # title = title[arch],
        # xlabel = xlabel,
        ylabel="Shape bias (bar), Accuracy (+)",
        ylim=(0, 1),
    )
    # plot shape bias
    for model_name in model_names:
        ax.bar(
            model_name,
            shape_bias[model_name],
            color=colors[model_name],
            hatch="///" if "vone_" in model_name else None,
            edgecolor="w",
        )
    ax.bar("Humans", 0.96, color=colors["humans"])

    # plot accuracy
    for model_name in model_names:
        ax.plot(
            model_name,
            cue_conf_acc[model_name],
            marker="+",
            color="k",
            markeredgewidth=1.5,
            markersize=6,
        )

    # ax.set_xticklabels(acc.columns, rotation=45, ha='right')
    ax.set_xticklabels(
        [rename_model_name(model_name=m, arch=arch) for m in model_names] + ["Humans"],
        rotation=45,
        ha="right",
    )
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # ax.grid(ls=":")

    # fig.show()
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    plt.close()
