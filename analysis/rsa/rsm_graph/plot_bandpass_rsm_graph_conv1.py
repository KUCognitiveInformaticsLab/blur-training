import os
import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as tick

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

from src.analysis.rsa.utils import load_rsms
from src.analysis.rsa.bandpass.bandpass_rsm_graph import compute_bandpass_values
from src.model.load_sin_pretrained_models import sin_names
from src.model.model_names import rename_model_name
from src.analysis.rsa.rsa import alexnet_layers, vone_alexnet_layers
from src.model.plot import colors, lines

if __name__ == "__main__":
    arch = "alexnet"
    num_classes = 1000
    epoch = 60

    metrics = "correlation"  # ("correlation", "1-covariance", "negative-covariance")
    analysis = f"bandpass_rsm_{metrics}"
    legend = True

    in_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/results/{analysis}/{num_classes}-class/"
    # out_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/plots/{analysis}/{num_classes}-class/"
    # in_dir = f"./results/{analysis}/{num_classes}-class/"
    out_dir = f"./plots/{analysis}_graph/{num_classes}-class/"

    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # models to compare
    model_names = [
        "raw_images",
        f"untrained_{arch}",
        f"{arch}_normal",
        f"{arch}_all_s04",
        f"{arch}_mix_s04",
        f"vone_{arch}",
        sin_names[arch],
    ]

    # (optional) set title
    plot_title = f"{analysis}, {num_classes}-class"

    # set filename
    num_filters = 6
    filename = (
        f"{analysis}_graph{num_classes}-class_f{num_filters}_conv1_legend.png"
        if legend
        else f"{analysis}_graph{num_classes}-class_f{num_filters}_conv1.png"
    )
    out_file = os.path.join(out_dir, filename)

    fig = plt.figure(dpi=300)

    for model_name in model_names:
        if model_name == "raw_images":
            in_file = os.path.join(
                pathlib.Path(in_dir).parent, f"{analysis}_{model_name}.pkl"
            )
        else:
            in_file = os.path.join(
                in_dir, f"{analysis}_{num_classes}-class_{model_name}.pkl"
            )

        rsms = load_rsms(file_path=in_file)

        if "vone" in model_name:
            layer = vone_alexnet_layers[0]  # vone-block
        else:
            layer = alexnet_layers[0]  # conv-relu-1

        renamed_model_name = rename_model_name(model_name=model_name, arch=arch)

        if model_name == "raw_images":
            y = compute_bandpass_values(rsms["raw"])
        else:
            y = compute_bandpass_values(rsms[layer])

        num_bandpass = int(len(y) / 2)
        x = [x for x in range(-num_bandpass, num_bandpass + 1)]

        ax = fig.add_subplot(111)
        ax.plot(
            x,
            y,
            label=renamed_model_name,
            marker=".",
            markersize=3,
            ls=lines[model_name],
            linewidth=1,
            color=colors[model_name],
        )

        # sns.set(font_scale=0.5)  # adjust the font size of labels
        # plt.rcParams["font.size"] = 8
        ax.set_title(layer, fontsize=8)
        plt.ylim((-0.1, 1.1))
        ax.xaxis.set_major_locator(tick.MultipleLocator(1))
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=8,
        )

    # fig.legend(
    #     bbox_to_anchor=(0, -0.1),
    #     loc='upper left',
    #     borderaxespad=0,
    #     fontsize=8,
    # )
    fig.tight_layout()
    # fig.show()
    fig.savefig(out_file)
