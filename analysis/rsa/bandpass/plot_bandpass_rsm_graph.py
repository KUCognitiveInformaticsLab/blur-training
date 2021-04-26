import os
import pathlib
import sys

import matplotlib.pyplot as plt

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

from src.analysis.rsa.utils import load_rsms
from src.analysis.rsa.bandpass.bandpass_rsm_graph import compute_bandpass_values
from src.model.load_sin_pretrained_models import sin_names
from src.analysis.rsa.rsa import alexnet_layers, vone_alexnet_layers
from src.analysis.plot import colors, lines

if __name__ == "__main__":
    arch = "alexnet"
    num_classes = 1000
    epoch = 60

    metrics = "correlation"  # ("correlation", "1-covariance", "negative-covariance")
    analysis = f"bandpass_rsm_{metrics}"

    in_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/results/{analysis}/{num_classes}-class/"
    # out_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/plots/{analysis}/{num_classes}-class/"
    # in_dir = f"./results/{analysis}/{num_classes}-class/"
    out_dir = f"./plots/{analysis}_graph/{num_classes}-class/"

    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # models to compare
    model_names = [
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
    filename = f"{analysis}_graph-one-side_{num_classes}-class_f{num_filters}.png"
    out_file = os.path.join(out_dir, filename)

    x = [1, 2, 3]
    fig = plt.figure(dpi=300)

    for model_name in model_names:
        in_file = os.path.join(
            in_dir, f"{analysis}_{num_classes}-class_{model_name}.pkl"
        )
        rsms = load_rsms(file_path=in_file)
        if "vone" in model_name:
            rsms["layers"] = vone_alexnet_layers
        else:
            rsms["layers"] = alexnet_layers

        for i, layer in enumerate(rsms["layers"]):
            a_avr, b_avr, c_avr = compute_bandpass_values(rsms[layer])
            ax = fig.add_subplot(2, 4, i + 1)
            # sns.set(font_scale=0.5)  # adjust the font size of labels
            ax.set_title(layer)

            ax.plot(
                x,
                [a_avr, b_avr, c_avr],
                label=model_name,
                marker="o",
                ls=lines[model_name],
                color=colors[model_name],
            )

            plt.ylim((0, 1))

            # if i == 3:
            #     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            # if i == 7:
            #     plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0,
            #                fontsize=10)

    # fig.legend(
    #     bbox_to_anchor=(0, -0.1),
    #     loc='upper left',
    #     borderaxespad=0,
    #     fontsize=10,
    # )
    fig.tight_layout()
    # fig.show()
    fig.savefig(out_file)
