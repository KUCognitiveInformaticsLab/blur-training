import os
import pathlib
import sys

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.analysis.rsa.rsa import alexnet_layers
from src.analysis.rsa.activations import load_activations
from src.analysis.rsa.rdm import save_rdms


def compute_mean_rdms(
    in_dir: str,
    num_filters: int = 6,
    num_images: int = 1600,
    metrics: str = "correlation",
) -> dict:
    """Computes RDM for each image and return mean RDMs.
    Args:
        in_dir: path to input directory
        num_filters: number of band-pass filter
        num_images: number of images
    Returns: Mean RDMs (Dict)
    """
    mean_rdms = {}
    mean_rdms["num_filters"] = num_filters
    mean_rdms["num_images"] = num_images
    # mean_rdms["target_id"] = target_id

    for layer in alexnet_layers:
        rdms = []
        # compute RDM for each image (with some filters applied)
        for image_id in range(num_images):
            file_name = f"image{image_id:04d}_f{num_filters:02d}.pkl"
            activations = load_activations(in_dir=in_dir, file_name=file_name)
            activation = activations[layer].reshape(num_filters + 1, -1)
            if metrics == "correlation":
                rdm = squareform(pdist(activation, metric=metrics))  # 1 - corr.
            elif metrics == "covariance":
                rdm = squareform(
                    pdist(
                        activation,
                        lambda u, v: np.average(
                            (u - np.average(u)) * (v - np.average(v))
                        ),
                    )
                )
            rdms.append(rdm)

        rdms = np.array(rdms)

        mean_rdms[layer] = rdms.mean(0)

    return mean_rdms


def plot_bandpass_rdms(
    rdms, num_filters, vmin=0, vmax=2, title="", out_file="rdms.png", show_plot=False
):
    """Plot several layers RDMs in one figure.
    Args:
        model_name: name of the model to examine.
    """
    fig = plt.figure(dpi=300)

    for i, layer in enumerate(alexnet_layers):
        ax = fig.add_subplot(2, 4, i + 1)
        # sns.set(font_scale=0.5)  # adjust the font size of labels
        ax.set_title(layer)

        sns.heatmap(
            rdms[layer],
            ax=ax,
            square=True,
            vmin=vmin,
            vmax=vmax,
            xticklabels=["0", "0-1", "1-2", "2-4", "4-8", "8-16", "16-"],
            yticklabels=["0", "0-1", "1-2", "2-4", "4-8", "8-16", "16-"],
            cmap="coolwarm",
            cbar=False,
            # cbar_ax=cbar_ax,
        )

        ax.hlines(
            [i for i in range(1, num_filters + 1)],
            *ax.get_xlim(),
            linewidth=0.1,
            colors="gray",
        )
        ax.vlines(
            [i for i in range(1, num_filters + 1)],
            *ax.get_ylim(),
            linewidth=0.1,
            colors="gray",
        )

    # show color bar
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])
    # sns.heatmap(rdms[layer], cbar=True, cbar_ax=cbar_ax,
    #             vmin=-1, vmax=1, cmap='coolwarm',
    #             xticklabels=False, yticklabels=False)

    # sns.set(font_scale=0.5)  # adjust the font size of title
    if title:
        fig.suptitle(title)
    # fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.tight_layout()
    plt.savefig(out_file)
    if show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    arch = "alexnet"
    num_classes = 1000
    epoch = 60
    metrics = "correlation"

    # I/O settings
    analysis_dir = "/home/sou/work/blur-training/analysis/rsa/bandpass"
    data_dir = os.path.join(analysis_dir, f"results/activations/{num_classes}-class-{arch}/")
    results_dir = os.path.join(analysis_dir, f"results/mean_rdms_{metrics}/{num_classes}-class-{arch}/")
    plots_dir = os.path.join(analysis_dir, f"plots/mean_rdms_{metrics}/{num_classes}-class-{arch}/")

    assert os.path.exists(data_dir), f"{data_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # models to compare
    model_names = [
        f"{arch}_normal",
        f"{arch}_multi-steps",
    ]
    modes = [
        f"{arch}_all",
        f"{arch}_mix",
        f"{arch}_random-mix",
        f"{arch}_single-step",
        # f"{arch}_fixed-single-step",
        # f"{arch}_reversed-single-step",
    ]

    # sigmas to compare
    sigmas_mix = [s for s in range(1, 6)] + [10]
    sigmas_random_mix = ["00-05", "00-10"]

    # add sigma to compare to the model names
    for mode in modes:
        if mode == f"{arch}_random-mix":
            for min_max in sigmas_random_mix:
                model_names += [f"{mode}_s{min_max}"]
        elif mode == f"{arch}_mix":
            for sigma in sigmas_mix:
                model_names += [f"{mode}_s{sigma:02d}"]
        else:
            for sigma in range(1, 5):
                model_names += [f"{mode}_s{sigma:02d}"]

    for model_name in model_names:
        in_dir = os.path.join(data_dir, f"{model_name}_e{epoch:02d}")
        assert os.path.exists(in_dir), f"{in_dir} does not exist."

        mean_rdms = compute_mean_rdms(
            in_dir=in_dir, num_filters=6, num_images=1600, metrics=metrics
        )

        result_file = f"{model_name}_e{epoch:02d}.pkl"
        result_path = os.path.join(results_dir, result_file)
        save_rdms(mean_rdms=mean_rdms, file_path=result_path)

        # get analysis parameters.
        num_images = mean_rdms["num_images"]
        num_filters = mean_rdms["num_filters"]

        # (optional) set title of the plot
        title = (
            f"RDM({metrics}), {num_classes}-class, {model_name}, epoch={epoch}"
        )

        # set the plot path
        plot_file = f"{num_classes}-class_mean-rdms_{model_name}_e{epoch}_f{num_filters}_n{num_images}.png"
        plot_path = os.path.join(plots_dir, plot_file)

        # plot
        plot_bandpass_rdms(
            rdms=mean_rdms,
            num_filters=num_filters,
            vmin=0,
            vmax=2,
            title=title,
            out_file=plot_path,
            show_plot=False,
        )
