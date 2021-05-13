import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.bandpass.bandpass_rdm import (
    load_compute_mean_rdms,
    plot_bandpass_rdms,
)
from src.analysis.rsa.utils import save_rdms

if __name__ == "__main__":
    arch = "alexnet"
    num_classes = 16
    epoch = 60
    metrics = (
        "negative-covariance"  # ("correlation", "1-covariance", "negative-covariance")
    )

    # I/O settings
    analysis_dir = "../"
    data_dir = os.path.join(
        analysis_dir, f"results/activations/{num_classes}-class-{arch}/"
    )
    results_dir = os.path.join(
        analysis_dir, f"results/mean_rdms_{metrics}/{num_classes}-class-{arch}/"
    )
    plots_dir = os.path.join(
        analysis_dir, f"plots/mean_rdms_{metrics}/{num_classes}-class-{arch}/"
    )

    assert os.path.exists(data_dir), f"{data_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # models to compare
    model_names = [
        f"{arch}_normal",
        # f"{arch}_multi-steps",
    ]
    modes = [
        f"{arch}_all",
        f"{arch}_mix",
        f"{arch}_random-mix",
        # f"{arch}_single-step",
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

    from src.analysis.rsa.rsa import alexnet_layers

    min_list = []
    max_list = []

    for model_name in model_names:
        in_dir = os.path.join(data_dir, f"{model_name}_e{epoch:02d}")
        assert os.path.exists(in_dir), f"{in_dir} does not exist."

        mean_rdms = load_compute_mean_rdms(
            in_dir=in_dir, num_filters=6, num_images=1600, metrics=metrics
        )

        for layer in alexnet_layers:
            min_list.append((mean_rdms[layer].min()))
            max_list.append((mean_rdms[layer].max()))

        result_file = f"{model_name}_e{epoch:02d}.pkl"
        result_path = os.path.join(results_dir, result_file)
        save_rdms(mean_rdms=mean_rdms, file_path=result_path)

        # get analysis parameters.
        num_images = mean_rdms["num_images"]
        num_filters = mean_rdms["num_filters"]

        # (optional) set title
        title = f"RDM ({metrics}), {num_classes}-class, {model_name}, epoch={epoch}"

        # set filename
        filename = f"mean_rdms_{metrics}_{num_classes}-class_{model_name}_e{epoch}_f{num_filters}_n{num_images}.png"
        out_file = os.path.join(plots_dir, filename)

        # colour value range of the plots
        vmin = 0 if metrics == "correlation" else -5
        vmax = 2 if metrics == "correlation" else 5

        # plot_rdms(rdms=diff_rdms, out_file=out_file, plot_show=True)
        plot_bandpass_rdms(
            rdms=mean_rdms,
            num_filters=num_filters,
            vmin=vmin,
            vmax=vmax,
            title=title,
            out_file=out_file,
            show_plot=False,
        )

    import numpy as np

    print(np.array(min_list).min())
    print(np.array(max_list).max())
