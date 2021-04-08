import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.bandpass.mean_rdms import (
    compute_mean_rdms,
    save_rdms,
    plot_bandpass_rdms,
)

if __name__ == "__main__":
    arch = "alexnet"
    num_classes = 1000
    epoch = 60
    metrics = "covariance"  # or "covariance"

    # I/O settings
    analysis_dir = "./"
    data_dir = os.path.join(
        analysis_dir, f"results/activations/{num_classes}-class-{arch}/"
    )
    results_dir = os.path.join(
        analysis_dir, f"results/mean_rdms_1-{metrics}/{num_classes}-class-{arch}/"
    )
    plots_dir = os.path.join(
        analysis_dir, f"plots/mean_rdms_1-{metrics}/{num_classes}-class-{arch}/"
    )

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
        title = f"RDM(1 - {metrics}), {num_classes}-class, {model_name}, epoch={epoch}"

        # set the plot path
        plot_file = f"mean-rdms_{metrics}_{num_classes}-class_{model_name}_e{epoch}_f{num_filters}_n{num_images}.png"
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
