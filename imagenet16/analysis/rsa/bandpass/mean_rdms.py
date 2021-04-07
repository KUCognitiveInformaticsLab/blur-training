import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.bandpass.mean_rdms import compute_mean_rdms, save_rdms, plot_rdms

if __name__ == "__main__":
    arch = "alexnet"
    mode = "normal"
    model_name = f"{arch}_{mode}"
    epoch = 60

    # I/O settings
    data_dir = "./results/activations/alexnet/"
    results_dir = f"./results/mean_rdms/{arch}/"
    plots_dir = f"./plots/mean_rdms/{arch}/"

    assert os.path.exists(data_dir), f"{data_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # models to compare
    modes = [
        "normal",
        "all",
        "mix",
        "random-mix",
        "single-step",
        "fixed-single-step",
        "reversed-single-step",
        "multi-steps",
    ]

    # sigmas to compare
    sigmas_mix = [s for s in range(1, 6)] + [10]
    sigmas_random_mix = ["00-05", "00-10"]

    # make model name list
    model_names = []
    for mode in modes:
        if mode in ("normal", "multi-steps"):
            model_names += [f"{arch}_{mode}"]
        elif mode == "random-mix":
            for min_max in sigmas_random_mix:
                model_names += [f"{arch}_{mode}_s{min_max}"]
        elif mode == "mix":
            for sigma in sigmas_mix:
                model_names += [f"{arch}_{mode}_s{sigma:02d}"]
        else:
            for s in range(4):
                model_names += [f"{arch}_{mode}_s{s + 1:02d}"]

    for model_name in model_names:
        in_dir = os.path.join(data_dir, f"{model_name}_e{epoch:02d}")
        assert os.path.exists(in_dir), f"{in_dir} does not exist."

        mean_rdms = compute_mean_rdms(in_dir=in_dir, num_filters=6, num_images=1600)

        save_rdms(mean_rdms=mean_rdms, out_dir=results_dir, model_name=model_name, epoch=epoch)

        plot_rdms(
            mean_rdms=mean_rdms, out_dir=plots_dir, model_name=model_name, epoch=epoch
        )
