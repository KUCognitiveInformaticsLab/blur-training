import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../")

from src.analysis.rsa.plot_mean_rdms import plot_rdms


if __name__ == "__main__":
    arch = "alexnet"
    epoch = 60
    analysis_name = f"{arch}_bandpass_all_comb_filters"
    in_dir = f"./results/{analysis_name}"
    out_dir = f"./plots/{analysis_name}"

    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # plot settings
    # models to plot
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

    # sigmas to plot
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

    # plot and save
    for model_name in model_names:
        plot_rdms(
            in_dir=in_dir,
            model_name=model_name,
            epoch=epoch,
            out_dir=out_dir,
        )
