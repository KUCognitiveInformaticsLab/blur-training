import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.bandpass.slr import compute_slr, plot_slr


if __name__ == "__main__":
    # arguments
    arch = "alexnet"
    epoch = 60
    num_filters = 6
    num_images = 1600

    # I/O settings
    data_dir = "./results/activations/alexnet/"
    results_dir = "./results/slr"

    assert os.path.exists(data_dir), f"{data_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)

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

    # create model names list
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
        in_dir = os.path.join(data_dir, model_name + f"_e{epoch:02d}")

        df_w, df_res, df_r2 = compute_slr(
            in_dir=in_dir,
            out_dir=results_dir,
            model_name=model_name,
            epoch=epoch,
            num_filters=num_filters,
            num_images=num_images,
        )

        ### plot (temp) ###
        plots_dir = f"./plots/slr/{model_name}_e{epoch:02d}"

        os.makedirs(plots_dir, exist_ok=True)

        plot_slr(df_w=df_w, df_res=df_res, df_r2=df_r2, out_dir=plots_dir)
