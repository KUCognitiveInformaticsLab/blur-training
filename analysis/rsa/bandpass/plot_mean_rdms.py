import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

from src.analysis.rsa.rdm import load_rdms
from src.analysis.rsa.bandpass.mean_rdms import plot_bandpass_rdms

if __name__ == "__main__":
    arch = "alexnet"
    num_classes = 16
    epoch = 60

    metrics = "covariance"

    in_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/results/mean_rdms_1-{metrics}/{num_classes}-class-{arch}/"
    out_dir = f"./plots/mean_rdms_1-{metrics}/{num_classes}-class-{arch}"
    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    # models to compare
    model_names = [
        f"{arch}_normal",
        # f"{arch}_multi-steps",
    ]
    modes = [
        # f"{arch}_all",
        # f"{arch}_mix",
        # f"{arch}_random-mix",
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

    for model_name in model_names:
        in_file = os.path.join(in_dir, f"{model_name}_e{epoch:02d}.pkl")
        rdms = load_rdms(file_path=in_file)

        # get analysis parameters.
        num_images = rdms["num_images"]
        num_filters = rdms["num_filters"]

        # (optional) set title
        title = f"mean_rdms(1-{metrics}), {num_classes}-class-{arch}, epoch={epoch}"

        # set filename
        filename = f"mean_rdms_1-{metrics}_{num_classes}-class_{model_name}_e{epoch}_f{num_filters}_n{num_images}.png"
        # add "target_id" if you need it.
        out_file = os.path.join(out_dir, filename)

        # plot_rdms(rdms=diff_rdms, out_file=out_file, plot_show=True)
        plot_bandpass_rdms(
            rdms=rdms,
            num_filters=num_filters,
            vmin=0,
            vmax=2,
            title=title,
            out_file=out_file,
            show_plot=False,
        )
