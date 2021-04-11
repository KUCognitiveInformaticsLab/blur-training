import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

from src.analysis.rsa.utils import load_rdms
from src.analysis.rsa.bandpass.mean_rdms import plot_bandpass_rdms

if __name__ == "__main__":
    arch = "alexnet"
    num_classes = 16
    epoch = 60

    mean = 0
    var = 0.01

    metrics = "correlation"  # ("correlation", "1-covariance", "negative-covariance")

    in_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/results/mean_rdms_bandpass-noise_no-all-random_correlation/{num_classes}-class-{arch}/"
    out_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/plots/mean_rdms_noise_bandpass-noise_no-all-random_correlation_annot/{num_classes}-class-{arch}/"
    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # models to compare
    model_names = [
        "alexnet_normal",
        "alexnet_all_s04",
        "alexnet_mix_s04",
    ]

    # model_names = [
    #     f"{arch}_normal",
    #     # f"{arch}_multi-steps",
    # ]
    # modes = [
    #     f"{arch}_all",
    #     f"{arch}_mix",
    #     f"{arch}_random-mix",
    #     f"{arch}_single-step",
    #     # f"{arch}_fixed-single-step",
    #     # f"{arch}_reversed-single-step",
    # ]
    #
    # # sigmas to compare
    # sigmas_mix = [s for s in range(1, 6)] + [10]
    # sigmas_random_mix = ["00-05", "00-10"]
    #
    # # add sigma to compare to the model names
    # for mode in modes:
    #     if mode == f"{arch}_random-mix":
    #         for min_max in sigmas_random_mix:
    #             model_names += [f"{mode}_s{min_max}"]
    #     elif mode == f"{arch}_mix":
    #         for sigma in sigmas_mix:
    #             model_names += [f"{mode}_s{sigma:02d}"]
    #     else:
    #         for sigma in range(1, 5):
    #             model_names += [f"{mode}_s{sigma:02d}"]

    from src.analysis.rsa.rsa import alexnet_layers

    min_list = []
    max_list = []

    for model_name in model_names:
        in_file = os.path.join(in_dir, f"{model_name}_e{epoch:02d}.pkl")
        rdms = load_rdms(file_path=in_file)

        for layer in alexnet_layers:
            min_list.append((rdms[layer].min()))
            max_list.append((rdms[layer].max()))

        # get analysis parameters.
        # num_images = rdms["num_images"]
        num_filters = rdms["num_filters"]

        # (optional) set title
        title = f"RDM ({metrics}) with Gaussian noise (mean={mean}, var={var}), {num_classes}-class, {model_name}, epoch={epoch}"

        # set filename
        filename = f"mean_rdms_{metrics}_{num_classes}-class_{model_name}_e{epoch}_f{num_filters}_mean{mean}_var{var}.png"
        out_file = os.path.join(out_dir, filename)

        # colour value range of the plots
        vmin = 0 if metrics == "correlation" else -5
        vmax = 2 if metrics == "correlation" else 5

        # plot_rdms(rdms=diff_rdms, out_file=out_file, plot_show=True)
        plot_bandpass_rdms(
            rdms=rdms,
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
