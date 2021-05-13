import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.analysis.rsa.utils import load_rdms
from src.analysis.rsa.bandpass.bandpass_rsm import plot_bandpass_RSMs
from src.model.load_sin_pretrained_models import sin_names
from src.analysis.rsa.rsa import alexnet_layers, vone_alexnet_layers

if __name__ == "__main__":
    arch = "alexnet"
    num_classes = 1000
    epoch = 60

    metrics = "correlation"  # ("correlation", "1-covariance", "negative-covariance")
    analysis = f"bandpass_rdm_{metrics}"
    new_analysis = f"bandpass_rsm_{metrics}"

    in_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/results/{analysis}/{num_classes}-class/"
    out_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/plots/{new_analysis}/{num_classes}-class/"
    # out_dir = f"plots/{new_analysis}/{num_classes}-class/"

    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # models to compare
    model_names = [
        f"{arch}_normal",
        f"{arch}_all_s04",
        f"{arch}_mix_s04",
        f"vone_{arch}",
        sin_names[arch],
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

    for model_name in model_names:
        in_file = os.path.join(in_dir, f"{analysis}_{model_name}.pkl")
        rdms = load_rdms(file_path=in_file)

        # RDM -> RSM
        rsms = {}
        if "vone" in model_name:
            for layer in vone_alexnet_layers:
                if layer == "last-outputs":
                    rsms[layer] = 1 - rdms["fc-relu-3"]
                else:
                    rsms[layer] = 1 - rdms[layer]
        else:
            for layer in alexnet_layers:
                rsms[layer] = 1 - rdms[layer]

        # get analysis parameters.
        # num_images = rdms["num_images"]
        num_filters = rdms["num_filters"]
        rsms["num_filters"] = rdms["num_filters"]

        # (optional) set title
        title = f"RSM, {num_classes}-class, {model_name}"

        # set filename
        filename = f"{new_analysis}_{model_name}_f{num_filters}.png"
        out_file = os.path.join(out_dir, filename)

        # colour value range of the plots
        vmin = -1
        vmax = 1

        # plot_rdms(rdms=diff_rdms, out_file=out_file, plot_show=True)
        plot_bandpass_RSMs(
            rsms=rsms,
            layers=vone_alexnet_layers if "vone" in model_name else alexnet_layers,
            num_filters=num_filters,
            vmin=vmin,
            vmax=vmax,
            title=title,
            out_file=out_file,
            show_plot=False,
        )
