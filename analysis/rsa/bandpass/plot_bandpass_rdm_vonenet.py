import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

from src.analysis.rsa.utils import load_rdms
from src.analysis.rsa.rsa import vone_alexnet_layers
from src.analysis.rsa.bandpass.bandpass_rdm import plot_bandpass_rdms

if __name__ == "__main__":
    arch = "alexnet"
    num_classes = 1000
    epoch = 60

    metrics = "correlation"  # ("correlation", "1-covariance", "negative-covariance")
    analysis = f"bandpass_rdm_{metrics}"

    in_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/results/{analysis}/{num_classes}-class/"
    out_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/plots/{analysis}/{num_classes}-class/"
    # out_dir = f"plots/{analysis}/{num_classes}-class/"

    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # models to compare
    model_names = [
        "vone_alexnet",
    ]

    for model_name in model_names:
        in_file = os.path.join(in_dir, f"{analysis}_{model_name}.pkl")
        rdms = load_rdms(file_path=in_file)

        # get analysis parameters.
        # num_images = rdms["num_images"]
        num_filters = rdms["num_filters"]

        # (optional) set title
        title = f"RDM, {num_classes}-class, {model_name}"

        # set filename
        filename = f"{analysis}_{model_name}_f{num_filters}.png"
        out_file = os.path.join(out_dir, filename)

        # colour value range of the plots
        vmin = 0 if metrics == "correlation" else -5
        vmax = 2 if metrics == "correlation" else 5

        # plot_rdms(rdms=diff_rdms, out_file=out_file, plot_show=True)
        plot_bandpass_rdms(
            rdms=rdms,
            layers=vone_alexnet_layers,
            num_filters=num_filters,
            vmin=vmin,
            vmax=vmax,
            title=title,
            out_file=out_file,
            show_plot=False,
        )
