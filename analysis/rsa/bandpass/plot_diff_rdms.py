import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

from src.analysis.rsa.rsa import alexnet_layers
from src.analysis.rsa.utils import load_rdms
from src.analysis.rsa.bandpass.bandpass_rdm import plot_bandpass_rdms

if __name__ == "__main__":
    arch = "alexnet"
    num_classes = int(sys.argv[1])
    epoch = 60

    metrics = "correlation"
    analysis = f"bandpass_rsm_{metrics}"

    in_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/results/mean_rdms_{metrics}/{num_classes}-class-{arch}/"
    out_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/plots/diff_rdms_{metrics}/{num_classes}-class-{arch}"
    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    mode = "normal"
    model_name = f"{arch}_{mode}"
    in_file = os.path.join(in_dir, f"{model_name}_e{epoch:02d}.pkl")
    rdms_normal = load_rdms(file_path=in_file)

    # models to compare
    blur_models = ["mix_s04", "all_s04"]

    for blur_model in blur_models:
        model_name = f"{arch}_{blur_model}"
        in_file = os.path.join(in_dir, f"{model_name}_e{epoch:02d}.pkl")
        rdms_blur = load_rdms(file_path=in_file)

        diff_rdms = {}
        for layer in alexnet_layers:
            diff_rdms[layer] = rdms_blur[layer] - rdms_normal[layer]

        # get analysis parameters.
        num_images = rdms_normal["num_images"]
        num_filters = rdms_normal["num_filters"]

        # (optional) set title
        # title = f"{num_classes}-class, {blur_model} - normal, epoch={epoch}"
        title = f"{num_classes}-class, B+S-Net (σ=4) - S-Net"

        # set filename
        filename = f"{num_classes}-class_mean_rdns_{metrics}diff_normal_{blur_model}_e{epoch}_f{num_filters}_n{num_images}.png"
        out_file = os.path.join(out_dir, filename)

        # plot_rdms(rdms=diff_rdms, out_file=out_file, plot_show=True)
        plot_bandpass_rdms(
            rdms=diff_rdms,
            num_filters=num_filters,
            vmin=-0.5,
            vmax=0.5,
            title=title,
            out_file=out_file,
            show_plot=False,
        )
