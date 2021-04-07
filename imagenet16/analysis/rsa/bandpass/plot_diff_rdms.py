import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.analysis.rsa.rsa import alexnet_layers
from src.analysis.rsa.rdm import load_rdms
from src.analysis.rsa.bandpass.mean_rdms import plot_bandpass_rdms


if __name__ == "__main__":
    in_dir = "./results/mean_rdms/alexnet/"
    out_dir = "./plots/diff_rdms/"
    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    arch = "alexnet"
    epoch = 60

    mode = "normal"
    model_name = f"{arch}_{mode}"
    file_path = os.path.join(in_dir, f"{model_name}_e{epoch:02d}.pkl")
    rdms_normal = load_rdms(file_path=file_path)

    # models to compare
    blur_models = ["mix_s04", "all_s04"]

    for blur_model in blur_models:
        model_name = f"{arch}_{blur_model}"
        file_path = os.path.join(in_dir, f"{model_name}_e{epoch:02d}.pkl")
        rdms_blur = load_rdms(file_path=file_path)

        diff_rdms = {}
        for layer in alexnet_layers:
            diff_rdms[layer] = rdms_blur[layer] - rdms_normal[layer]

        # get analysis parameters.
        num_images = rdms_normal["num_images"]
        num_filters = rdms_normal["num_filters"]

        # (optional) set title
        title = f"16-class-{arch}, {blur_model} - normal, epoch={epoch}"

        # set filename
        filename = "16-class_diff_normal_{}_e{}_f{}_n{}.png".format(
            blur_model, epoch, num_filters, num_images
        )  # add "target_id" if you need it.
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