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

    in_dir = f"./results/mean_rdms_1-{metrics}/{num_classes}-class-{arch}/"
    out_dir = f"./plots/mean_rdms_1-{metrics}/{num_classes}-class-{arch}"
    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    mode = "normal"
    model_name = f"{arch}_{mode}"
    rdms = load_rdms(file_path=epoch)

    # get analysis parameters.
    num_images = rdms["num_images"]
    num_filters = rdms["num_filters"]

    # (optional) set title
    title = f"mean_rdms(1-{metrics}), {num_classes}-class-{arch}, epoch={epoch}"

    # set filename
    filename = f"mean_rdms_1-{metrics}_{num_classes}-class-{arch}_e{epoch}_f{num_filters}_n{num_images}.png"
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
