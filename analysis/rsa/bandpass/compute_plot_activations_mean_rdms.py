import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.bandpass.mean_rdms import (
    compute_mean_rdms_with_bandpass,
    plot_bandpass_rdms,
)
from src.analysis.rsa.rdm import save_rdms

if __name__ == "__main__":
    arch = "alexnet"
    num_classes = 16
    epoch = 60
    metrics = "correlation"  # "1-covariance", "negative-covariance"

    # I/O settings
    results_dir = f".results/mean_rdms_{metrics}/{num_classes}-class-{arch}/"
    plots_dir = f"./plots/mean_rdms_{metrics}/{num_classes}-class-{arch}/"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # models to compare
    model_names = [
        f"{arch}_normal",
        # f"{arch}_multi-steps",
    ]
    modes = [
        f"{arch}_all",
        f"{arch}_mix",
        f"{arch}_random-mix",
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


    # random seed settings
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model_name in model_names:
        mean_rdms = compute_mean_rdms_with_bandpass(
            model=model, data_loader=test_loader, filters=filters, device=device, metrics=metrics
        )

        for layer in alexnet_layers:
            min_list.append((mean_rdms[layer].min()))
            max_list.append((mean_rdms[layer].max()))

        result_file = f"{model_name}_e{epoch:02d}.pkl"
        result_path = os.path.join(results_dir, result_file)
        save_rdms(mean_rdms=mean_rdms, file_path=result_path)

        # get analysis parameters.
        num_images = mean_rdms["num_images"]
        num_filters = mean_rdms["num_filters"]

        # (optional) set title
        title = f"RDM ({metrics}), {num_classes}-class, {model_name}, epoch={epoch}"

        # set filename
        filename = f"mean_rdms_{metrics}_{num_classes}-class_{model_name}_e{epoch}_f{num_filters}_n{num_images}.png"
        out_file = os.path.join(plots_dir, filename)

        # colour value range of the plots
        vmin = 0 if metrics == "correlation" else -5
        vmax = 2 if metrics == "correlation" else 5

        # plot_rdms(rdms=diff_rdms, out_file=out_file, plot_show=True)
        plot_bandpass_rdms(
            rdms=mean_rdms,
            num_filters=num_filters,
            vmin=vmin,
            vmax=vmax,
            title=title,
            out_file=out_file,
            show_plot=False,
        )
