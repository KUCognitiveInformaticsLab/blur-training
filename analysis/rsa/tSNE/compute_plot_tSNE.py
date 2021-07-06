import argparse
import os
import pathlib
import sys

import numpy as np
import torch
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.bandpass.t_sne import (
    compute_tSNE_each_bandpass,
    compute_tSNE_all_bandpass,
    compute_tSNE_h_l,
    save_embedded_activations,
    load_embedded_activations,
    plot_tSNE_each_bandpass,
    plot_tSNE_all_bandpass,
    plot_tSNE_s_b,
    plot_tSNE_h_l,
)
from src.analysis.rsa.rsa import (
    AlexNetRSA,
    VOneNetAlexNetRSA,
    alexnet_layers,
    vone_alexnet_layers,
)
from src.dataset.imagenet16 import make_local_in16_test_loader
from src.image_process.bandpass_filter import make_bandpass_filters, make_blur_filters
from src.model.utils import load_model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--stimuli",
    default="each_bandpass",
    type=str,
    choices=["each_bandpass", "all_bandpass", "s-b", "h-l"],
)
parser.add_argument(
    "--models",
    default="vss",
    type=str,
)
parser.add_argument(
    "--compute",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--plot",
    action="store_true",
    default=False,
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="alexnet",
)
parser.add_argument(
    "--num_classes",
    default=16,
    type=int,
)
parser.add_argument(
    "--epoch",
    default=60,
    type=int,
)
parser.add_argument("--data_dir", default="/mnt/data1", type=str)
parser.add_argument("--model_names", nargs="+", type=str)
parser.add_argument(
    "--num_filters",
    default=6,
    type=int,
)
parser.add_argument(
    "-d",
    "--num_dim",
    default=2,
    type=int,
)
parser.add_argument(
    "--perplexity",
    default=30,
    type=int,
)
parser.add_argument(
    "--n_iter",
    default=1000,
    type=int,
)
parser.add_argument("--server", default="gpu2", type=str)
parser.add_argument("--machine", default="server", type=str)
# parser.add_argument(
#     "--compute",
#     type=strtobool,
#     default=1,
# )


if __name__ == "__main__":
    # python compute_plot_tSNE.py --compute --plot --stimuli each_bandpass
    # ===== args =====
    args = parser.parse_args()

    arch = args.arch
    num_classes = args.num_classes
    epoch = args.epoch

    stimuli = args.stimuli  # "each_bandpass", "all_bandpass"
    analysis = f"tSNE_{stimuli}"
    compute = args.compute
    plot = args.plot
    num_filters = args.num_filters
    if stimuli == "s-b":
        num_filters = 1
    num_dim = args.num_dim
    perplexity = args.perplexity
    n_iter = args.n_iter

    # I/O settings
    if args.compute:
        # imagenet_path = os.path.join(args.data_dir, "ImageNet/ILSVRC2012/")
        in16_test_path = os.path.join(args.data_dir, "imagenet16/test/")
        if args.server != "gpu2":
            args.data_dir = "/mnt/data"
        models_dir = os.path.join(
            args.data_dir,
            "pretrained_models/blur-training/imagenet{}/models/".format(
                16 if num_classes == 16 else 1000  # else is (num_classes == 1000)
            ),
        )
        # assert os.path.exists(imagenet_path), f"{imagenet_path} does not exist."
        assert os.path.exists(in16_test_path), f"{in16_test_path} does not exist."
        assert os.path.exists(models_dir), f"{models_dir} does not exist."

    results_dir = (
        f"./results/{analysis}/{num_classes}-class/"
        if args.machine == "server"
        else f"/Users/sou/lab2-work/blur-training-dev/analysis/rsa/tSNE/results/{analysis}/{num_classes}-class/"
    )
    if args.machine == "server" and args.server != "gpu2":
        results_dir = f"/mnt/home/sou/work/blur-training-dev/analysis/rsa/tSNE/results/{analysis}/{num_classes}-class/"
    os.makedirs(results_dir, exist_ok=True)

    if args.plot:
        plots_dir = f"./plots/{analysis}/{num_classes}-class/"
        # plots_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/rsa/tSNE/plots/{analysis}/{num_classes}-class/"
        if args.server != "gpu2":
            plots_dir = f"/mnt/home/sou/work/blur-training-dev/analysis/rsa/tSNE/plots/{analysis}/{num_classes}-class/"
        os.makedirs(plots_dir, exist_ok=True)

    # models to compare
    from src.model.model_names import get_model_names

    model_names = get_model_names(arch=arch, models=args.models, num_classes=args.num_classes)

    print("===== arguments =====")
    print("analysis:", analysis)
    print("compute:", compute)
    print("plot:", plot)
    print("num_classes:", num_classes)
    print("num_filters:", num_filters)
    print("num_dim:", num_dim)
    print("perplexity:", perplexity)
    print("n_iter:", n_iter)

    print("===== I/O =====")
    if args.compute:
        print("IN, models_dir:", models_dir)
        print("OUT, results_dir:", results_dir)
    if args.plot:
        print("IN, results_dir:", results_dir)
        print("OUT, plots_dir:", plots_dir)
    print()

    print("===== models to analyze =====")
    print(model_names)
    print()

    # ===== main =====
    print("===== main =====")
    print()

    # random seed settings
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if args.compute:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # make Dataloader
        # ** batch_size must be 1 **
        # _, test_loader = load_imagenet16(imagenet_path=imagenet_path, batch_size=1)
        test_loader = make_local_in16_test_loader(
            data_path=in16_test_path, batch_size=1, shuffle=False
        )

        # make filters
        filters = make_bandpass_filters(num_filters=num_filters)
        if stimuli == "s-b":
            filters = make_blur_filters(sigmas=[4])  # blur filters (sigma=sigmas)
        elif args.stimuli == "h-l":
            filters = {}
            filters[0] = [1, 2]  # high-pass
            filters[1] = [4, None]  # low-pass

    for model_name in tqdm(model_names, desc="models"):
        # ===== compute RSM =====
        if args.compute:
            # load model
            if "untrained" in model_name:
                model_path = ""  # load untrained model
            else:
                model_path = os.path.join(
                    models_dir, model_name, f"epoch_{epoch:02d}.pth.tar"
                )
            model = load_model(
                arch=arch, num_classes=num_classes, model_path=model_path
            ).to(device)

            # make RSA instance
            if "vone" in model_name:
                RSA = VOneNetAlexNetRSA(model)
            else:
                RSA = AlexNetRSA(model)

            # compute bandpass tSNE
            print(f"{model_name} computing...")
            if stimuli == "each_bandpass":
                embed, labels = compute_tSNE_each_bandpass(
                    RSA=RSA,
                    num_images=test_loader.num_images,
                    data_loader=test_loader,
                    filters=filters,
                    num_dim=num_dim,
                    perplexity=perplexity,
                    n_iter=n_iter,
                    device=device,
                )  # (F+1, L, N, D), (N)
            elif stimuli == "all_bandpass" or stimuli == "s-b":
                embed, labels = compute_tSNE_all_bandpass(
                    RSA=RSA,
                    num_images=test_loader.num_images,
                    data_loader=test_loader,
                    filters=filters,
                    num_dim=num_dim,
                    perplexity=perplexity,
                    n_iter=n_iter,
                    device=device,
                )  # (L, N * (F+1), D), (N)
            elif stimuli == "h-l":
                embed, labels = compute_tSNE_h_l(
                    RSA=RSA,
                    num_images=test_loader.num_images,
                    data_loader=test_loader,
                    filters=filters,
                    num_dim=num_dim,
                    perplexity=perplexity,
                    n_iter=n_iter,
                    device=device,
                )  # (L, N * F, D), (N)

        result_file = f"{analysis}_embedded_activations_{num_dim}d_p{perplexity}_i{n_iter}_{num_classes}-class_{model_name}.npy"
        result_path = os.path.join(results_dir, result_file)

        if compute:
            # save t-SNE embedded activations
            save_embedded_activations(
                embedded_activations=embed, labels=labels, file_path=result_path
            )

        # === plot t-SNE ===
        if plot:
            embed, labels = load_embedded_activations(
                file_path=result_path
            )  # (F+1, L, N, D), (N) or (L, N * (F+1), D), (N * (F+1))

            if "vone" in model_name:
                layers = vone_alexnet_layers
            else:
                layers = alexnet_layers

            if stimuli == "each_bandpass":
                plot_tSNE_each_bandpass(
                    embedded_activations=embed,
                    labels=labels,
                    num_filters=num_filters,
                    layers=layers,
                    num_dim=num_dim,
                    plots_dir=plots_dir,
                    analysis=analysis,
                    perplexity=perplexity,
                    n_iter=n_iter,
                    num_classes=num_classes,
                    model_name=model_name,
                    title=True,
                )
            elif stimuli == "all_bandpass":
                plot_tSNE_all_bandpass(
                    embedded_activations=embed,
                    labels=labels,
                    layers=layers,
                    num_dim=num_dim,
                    plots_dir=plots_dir,
                    analysis=analysis,
                    perplexity=perplexity,
                    n_iter=n_iter,
                    num_classes=num_classes,
                    model_name=model_name,
                    title=True,
                )
            elif stimuli == "s-b":
                plot_tSNE_s_b(
                    embedded_activations=embed,
                    labels=labels,
                    layers=layers,
                    num_dim=num_dim,
                    plots_dir=plots_dir,
                    analysis=analysis,
                    perplexity=perplexity,
                    n_iter=n_iter,
                    num_classes=num_classes,
                    model_name=model_name,
                    title=True,
                )
                # plot each layer
                # plot_tSNE_s_b_each_layer(
                #     embedded_activations=embed,
                #     labels=labels,
                #     layers=layers,
                #     num_dim=num_dim,
                #     plots_dir=plots_dir,
                #     analysis=analysis,
                #     perplexity=perplexity,
                #     n_iter=n_iter,
                #     num_classes=num_classes,
                #     model_name=model_name,
                #     title=True,
                # )
            elif stimuli == "h-l":
                plot_tSNE_h_l(
                    embedded_activations=embed,
                    labels=labels,
                    layers=layers,
                    num_dim=num_dim,
                    plots_dir=plots_dir,
                    analysis=analysis,
                    perplexity=perplexity,
                    n_iter=n_iter,
                    num_classes=num_classes,
                    model_name=model_name,
                    title=True,
                )
