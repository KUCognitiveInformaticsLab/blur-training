import argparse
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.rsa import (
    AlexNetRSA,
    VOneNetAlexNetRSA,
    VOneNetAlexNetRSAParallel,
    alexnet_layers,
    vone_alexnet_layers,
)
from src.dataset.imagenet16 import make_local_in16_test_loader
from src.image_process.bandpass_filter import make_bandpass_filters, make_blur_filters
from src.model.utils import load_model
from src.analysis.rsa.bandpass.dist import (
    compute_corr2dist,
    compute_corr2dist_h_l,
    compute_corr2dist_s_h,
    plot_dist,
)
from src.model.model_names import rename_model_name

parser = argparse.ArgumentParser()
parser.add_argument(
    "--stimuli",
    default="s-b",
    type=str,
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
    "--full",
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
parser.add_argument("--server", default="gpu2", type=str)
parser.add_argument("--machine", default="server", type=str)


if __name__ == "__main__":
    # ===== args =====
    args = parser.parse_args()

    pretrained_vone = False  # True if you want to use pretrained vone_alexnet.

    imagenet_path = "/mnt/data1/ImageNet/ILSVRC2012/"

    num_filters = 6
    if args.stimuli == "s-b":
        num_filters = 1
    if args.stimuli == "h-l":
        num_filters = 2
    add_noise = False
    metric = "correlation"
    analysis = f"dist_{metric}_{args.stimuli}"

    # I/O settings
    if args.compute:
        # imagenet_path = os.path.join(args.data_dir, "ImageNet/ILSVRC2012/")
        in16_test_path = os.path.join(args.data_dir, "imagenet16/test/")
        if args.server != "gpu2":
            args.data_dir = "/mnt/data"
        models_dir = os.path.join(
            args.data_dir,
            "pretrained_models/blur-training/imagenet{}/models/".format(
                16
                if args.num_classes == 16
                else 1000  # else is (args.num_classes == 1000)
            ),
        )
        # assert os.path.exists(imagenet_path), f"{imagenet_path} does not exist."
        assert os.path.exists(in16_test_path), f"{in16_test_path} does not exist."
        assert os.path.exists(models_dir), f"{models_dir} does not exist."

    results_dir = (
        f"./results/{analysis}/{args.num_classes}-class/"
        if args.machine == "server"
        else f"/Users/sou/lab2-work/blur-training-dev/analysis/rsa/dist/results/{analysis}/{args.num_classes}-class/"
    )
    if args.machine == "server" and args.server != "gpu2":
        results_dir = f"/mnt/home/sou/work/blur-training-dev/analysis/rsa/dist/results/{analysis}/{args.num_classes}-class/"

    os.makedirs(results_dir, exist_ok=True)

    if args.plot:
        plots_dir = f"./plots/{analysis}/{args.num_classes}-class/"
        # plots_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/rsa/dist/plots/{analysis}/{args.num_classes}-class/"
        if args.server != "gpu2":
            plots_dir = f"/mnt/home/sou/work/blur-training-dev/analysis/rsa/dist/plots/{analysis}/{args.num_classes}-class/"
        os.makedirs(plots_dir, exist_ok=True)

    # models to compare
    from src.model.model_names import get_model_names

    model_names = get_model_names(arch=args.arch, models=args.models, num_classes=args.num_classes)

    print("===== arguments =====")
    print("num_classes:", args.num_classes)
    print("num_filters:", num_filters)
    print("add_noise:", add_noise)
    print("metric:", metric)
    print()

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
        if args.stimuli == "s-b":
            filters = make_blur_filters(sigmas=[4])  # blur filters (sigma=sigmas)
        elif args.stimuli == "h-l":
            filters = {}
            filters[0] = [1, 2]  # high-pass
            filters[1] = [4, None]  # low-pass
        elif args.stimuli == "s-h":
            filters = {}
            filters[0] = [1, 2]  # high-pass

    for model_name in tqdm(model_names, desc="models"):
        if "1label" in model_name:
            excluded_labels = [15]
        elif "8label" in model_name:
            excluded_labels = [i for i in range(8, 16)]
        else:
            excluded_labels = []

        # ===== compute =====
        if args.compute:
            print(f"{model_name} computing...")
            # load model
            if "untrained" in model_name:
                model_path = ""  # load untrained model
            else:
                model_path = os.path.join(
                    models_dir, model_name, f"epoch_{args.epoch:02d}.pth.tar"
                )
            model = load_model(
                arch=args.arch, num_classes=args.num_classes, model_path=model_path, model_name=model_name,
            ).to(device)

            # make RSA instance
            if "vone" in model_name:
                if args.num_classes == 1000:
                    RSA = VOneNetAlexNetRSAParallel(model)
                else:
                    RSA = VOneNetAlexNetRSA(model)
            else:
                RSA = AlexNetRSA(model)

            # compute dist
            if args.stimuli == "s-b":
                df_dist = compute_corr2dist(
                    RSA=RSA,
                    data_loader=test_loader,
                    filters=filters,
                    device=device,
                    excluded_labels=excluded_labels,
                )
            elif args.stimuli == "h-l":
                df_dist = compute_corr2dist_h_l(
                    RSA=RSA,
                    data_loader=test_loader,
                    filters=filters,
                    device=device,
                )
            elif args.stimuli == "s-h":
                df_dist = compute_corr2dist_s_h(
                    RSA=RSA,
                    data_loader=test_loader,
                    filters=filters,
                    device=device,
                )

        result_file = f"{analysis}_{args.num_classes}-class_{model_name}.csv"
        result_path = os.path.join(results_dir, result_file)

        if args.compute:
            # save dist
            df_dist.to_csv(result_path)

        # ===== plot =====
        if args.plot:
            if "vone" in model_name:
                layers = vone_alexnet_layers
            else:
                layers = alexnet_layers

            # load dist
            df_dist = pd.read_csv(result_path, index_col=0)

            plot_file = f"{analysis}_{args.num_classes}-class_{model_name}_separate.png"
            if args.full:
                plot_file = plot_file.replace(".png", "_full.png")
            plot_path = os.path.join(plots_dir, plot_file)

            plot_dist(dist=df_dist, stimuli=args.stimuli, compare=f"separate", layers=layers,
                      title=f"{args.num_classes}-class, {rename_model_name(model_name)}", plot_path=plot_path,
                      excluded_labels=excluded_labels, full=args.full)

            plot_file = f"{analysis}_{args.num_classes}-class_{model_name}_cross.png"
            if args.full:
                plot_file = plot_file.replace(".png", "_full.png")
            plot_path = os.path.join(plots_dir, plot_file)

            plot_dist(dist=df_dist, stimuli=args.stimuli, compare="cross", layers=layers,
                      title=f"{args.num_classes}-class, {rename_model_name(model_name)}", plot_path=plot_path,
                      excluded_labels=excluded_labels, full=args.full)

    print("All done!!")
