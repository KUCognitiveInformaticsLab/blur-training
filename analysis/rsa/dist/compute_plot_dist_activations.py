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

from src.analysis.rsa.rsa import (
    AlexNetRSA,
    VOneNetAlexNetRSA,
)
from src.dataset.imagenet16 import load_imagenet16, make_local_in16_test_loader
from src.image_process.bandpass_filter import make_bandpass_filters, make_blur_filters
from src.model.utils import load_model
from src.analysis.rsa.bandpass.dist import compute_dist_sharp

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

    arch = args.arch
    num_classes = args.num_classes
    epoch = args.epoch

    stimuli = args.stimuli

    models = args.models
    server = args.server

    pretrained_vone = False  # True if you want to use pretrained vone_alexnet.

    imagenet_path = "/mnt/data1/ImageNet/ILSVRC2012/"

    num_filters = 6
    if stimuli == "s-b":
        num_filters = 1
    add_noise = False
    metric = "correlation"  # "1-covariance", "negative-covariance"
    analysis = f"dist_{metric}_{stimuli}"

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

    if args.plot:
        plots_dir = f"./plots/{analysis}/{num_classes}-class/"
        # plots_dir = f"/Users/sou/lab2-work/blur-training-dev/analysis/rsa/tSNE/plots/{analysis}/{num_classes}-class/"
        if args.server != "gpu2":
            plots_dir = f"/mnt/home/sou/work/blur-training-dev/analysis/rsa/tSNE/plots/{analysis}/{num_classes}-class/"
        os.makedirs(plots_dir, exist_ok=True)

    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # models to compare
    from src.model.model_names import get_model_names
    model_names = get_model_names(arch=arch, models=models)

    print("===== arguments =====")
    print("num_classes:", num_classes)
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make Dataloader
    # ** batch_size must be 1 **
    _, test_loader = load_imagenet16(imagenet_path=imagenet_path, batch_size=1)

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

    for model_name in tqdm(model_names, desc="models"):
        # ===== compute =====
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

            # compute dist
            print(f"{model_name} computing...")
            dist_within, dist_btw = compute_dist_sharp(
                RSA=RSA,
                data_loader=test_loader,
                filters=filters,
                metric=metric,
                device=device,
            )

        # ===== plot =====
        import matplotlib.pyplot as plt
        if args.plot:
            fig = plt.figure(dpi=150)
            ax = fig.add_subplot(
                1,
                1,
                1,
                # xlabel="layers",
                ylabel=f"Distance",
            )
            ax.plot(RSA.layers, dist_within, label="within classes")
            ax.plot(RSA.layers, dist_btw, label="between classes")

            ax.set_xticklabels(RSA.layers, rotation=45, ha="right")
            ax.legend()

            plot_file = f"{analysis}_{num_classes}-class_{model_name}.png"
            plot_path = os.path.join(plots_dir, plot_file)
            plt.savefig(plot_path)
            plt.close()

    print("All done!!")
