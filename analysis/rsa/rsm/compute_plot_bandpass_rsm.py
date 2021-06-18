import os
import pathlib
import sys

import numpy as np
import torch
import vonenet
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../../")

from src.analysis.rsa.bandpass.bandpass_rsm import (
    compute_bandpass_RSMs,
    plot_bandpass_RSMs,
)
from src.analysis.rsa.rsa import (
    AlexNetRSA,
    VOneNetAlexNetRSA,
    VOneNetAlexNetRSAParallel,
)
from src.analysis.rsa.utils import save_rsms
from src.dataset.imagenet16 import load_imagenet16
from src.image_process.bandpass_filter import make_bandpass_filters
from src.model.utils import load_model
from src.model.load_sin_pretrained_models import load_sin_model

if __name__ == "__main__":
    # ===== args =====
    arch = str(sys.argv[1])  # e.g.: ("alexnet", "vone_alexnet")
    num_classes = int(sys.argv[2])
    compare = str(
        sys.argv[3]
    )  # models to compare e.g.: ("vss", "all_blur-training", "mix_no-blur", "mix_no-sharp")

    epoch = 60

    pretrained_vone = False  # True if you want to use pretrained vone_alexnet.

    imagenet_path = "/mnt/data1/ImageNet/ILSVRC2012/"

    num_filters = 6
    add_noise = False
    metrics = "correlation"  # "1-covariance", "negative-covariance"
    analysis = f"bandpass_rsm_{metrics}"

    # I/O settings
    models_dir = "/mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
        16 if num_classes == 16 else 1000
    )
    results_dir = f"./results/{analysis}/{num_classes}-class/"
    plots_dir = f"./plots/{analysis}/{num_classes}-class/"
    server = str(sys.argv[4])

    if server == "gpu1":
        models_dir = models_dir.replace("data1", "data")
        results_dir = f"/mnt/home/sou/work/blur-training-dev/analysis/rsa/bandpass/rsm/results/{analysis}/{num_classes}-class/"
        plots_dir = f"/mnt/home/sou/work/blur-training-dev/analysis/rsa/bandpass/rsm/plots/{analysis}/{num_classes}-class/"

    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    from src.model.model_names import get_model_names

    model_names = get_model_names(arch=arch, models=compare)

    print("===== arguments =====")
    print("num_classes:", num_classes)
    print("num_filters:", num_filters)
    print("add_noise:", add_noise)
    # print("mean:", mean)
    # print("var:", var)
    print("metrics:", metrics)
    print()

    print("===== I/O =====")
    print("IN, models_dir:", models_dir)
    print("OUT, results_dir:", results_dir)
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

    # make filters
    filters = make_bandpass_filters(num_filters=num_filters)

    for model_name in tqdm(model_names, desc="models"):
        # ===== compute RSM =====
        print()
        print(f"{model_name}: computing RSM...")
        # make RSA instance

        if num_classes == 1000 and "SIN" in model_name:
            # Stylized-ImageNet
            model = load_sin_model(model_name).to(device)
            model.features = model.features.module
            RSA = AlexNetRSA(model)
        elif "vone" in model_name:
            # if pretrained_vone:
            #     model = vonenet.get_model(model_arch=arch, pretrained=True).to(device)
            #     RSA = VOneNetAlexNetRSAParallel(model)
            # else:
            if "untrained" in model_name:
                model_path = ""  # load untrained
            else:
                model_path = os.path.join(
                    models_dir, model_name, f"epoch_{epoch:02d}.pth.tar"
                )
            model = load_model(
                arch=arch, num_classes=num_classes, model_path=model_path
            ).to(device)
            RSA = VOneNetAlexNetRSA(model)
        elif "untrained" in model_name:
            model_path = ""  # load untrained model
            model = load_model(
                arch=arch, num_classes=num_classes, model_path=model_path
            ).to(device)
            RSA = AlexNetRSA(model)
        else:
            model_path = os.path.join(
                models_dir, model_name, f"epoch_{epoch:02d}.pth.tar"
            )
            model = load_model(
                arch=arch, num_classes=num_classes, model_path=model_path
            ).to(device)
            RSA = AlexNetRSA(model)

        # compute mean RSMs
        mean_rsms = compute_bandpass_RSMs(
            RSA=RSA,
            data_loader=test_loader,
            filters=filters,
            add_noise=add_noise,
            # mean=mean,
            # var=var,
            metrics=metrics,
            device=device,
        )

        # save mean RSMs
        # print("saving RSM...")
        result_file = f"{analysis}_{num_classes}-class_{model_name}.pkl"
        result_path = os.path.join(results_dir, result_file)
        save_rsms(mean_rsms=mean_rsms, file_path=result_path)

        # ===== plot RSM =====
        # print(f"{model_name}: plotting RSM...")
        # get analysis parameters.
        # num_images = mean_rsms["num_images"]
        num_filters = mean_rsms["num_filters"]

        # (optional) set title
        plot_title = f"{analysis}, {num_classes}-class, {model_name}"

        # set plot filename
        plot_file = f"{analysis}_{num_classes}-class_{model_name}.png"
        plot_path = os.path.join(plots_dir, plot_file)

        # colour value range of the plots
        vmin = -1
        vmax = 1

        # plot_rsms(rsms=diff_rsms, out_file=out_file, plot_show=True)
        plot_bandpass_RSMs(
            rsms=mean_rsms,
            layers=RSA.layers,
            num_filters=num_filters,
            vmin=vmin,
            vmax=vmax,
            title=plot_title,
            out_file=plot_path,
            show_plot=False,
        )

    print("All done!!")
