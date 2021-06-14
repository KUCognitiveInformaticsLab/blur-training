import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import vonenet
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

from src.dataset.imagenet16 import load_imagenet16
from src.dataset.imagenet import load_imagenet
from src.model.utils import load_model
from src.model.load_sin_pretrained_models import load_sin_model
from src.image_process.bandpass_filter import make_bandpass_filters
from src.analysis.bandpass_acc.bandpass_acc import compute_confusion_matrix


if __name__ == "__main__":
    # ===== args =====
    arch = str(sys.argv[1])  # e.g.: ("alexnet", "vone_alexnet")
    num_classes = int(sys.argv[2])  # number of last output of the models
    test_dataset = str(sys.argv[3])  # test_dataset to use
    compare = str(
        sys.argv[4]
    )  # models to compare e.g.: ("vss", "all_blur-training", "mix_no-blur", "mix_no-sharp")

    analysis = "bandpass_confusion_matrix"
    epoch = 60
    batch_size = 64

    pretrained = False  # True if you want to use pretrained vone_alexnet.

    machine = "server"  # ("server", "local")

    imagenet_path = "/mnt/data1/ImageNet/ILSVRC2012/"

    num_filters = 6  # the number of bandpass filters

    # I/O
    models_dir = (
        "/mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
            16 if num_classes == 16 else 1000  # else is (num_classes == 1000)
        )
        if machine == "server"
        else (
            "/Users/sou/lab2-mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
                16 if num_classes == 16 else 1000  # else means (num_classes == 1000)
            )
        )
    )
    results_dir = f"./results/{analysis}/{num_classes}-class/"
    plots_dir = f"./plots/{analysis}/{num_classes}-class/"

    server = str(sys.argv[5])
    if server == "gpu1":
        models_dir = models_dir.replace("data1", "data")
        results_dir = f"/mnt/home/sou/work/blur-training-dev/analysis/bandpass_acc/results/{analysis}/{num_classes}-class/"
        plots_dir = f"/mnt/home/sou/work/blur-training-dev/analysis/bandpass_acc/plots/{analysis}/{num_classes}-class/"

    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # models to compare
    from src.model.model_names import get_model_names

    model_names = get_model_names(arch=arch, models=compare)

    print("===== arguments =====")
    print("num_classes:", num_classes)
    print("num_filters:", num_filters)
    print("batch_size:", batch_size)
    print("test_dataset:", test_dataset)
    print()

    print("===== I/O =====")
    print("IN, models_dir:", models_dir)
    print("OUT, results_dir:", results_dir)
    print()

    print("===== models to analyze =====")
    print(model_names)
    print()

    # ===== main =====
    print("===== main =====")

    cudnn.benchmark = True  # for fast running

    # random seed settings
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    # loading data
    if test_dataset == "imagenet16":
        _, test_loader = load_imagenet16(
            imagenet_path=imagenet_path, batch_size=batch_size
        )
    elif test_dataset == "imagenet":
        _, _, test_loader = load_imagenet(
            imagenet_path=imagenet_path,
            batch_size=batch_size,
            distributed=False,
            workers=4,
        )

    # make bandpass bandpass_filters
    bandpass_filters = make_bandpass_filters(num_filters=num_filters)

    for model_name in tqdm(model_names, desc="models", leave=False):
        print()
        print(f"{model_name}: computing bandpass confusion matrix...")
        # load model
        if "SIN" in model_name:
            if test_dataset == "imagenet16":
                continue
            # Stylized-ImageNet
            model = load_sin_model(model_name).to(device)
            model.num_classes = num_classes
        # elif "vone" in model_name and pretrained:
        #     model = vonenet.get_model(model_arch=arch, pretrained=True).to(device)
        #     model.num_classes = num_classes
        elif "untrained" in model_name:
            model_path = ""  # load untrained model
            model = load_model(
                arch=arch, num_classes=num_classes, model_path=model_path
            ).to(device)
            model.num_classes = num_classes
        else:
            model_path = os.path.join(
                models_dir, model_name, "epoch_{}.pth.tar".format(epoch)
            )
            model = load_model(
                arch=arch,
                num_classes=num_classes,
                model_path=model_path,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            ).to(device)
            model.num_classes = num_classes

        model.num_classes = num_classes

        # confusion matrix with raw images
        conf_matrix = compute_confusion_matrix(
            model=model, test_loader=test_loader, device=device, raw=True
        )

        # save confusion matrix
        result_name = f"{num_classes}-class_{model_name}_{analysis}_f0.png"
        result_path = os.path.join(results_dir, result_name)
        np.savetxt(result_path, conf_matrix, delimiter=",")

        # plot confusion matrix
        title = f"{analysis} f0, {num_classes}-class, {model_name}"
        plt.title(title)
        sns.heatmap(conf_matrix)
        plot_name = f"{num_classes}-class_{model_name}_{analysis}_f0.png"
        plot_path = os.path.join(plots_dir, plot_name)
        plt.savefig(plot_path)
        plt.close()

        for f, (s1, s2) in tqdm(
            enumerate(bandpass_filters.values(), 1),
            desc="bandpass filters",
            leave=False,
        ):
            conf_matrix = compute_confusion_matrix(
                model=model,
                test_loader=test_loader,
                sigma1=s1,
                sigma2=s2,
                device=device,
            )

            # save confusion matrix
            result_name = f"{num_classes}-class_{model_name}_{analysis}_f{f}.csv"
            result_path = os.path.join(results_dir, result_name)
            np.savetxt(result_path, conf_matrix, delimiter=",")

            # plot confusion matrix
            sns.heatmap(conf_matrix)
            title = f"{analysis} f{f}, {num_classes}-class, {model_name}"
            plt.title(title)
            plot_name = f"{num_classes}-class_{model_name}_{analysis}_f{f}.png"
            plot_path = os.path.join(plots_dir, plot_name)
            plt.savefig(plot_path)
            plt.close()

    print("All done!!")
