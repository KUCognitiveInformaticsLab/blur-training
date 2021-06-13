import os
import pathlib
import sys

import torch

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

from src.analysis.shape_bias.shape_bias import compute_shape_bias
from src.model.load_sin_pretrained_models import load_sin_model
from src.model.utils import load_model

import vonenet


if __name__ == "__main__":
    arch = str(sys.argv[1])  # e.g.: ("alexnet", "vone_alexnet")
    num_classes = int(sys.argv[2])  # number of last output of the models
    compare = str(sys.argv[3])  # models to compare e.g.: ("vss", "all_blur-training", "mix_no-blur", "mix_no-sharp")

    epoch = 60
    pretrained = False  # True if you want to use pretrained vone_alexnet.

    # I/O
    models_dir = "/mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
        16 if num_classes == 16 else ""  # else is (num_classes == 1000)
    )
    results_dir = f"./results/{num_classes}-class-{arch}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # models to compare
    from src.model.model_names import get_model_names

    model_names = get_model_names(arch=arch, compare=compare)

    # VOneNet
    model_names += ["{}_vonenet".format(arch)]

    # Stylized-ImageNet
    sin_names = {
        "alexnet": "alexnet_trained_on_SIN",
        "vgg16": "vgg16_trained_on_SIN",
        "resnet50": "resnet50_trained_on_SIN",
    }
    model_names += [sin_names[arch]]

    for model_name in model_names:
        print(model_name)
        # load model
        if "SIN" in model_name:
            # Stylized-ImageNet
            model = load_sin_model(model_name).to(device)
            model.num_classes = num_classes
            all_file = os.path.join(
                results_dir, "all_decisions_{}.csv".format(model_name)
            )
            correct_file = os.path.join(
                results_dir, "correct_decisions_{}.csv".format(model_name)
            )
        elif "vone" in model_name and pretrained:
            model = vonenet.get_model(model_arch=arch, pretrained=True).to(device)
            model.num_classes = num_classes
            all_file = os.path.join(
                results_dir, "all_decisions_{}.csv".format(model_name)
            )
            correct_file = os.path.join(
                results_dir, "correct_decisions_{}.csv".format(model_name)
            )
        elif "untrained" in model_name:
            model_path = ""  # load untrained model
            model = load_model(
                arch=arch, num_classes=num_classes, model_path=model_path
            ).to(device)
            model.num_classes = num_classes
            all_file = os.path.join(
                results_dir, "all_decisions_{}.csv".format(model_name)
            )
            correct_file = os.path.join(
                results_dir, "correct_decisions_{}.csv".format(model_name)
            )
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
            all_file = os.path.join(
                results_dir, "all_decisions_{}_e{}.csv".format(model_name, epoch)
            )
            correct_file = os.path.join(
                results_dir, "correct_decisions_{}_e{}.csv".format(model_name, epoch)
            )

        # compute
        df_all_decisions, df_correct_decisions = compute_shape_bias(model=model, num_classes=num_classes)

        # save
        df_all_decisions.to_csv(all_file)
        df_correct_decisions.to_csv(correct_file)
