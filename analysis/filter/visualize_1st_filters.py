#!/usr/bin/env python
# coding: utf-8

import os
import pathlib
import sys

from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

from src.model.utils import load_model
from src.model.load_sin_pretrained_models import load_sin_model, sin_names
from src.analysis.filter.filter_visualization import plot_filters
from src.model.model_names import rename_model_name

if __name__ == "__main__":
    arch = "alexnet"
    num_classes = int(sys.argv[1])  # number of last output of the models
    epoch = 60

    analysis = "1st-filter_visualization"
    layer_num = 0  # index of 1st layer

    # I/O settings
    models_dir = "/mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
        16 if num_classes == 16 else 1000  # else is (num_classes == 1000)
    )
    plots_dir = f"./plots/{analysis}/{num_classes}-class/"
    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    os.makedirs(plots_dir, exist_ok=True)

    # models to compare
    # model_names = [
    #     "untrained_alexnet",
    #     "alexnet_normal",
    #     sin_names[arch],
    #     # "vone_alexnet",
    # ]

    from src.model.model_names import get_model_names

    model_names = get_model_names(arch=arch)

    print("===== models to analyze =====")
    print(model_names)
    print()

    for model_name in tqdm(model_names, desc="models", leave=False):
        # load model
        if (num_classes == 1000) and ("SIN" in model_name):
            # Stylized-ImageNet
            model = load_sin_model(model_name).cpu()
            model.features = model.features.module
        # elif num_classes == 1000 and "vone" in model_name:
        #     model = vonenet.get_model(model_arch=arch, pretrained=True)
        elif (num_classes == 16) and ("SIN" in model_name) or ("vone" in model_name):
            continue
        elif "untrained" in model_name:
            model_path = ""  # load untrained model
            model = load_model(
                arch=arch, num_classes=num_classes, model_path=model_path
            ).cpu()
        else:
            model_path = os.path.join(
                models_dir, model_name, f"epoch_{epoch:02d}.pth.tar"
            )
            model = load_model(
                arch=arch, num_classes=num_classes, model_path=model_path
            ).cpu()

        # (optional) set title
        plot_title = f"{num_classes}-class, {rename_model_name(model_name)}"

        # plot file path
        output_path = os.path.join(
            plots_dir, f"{analysis}_{num_classes}-class_{model_name}.jpg"
        )

        # visualization of 1st layer filters
        plot_filters(
            model=model, layer_num=layer_num, title=plot_title, file_name=output_path
        )
