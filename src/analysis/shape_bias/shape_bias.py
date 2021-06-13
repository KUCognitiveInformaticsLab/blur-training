#!/usr/bin/env python
# coding: utf-8

import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

# Ref: https://github.com/rgeirhos/texture-vs-shape/tree/master/code
from src.utils.mapping import probabilities_to_decision
from src.model.load_sin_pretrained_models import load_sin_model
from src.model.utils import load_model
from src.dataset.imagenet16 import label_map
from src.dataset.cue_conflict import load_cue_conflict
from src.utils.dictionary import get_key_from_value
from vonenet import get_model


# create mapping
mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()


def compute_shape_bias(model, num_classes, cue_conf_data_path, device):
    """Test with cue-conflict images and record correct decisions.
    * You need to exclude images without a cue conflict (e.g. texture=cat, shape=cat)
    Dataset: Cue-conflict
        Size: 224 x 224 (RGB)
        Dataset size: 1,200 images
            # The number of total images is 1,280.
            # The number of images without a cue conflict is 80 (5 per each shape category)
            # Thus, the number of the images becomes 1,200.
            # The number of shape images: 75 (per each shape category)
            # The number of texture images: 75 (per each tecture category)
    """
    num_labels = 16  # number of classes

    # === compute shape-vs-texture decisions ===
    # make numpy arrays for recording
    correct_shape_decisions = np.zeros(num_labels)
    correct_texture_decisions = np.zeros(num_labels)
    all_results = []
    all_file_names = []

    # make dataloader
    cue_conf_loader = load_cue_conflict(data_path=cue_conf_data_path)

    model.eval()
    with torch.no_grad():
        for images, _, file_names in tqdm(
            cue_conf_loader, desc="cue-conf images", leave=False
        ):
            all_file_names.extend(file_names)
            images = images.to(device)
            # get labels from file names
            # first one is shape label. second one is texture label
            labels = [re.sub("\d+", "", f.split(".")[0]).split("-") for f in file_names]
            outputs = model(images)

            for i in range(outputs.shape[0]):
                # get label keys from label_map (type: int)
                shape_id, texture_id = [
                    get_key_from_value(label_map, v) for v in labels[i]
                ]

                # get model decision
                if num_classes == 1000:
                    outputs = torch.nn.Softmax(dim=1)(outputs)  # softmax
                    # get model_decision (str) by mappig outputs from 1,000 to 16
                    model_decision = mapping.probabilities_to_decision(
                        outputs[i].detach().cpu().numpy()
                    )
                elif num_classes == 16:
                    # outputs = torch.nn.Softmax(dim=1)(outputs)  # softmax
                    _, pred = outputs[i].topk(1)
                    model_decision = label_map[pred.item()]

                # record all results
                all_results.append(
                    [
                        label_map[shape_id],  # shape label
                        label_map[texture_id],  # texture label
                        model_decision,  # model decision (str)
                        # *outputs[i].cpu().detach().numpy()  # remove all outputs
                    ]
                )

                if (
                    not shape_id == texture_id
                ):  # Exclude images without a cue conflict (e.g. texture=cat, shape=cat)
                    # Sum up shape or texture decisions (when either shape or texture category is correctly predicted).
                    correct_shape_decisions[shape_id] += float(
                        (model_decision == label_map[shape_id])
                    )
                    correct_texture_decisions[texture_id] += float(
                        (model_decision == label_map[texture_id])
                    )

    # save all decisions
    df_all_decisions = pd.DataFrame(
        all_results,
        index=all_file_names,
        columns=np.array(
            [
                "shape_label",
                "texture_label",
                "model_decision",
                # *label_map.values()  # remove all outputs
            ]
        ),
    )

    # save correct decisions
    correct_results = np.concatenate(
        [
            correct_shape_decisions.reshape(1, -1),
            correct_texture_decisions.reshape(1, -1),
        ]
    )
    df_correct_decisions = pd.DataFrame(
        correct_results,
        index=["correct_shape_decisions", "correct_texture_decisions"],
        columns=list(label_map.values()),
    )

    return df_all_decisions, df_correct_decisions


if __name__ == "__main__":
    arch = "alexnet"
    num_classes = 1000
    epoch = 60
    models_dir = "../../logs/models/"  # model directory
    results_dir = "./results/{}".format(arch)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # models to compare
    model_names = [
        f"{arch}_normal",
        f"{arch}_multi-steps",
    ]
    modes = [
        f"{arch}_all",
        f"{arch}_mix",
        f"{arch}_random-mix",
        f"{arch}_single-step",
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

    # VOneNet
    model_names += ["{}_vonenet".format(arch)]

    # Stylized-ImageNet
    sin_names = {
        "alexnet": "alexnet_trained_on_SIN",
        "vgg16": "vgg16_trained_on_SIN",
        "resnet50": "resnet50_trained_on_SIN",
    }
    model_names += sin_names[arch]

    for model_name in model_names:
        print(model_name)
        # load model
        if "vonenet" in model_name:
            model = get_model(model_arch=arch, pretrained=True).to(device)
            all_file = os.path.join(
                results_dir, "all_decisions_{}.csv".format(model_name)
            )
            correct_file = os.path.join(
                results_dir, "correct_decisions_{}.csv".format(model_name)
            )
        elif "SIN" in model_name:
            model = load_sin_model(model_name).to(device)
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
            model = load_model(model_path).to(device)
            all_file = os.path.join(
                results_dir, "all_decisions_{}_e{}.csv".format(model_name, epoch)
            )
            correct_file = os.path.join(
                results_dir, "correct_decisions_{}_e{}.csv".format(model_name, epoch)
            )

        # compute
        df_all_decisions, df_correct_decisions = compute_shape_bias(
            model=model, num_classes=num_classes
        )

        # save
        df_all_decisions.to_csv(all_file)
        df_correct_decisions.to_csv(correct_file)
