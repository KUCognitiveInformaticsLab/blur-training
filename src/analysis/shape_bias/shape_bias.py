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


# save
# df_all_decisions.to_csv(file_path)
# df_correct_decisions.to_csv(file_path)


def get_shape_bias(file_path):
    correct_decisions = pd.read_csv(file_path, index_col=0).values
    # compute shape bias
    shape_bias = correct_decisions[0].sum() / (
        correct_decisions[0].sum() + correct_decisions[1].sum()
    )
    return shape_bias


def get_cue_conf_acc(file_path):
    correct_df = pd.read_csv(file_path, index_col=0)
    correct_decisions = correct_df.values.sum()
    return correct_decisions / 1200
