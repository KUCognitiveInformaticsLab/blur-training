#!/usr/bin/env python
# coding: utf-8

import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

from src.image_process.lowpass_filter import GaussianBlurAll
from src.utils.accuracy import AverageMeter, accuracy
from src.utils.mapping import probabilities_to_decision
from src.utils.dictionary import get_key_from_value
from src.dataset.imagenet16 import label_map


def test_performance(
    model, test_loader, out_path, max_sigma=10, device=torch.device("cuda:0")
):
    """
    compute performance of the model
    and return the results as lists
    """
    acc1_list = []
    acc5_list = []
    for i in tqdm(range(max_sigma + 1), desc="lowpass filters", leave=False):
        acc1, acc5 = calc_lowpass_acc(
            model,
            test_loader=test_loader,
            sigma=i,
            device=device,
        )
        acc1_list.append(acc1.item())
        acc5_list.append(acc5.item())

    # range of sigma
    s = [i for i in range(max_sigma + 1)]

    # save acc1
    df = pd.DataFrame(np.array(acc1_list).reshape(1, -1), columns=s)
    df.to_csv(out_path)

    # save acc5
    df = pd.DataFrame(np.array(acc5_list).reshape(1, -1), columns=s)
    out_path = out_path.replace("acc1", "acc5")
    df.to_csv(out_path)


def calc_lowpass_acc(model, test_loader, sigma, device=torch.device("cuda:0")):
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="test images", leave=False):
            inputs, labels = data[0], data[1].to(device)
            if sigma != 0:
                inputs = GaussianBlurAll(imgs=inputs, sigma=sigma)

            inputs = inputs.to(device)

            outputs = model(inputs)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

    return top1.avg, top5.avg


def compute_confusion_matrix(
    model, test_loader, sigma=1, device=torch.device("cuda:0")
):
    """Computes model decisions and confusion matrix.
    Returns:
        conf_matrix: ndarray of shape (16, 16)
        pred: predictions (acc1). ndarray of shape (N)
        targets: correct labels. ndarray of shape (N)
    """
    model.eval()
    pred = []
    targets = []

    # create mapping
    mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()

    with torch.no_grad():
        for data in tqdm(test_loader, desc="test images", leave=False):
            inputs, labels = data[0], data[1]

            # add target ids
            targets += [labels.numpy()]

            if sigma != 0:
                inputs = GaussianBlurAll(inputs, sigma)
            inputs = inputs.to(device)

            outputs = model(inputs)  # torch.Size([batch_size, num_labels])

            if model.num_classes == 1000 and test_loader.num_classes == 16:
                outputs = torch.nn.Softmax(dim=1)(outputs)  # softmax
                # get model_decision (str) by mapping outputs from 1,000 to 16
                correct = 0
                for i in range(outputs.shape[0]):
                    model_decision = mapping.probabilities_to_decision(
                        outputs[i].detach().cpu().numpy()  # 一個ずつじゃ無いとダメ(多分)
                    )

                    model_decision_id = get_key_from_value(label_map, model_decision)

                    pred += [model_decision_id]
            elif model.num_classes == 16:
                pred += [outputs.topk(1)[1].view(-1).cpu().numpy()]

    targets = np.array(targets).reshape(-1)
    pred = np.array(pred).reshape(-1)

    conf_matrix = confusion_matrix(targets, pred)

    return conf_matrix


def plot_confusion_matrix(
    confusion_matrix, vmin=0, vmax=1, title="", out_path="", show=False
):
    sns.heatmap(confusion_matrix, vmin=vmin, vmax=vmax, cmap="Reds")

    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")

    if title:
        plt.title(title)
    if out_path:
        plt.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close()
