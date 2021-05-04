#!/usr/bin/env python
# coding: utf-8

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

from src.image_process.lowpass_filter import GaussianBlurAll
from src.utils.accuracy import AverageMeter, accuracy


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
            model, test_loader=test_loader, sigma=i, device=torch.device("cuda:0")
        )
        acc1_list.append(acc1.item())
        acc5_list.append(acc5.item())

    # range of sigma
    s = [i for i in range(max_sigma + 1)]

    # save acc1
    df = pd.DataFrame(np.array(acc1_list).reshape(1, -1), index=[model_name], columns=s)
    df.to_csv(out_path)

    # save acc5
    df = pd.DataFrame(np.array(acc5_list).reshape(1, -1), index=[model_name], columns=s)
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
                inputs = GaussianBlurAll(inputs, sigma)
            inputs = inputs.to(device)
            outputs = model(inputs)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

    return top1.avg, top5.avg
