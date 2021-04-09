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

from src.image_process.bandpass_filter import (
    apply_bandpass_filter,
)
from src.utils.accuracy import AverageMeter, accuracy
from src.utils.mapping import probabilities_to_decision


def test_performance(model, test_loader, bandpass_filters, device, out_file):
    """
    compute performance of the model
    and return the results as lists
    """
    acc1_list = []

    # acc. of raw test images
    acc1 = compute_bandpass_acc(
        model=model, test_loader=test_loader, device=device, raw=True
    )
    acc1_list.append(acc1.item())

    # acc. of bandpass test images
    for s1, s2 in tqdm(bandpass_filters.values(), desc="bandpass filters", leave=False):
        acc1 = compute_bandpass_acc(
            model=model, test_loader=test_loader, sigma1=s1, sigma2=s2, device=device
        )
        acc1_list.append(acc1.item())

    # range of sigma
    bandpass_sigmas = ["0"] + [f"{s1}-{s2}" for s1, s2 in bandpass_filters.values()]

    # make dataframe and save
    df = pd.DataFrame(
        np.array(acc1_list).reshape(1, -1), index=[model_name], columns=bandpass_sigmas
    )
    df.to_csv(out_file)


# create mapping
mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()


def compute_bandpass_acc(
    model,
    test_loader: iter,
    sigma1: int = 0,
    sigma2: int = 1,
    raw: bool = False,
    device=torch.device("cuda:0"),
):
    """
    Args:
        model: model to test
        sigma1, sigma2: bandpass images are made by subtracting
            GaussianBlur(sigma1) - GaussianBlur(sigma2)
        raw: if True, calculate accuracy of raw images
    return: accuracy of bandpass images
        :: when raw == True, return accuracy of raw images
    """
    top1 = AverageMeter("Acc@1", ":6.2f")
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="test images", leave=False):
            inputs, labels = data[0], data[1].to(device)
            if not raw:
                inputs = apply_bandpass_filter(
                    images=inputs, sigma1=sigma1, sigma2=sigma2
                )
            inputs = inputs.to(device)
            outputs = model(inputs)
            if model.num_classes == 1000 and test_loader.num_classes == 16:
                outputs = torch.nn.Softmax(dim=1)(outputs)  # softmax
                # get model_decision (str) by mappig outputs from 1,000 to 16
                # model_decision = mapping.probabilities_to_decision(
                #     outputs[i].detach().cpu().numpy()  # 一個ずつじゃ無いとダメ？
                # )
                pass
                # map outputs from 1000 to 16
            acc1 = accuracy(outputs, labels, topk=(1,))
            top1.update(acc1[0], inputs.size(0))

    return top1.avg
