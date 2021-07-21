import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

from src.dataset.imagenet16 import label_map
from src.image_process.bandpass_filter import (
    apply_bandpass_filter,
)
from src.utils.accuracy import AverageMeter, accuracy
from src.utils.mapping import probabilities_to_decision
from src.utils.dictionary import get_key_from_value


def test_performance(model, test_loader, bandpass_filters, device, out_file):
    """
    compute performance of the model
    and return the results as lists
    """
    acc1_list = []
    acc5_list = []

    # acc. of raw test images
    acc1, acc5 = compute_bandpass_acc(
        model=model, test_loader=test_loader, device=device, raw=True
    )
    acc1_list.append(acc1)
    acc5_list.append(acc5)

    # acc. of bandpass test images
    for s1, s2 in tqdm(bandpass_filters.values(), desc="bandpass filters", leave=False):
        acc1 = compute_bandpass_acc(
            model=model, test_loader=test_loader, sigma1=s1, sigma2=s2, device=device
        )
        acc1_list.append(acc1)

    # range of sigma
    bandpass_sigmas = ["0"] + [f"{s1}-{s2}" for s1, s2 in bandpass_filters.values()]

    # make dataframe and save
    df = pd.DataFrame(np.array(acc1_list).reshape(1, -1), columns=bandpass_sigmas)
    df.to_csv(out_file)
    df = pd.DataFrame(np.array(acc5_list).reshape(1, -1), columns=bandpass_sigmas)
    out_file = out_file.replace("acc1", "acc5")
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
    top5 = AverageMeter("Acc@5", ":6.2f")

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="test images", leave=False):
            inputs, targets = data[0], data[1].to(device)
            if not raw:
                inputs = apply_bandpass_filter(
                    images=inputs, sigma1=sigma1, sigma2=sigma2
                )
            inputs = inputs.to(device)
            outputs = model(inputs)  # torch.Size([batch_size, num_labels])
            if model.num_classes == 1000 and test_loader.num_classes == 16:
                outputs = torch.nn.Softmax(dim=1)(outputs)  # softmax

                correct = 0
                for i in range(outputs.shape[0]):
                    # get model_decision (str) by mapping outputs from 1,000 to 16
                    model_decision = mapping.probabilities_to_decision(
                        outputs[i]
                        .detach()
                        .cpu()
                        .numpy()  # It needs to be a single output.
                    )  # Returns: label name (str)

                    # label name (str) -> label id (int)
                    model_decision_id = get_key_from_value(label_map, model_decision)
                    correct += float(model_decision_id == targets[i])

                acc1 = (correct / outputs.shape[0]) * 100
                top1.update(acc1, outputs.shape[0])

            else:
                acc1, acc5 = accuracy(outputs, targets, topk=(1,5))
                top1.update(acc1[0].item(), outputs.shape[0])
                top5.update(acc5[0].item(), outputs.shape[0])

    return top1.avg, top5.avg

def test_performance_bak(model, test_loader, bandpass_filters, device, out_file):
    """
    compute performance of the model
    and return the results as lists
    """
    acc1_list = []

    # acc. of raw test images
    acc1 = compute_bandpass_acc(
        model=model, test_loader=test_loader, device=device, raw=True
    )
    acc1_list.append(acc1)

    # acc. of bandpass test images
    for s1, s2 in tqdm(bandpass_filters.values(), desc="bandpass filters", leave=False):
        acc1 = compute_bandpass_acc(
            model=model, test_loader=test_loader, sigma1=s1, sigma2=s2, device=device
        )
        acc1_list.append(acc1)

    # range of sigma
    bandpass_sigmas = ["0"] + [f"{s1}-{s2}" for s1, s2 in bandpass_filters.values()]

    # make dataframe and save
    df = pd.DataFrame(np.array(acc1_list).reshape(1, -1), columns=bandpass_sigmas)
    df.to_csv(out_file)


# create mapping
mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()


def compute_bandpass_acc_bak(
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
            inputs, targets = data[0], data[1].to(device)
            if not raw:
                inputs = apply_bandpass_filter(
                    images=inputs, sigma1=sigma1, sigma2=sigma2
                )
            inputs = inputs.to(device)
            outputs = model(inputs)  # torch.Size([batch_size, num_labels])
            if model.num_classes == 1000 and test_loader.num_classes == 16:
                outputs = torch.nn.Softmax(dim=1)(outputs)  # softmax

                correct = 0
                for i in range(outputs.shape[0]):
                    # get model_decision (str) by mapping outputs from 1,000 to 16
                    model_decision = mapping.probabilities_to_decision(
                        outputs[i]
                        .detach()
                        .cpu()
                        .numpy()  # It needs to be a single output.
                    )  # Returns: label name (str)

                    # label name (str) -> label id (int)
                    model_decision_id = get_key_from_value(label_map, model_decision)
                    correct += float(model_decision_id == targets[i])

                acc1 = (correct / outputs.shape[0]) * 100
                top1.update(acc1, outputs.shape[0])

            else:
                acc1 = accuracy(outputs, targets, topk=(1,))
                top1.update(acc1[0].item(), outputs.shape[0])

    return top1.avg

def compute_confusion_matrix(
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
    model.eval()
    pred = []
    targets = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="test images", leave=False):
            inputs, labels = data[0], data[1]

            # add target ids
            targets += [labels.numpy()]

            if not raw:
                inputs = apply_bandpass_filter(
                    images=inputs, sigma1=sigma1, sigma2=sigma2
                )
            inputs = inputs.to(device)

            outputs = model(inputs)  # torch.Size([batch_size, num_labels])

            if model.num_classes == 1000 and test_loader.num_classes == 16:
                outputs = torch.nn.Softmax(dim=1)(outputs)  # softmax
                # get model_decision (str) by mapping outputs from 1,000 to 16
                correct = 0
                for i in range(outputs.shape[0]):
                    model_decision = mapping.probabilities_to_decision(
                        outputs[i].detach().cpu().numpy()  # 一個ずつじゃ無いとダメ？
                    )
                    model_decision_id = get_key_from_value(label_map, model_decision)
                    pred += [model_decision_id]
            else:
                pred += [outputs.topk(1)[1].view(-1).cpu().numpy()]

    targets = np.array(targets).reshape(-1)
    pred = np.array(pred).reshape(-1)

    conf_matrix = confusion_matrix(targets, pred)

    return conf_matrix
