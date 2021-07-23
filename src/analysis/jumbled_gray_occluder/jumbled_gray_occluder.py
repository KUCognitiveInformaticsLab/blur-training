import os
import pathlib
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

from src.utils.mapping import probabilities_to_decision
from src.utils.dictionary import get_key_from_value
from src.dataset.imagenet16 import label_map
from src.image_process.jumble import jumble_images, jumble_images_with_glay_occluder
from src.image_process.gray_occluder import gray_occlude_images
from src.utils.accuracy import AverageMeter, accuracy


def compute_confusion_matrix(
    model, test_loader, stimuli, div_v, div_h, device=torch.device("cuda:0")
):
    """Computes model decisions and confusion matrix.
    Args:
        stimuli: ("jumbled", "gray_occluder", "jumbled_with_gray_occluder")
        div_v: # of vertical splits
        div_h: # of horizontal splits

    Returns:
        conf_matrix: ndarray of shape (16, 16)
        acc: accuracy
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

            if stimuli == "jumbled":
                inputs = jumble_images(imgs=inputs, div_v=div_v, div_h=div_h).to(device)
            elif stimuli == "gray_occluder":
                inputs = gray_occlude_images(imgs=inputs, div_v=div_v, div_h=div_h).to(
                    device
                )
            elif stimuli == "jumbled_with_gray_occluder":
                inputs = jumble_images_with_glay_occluder(
                    imgs=inputs, div_v=div_v, div_h=div_h
                ).to(device)

            outputs = model(inputs)  # torch.Size([batch_size, num_labels])

            if model.num_classes == 1000 and test_loader.num_classes == 16:
                outputs = torch.nn.Softmax(dim=1)(outputs)  # softmax
                # get model_decision (str) by mapping outputs from 1,000 to 16
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

    # compute acc
    acc = np.diag(conf_matrix).sum() / conf_matrix.sum()

    return conf_matrix, acc


def compute_acc(
    model, test_loader, stimuli, div_v, div_h, device=torch.device("cuda:0")
):
    """Computes model decisions and confusion matrix.
    Args:
        stimuli: ("jumbled", "gray_occluder", "jumbled_with_gray_occluder")
        div_v: # of vertical splits
        div_h: # of horizontal splits

    Returns:
        conf_matrix: ndarray of shape (16, 16)
        acc: accuracy
    """
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    model.eval()

    with torch.no_grad():
        for data in tqdm(test_loader, desc="test images", leave=False):
            inputs, labels = data[0], data[1]

            if stimuli == "jumbled":
                inputs = jumble_images(imgs=inputs, div_v=div_v, div_h=div_h).to(device)
            elif stimuli == "gray_occluder":
                inputs = gray_occlude_images(imgs=inputs, div_v=div_v, div_h=div_h).to(
                    device
                )
            elif stimuli == "jumbled_with_gray_occluder":
                inputs = jumble_images_with_glay_occluder(
                    imgs=inputs, div_v=div_v, div_h=div_h
                ).to(device)

            outputs = model(inputs)  # torch.Size([batch_size, num_labels])

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            top1.update(acc1[0].item(), inputs.size(0))
            top5.update(acc5[0].item(), inputs.size(0))

    return top1.avg, top5.avg
