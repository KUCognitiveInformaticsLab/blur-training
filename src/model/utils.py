import os
import shutil

import torch
import torch.nn as nn
import torchvision.models as models


def load_model(
    arch: str,
    num_classes: int = 16,
    paralell: bool = False,
    model_path: str = "",
    device: str = "cuda:0",
):
    """
    Load model from pytorch model zoo and change the number of final layser's units
    Args:
        arch (str): name of architecture.
        num_classes (int): number of last layer's units.
        paralell (bool): use a parallel model or not.
        model_path (str): path to trained model's weights.
        device (str): device for map_location for loading weights. (e.g. "cuda:0")
    Returns: model (torch.model)
    """
    model = models.__dict__[arch]()
    model.num_classes = num_classes
    if num_classes == 1000:
        checkpoint = torch.load(model_path, map_location=device)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError:
            model.features = torch.nn.DataParallel(model.features)
            model.load_state_dict(checkpoint["state_dict"])
            if not paralell:
                model.features = model.features.module
                # TODO: This part is different when a model is "resnet".
        return model
    else:  # num_classes == 16
        if (
            arch.startswith("alexnet")
            or arch.startswith("vgg")
            or arch.startswith("mnasnet")
            or arch.startswith("mobilenet")
        ):
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, num_classes
            )
        elif (
            arch.startswith("resne")
            or arch.startswith("shufflenet")
            or arch.startswith("inception")
            or arch.startswith("wide_resnet")
        ):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif arch.startswith("densenet"):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif arch.startswith("squeezenet"):
            model.classifier[1] = nn.Conv2d(
                model.classifier[1].in_channels,
                num_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
            )

        # load trained weights.
        if model_path:
            if device:
                checkpoint = torch.load(model_path, map_location=device)
            else:
                checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["state_dict"])

        return model


def save_model(state, param_path, epoch=None):
    if not epoch:
        filename = os.path.join(param_path, "checkpoint.pth.tar")
    else:
        filename = os.path.join(param_path, "epoch_{}.pth.tar".format(epoch))
    torch.save(state, filename)


def save_checkpoint(state, is_best, param_path, epoch):
    filename = param_path + "checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, param_path + "model_best.pth.tar")
