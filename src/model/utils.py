import os
import shutil

import torch
import torch.nn as nn
import torchvision.models as models

import vonenet
from src.model.load_sin_pretrained_models import load_sin_model
from src.model.resnet_wider import resnet50x1, resnet50x2, resnet50x4


def load_model(
    arch: str = "",
    model_name: str = "",
    num_classes: int = 16,
    parallel: bool = False,
    model_path: str = "",
    device: str = "cuda:0",
):
    """
    Load model from pytorch model zoo and change the number of final layser's units
    Args:
        arch (str): name of architecture.
        num_classes (int): number of last layer's units.
        parallel (bool): use a parallel model or not.
        model_path (str): path to trained model's weights.
        device (str): device for map_location for loading weights. (e.g. "cuda:0")
    Returns: model (torch.model)
    """
    # pretrained models
    if "SIN" in model_name:
        # Stylized-ImageNet
        # model = load_sin_model(model_name).to(device)
        # === For the bug that the pretrained-model's url doesn't work.
        import torchvision
        print("Using the AlexNet architecture.")
        model = torchvision.models.alexnet(pretrained=False)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        try:
            model_path = "/mnt/data1/pretrained_models/sin/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar"
            checkpoint = torch.load(model_path, map_location=device)
        except:
            model_path = "/mnt/data/pretrained_models/sin/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar"
            checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        # ===
        if not parallel:
            model.features = model.features.module

        return model
    elif model_name == "vone_alexnet":  # pretrained vonenet
        model = vonenet.get_model(model_arch=arch, pretrained=True).to(device)

        return model
    elif "simclr" in model_name:  # SimCLR
        if "resnet50-1x" in model_name:
            model = resnet50x1()
        elif "resnet50-2x" in model_name:
            model = resnet50x2()
        elif "resnet50-4x" in model_name:
            model = resnet50x4()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

        return model

    # load model arch
    if arch == "vone_alexnet":
        model = vonenet.get_model(model_arch=arch.split("_")[1], pretrained=False)
        if num_classes == 16:
            model.model.classifier[-1] = nn.Linear(
                model.model.classifier[-1].in_features, num_classes
            )
    else:
        model = models.__dict__[arch]()
        if num_classes == 16:
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
        if num_classes == 1000:
            checkpoint = torch.load(model_path, map_location=device)
            try:
                model.load_state_dict(checkpoint["state_dict"])
            except RuntimeError:  # Apply parallel for loading weights
                if arch == "vone_alexnet":
                    model = torch.nn.DataParallel(model)
                else:
                    model.features = torch.nn.DataParallel(model.features)
                model.load_state_dict(checkpoint["state_dict"])

                # Disable parallel
                if not parallel:
                    if arch == "vone_alexnet":
                        model = model.module
                    else:
                        model.features = model.features.module
                        # TODO: This part is different when a model is "resnet".

            return model
        elif num_classes == 16:
            if device:
                checkpoint = torch.load(model_path, map_location=device)
            else:
                checkpoint = torch.load(model_path)

            model.load_state_dict(checkpoint["state_dict"])

            return model

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
