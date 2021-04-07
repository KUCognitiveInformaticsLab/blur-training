#!/usr/bin/env python
# coding: utf-8

import os
import sys

sys.path.append("../../")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from robustness import datasets
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy

from training.utils import GaussianBlurAll, AverageMeter, accuracy

arch = sys.argv[1]
RESULTS_DIR = "./results/{}/".format(arch)
MODELS_DIR = "/mnt/work/blur-training/imagenet16/logs/models/"
epoch = 60
max_sigma = 40  # max sigma of guassian kernel to test the models with

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

cudnn.benchmark = True


def load_data(
    batch_size, in_path="/mnt/data/ImageNet/ILSVRC2012/", in_info_path="../info/"
):
    """
    load 16-class-ImageNet
    :param batch_size: the batch size used in training and test
    :param in_path: the path to ImageNet
    :param in_info_path: the path to the directory
                              that contains imagenet_class_index.json, wordnet.is_a.txt, words.txt
    :return: train_loader, test_loader
    """

    in_hier = ImageNetHierarchy(in_path, in_info_path)
    superclass_wnid = common_superclass_wnid("geirhos_16")  # 16-class-imagenet
    class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

    custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
    # data augumentation for imagenet in robustness library is:
    # https://github.com/MadryLab/robustness/blob/master/robustness/data_augmentation.py

    ### parameters for normalization: choose one of them if you want to use normalization #############
    # normalize = transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])  # norm
    # https://github.com/MadryLab/robustness/blob/master/robustness/datasets.py

    # If you want to use normalization parameters of ImageNet from pyrotch:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )  # norm-in

    # 16-class-imagenet
    # normalize = transforms.Normalize(mean=[0.4677, 0.4377, 0.3986], std=[0.2769, 0.2724, 0.2821])  # norm16
    # normalize = transforms.Normalize(mean=[0.4759, 0.4459, 0.4066], std=[0.2768, 0.2723, 0.2827])  # norm16-2
    ############################################################################
    # add normalization
    custom_dataset.transform_train.transforms.append(normalize)
    custom_dataset.transform_test.transforms.append(normalize)

    # train_loader, test_loader = custom_dataset.make_loaders(workers=10,
    #                                                         batch_size=batch_size)
    train_loader, test_loader = custom_dataset.make_loaders(
        workers=10, batch_size=batch_size, only_val=True
    )

    return train_loader, test_loader


def load_model(model_path, arch, num_classes=16):
    """
    :param model_path: path to the pytorch saved file of the model you want to use
    """
    model = models.__dict__[arch]()
    # change the number of last layer's units
    model = models.__dict__[arch]()
    if (
        arch.startswith("alexnet")
        or arch.startswith("vgg")
        or arch.startswith("mnasnet")
        or arch.startswith("mobilenet")
    ):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
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

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])

    return model


def calc_acc(model, sigma):
    global test_loader
    global device

    top1 = AverageMeter("Acc@1", ":6.2f")
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0], data[1].to(device)
            if sigma != 0:
                inputs = GaussianBlurAll(inputs, sigma)
            inputs = inputs.to(device)
            outputs = model(inputs)
            acc1 = accuracy(outputs, labels, topk=(1,))
            top1.update(acc1[0], inputs.size(0))

    return top1.avg


def test_performance(model_name, arch, epoch=60, max_sigma=10):
    """
    compute performance of the model
    and return the results as lists
    """
    # set paths
    model_path = os.path.join(MODELS_DIR, model_name, "epoch_{}.pth.tar".format(epoch))
    save_path = os.path.join(
        RESULTS_DIR, "{}_e{}_acc1.csv".format(model_name, epoch)
    )  # save path of the results

    # load model
    model = load_model(model_path, arch).to(device)

    acc1_list = []
    for i in range(max_sigma + 1):
        acc1 = calc_acc(model, i)
        acc1_list.append(acc1.item())

    # range of sigma
    s = [i for i in range(max_sigma + 1)]

    # make dataframe and save
    df = pd.DataFrame(np.array(acc1_list).reshape(1, -1), index=[model_name], columns=s)
    df.to_csv(save_path)


# random seed settings
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# data settings
batch_size = 64

_, test_loader = load_data(
    batch_size, in_path="/mnt/data/ImageNet/ILSVRC2012/", in_info_path="../../info/"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# models to compare
modes = [
    "normal",
    "all",
    "mix",
    "random-mix",
    "single-step",
    "fixed-single-step",
    "reversed-single-step",
    "multi-steps",
]

# sigmas to compare
sigmas_mix = [s for s in range(1, 6)] + [10]
sigmas_random_mix = ["00-05", "00-10"]

# make model name list
model_names = []
for mode in modes:
    if mode in ("normal", "multi-steps"):
        model_names += [f"{arch}_{mode}"]
    elif mode == "random-mix":
        for min_max in sigmas_random_mix:
            model_names += [f"{arch}_{mode}_s{min_max}"]
    elif mode == "mix":
        for sigma in sigmas_mix:
            model_names += [f"{arch}_{mode}_s{sigma:02d}"]
    else:
        for s in range(4):
            model_names += [f"{arch}_{mode}_s{s + 1:02d}"]

for model_name in model_names:
    test_performance(model_name, arch, epoch, max_sigma)
