import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

from src.image_process.bandpass_filter import (
    make_bandpass_filters,
    apply_bandpass_filter,
)
from src.utils.accuracy import AverageMeter, accuracy
from src.model.utils import load_model
from src.dataset.imagenet16 import load_imagenet16
from src.dataset.imagenet import load_imagenet
from src.model.mapping import probabilities_to_decision


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
    for s1, s2 in bandpass_filters.values():
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
        for data in test_loader:
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
                model_decision = mapping.probabilities_to_decision(
                    outputs[i].detach().cpu().numpy()  # 一個ずつじゃ無いとダメ？
                )
                pass
                # map outputs from 1000 to 16
            acc1 = accuracy(outputs, labels, topk=(1,))
            top1.update(acc1[0], inputs.size(0))

    return top1.avg


if __name__ == "__main__":
    # args
    arch = "alexnet"
    num_classes = 16  # number of last output of the models
    epoch = 60
    batch_size = 64
    imagenet_path = "/Users/sou/lab1-mnt/data1/ImageNet/ILSVRC2012/"
    dataset = "imagenet16"  # dataset to use
    num_filters = 6

    models_dir = "/Users/sou/lab1-mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
        16 if num_classes == 16 else ""  # else is (num_classes == 1000)
    )
    results_dir = f"/Users/sou/work/blur-training/analysis/bandpass_acc/results/{num_classes}-class/{arch}/"
    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    cudnn.benchmark = True  # for fast running

    # random seed settings
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    # loading data
    if dataset == "imagenet16":
        _, test_loader = load_imagenet16(
            imagenet_path=imagenet_path, batch_size=batch_size
        )
    elif dataset == "imagenet":
        _, _, test_loader = load_imagenet(
            imagenet_path=imagenet_path,
            batch_size=batch_size,
            distributed=False,
            workers=4,
        )

    # make bandpass bandpass_filters
    bandpass_filters = make_bandpass_filters(num_filters=num_filters)

    # models to compare
    modes = [
        "normal",
        "all",
        # "mix",
        # "random-mix",
        # "single-step",
        # "fixed-single-step",
        # "reversed-single-step",
        # "multi-steps",
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
        # load model
        model_path = os.path.join(
            models_dir, model_name, "epoch_{}.pth.tar".format(epoch)
        )
        model = load_model(
            arch=arch, num_classes=num_classes, model_path=model_path, device="cpu"
        ).to(device)
        print(model)
        if model.num_classes == 1000 and test_loader.num_classes == 16:
            print("need 1000 -> 16")
        hoge

        # set path to output
        out_file = os.path.join(
            results_dir, f"{num_classes}-class_{model_name}_e{epoch}_acc1.csv"
        )

        test_performance(
            model=model,
            test_loader=test_loader,
            bandpass_filters=bandpass_filters,
            device=device,
            out_file=out_file,
        )
