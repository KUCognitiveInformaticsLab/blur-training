import argparse
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

# add a path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../")

from src.image_process.lowpass_filter import GaussianBlurAll, RandomGaussianBlurAll
from src.dataset.imagenet16 import load_imagenet16
from src.utils.model import load_model, save_model
from src.utils.adjust import (
    adjust_learning_rate,
    adjust_multi_steps,
    adjust_multi_steps_cbt,
)
from src.utils.accuracy import AverageMeter, accuracy
from src.utils.print import print_settings


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
)  # Remember no-cuda becomes no_cuda in the code. ("-" becomes "_")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="alexnet",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: alexnet)",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=[
        "normal",
        "all",
        "mix",
        "random-mix",
        "single-step",
        "reversed-single-step",
        "fixed-single-step",
        "multi-steps",
        "multi-steps-cbt",
    ],
    help="Training mode.",
)
parser.add_argument("--exp_name", "-n", type=str, default="", help="Experiment name.")
parser.add_argument(
    "--log_dir",
    type=str,
    default=str(current_dir) + "/../train-logs/imagenet16",
    help="Path to log directory to store trained models, tensorboard, stdout, and stderr.",
)
parser.add_argument(
    "--sigma",
    "-s",
    type=float,
    default=1,
    help="Sigma of Gaussian Kernel (Gaussian Blur).",
)
parser.add_argument(
    "--min_sigma",
    type=float,
    default=0,
    help="Minimum sigma of Gaussian Kernel (Gaussian Blur) for random-mix training.",
)
parser.add_argument(
    "--max_sigma",
    type=float,
    default=5,
    help="Maximum sigma of Gaussian Kernel (Gaussian Blur) for random-mix training.",
)
# parser.add_argument('--init-sigma', type=float, default=2,
#                    help='Initial Sigma of Gaussian Blur. (multi-steps-cbt)')
parser.add_argument(
    "--cbt_rate", type=float, default=0.9, help="Blur decay rate (multi-steps-cbt)"
)
parser.add_argument(
    "--kernel_size",
    "-k",
    type=int,
    nargs=2,
    default=(0, 0),
    help="Kernel size of Gaussian Blur.",
)
parser.add_argument(
    "--epochs", "-e", type=int, default=60, help="Number of epochs to train."
)
parser.add_argument(
    "--blur_val", action="store_true", default=False, help="Blur validation data."
)
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size.")
parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    "-w",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)


def main():
    args = parser.parse_args()
    if args.exp_name == "":
        print(
            "ERROR: USE '--exp_name' or '-n' option to define this experiment's name."
        )
        sys.exit()

    # directories settings
    os.makedirs(os.path.join(args.log_dir, "outputs"), exist_ok=True)
    os.makedirs(
        os.path.join(args.log_dir, "models/{}".format(args.exp_name)), exist_ok=True
    )

    output_path = os.path.join(args.log_dir, "outputs/{}.log".format(args.exp_name))
    model_path = os.path.join(args.log_dir, "models/{}/".format(args.exp_name))

    # check if "exp_name" is already in use or not (except --resume mode)
    if not args.resume and os.path.exists(output_path):
        print(
            "ERROR: This '--exp_name' is already used. \
                Use another name for this experiment."
        )
        sys.exit()

    # recording outputs
    sys.stdout = open(output_path, "a")
    sys.stderr = open(output_path, "a")

    # tensorboardX
    writer = SummaryWriter(
        log_dir=os.path.join(args.log_dir, "tb/{}".format(args.exp_name))
    )

    # cuda settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print("device: {}".format(device))

    # for fast training
    cudnn.benchmark = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # data settings
    trainloader, testloader = load_imagenet16(batch_size=args.batch_size)

    # Model, Criterion, Optimizer
    model = load_model(args.arch)  # remember the number of final outputs is 16.
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # print settings
    print_settings(model, args)

    # training
    sigma_blur = args.sigma  # save sigma of blur (for reversed-single-step)
    print("Start Training...")
    train_time = time.time()
    for epoch in range(
        args.start_epoch, args.epochs
    ):  # loop over the dataset multiple times
        # blur settings
        if args.mode == "normal":
            args.sigma = 0
        elif args.mode == "multi-steps-cbt":
            # args.sigma = adjust_sigma(epoch, args)  # sigma decay every 5 epoch
            args.sigma = adjust_multi_steps_cbt(args.sigma, epoch, args.cbt_rate)
        elif args.mode == "multi-steps":
            args.sigma = adjust_multi_steps(epoch)
        elif args.mode == "single-step":
            if epoch >= args.epochs // 2:
                args.sigma = 0  # no blur
        elif args.mode == "fixed-single-step":
            if epoch >= args.epochs // 2:
                args.sigma = 0  # no blur
                # fix parameters of 1st Conv layer
                model.features[0].weight.requires_grad = False
                model.features[0].bias.requires_grad = False
        elif args.mode == "reversed-single-step":
            if epoch < args.epochs // 2:
                args.sigma = 0
            else:
                args.sigma = sigma_blur

        adjust_learning_rate(optimizer, epoch, args)  # decay by 10 every 20 epoch

        # ===== train mode =====
        train_acc = AverageMeter("train_acc", ":6.2f")
        train_loss = AverageMeter("train_loss", ":.4e")
        model.train()
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0], data[1].to(device)

            # Blur images
            if args.mode == "mix":
                half1, half2 = inputs.chunk(2)
                # blur first half images
                half1 = GaussianBlurAll(half1, args.sigma)
                inputs = torch.cat((half1, half2))
            elif args.mode == "random-mix":
                half1, half2 = inputs.chunk(2)
                # blur first half images
                half1 = RandomGaussianBlurAll(half1, args.min_sigma, args.max_sigma)
                inputs = torch.cat((half1, half2))
            else:
                inputs = GaussianBlurAll(inputs, args.sigma)
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + record
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc1 = accuracy(outputs, labels, topk=(1,))
            train_loss.update(loss.item(), inputs.size(0))
            train_acc.update(acc1[0], inputs.size(0))

            # backward + optimize
            loss.backward()
            optimizer.step()

        # record the values in tensorboard
        writer.add_scalar("loss/train", train_loss.avg, epoch + 1)  # average loss
        writer.add_scalar("acc/train", train_acc.avg, epoch + 1)  # average acc

        # ===== val mode =====
        val_acc = AverageMeter("val_acc", ":6.2f")
        val_loss = AverageMeter("val_loss", ":.4e")
        model.eval()
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0], data[1].to(device)
                if args.blur_val:
                    inputs = GaussianBlurAll(inputs, args.sigma)
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc1 = accuracy(outputs, labels, topk=(1,))
                val_loss.update(loss.item(), inputs.size(0))
                val_acc.update(acc1[0], inputs.size(0))

        # record the values in tensorboard
        writer.add_scalar("loss/val", val_loss.avg, epoch + 1)  # average loss
        writer.add_scalar("acc/val", val_acc.avg, epoch + 1)  # average acc

        # ===== save the model =====
        # checkpoint
        save_model(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "val_loss": val_loss.avg,
                "val_acc": val_acc.avg,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            model_path,
        )
        # save every 10 epoch
        if (epoch + 1) % 10 == 0:
            save_model(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "val_loss": val_loss.avg,
                    "val_acc": val_acc.avg,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                model_path,
                epoch + 1,
            )

    print("Finished Training")
    print("Training time elapsed: {:.4f}mins".format((time.time() - train_time) / 60))
    print()

    writer.close()  # close tensorboardX writer


if __name__ == "__main__":
    run_time = time.time()
    main()
    mins = (time.time() - run_time) / 60
    hours = mins / 60
    print("Total run time: {:.4f}mins, {:.4f}hours".format(mins, hours))
