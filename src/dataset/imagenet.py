import os
from typing import Tuple, Any

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_imagenet(
    imagenet_path: str = "/mnt/data1/ImageNet/ILSVRC2012/",
    batch_size: int = 256,
    distributed: bool = False,
    workers: int = 4,
) -> Tuple[Any, Any, Any]:
    traindir = os.path.join(imagenet_path, "train")
    valdir = os.path.join(imagenet_path, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    train_loader.num_classes = 1000
    val_loader.num_classes = 1000

    return train_loader, train_sampler, val_loader
