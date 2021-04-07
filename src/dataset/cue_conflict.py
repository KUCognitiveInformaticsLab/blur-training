import torch
import torchvision.transforms as transforms

from .folder import ImageFolderWithFileName


def load_cue_conflict(data_path="/mnt/data/shape-texture-cue-conflict/", batch_size=64):
    """
    Args:
        data_path: path to the directory that contains cue conflict images
        bath_size: the size of each batch set
    """
    # normalization of cue-conflict images:
    # normalize = transforms.Normalize(mean=[0.5374, 0.4923, 0.4556], std=[0.2260, 0.2207, 0.2231])
    # standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # data augumentation for imagenet in robustness library is:
    # https://github.com/MadryLab/robustness/blob/master/robustness/data_augmentation.py
    dataset = ImageFolderWithFileName(
        data_path,
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True
    )

    return dataloader
