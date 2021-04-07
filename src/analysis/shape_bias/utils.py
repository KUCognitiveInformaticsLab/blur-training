import os

import torch
import torch.nn as nn
from torchvision import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# create label_map dictionary
labels =sorted(["knife", "keyboard", "elephant", "bicycle", "airplane",
            "clock", "oven", "chair", "bear", "boat", "cat",
            "bottle", "truck", "car", "bird", "dog"])
label_map =  {k:v for k, v in enumerate(labels)}


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None


def load_model(model_path, arch='alexnet'):
    """Load model. 
    Args:
        model_path: Path to the pytorch saved file of the model you want to use
        arch: Architecture of CNN
    Returns: CNN model 
    """
    checkpoint = torch.load(model_path, map_location='cuda:0')
    model = models.__dict__[arch]()
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError:
        model.features = torch.nn.DataParallel(model.features)
        model.load_state_dict(checkpoint['state_dict'])
    return model


def make_dataloader(data_path='/mnt/data/shape-texture-cue-conflict/' , 
                            batch_size=64):
    """
    Args: 
        data_path: path to the directory that contains cue conflict images
        bath_size: the size of each batch set
    """
    # normalization of cue-conflict images:
    #normalize = transforms.Normalize(mean=[0.5374, 0.4923, 0.4556], std=[0.2260, 0.2207, 0.2231])
    # standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # data augumentation for imagenet in robustness library is:
    # https://github.com/MadryLab/robustness/blob/master/robustness/data_augmentation.py
    dataset = ImageFolderWithFileName(
                data_path,
                transforms.Compose([
                transforms.ToTensor(),
                normalize,  
            ]))

    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=5, pin_memory=True)
    
    return dataloader


class ImageFolderWithFileName(datasets.ImageFolder):
    """Custom dataset so that it returns the image file name as well.
    Extends torchvision.datasets.ImageFolder.
    ref: https://discuss.pytorch.org/t/dataloader-filenames-in-each-batch/4212/4
    source code:
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L128
    """
    # override the __getitem__ method that dataloader calls
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        # get filename from path
        filename = os.path.basename(path)

        return sample, target, filename  # add filename

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    