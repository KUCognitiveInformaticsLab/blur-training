import os

import torchvision.datasets as datasets


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
