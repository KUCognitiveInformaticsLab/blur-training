import random

import cv2
import numpy as np
import torch


def GaussianBlurAll(imgs, sigma, kernel_size=(0, 0)) -> torch.Tensor:
    """
    Args:
        imgs: Images (torch.Tensor)
            size: (N, 3, 224, 224)
        sigma: Standard deviation of Gaussian kernel.
        kernel_size: This size will be automatically adjusted.
    Returns: Blurred images (torch.Tensor)
            size: (N, 3, 224, 224)
    """
    if sigma == 0:
        return imgs  # do nothing
    else:
        imgs = imgs.numpy()

        imgs_list = []
        for img in imgs:
            imgs_list.append(
                cv2.GaussianBlur(img.transpose(1, 2, 0), kernel_size, sigma)
            )

        imgs_list = np.array(imgs_list)
        # Change the order of dimension for pytorch (B, C, H, W)
        imgs_list = imgs_list.transpose(0, 3, 1, 2)

        return torch.from_numpy(imgs_list)


def RandomGaussianBlurAll(
    imgs, min_sigma, max_sigma, kernel_size=(0, 0)
) -> torch.Tensor:
    """Return Blurred images by random sigma.
    Each image is blurred by a sigma chosen by randomly from [min_sigma, max_sigma].

    Args:
        imgs: Images (torch.Tensor)
            size: (N, 3, 224, 224)
        kernel_size: This size will be automatically adjusted.
    Returns: Blurred images (torch.Tensor)
            size: (N, 3, 224, 224)
    """
    imgs = imgs.numpy()

    imgs_list = []
    for img in imgs:
        # Choose a random sigma for each image
        sigma = random.uniform(min_sigma, max_sigma)

        imgs_list.append(cv2.GaussianBlur(img.transpose(1, 2, 0), kernel_size, sigma))

    imgs_list = np.array(imgs_list)
    # Change the order of dimension for pytorch (B, C, H, W)
    imgs_list = imgs_list.transpose(0, 3, 1, 2)

    return torch.from_numpy(imgs_list)
