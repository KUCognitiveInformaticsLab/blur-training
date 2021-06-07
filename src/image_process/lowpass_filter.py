import random

import cv2
import numpy as np
import torch


def GaussianBlurAll(imgs, sigma, kernel_size=(0, 0)) -> torch.Tensor:
    """
    Args:
        imgs: Images (torch.Tensor)
            size: (N, C, H, W)
        sigma: Standard deviation of Gaussian kernel.
        kernel_size: This size will be automatically adjusted.
    Returns: Blurred images (torch.Tensor)
            size: (N, C, H, W)
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


def GaussianBlurAllNotInExcludedLabels(
    images, labels, excluded_labels=[], sigma=1, kernel_size=(0, 0)
) -> torch.Tensor:
    """
    Args:
        images (torch.Tensor): Images, size: (N, C, H, W)
        labels (torch.Tensor): Labels, size: (N)
        sigma: Standard deviation of Gaussian kernel.
        kernel_size: This size will be automatically adjusted.
    Returns: Blurred images (torch.Tensor)
            size: (N, C, H, W)
    """
    if sigma == 0:
        return images  # do nothing
    else:
        images = images.numpy()

        images_list = []
        for image, label in zip(images, labels):
            if label not in excluded_labels:
                images_list.append(
                    cv2.GaussianBlur(image.transpose(1, 2, 0), kernel_size, sigma)
                )  # blur
            else:
                images_list.append(image.transpose(1, 2, 0))  # sharp (no blur)

        images_list = np.array(images_list)
        # Change the order of dimension for pytorch (B, C, H, W)
        images_list = images_list.transpose(0, 3, 1, 2)

        return torch.from_numpy(images_list)


def GaussianBlurAllInExcludedLabels(
    images, labels, excluded_labels=[], sigma=1, kernel_size=(0, 0)
) -> torch.Tensor:
    """
    Args:
        images (torch.Tensor): Images, size: (N, C, H, W)
        labels (torch.Tensor): Labels, size: (N)
        sigma: Standard deviation of Gaussian kernel.
        kernel_size: This size will be automatically adjusted.
    Returns: Blurred images (torch.Tensor)
            size: (N, C, H, W)
    """
    if sigma == 0:
        return images  # do nothing
    else:
        images = images.numpy()

        images_list = []
        for image, label in zip(images, labels):
            if label in excluded_labels:
                images_list.append(
                    cv2.GaussianBlur(image.transpose(1, 2, 0), kernel_size, sigma)
                )  # blur
            else:  # if label not in excluded_labels
                images_list.append(image.transpose(1, 2, 0))  # sharp (no blur)

        images_list = np.array(images_list)
        # Change the order of dimension for pytorch (B, C, H, W)
        images_list = images_list.transpose(0, 3, 1, 2)

        return torch.from_numpy(images_list)


def GaussianBlurAllRandomSigma(
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


def GaussianBlurProb(images, sigma, p_blur, kernel_size=(0, 0)) -> torch.Tensor:
    """
    Args:
        images (torch.Tensor): Images, size: (N, C, H, W)
        sigma: Standard deviation of Gaussian kernel.
        p_blur: percentage of blur
        kernel_size: This size will be automatically adjusted.
    Returns: Blurred images (torch.Tensor)
            size: (N, C, H, W)
    """
    if sigma == 0:
        return images  # do nothing
    else:
        images = images.numpy()

        images_list = []
        for image in images:
            if np.random.choice([0, 1], 1, p=[1 - p_blur, p_blur]):
                images_list.append(
                    cv2.GaussianBlur(image.transpose(1, 2, 0), kernel_size, sigma)
                )  # blur
            else:
                images_list.append(image.transpose(1, 2, 0))  # no blur

        images_list = np.array(images_list)
        # Change the order of dimension for pytorch (B, C, H, W)
        images_list = images_list.transpose(0, 3, 1, 2)

        return torch.from_numpy(images_list)


def GaussianBlurProbExcludeLabels(
    images,
    labels,
    p_blur,
    sigma,
    excluded_labels=[],
    min_sigma=0,
    max_sigma=0,
    kernel_size=(0, 0),
) -> torch.Tensor:
    """
    Args:
        images (torch.Tensor): Images, size: (N, C, H, W)
        labels (torch.Tensor): Labels, size: (N)
        excluded_labels (list): Excluded labels
        p_blur: percentage of blur
        sigma: Standard deviation of Gaussian kernel. Or set to "random".
        min_sigma: for random sigma
        max_sigma: for random sigma
        kernel_size: This size will be automatically adjusted.
    Returns: Blurred images (torch.Tensor)
            size: (N, C, H, W)
    """
    if sigma == 0:
        return images  # do nothing
    else:
        images = images.numpy()

        images_list = []
        for image, label in zip(images, labels):
            if label not in excluded_labels and np.random.choice(
                [0, 1], 1, p=[1 - p_blur, p_blur]
            ):
                if sigma == "random":  # blur (random sigma)
                    images_list.append(
                        cv2.GaussianBlur(
                            image.transpose(1, 2, 0),
                            kernel_size,
                            sigmaX=random.uniform(min_sigma, max_sigma),  # random sigma
                        )  # sigmaY will be same as sigmaX
                    )
                else:  # blur
                    images_list.append(
                        cv2.GaussianBlur(
                            image.transpose(1, 2, 0), kernel_size, sigmaX=sigma
                        )  # sigmaY will be same as sigmaX
                    )

            else:  # no blur
                images_list.append(image.transpose(1, 2, 0))

        images_list = np.array(images_list)
        # Change the order of dimension for pytorch (B, C, H, W)
        images_list = images_list.transpose(0, 3, 1, 2)

        return torch.from_numpy(images_list)
