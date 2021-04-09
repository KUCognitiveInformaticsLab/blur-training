from typing import Union

import numpy as np
import torch


# Ref: https://www.javaer101.com/ja/article/3437291.html
def gaussian_noise(
    image: Union[np.array, torch.Tensor], mean=0, var=0.1
) -> Union[np.array, torch.Tensor]:
    """Adds Gaussian noise to input image.
    Args:
        image: an image. np.array=(H, W, C) or torch.Tensor=(C, H, W)
    Return:
        noisy: a noisy image. np.array=(H, W, C) or torch.Tensor=(C, H, W)
    """
    return_torch = False
    if type(image) == torch.Tensor:
        return_torch = True
        image = image.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    row, col, ch = image.shape

    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)

    noisy = image + gauss

    if return_torch:
        noisy = noisy.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return torch.from_numpy(noisy)
    return noisy
