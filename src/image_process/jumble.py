import numpy as np

from .block import split_image_cut


def jumble_images(imgs, div_v: int, div_h: int):
    """
    Args:
        imgs (torch.Tensor): Images
            size: (N, C, H, W)
        div_v: # of vertical splits
        div_h: # of horizontal splits

    Returns:
        imgs (torch.Tensor): Jumbled images
            Blocks randomly permuted horizontally.
            "imgs" and "blocks" have shared memory (unless copy() is used).
            size: (N, C, H, W)
    """
    for img in imgs:
        blocks = split_image_cut(
            img.numpy().transpose(1, 2, 0), div_v=div_v, div_h=div_h
        ).squeeze()

        for v in range(div_v):
            # blocks randomly permuted horizontally
            blocks[v] = blocks[v, np.random.permutation(div_v), ...]

    return imgs


def jumble_images_with_glay_occluder(imgs, div_v: int, div_h: int):
    """
    Args:
        imgs (torch.Tensor): Images
            size: (N, C, H, W)
        div_v: # of vertical splits
        div_h: # of horizontal splits

    Returns:
        imgs (torch.Tensor): Jumbled images with gray occluder.
            Blocks randomly permuted horizontally.
            "imgs" and "blocks" have shared memory (unless copy() is used).
            size: (N, C, H, W)
    """
    gray_idx_0 = [i for i in range(0, div_v, 2)]
    gray_idx_1 = [i for i in range(1, div_v, 2)]

    for img in imgs:
        blocks = split_image_cut(
            img.numpy().transpose(1, 2, 0), div_v=div_v, div_h=div_h
        ).squeeze()

        for v in range(0, div_v, 2):
            blocks[v] = blocks[v, np.random.permutation(div_v), ...]
            blocks[v, gray_idx_0, ...] = 0.5  # make blocks gray (every one block)

        for v in range(1, div_v, 2):
            blocks[v] = blocks[v, np.random.permutation(div_v), ...]
            blocks[v, gray_idx_1, ...] = 0.5  # make blocks gray (every one block)

    return imgs
