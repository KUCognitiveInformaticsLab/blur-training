import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def imshow(
    img, unnormalize=False, dpi=None, scale=True, title="", filename="", save_raw=False
):
    if unnormalize:
        img = img / 4 + 0.5

    if type(img) == torch.Tensor:
        img = img.numpy().transpose(1, 2, 0)

    if dpi:
        plt.figure(dpi=dpi)

    plt.imshow(img)

    if not scale:
        plt.xticks([])  # if you want to remove scale axes
        plt.yticks([])

    if title:
        plt.title(title)

    if filename:
        if save_raw:
            # clipping
            img = np.where(img > 1, 1, img)
            img = np.where(img < 0, 0, img)

            matplotlib.image.imsave(filename, img)  # save the raw image

        else:
            plt.savefig(filename)

    plt.show()


def imsave(img, filename, unnormalize=False):
    if unnormalize:
        img = img / 4 + 0.5

    if type(img) == torch.Tensor:
        img = img.numpy().transpose(1, 2, 0)

    # clipping
    img = np.where(img > 1, 1, img)
    img = np.where(img < 0, 0, img)

    matplotlib.image.imsave(filename, img)  # save the raw image
