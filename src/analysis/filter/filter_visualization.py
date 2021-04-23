import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import models


def plot_filters_multi_channel(t, file_name):
    # get the number of kernals
    num_kernels = t.shape[0]

    # define number of columns for subplots
    num_cols = 12
    # rows = num of kernels
    # num_rows = num_kernels
    num_rows = -(-num_kernels // num_cols)  # round down

    # set the figure size
    fig = plt.figure(figsize=(num_cols, num_rows))

    # looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)

        # for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].numpy(), np.float32)
        # convert [-1, 1] to [0, 1]
        npimg = (npimg + 1) / 2
        # standardize the numpy image
        # npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        # npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis("off")
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.tight_layout()
    # mappable = ScalarMappable(cmap='viridis')  # for colorbar. default: 'viridis'
    # plt.colorbar(mappable)
    plt.savefig(file_name)
    # plt.show()
    plt.close()


def plot_filters(model, layer_num, file_name):
    # extracting the model features at the particular layer number
    layer = model.features[layer_num]
    # checking whether the layer is convolution layer or not
    if isinstance(layer, nn.Conv2d):
        # getting the weight tensor data
        weight_tensor = layer.weight.data

        if weight_tensor.shape[1] == 3:
            plot_filters_multi_channel(weight_tensor, file_name)
        else:
            print(
                "Can only plot weights with three channels with single channel = False"
            )
    else:
        print("Can only visualize layers which are convolutional")


def plot_filters_fft(model, layer_num, file_name):
    # extracting the model features at the particular layer number
    layer = model.features[layer_num]
    weight = layer.weight.data.numpy()

    # get the number of kernals
    num_kernels = weight.shape[0]

    # define number of columns for subplots
    num_cols = 12
    # rows = num of kernels
    # num_rows = num_kernels
    num_rows = -(-num_kernels // num_cols)  # round down

    # set the figure size
    fig = plt.figure(figsize=(num_cols, num_rows))

    # normaliza and cmap
    # norm = Normalize(vmin=t.min(), vmax=t.max())
    norm = None
    cmap = "gray"  # default: 'viridis'

    # looping through all the kernels
    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)

        # fft
        img = weight[i].transpose(1, 2, 0)  # change color index
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        ax1.imshow(magnitude_spectrum, origin="lower", cmap=cmap)
        ax1.axis("off")
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.tight_layout()
    mappable = ScalarMappable(norm=norm, cmap=cmap)  # for colorbar
    plt.colorbar(mappable)
    plt.savefig(file_name)
    # plt.show()
    plt.close()
