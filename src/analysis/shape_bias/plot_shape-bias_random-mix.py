# # import, functions

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch


# # arguments, settings

# In[2]:


arch = "alexnet"
epoch = 60

MODELS_DIR = "../../logs/models/"  # trained models directory
DATA_DIR = f"./results/{arch}"
OUTPUTS_DIR = f"./plots/{arch}_mix-s5-10_random-mix"

assert os.path.exists(MODELS_DIR), "The path does not exist."
assert os.path.exists(DATA_DIR), "The path does not exist."
if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)


# In[7]:


filename = f"shape-bias_acc1_e{epoch}.png"

# modes to compare
modes = [
    "normal",
    #     "all",
    "mix",
    "random-mix",
    "single-step",
    #     "fixed-single-step",
    #     "reversed-single-step",
    "multi-steps",
    "SIN",
    "vonenet",
]

# sigmas to compare
sigmas_mix = [s for s in range(1, 6)] + [10]
sigmas_random_mix = ["00-05", "00-10"]

sin_names = {
    "alexnet": "alexnet_trained_on_SIN",
    "vgg16": "vgg16_trained_on_SIN",
    "resnet50": "resnet50_trained_on_SIN",
}


# In[8]:


colors = {
    "normal": "#1f77b4",
    "all": "yellowgreen",
    "mix": "darkgreen",
    "random-mix": "deepskyblue",
    "single-step": "orchid",
    "reversed-single-step": "plum",
    "fixed-single-step": "grey",
    "multi-steps": "mediumvioletred",
    "SIN": "darkorange",
    "vonenet": "brown",
}


# # functions

# In[9]:


def get_shape_bias(model_name, epoch):
    if epoch == 0:  # for pre-trained model
        file_path = os.path.join(
            DATA_DIR, "correct_decisions_{}.csv".format(model_name)
        )
    else:
        file_path = os.path.join(
            DATA_DIR, "correct_decisions_{}_e{}.csv".format(model_name, epoch)
        )
    correct_decisions = pd.read_csv(file_path, index_col=0).values
    # compute shape bias
    shape_bias = correct_decisions[0].sum() / (
        correct_decisions[0].sum() + correct_decisions[1].sum()
    )
    return shape_bias


def get_acc1(model_name, epoch):
    """Return top-1 accuracy from saved model"""
    model_path = os.path.join(MODELS_DIR, model_name, "epoch_{}.pth.tar".format(epoch))
    checkpoint = torch.load(model_path, map_location="cpu")

    return checkpoint["val_acc"].item() / 100


# # main

# ## compute shape bias

# In[10]:


shape_bias = {}

for mode in modes:
    if mode in ("normal", "multi-steps"):
        model_name = f"{arch}_{mode}"
        shape_bias[model_name] = get_shape_bias(model_name, epoch)
    elif mode == "mix":
        for sigma in sigmas_mix:
            model_name = f"{arch}_{mode}_s{sigma:02d}"
            shape_bias[model_name] = get_shape_bias(model_name, epoch)
    elif mode == "random-mix":
        for min_max in sigmas_random_mix:
            model_name = f"{arch}_{mode}_s{min_max}"
            shape_bias[model_name] = get_shape_bias(model_name, epoch)
    elif mode == "SIN":
        model_name = sin_names[arch]
        shape_bias[model_name] = get_shape_bias(model_name, epoch=0)
    elif mode == "vonenet":
        model_name = f"{arch}_{mode}"
        shape_bias[model_name] = get_shape_bias(model_name, epoch=0)
    else:
        for sigma in range(1, 5):
            model_name = f"{arch}_{mode}_s{sigma:02d}"
            shape_bias[model_name] = get_shape_bias(model_name, epoch)

# dictionary of shape bias of SIN
# sb_sin = {'alexnet':0.755, 'vgg16':0.78, 'resnet50':0.8137}


# In[11]:


shape_bias


# ## get top-1 accuracy

# In[12]:


acc1 = {}

for mode in modes:
    if mode in ("normal", "multi-steps"):
        model_name = f"{arch}_{mode}"
        acc1[model_name] = get_acc1(model_name, epoch)
    elif mode == "mix":
        for sigma in sigmas_mix:
            model_name = f"{arch}_{mode}_s{sigma:02d}"
            acc1[model_name] = get_acc1(model_name, epoch)
    elif mode == "random-mix":
        for min_max in sigmas_random_mix:
            model_name = f"{arch}_{mode}_s{min_max}"
            acc1[model_name] = get_acc1(model_name, epoch)
    elif mode == "SIN":
        pass
    #         model_name = sin_names[arch]
    #         acc1[model_name] = get_acc1(model_name, epoch=0)
    elif mode == "vonenet":
        pass
    #         model_name = f"{arch}_{mode}"
    #         acc1[model_name] = get_acc1(model_name, epoch=0)
    else:
        for sigma in range(1, 5):
            model_name = f"{arch}_{mode}_s{sigma:02d}"
            acc1[model_name] = get_acc1(model_name, epoch)


# In[13]:


acc1


# ## plot

# In[16]:


fig = plt.figure(dpi=150)
ax = fig.add_subplot(
    1,
    1,
    1,
    title="{} (16-class)".format(arch).capitalize(),
    xlabel="Models",
    ylabel="Shape bias, Top-1 accuracy",
    ylim=(0, 1),
)
# plot shape bias & acc1
for mode in modes:
    if mode in ("normal", "multi-steps"):
        model_name = f"{arch}_{mode}"
        ax.bar(
            model_name,
            shape_bias[model_name],
            color=colors[mode],
        )
        ax.plot(
            model_name,
            acc1[model_name],
            marker="+",
            linestyle="None",
            color="k",
        )
    elif mode == "mix":
        for sigma in sigmas_mix:
            model_name = f"{arch}_{mode}_s{sigma:02d}"
            ax.bar(
                model_name,
                shape_bias[model_name],
                color=colors[mode],
            )
            ax.plot(
                model_name,
                acc1[model_name],
                marker="+",
                linestyle="None",
                color="k",
            )
    elif mode == "random-mix":
        for min_max in sigmas_random_mix:
            model_name = f"{arch}_{mode}_s{min_max}"
            ax.bar(
                model_name,
                shape_bias[model_name],
                color=colors[mode],
            )
            ax.plot(
                model_name,
                acc1[model_name],
                marker="+",
                linestyle="None",
                color="k",
            )
    elif mode == "SIN":
        model_name = sin_names[arch]
        ax.bar(
            model_name,
            shape_bias[model_name],
            color=colors[mode],
        )
        # ax.plot(model_name, acc1[model_name], marker="+", linestyle="None", color="k")
    elif mode == "vonenet":
        model_name = f"{arch}_{mode}"
        ax.bar(
            model_name,
            shape_bias[model_name],
            color=colors[mode],
        )
        # ax.plot(model_name, acc1[model_name], marker="+", linestyle="None", color="k")
    else:
        for sigma in range(1, 5):
            model_name = f"{arch}_{mode}_s{sigma:02d}"
            ax.bar(
                model_name,
                shape_bias[model_name],
                color=colors[mode],
            )
            ax.plot(
                model_name,
                acc1[model_name],
                marker="+",
                linestyle="None",
                color="k",
            )

x = [key.replace(f"{arch}_", "") for key in shape_bias.keys()]
ax.set_xticklabels(x, rotation=45, ha="right")
# ax.set_xticklabels(shape_bias.keys(), rotation=45, ha="right")
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.grid(ls=":")
fig.show()
fig.savefig(os.path.join(OUTPUTS_DIR, filename), bbox_inches="tight")


# In[ ]:
