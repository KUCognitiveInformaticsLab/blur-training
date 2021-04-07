import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

max_sigma = 10  # max sigma of guassian kernel to test the models with

arch = "alexnet"
epoch = 60
# directories and model settings
DATA_DIR = "./results/{}".format(arch)
OUTPUTS_DIR = "./plots/{}".format(arch)
if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)


def read_result(model_name, epoch=60, value="acc1"):
    file_path = os.path.join(DATA_DIR, "{}_e{}_{}.csv".format(model_name, epoch, value))
    return pd.read_csv(file_path, index_col=0)


def plot_common():
    ax.set_xlabel("σ (Gaussian Blur)")
    ax.set_ylabel("Top1-Accuracy (%)")
    ax.legend()
    # ax.invert_xaxis()
    # ax.set_xticks(np.arange(0, max_sigma+1, 5))
    if max_sigma in (10, 20):
        ax.set_xticks(np.arange(0, max_sigma + 1, 1))
    elif max_sigma in (30, 40, 50):
        plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(10))
    ax.grid(which="major")
    ax.grid(which="minor")
    plt.xlim(-1, max_sigma + 1)
    plt.ylim(0, 100)
    fig.show()


# read results
model_name = "{}_normal".format(arch)
normal_acc1 = read_result(model_name, epoch).values[0]


all_acc1 = []
for i in range(1, 5):
    model_name = "{}_all_s{}".format(arch, i)
    all_acc1.append(read_result(model_name, epoch).values[0])

mix_acc1 = []
for i in range(1, 5):
    model_name = "{}_mix_s{}".format(arch, i)
    mix_acc1.append(read_result(model_name, epoch).values[0])

single_step_acc1 = []
for i in range(1, 5):
    model_name = "{}_single-step_s{}".format(arch, i)
    single_step_acc1.append(read_result(model_name, epoch).values[0])

r_single_step_acc1 = []
for i in range(1, 5):
    model_name = "{}_reversed-single-step_s{}".format(arch, i)
    r_single_step_acc1.append(read_result(model_name, epoch).values[0])

model_name = "{}_multi-steps".format(arch)
multi_steps_acc1 = read_result(model_name, epoch).values[0]


# plot results
x = [i for i in range(max_sigma + 1)]

# all
fig, ax = plt.subplots(dpi=150)
plt.title("{} All".format(arch.capitalize()))
ax.plot(x, normal_acc1[: max_sigma + 1], label="normal", marker="o")
for i in range(4):
    ax.plot(x, all_acc1[i][: max_sigma + 1], label="σ={}".format(i + 1), marker="o")
plot_common()
fig.savefig(os.path.join(OUTPUTS_DIR, f"{arch}_all_e{epoch}_max-s{max_sigma}.png"))

# mix
fig, ax = plt.subplots(dpi=150)
plt.title("{} Mix".format(arch.capitalize()))
ax.plot(x, normal_acc1[: max_sigma + 1], label="normal", marker="o")
for i in range(4):
    ax.plot(x, mix_acc1[i][: max_sigma + 1], label="σ={}".format(i + 1), marker="o")
plot_common()
fig.savefig(os.path.join(OUTPUTS_DIR, f"{arch}_mix_acc1_e{epoch}_max-s{max_sigma}.png"))

# sigle-step
fig, ax = plt.subplots(dpi=150)
plt.title("{} Single-step".format(arch.capitalize()))
ax.plot(x, normal_acc1[: max_sigma + 1], label="normal", marker="o")
for i in range(4):
    ax.plot(
        x, single_step_acc1[i][: max_sigma + 1], label="σ={}".format(i + 1), marker="o"
    )
plot_common()
fig.savefig(
    os.path.join(OUTPUTS_DIR, f"{arch}_single-step_e{epoch}_max-s{max_sigma}.png")
)

# reversed-single-step
fig, ax = plt.subplots(dpi=150)
plt.title("{} Reversed-single-step".format(arch.capitalize()))
ax.plot(x, normal_acc1[: max_sigma + 1], label="normal", marker="o")
for i in range(4):
    ax.plot(
        x,
        r_single_step_acc1[i][: max_sigma + 1],
        label="σ={}".format(i + 1),
        marker="o",
    )
plot_common()
fig.savefig(
    os.path.join(
        OUTPUTS_DIR, f"{arch}_reversed-single-step_e{epoch}_max-s{max_sigma}.png"
    )
)

# multi-steps
fig, ax = plt.subplots(dpi=150)
plt.title("{} Multi-steps".format(arch.capitalize()))
ax.plot(x, normal_acc1[: max_sigma + 1], label="normal", marker="o")
ax.plot(
    x,
    multi_steps_acc1[: max_sigma + 1],
    label="multi-steps",
    marker="o",
    c="mediumvioletred",
)
plot_common()
fig.savefig(
    os.path.join(OUTPUTS_DIR, f"{arch}_multi-steps_acc1_e{epoch}_max-s{max_sigma}.png")
)


##### plot each sigma gradually #####
# all
for s in range(5):
    fig, ax = plt.subplots(dpi=150)
    plt.title("{} All".format(arch.capitalize()))
    ax.plot(x, normal_acc1[: max_sigma + 1], label="normal", marker="o")
    if s != 0:
        for i in range(s):
            ax.plot(
                x, all_acc1[i][: max_sigma + 1], label="σ={}".format(i + 1), marker="o"
            )
    plot_common()
    fig.savefig(
        os.path.join(OUTPUTS_DIR, f"{arch}_all_acc1_e{epoch}_max-s{max_sigma}_{s}.png")
    )


# mix
for s in range(5):
    fig, ax = plt.subplots(dpi=150)
    plt.title("{} Mix".format(arch.capitalize()))
    ax.plot(x, normal_acc1[: max_sigma + 1], label="normal", marker="o")
    if s != 0:
        for i in range(s):
            ax.plot(
                x, mix_acc1[i][: max_sigma + 1], label="σ={}".format(i + 1), marker="o"
            )
    plot_common()
    fig.savefig(
        os.path.join(OUTPUTS_DIR, f"{arch}_mix_acc1_e{epoch}_max-s{max_sigma}_{s}.png")
    )


# sigle-step
for s in range(5):
    fig, ax = plt.subplots(dpi=150)
    plt.title("{} Single-step".format(arch.capitalize()))
    ax.plot(x, normal_acc1[: max_sigma + 1], label="normal", marker="o")
    if s != 0:
        for i in range(s):
            ax.plot(
                x,
                single_step_acc1[i][: max_sigma + 1],
                label="σ={}".format(i + 1),
                marker="o",
            )
    plot_common()
    fig.savefig(
        os.path.join(
            OUTPUTS_DIR, f"{arch}_single-step_e{epoch}_max-s{max_sigma}_{s}.png"
        )
    )


# reversed-single-step
for s in range(5):
    fig, ax = plt.subplots(dpi=150)
    plt.title("{} Reversed-single-step".format(arch.capitalize()))
    ax.plot(x, normal_acc1[: max_sigma + 1], label="normal", marker="o")
    if s != 0:
        for i in range(s):
            ax.plot(
                x,
                r_single_step_acc1[i][: max_sigma + 1],
                label="σ={}".format(i + 1),
                marker="o",
            )
    plot_common()
    fig.savefig(
        os.path.join(
            OUTPUTS_DIR,
            f"{arch}_reversed-single-step_e{epoch}_max-s{max_sigma}_{s}.png",
        )
    )
