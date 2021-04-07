import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

arch = "alexnet"
num_classes = 16
epoch = 60

# directories and model settings
in_dir = f"./results/{num_classes}-class/{arch}"
out_dir = f"./plots/{num_classes}-class/{arch}"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def read_result(model_name, epoch=60, value="acc1"):
    file_path = os.path.join(in_dir, "{}_e{}_{}.csv".format(model_name, epoch, value))
    return pd.read_csv(file_path, index_col=0)


'''
def read_raw_acc1(model_name, epoch):
    """Return top-1 accuracy from saved model"""
    model_path = os.path.join(MODELS_DIR, model_name, 'epoch_{}.pth.tar'.format(epoch))
    checkpoint = torch.load(model_path, map_location='cpu')

    return checkpoint['val_acc'].item()
'''

colors = {
    f"{arch}_normal": "#1f77b4",
    f"{arch}_mix_s02": "darkorange",
    f"{arch}_mix_s04": "limegreen",
    f"{arch}_mix_s10": "hotpink",
    f"{arch}_random-mix_s00-05": "green",
    f"{arch}_random-mix_s00-10": "mediumvioletred",
}

lines = {
    f"{arch}_normal": ":",
    f"{arch}_mix_s02": "-",
    f"{arch}_mix_s04": "-",
    f"{arch}_mix_s10": "-",
    f"{arch}_random-mix_s00-05": "-",
    f"{arch}_random-mix_s00-10": "-",
}

# models to plot
modes = [
    "normal",
    # "all",
    "mix",
    "random-mix",
    # "single-step",
    # "fixed-single-step",
    # "reversed-single-step",
    # "multi-steps",
]

# sigmas to plot
sigmas_mix = [2, 4, 10]
sigmas_random_mix = ["00-05", "00-10"]

# make model name list
model_names = []
for mode in modes:
    if mode in ("normal", "multi-steps"):
        model_names += [f"{arch}_{mode}"]
    elif mode == "random-mix":
        for min_max in sigmas_random_mix:
            model_names += [f"{arch}_{mode}_s{min_max}"]
    elif mode == "mix":
        for sigma in sigmas_mix:
            model_names += [f"{arch}_{mode}_s{sigma:02d}"]
    else:
        for s in range(4):
            model_names += [f"{arch}_{mode}_s{s + 1:02d}"]

# read band-pass accuracy results
acc1 = {}
for model_name in model_names:
    acc1[model_name] = read_result(model_name, epoch).values[0]

x = ["σ{}-σ{}".format(2 ** i, 2 ** (i + 1)) for i in range(4)] + ["σ16-"]
x.insert(0, "σ0 - σ1")
x.insert(0, "raw(σ0)")


fig = plt.figure(dpi=150)
ax = fig.add_subplot(
    1,
    1,
    1,
    title="Top-1 Accuracy of Band-Pass Images, {} (16-class)".format(
        arch.capitalize(),
    ),
    xlabel="Test images",
    ylabel="Top-1 accuracy",
    ylim=(0, 1),
)
for model_name in model_names:
    ax.plot(x[0], acc1[model_name][0], marker="o", color=colors[model_name])
    ax.plot(
        x[1:],
        acc1[model_name][1:],
        label=model_name,
        marker="o",
        color=colors[model_name],
    )

ax.legend()
# ax.set_xticks(np.arange(0, max_sigma+1, 5))
plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(10))
# ax.xaxis.set_major_locator(tick.MultipleLocator(1))
ax.grid(which="major")
ax.grid(which="minor")
# plt.xlim()
plt.ylim(0, 100)
fig.show()
filename = "bandpass-acc1_{}_mix_s5-10_e{}.png".format(arch, epoch)
fig.savefig(os.path.join(out_dir, filename))
