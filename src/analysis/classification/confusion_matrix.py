import seaborn as sns

import matplotlib.pyplot as plt


def plot_confusion_matrix(
    confusion_matrix, vmin=0, vmax=1, title="", out_path="", show=False
):
    sns.heatmap(confusion_matrix, vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    if out_path:
        plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()
